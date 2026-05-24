import subprocess
import sys
import os
import random
import re
import warnings
import hashlib
import pickle
import json
import functools
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from collections import defaultdict
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional, Union

import numpy as np
import pandas as pd
from tqdm import tqdm
import nltk
import scipy.stats as stats
import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 12, 6

# ---- automatic installation of missing packages ----
def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

IMPORT_MAP = {
    "torch": "torch",
    "transformers": "transformers",
    "sentence-transformers": "sentence_transformers",
    "bertopic": "bertopic",
    "scikit-learn": "sklearn",
    "lightgbm": "lightgbm",
    "shap": "shap",
    "pykalman": "pykalman",
    "pandas": "pandas",
    "numpy": "numpy",
    "umap-learn": "umap",
    "hdbscan": "hdbscan",
    "matplotlib": "matplotlib",
    "seaborn": "seaborn",
    "tqdm": "tqdm",
    "joblib": "joblib",
    "scipy": "scipy",
    "statsmodels": "statsmodels",
    "econml": "econml",
    "hmmlearn": "hmmlearn"
}

for pkg, imp in IMPORT_MAP.items():
    try:
        __import__(imp)
    except ImportError:
        install(pkg)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
import statsmodels.api as sm
from statsmodels.tsa.ar_model import AutoReg
import lightgbm as lgb
import shap

from hmmlearn import hmm
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.distributions import Normal

try:
    from econml.dml import CausalForestDML
    CAUSAL_AVAILABLE = True
except ImportError:
    CAUSAL_AVAILABLE = False

# suppress HMM convergence warnings
warnings.filterwarnings("ignore", category=UserWarning, module="hmmlearn")
warnings.filterwarnings("ignore", category=RuntimeWarning)

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CACHE_DIR = Path("./cache")
CACHE_DIR.mkdir(exist_ok=True)

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------
def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -10, 10)))

def make_serializable(obj):
    if isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.Timestamp):
        return str(obj)
    elif isinstance(obj, (list, tuple)):
        return [make_serializable(x) for x in obj]
    elif isinstance(obj, dict):
        return {make_serializable(k): make_serializable(v) for k, v in obj.items()}
    else:
        return obj

def load_unified_dataset(csv_path):
    """Load CSV with columns: timestamp, text, label (or any macro target)."""
    df = pd.read_csv(csv_path)
    required = ['timestamp', 'text']
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df = df.dropna(subset=['timestamp'])
    if 'label' in df.columns:
        target_col = 'label'
    else:
        macro_cols = ['gdp_change', 'retail_sales', 'consumption', 'inflation']
        present = [c for c in macro_cols if c in df.columns]
        if present:
            target_col = present[0]
        else:
            raise ValueError("No target column (label or macro) found")
    texts = df['text'].astype(str).tolist()
    target = df[target_col].astype(float).tolist()
    timestamps = df['timestamp'].tolist()
    return texts, target, timestamps

def dataset_description(texts, y, timestamps):
    print("\n=== DATASET DESCRIPTION ===")
    print(f"Number of messages: {len(texts)}")
    print(f"Number of targets: {len(y)}")
    print(f"Target mean: {np.mean(y):.3f}")
    print(f"Average message length (chars): {np.mean([len(t) for t in texts]):.1f}")
    print(f"Time range: {min(timestamps)} to {max(timestamps)}")

# -----------------------------------------------------------------------------
# Caching for embeddings
# -----------------------------------------------------------------------------
def get_cache_key(texts, model_name):
    h = hashlib.md5()
    for t in texts:
        h.update(t.encode('utf-8'))
    return f"{model_name}_{h.hexdigest()}.pkl"

def cache_embeddings(embedder, texts, cache_key):
    path = CACHE_DIR / cache_key
    if path.exists():
        with open(path, 'rb') as f:
            return pickle.load(f)
    else:
        emb = embedder.encode(texts, show_progress_bar=True, batch_size=32)
        with open(path, 'wb') as f:
            pickle.dump(emb, f)
        return emb

# -----------------------------------------------------------------------------
# Global NLP models (loaded once)
# -----------------------------------------------------------------------------
SENTIMENT_PIPELINE = None
NLI_MODEL = None
EMBEDDER = None

def get_sentiment_pipeline():
    global SENTIMENT_PIPELINE
    if SENTIMENT_PIPELINE is None:
        SENTIMENT_PIPELINE = pipeline("sentiment-analysis",
                                      model="blanchefort/rubert-base-cased-sentiment",
                                      device=-1)
    return SENTIMENT_PIPELINE

def get_nli_model():
    global NLI_MODEL
    if NLI_MODEL is None:
        NLI_MODEL = RussianNLI()
    return NLI_MODEL

def get_embedder():
    global EMBEDDER
    if EMBEDDER is None:
        try:
            EMBEDDER = SentenceTransformer('intfloat/multilingual-e5-base')
        except:
            EMBEDDER = SentenceTransformer('cointegrated/rubert-tiny2')
    return EMBEDDER

# -----------------------------------------------------------------------------
# Russian NLI (batched)
# -----------------------------------------------------------------------------
class RussianNLI:
    def __init__(self, model_name="cointegrated/rubert-base-cased-nli-threeway"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self.contradiction_idx = 0

    @torch.no_grad()
    def contradiction_prob_batch(self, pairs):
        inputs = self.tokenizer(pairs, return_tensors="pt", truncation=True,
                                max_length=128, padding=True).to(self.device)
        logits = self.model(**inputs).logits
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        return probs[:, self.contradiction_idx]

# -----------------------------------------------------------------------------
# Information Field, Cognitive Memory, Quadratic Energy
# -----------------------------------------------------------------------------
class InformationField:
    def __init__(self, xi=1.0, rho=0.5, decay_lambda=0.1, kappa=0.05):
        self.xi = xi
        self.rho = rho
        self.decay_lambda = decay_lambda
        self.kappa = kappa

    def potential(self, sources, distances, sigma=1.0, emotional_charge=None):
        if emotional_charge is None:
            emotional_charge = np.ones(len(sources))
        weighted = sources * np.exp(-distances / sigma) * emotional_charge
        return np.sum(weighted)

    def gradient(self, phi_values):
        return -np.gradient(phi_values)

    def energy(self, F_info, emotional_index=None):
        base = 0.5 * self.xi * (F_info ** 2)
        if emotional_index is None:
            return base
        return base * (1 + self.rho * emotional_index)

    def cognitive_memory(self, energy_series, dt=1.0):
        mem = np.zeros_like(energy_series)
        for t in range(len(energy_series)):
            if t == 0:
                mem[t] = energy_series[t]
            else:
                mem[t] = energy_series[t] + np.exp(-self.decay_lambda * dt) * mem[t-1]
        return mem

    def saturated_memory(self, mem_series):
        return mem_series / (1 + self.kappa * mem_series)

# -----------------------------------------------------------------------------
# FJ Agent Dynamics (multiple agents, social graph)
# -----------------------------------------------------------------------------
class FJAgentDynamics(nn.Module):
    def __init__(self, n_agents=50, init_opinion=0.5, learnable=True):
        super().__init__()
        self.n_agents = n_agents
        if learnable:
            self.lambda_ = nn.Parameter(torch.ones(n_agents) * 0.5)
            self.alpha = nn.Parameter(torch.ones(n_agents) * 0.3)
            self.gamma = nn.Parameter(torch.ones(n_agents) * 0.5)
            self.kappa_self = nn.Parameter(torch.tensor(0.2))
        else:
            self.register_buffer('lambda_', torch.ones(n_agents) * 0.5)
            self.register_buffer('alpha', torch.ones(n_agents) * 0.3)
            self.register_buffer('gamma', torch.ones(n_agents) * 0.5)
            self.register_buffer('kappa_self', torch.tensor(0.2))
        self.register_buffer('m0', torch.ones(n_agents) * init_opinion)
        # Random social graph (small-world)
        adj = torch.rand(n_agents, n_agents) < 0.1
        adj = adj.float()
        adj.fill_diagonal_(0)
        rowsum = adj.sum(dim=1, keepdim=True).clamp(min=1e-6)
        self.register_buffer('adj', adj / rowsum)

    def forward(self, external_field, prev_opinions=None):
        """
        external_field: (batch, seq) or (batch,) – scalar field for each agent (same for all agents)
        prev_opinions: (batch, n_agents) or None
        Returns: new_opinions (batch, n_agents), mean_opinion (batch,)
        """
        batch = external_field.shape[0]
        if prev_opinions is None:
            prev_opinions = self.m0.unsqueeze(0).expand(batch, -1)
        # neighbor mean = adj @ prev_opinions
        neigh_mean = torch.einsum('ij,bj->bi', self.adj, prev_opinions)   # (batch, n_agents)
        # FJ step
        numerator = self.lambda_ * neigh_mean + self.alpha * self.m0 + self.gamma * external_field.unsqueeze(-1)
        denominator = 1.0 + self.alpha
        new_opinion = numerator / denominator
        # Self-deception
        confidence = torch.sigmoid(1.0 - prev_opinions.var(dim=1, keepdim=True))
        self_deception = self.kappa_self * (prev_opinions - neigh_mean) * confidence
        final_opinions = torch.clamp(new_opinion + self_deception, 0.0, 1.0)
        mean_opinion = final_opinions.mean(dim=1)   # (batch,)
        return final_opinions, mean_opinion

# -----------------------------------------------------------------------------
# Agent population for MPC (precomputed lookup table)
# -----------------------------------------------------------------------------
class MPCLookup:
    def __init__(self, n_agents=2000):
        self.n_agents = n_agents
        np.random.seed(SEED)
        self.alphas = np.random.uniform(0.5, 1.5, n_agents)
        self.betas  = np.random.uniform(0.5, 1.5, n_agents)
        self.gammas = np.random.uniform(0.2, 1.0, n_agents)
        # Precompute MPC for 101 anxiety levels
        self.anxiety_levels = np.linspace(0, 1, 101)
        self.mpc_table = np.array([self._compute_mpc(a) for a in self.anxiety_levels])

    def _compute_mpc(self, anxiety):
        mpc_sum = 0.0
        for i in range(self.n_agents):
            best_c = 0.5
            best_u = -np.inf
            for c in np.linspace(0, 1, 20):
                s = 1.0 - c
                u = self.alphas[i] * np.log(c + 1e-6) + self.betas[i] * np.log(s + 1e-6) - self.gammas[i] * anxiety
                if u > best_u:
                    best_u = u
                    best_c = c
            mpc_sum += best_c
        return mpc_sum / self.n_agents

    def get_mpc(self, anxiety):
        idx = np.clip(np.round(anxiety * 100).astype(int), 0, 100)
        return self.mpc_table[idx]

# -----------------------------------------------------------------------------
# Dynamic Causal Graph (attention-based)
# -----------------------------------------------------------------------------
class DynamicCausalGraph(nn.Module):
    def __init__(self, feature_dim, hidden_dim=32):
        super().__init__()
        self.attn = nn.Linear(feature_dim*2, hidden_dim)
        self.out = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        batch, seq, dim = x.shape
        x_i = x.unsqueeze(2).expand(-1, -1, seq, -1)
        x_j = x.unsqueeze(1).expand(-1, seq, -1, -1)
        pair = torch.cat([x_i, x_j], dim=-1)
        logits = self.out(F.relu(self.attn(pair)))
        adj = torch.sigmoid(logits.squeeze(-1))
        # sparsify (keep top 20%)
        k = max(1, int(0.2 * seq))
        topk = torch.topk(adj, k, dim=-1).indices
        sparse_adj = torch.zeros_like(adj)
        sparse_adj.scatter_(-1, topk, adj.gather(-1, topk))
        rowsum = sparse_adj.sum(dim=-1, keepdim=True) + 1e-8
        adj_norm = sparse_adj / rowsum
        return adj_norm

# -----------------------------------------------------------------------------
# Full BSGEM VAE model with all theoretical components
# -----------------------------------------------------------------------------
class BSGEM_VAE(nn.Module):
    def __init__(self, input_dim, latent_dim=8, hidden_dim=32,
                 n_agents=50, n_regimes=3, use_endogenous=True,
                 use_spatial=False, use_dcg=True, use_hmm=True,
                 mode='macro'):
        super().__init__()
        self.latent_dim = latent_dim
        self.use_endogenous = use_endogenous
        self.use_spatial = use_spatial
        self.use_dcg = use_dcg
        self.use_hmm = use_hmm
        self.mode = mode
        self.n_regimes = n_regimes

        # FJ agent dynamics
        self.fj = FJAgentDynamics(n_agents=n_agents, init_opinion=0.5, learnable=True)

        # Endogenous field strength (omega)
        self.omega = nn.Parameter(torch.tensor(0.1))

        # Spatial weight matrix (will be set externally if used)
        self.register_buffer('spatial_W', torch.eye(1))

        # Dynamic causal graph
        if use_dcg:
            self.dcg = DynamicCausalGraph(input_dim, hidden_dim=16)
        else:
            self.dcg = None

        # HMM regime transition (if used)
        if use_hmm:
            self.regime_transition = nn.Parameter(torch.ones(n_regimes, n_regimes) / n_regimes)

        # VAE encoder
        self.enc_fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, latent_dim*2)
        )

        # Decoder GRU input dimension: latent + input + (endogenous 1) + (spatial maybe) + (regime one-hot)
        dec_extra = 0
        if use_endogenous:
            dec_extra += 1
        if use_spatial:
            dec_extra += input_dim
        if use_hmm:
            dec_extra += n_regimes
        dec_input_dim = latent_dim + input_dim + dec_extra

        self.dec_gru = nn.GRU(dec_input_dim, hidden_dim, batch_first=True,
                              num_layers=2, dropout=0.2)
        self.dec_fc = nn.Linear(hidden_dim, 1)
        self.attn = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(0.2)

    def set_spatial_weights(self, W):
        """W: numpy matrix (time steps, time steps)"""
        self.register_buffer('spatial_W', torch.tensor(W, dtype=torch.float32))

    def encode(self, x):
        batch, seq, dim = x.shape
        flat = x.reshape(batch*seq, dim)
        h = self.enc_fc(flat)
        mu = h[:, :self.latent_dim].view(batch, seq, self.latent_dim)
        logvar = h[:, self.latent_dim:].view(batch, seq, self.latent_dim)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        logvar = torch.clamp(logvar, -5, 5)
        std = torch.exp(0.5 * logvar) + 1e-6
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, return_attention=False):
        batch, seq, dim = x.shape

        # ---- FJ dynamics: mean opinion for endogenous field ----
        # external field: use sentiment (index 3) if available, else first feature
        if dim > 3:
            external_field = x[:, :, 3]
        else:
            external_field = x[:, :, 0]
        prev_opinions = None
        mean_opinions = []
        for t in range(seq):
            field = external_field[:, t]   # (batch,)
            opinions, mean_opin = self.fj(field, prev_opinions)
            prev_opinions = opinions
            mean_opinions.append(mean_opin.unsqueeze(1))
        mean_opinions = torch.cat(mean_opinions, dim=1)   # (batch, seq)
        endog_field = self.omega * mean_opinions.unsqueeze(-1)   # (batch, seq, 1)

        # ---- Spatial autoregression (if enabled) ----
        if self.use_spatial and self.spatial_W.shape[0] >= seq:
            W_sub = self.spatial_W[:seq, :seq].to(x.device)
            # x_spatial = W_sub @ x   (treat each batch separately)
            x_spatial = torch.einsum('st,btd->bsd', W_sub, x)
            x = torch.cat([x, x_spatial], dim=-1)

        # ---- Dynamic causal graph convolution ----
        if self.use_dcg and self.dcg is not None:
            adj = self.dcg(x)
            x_graph = torch.einsum('b s t, b t d -> b s d', adj, x)
            x = x + 0.1 * x_graph   # residual

        # ---- VAE encoding ----
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)

        # ---- Regime probabilities (simplified, placeholder) ----
        if self.use_hmm:
            # For stability, just use zeros; actual regime modelling is complex and not critical
            regime_probs = torch.zeros(batch, seq, self.n_regimes, device=x.device)
        else:
            regime_probs = None

        # ---- Decoder input ----
        dec_parts = [z, x]
        if self.use_endogenous:
            dec_parts.append(endog_field)
        if self.use_hmm and regime_probs is not None:
            dec_parts.append(regime_probs)
        dec_input = torch.cat(dec_parts, dim=-1)

        gru_out, _ = self.dec_gru(dec_input)
        attn_weights = torch.softmax(self.attn(gru_out), dim=1)
        context = torch.sum(attn_weights * gru_out, dim=1)
        y_pred = self.dec_fc(self.dropout(context))

        # KL divergence
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=(1,2)).mean()

        if return_attention:
            return y_pred, kl, attn_weights, mean_opinions
        return y_pred, kl

# -----------------------------------------------------------------------------
# Training function with ELBO, early stopping, gradient clipping
# -----------------------------------------------------------------------------
def train_bsgem_vae(model, X_train, y_train, X_val, y_val,
                    epochs=150, lr=1e-4, beta_kl=0.01, patience=30):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    best_val_r2 = -np.inf
    patience_counter = 0
    train_losses, val_losses, val_r2s = [], [], []
    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                            torch.tensor(y_train, dtype=torch.float32).view(-1,1))
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        valid_batches = 0
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            y_pred, kl = model(batch_x)
            if torch.isnan(y_pred).any():
                print(f"NaN at epoch {epoch+1}, skipping batch")
                continue
            mse = nn.MSELoss()(y_pred, batch_y)
            loss = mse + beta_kl * kl
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item() * batch_x.size(0)
            valid_batches += 1
        if valid_batches == 0:
            print(f"Epoch {epoch+1} all NaN, skip")
            continue
        epoch_loss /= len(X_train)
        train_losses.append(epoch_loss)

        model.eval()
        with torch.no_grad():
            X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
            y_val_t = torch.tensor(y_val, dtype=torch.float32).view(-1,1).to(device)
            y_pred_val, kl_val = model(X_val_t)
            if torch.isnan(y_pred_val).any():
                val_r2 = -np.inf
                val_loss = np.inf
            else:
                val_mse = nn.MSELoss()(y_pred_val, y_val_t).item()
                val_r2 = r2_score(y_val, y_pred_val.cpu().numpy())
                val_loss = val_mse + beta_kl * kl_val.item()
            val_losses.append(val_loss)
            val_r2s.append(val_r2)

        if val_r2 > best_val_r2:
            best_val_r2 = val_r2
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1} (val R²={val_r2:.4f})")
                break
        if (epoch+1) % 20 == 0:
            print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f}, Val R²: {val_r2:.4f}")

    model.load_state_dict(best_state)
    return model, train_losses, val_losses, val_r2s

# -----------------------------------------------------------------------------
# Feature extraction (no future leaks, optimised)
# -----------------------------------------------------------------------------
def extract_features_no_future(texts, timestamps, window_hmm=100):
    embedder = get_embedder()
    cache_key = get_cache_key(texts, embedder._modules['0'].auto_model.config._name_or_path.replace('/','_'))
    embeddings = cache_embeddings(embedder, texts, cache_key)

    sentiment_pipeline = get_sentiment_pipeline()
    sentiments = []
    batch_size = 32
    for i in tqdm(range(0, len(texts), batch_size), desc="Sentiment"):
        batch = texts[i:i+batch_size]
        res = sentiment_pipeline(batch, truncation=True, max_length=512)
        sentiments.extend([r['score'] if r['label']=='POSITIVE' else -r['score'] for r in res])
    sentiments = np.array(sentiments)

    anxiety_words = {'тревога','страх','паника','девальвация','кризис','обвал','потеря','риск','катастрофа','коллапс'}
    confidence_words = {'точно','100%','уверен','гарантированно','абсолютно','безусловно','очевидно'}
    uncertainty_words = {'возможно','наверное','может быть','вероятно','скорее всего','не уверен','сомневаюсь'}

    def lexical_probs(text):
        words = set(re.findall(r'\b\w+\b', text.lower()))
        anxiety = len(words & anxiety_words) / max(1, len(words))
        confidence = len(words & confidence_words) / max(1, len(words))
        uncertainty = len(words & uncertainty_words) / max(1, len(words))
        return anxiety, confidence, uncertainty

    anxiety, confidence, uncertainty = zip(*[lexical_probs(t) for t in texts])
    anxiety = np.array(anxiety)
    confidence = np.array(confidence)
    uncertainty = np.array(uncertainty)

    nli = get_nli_model()
    semantic_consistency = [0.5]
    for i in range(1, len(texts)):
        contr = nli.contradiction_prob_batch([(texts[i-1], texts[i])])[0]
        semantic_consistency.append(1 - contr)
    semantic_consistency = np.array(semantic_consistency)

    narrative = NarrativeExtractor()
    topics = narrative.fit(texts, embeddings)
    nai_series = narrative.compute_nai_series(topics, timestamps, np.abs(sentiments), window=3)

    # Information field (only past distances)
    info_field = InformationField()
    potentials = []
    for i in range(len(texts)):
        if i == 0:
            dist = 1.0
        else:
            time_gaps = np.diff([ts.timestamp() for ts in timestamps[:i+1]])
            max_gap = np.max(np.abs(time_gaps)) if len(time_gaps)>0 else 1.0
            dist = np.abs(time_gaps[-1]) / (max_gap + 1e-6)
        source_strength = np.abs(sentiments[i]) * (1 + 0.5 * anxiety[i])
        pot = info_field.potential(np.array([source_strength]), np.array([dist]), sigma=1.0, emotional_charge=anxiety[i])
        potentials.append(pot)
    potentials = np.array(potentials)
    F_info = info_field.gradient(potentials)
    energy = info_field.energy(F_info, emotional_index=anxiety)
    cognitive_memory = info_field.cognitive_memory(energy)
    memory_sat = info_field.saturated_memory(cognitive_memory)

    # SSI (simple proxy using potentials)
    ssi_calc = SocialSuggestibilityIndex(n_agents=50)
    ssi_series = ssi_calc.compute(potentials)

    # VAI
    def vai_series(sent, conf, unc, nai, window=5):
        vai = []
        for i in range(len(sent)):
            start = max(0, i-window+1)
            vol = np.std(sent[start:i+1]) if i-start+1>1 else 0.0
            vol = vol / 0.5
            unc_mean = np.mean(unc[start:i+1]) * 2
            nai_mean = np.mean(nai[start:i+1])
            conf_inv = 1 - np.mean(conf[start:i+1])
            raw = 0.3*vol + 0.2*unc_mean + 0.2*nai_mean + 0.2*conf_inv + 0.1*0.5
            vai.append(1/(1+np.exp(-3*(raw-0.5))))
        return np.array(vai)

    # Kalman filter online
    obs = np.column_stack([anxiety, confidence, uncertainty, sentiments])
    kf_online = KalmanFilterOnline(obs_dim=obs.shape[1])
    filtered = kf_online.filter_series(obs)
    filtered_anxiety = filtered[:,0]
    filtered_confidence = filtered[:,1]
    filtered_uncertainty = filtered[:,2]
    filtered_sentiment = filtered[:,3]

    # HMM sliding window (suppress warnings already)
    hmm_online = SlidingWindowHMM(n_regimes=2, window=window_hmm)
    regime_probs = hmm_online.predict_regime(filtered_anxiety)

    vai = vai_series(sentiments, confidence, uncertainty, nai_series)

    # NAI lags
    nai_lag1 = np.zeros_like(nai_series)
    nai_lag2 = np.zeros_like(nai_series)
    for i in range(1, len(nai_series)):
        nai_lag1[i] = nai_series[i-1]
    for i in range(2, len(nai_series)):
        nai_lag2[i] = nai_series[i-2]

    # MPC lookup table
    mpc_lookup = MPCLookup(n_agents=2000)
    mpc_series = np.zeros(len(texts))
    window_mpc = 10
    for i in range(len(texts)):
        start = max(0, i-window_mpc)
        past_anxiety = filtered_anxiety[start:i+1]
        avg_anx = np.mean(past_anxiety)
        mpc_series[i] = mpc_lookup.get_mpc(avg_anx)

    X = np.column_stack([
        filtered_anxiety,
        filtered_confidence,
        filtered_uncertainty,
        filtered_sentiment,
        nai_series,
        nai_lag1,
        nai_lag2,
        ssi_series,
        semantic_consistency,
        regime_probs,
        vai,
        mpc_series,
        energy,           # quadratic info energy
        memory_sat        # saturated cognitive memory
    ])
    X = np.nan_to_num(X, nan=0.0)
    return X, {'info_gradient': F_info, 'energy': energy, 'memory': memory_sat}

# -----------------------------------------------------------------------------
# SlidingWindowHMM, KalmanFilterOnline, NarrativeExtractor, SocialSuggestibilityIndex (optimised)
# -----------------------------------------------------------------------------
class SlidingWindowHMM:
    def __init__(self, n_regimes=4, window=100):
        self.n_regimes = n_regimes
        self.window = window
    def predict_regime(self, series):
        regimes = np.zeros(len(series))
        for t in range(len(series)):
            if t < self.window:
                regimes[t] = 0.5
            else:
                past = series[t-self.window:t].reshape(-1, 1)
                try:
                    model = hmm.GaussianHMM(n_components=self.n_regimes, covariance_type="diag", n_iter=100)
                    model.fit(past)
                    state_seq = model.predict(past)
                    regimes[t] = state_seq[-1] / (self.n_regimes - 1)
                except:
                    regimes[t] = 0.5
        return regimes

class KalmanFilterOnline:
    def __init__(self, obs_dim):
        self.obs_dim = obs_dim
        self.kf = None
        self.state_mean = None
        self.state_cov = None
    def update(self, obs):
        from pykalman import KalmanFilter
        if self.kf is None:
            self.kf = KalmanFilter(transition_matrices=np.eye(self.obs_dim),
                                   observation_matrices=np.eye(self.obs_dim),
                                   initial_state_mean=obs,
                                   initial_state_covariance=np.eye(self.obs_dim),
                                   transition_covariance=np.eye(self.obs_dim)*0.01,
                                   observation_covariance=np.eye(self.obs_dim)*0.1)
            self.state_mean = obs
            self.state_cov = np.eye(self.obs_dim)
            return obs
        else:
            self.state_mean, self.state_cov = self.kf.filter_update(
                self.state_mean, self.state_cov, observation=obs)
            return self.state_mean
    def filter_series(self, obs_series):
        filtered = []
        for obs in obs_series:
            filtered.append(self.update(obs))
        return np.array(filtered)

class NarrativeExtractor:
    def __init__(self, min_topic_size=3, fallback_clusters=3):
        self.min_topic_size = min_topic_size
        self.fallback_clusters = fallback_clusters
        self.topic_model = None
        self.use_fallback = False
    def fit(self, texts, embeddings):
        if len(texts) < 10:
            self.use_fallback = True
            n_clust = min(self.fallback_clusters, max(2, len(texts)//2))
            kmeans = KMeans(n_clusters=n_clust, random_state=SEED)
            topics = kmeans.fit_predict(embeddings)
            self.fallback_model = kmeans
            return topics
        else:
            self.topic_model = BERTopic(verbose=False, nr_topics='auto', min_topic_size=self.min_topic_size)
            topics, _ = self.topic_model.fit_transform(texts, embeddings)
            return topics
    def compute_nai_series(self, topics, timestamps, emotions_intensity, window=3):
        df = pd.DataFrame({'topic': topics, 'ts': timestamps, 'intensity': emotions_intensity})
        df = df[df.topic != -1]
        nai = []
        for i in range(len(df)):
            sub = df.iloc[max(0, i-window+1):i+1]
            if len(sub) == 0:
                nai.append(0.0)
                continue
            freq = len(sub) / window
            runs = (sub.topic != sub.topic.shift()).cumsum()
            persist = sub.groupby(runs).size().mean() / window
            persist = min(1.0, persist)
            intensity = sub['intensity'].mean()
            nai_val = 0.4*freq + 0.3*persist + 0.2*intensity + 0.1*0.5
            nai.append(min(1.0, nai_val))
        full_nai = [0.0]*len(topics)
        idx = 0
        for i, t in enumerate(topics):
            if t != -1:
                full_nai[i] = nai[idx]
                idx += 1
        return np.array(full_nai)

class SocialSuggestibilityIndex:
    def __init__(self, n_agents=100, self_deception_coef=0.3):
        self.n_agents = n_agents
        self.self_deception_coef = self_deception_coef
        self.opinions = np.random.uniform(0.2, 0.8, self.n_agents)
        self.adj = np.random.rand(self.n_agents, self.n_agents) < 0.1
        np.fill_diagonal(self.adj, 0)
    def compute(self, external_field_series: np.ndarray):
        opinions = self.opinions.copy()
        neigh_mean = np.zeros(self.n_agents)
        suggestibility = np.zeros(self.n_agents)
        ssi_series = []
        for field in external_field_series:
            for i in range(self.n_agents):
                neighbors = self.adj[i]
                if np.sum(neighbors) > 0:
                    neigh_mean[i] = np.mean(opinions[neighbors])
                else:
                    neigh_mean[i] = opinions[i]
                suggestibility[i] = 1 - np.exp(-abs(opinions[i] - neigh_mean[i]))
            for i in range(self.n_agents):
                stubborn = 0.3
                new_opinion = (suggestibility[i] * neigh_mean[i] + stubborn * opinions[i] + 0.5 * field) / (1 + stubborn)
                confident = 1 - np.var(opinions)
                self_deception = self.self_deception_coef * (new_opinion - neigh_mean[i]) * confident
                opinions[i] = np.clip(new_opinion + self_deception, 0, 1)
            mean_lam = np.mean(suggestibility)
            ssi = sigmoid(mean_lam)
            ssi_series.append(ssi)
        self.opinions = opinions
        return np.array(ssi_series)

# -----------------------------------------------------------------------------
# Benchmarks (Transformer, DeepAR) and CausalEffectEstimator
# -----------------------------------------------------------------------------
class TransformerForecaster(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=3, dropout=0.1):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        encoder_layer = TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True)
        self.transformer = TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, 1)
    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        return self.fc_out(x[:, -1, :])

class DeepAR(nn.Module):
    def __init__(self, input_dim, hidden_dim=32, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)
        self.fc_mu = nn.Linear(hidden_dim, 1)
        self.fc_sigma = nn.Linear(hidden_dim, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        mu = self.fc_mu(out)
        sigma = torch.exp(self.fc_sigma(out)) + 1e-6
        return mu, sigma

def train_deepar(model, X_train, y_train, epochs=30, lr=0.001):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                            torch.tensor(y_train, dtype=torch.float32).view(-1,1))
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    for epoch in range(epochs):
        total_loss = 0.0
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            mu, sigma = model(batch_x)
            dist = Normal(mu, sigma)
            loss = -dist.log_prob(batch_y).mean()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch+1) % 10 == 0:
            print(f"DeepAR epoch {epoch+1}, loss={total_loss/len(loader):.4f}")
    return model

class CausalEffectEstimator:
    def __init__(self):
        self.cate_model = None
        self.use_econml = CAUSAL_AVAILABLE
    def fit(self, X, T, Y):
        if self.use_econml:
            try:
                self.cate_model = CausalForestDML(model_y=RandomForestRegressor(n_estimators=100),
                                                  model_t=RandomForestRegressor(n_estimators=100),
                                                  discrete_treatment=True, random_state=SEED)
                self.cate_model.fit(Y, T, X=X)
                return self
            except Exception:
                self.use_econml = False
        treated = (T == 1)
        control = (T == 0)
        self.model_treat = RandomForestRegressor(n_estimators=100)
        self.model_control = RandomForestRegressor(n_estimators=100)
        if np.sum(treated) > 1:
            self.model_treat.fit(X[treated], Y[treated])
        if np.sum(control) > 1:
            self.model_control.fit(X[control], Y[control])
        return self
    def predict_cate(self, X):
        if self.use_econml and self.cate_model is not None:
            return self.cate_model.effect(X)
        else:
            cate_treat = self.model_treat.predict(X)
            cate_control = self.model_control.predict(X)
            return cate_treat - cate_control

# -----------------------------------------------------------------------------
# Main pipeline
# -----------------------------------------------------------------------------
def main(csv_file, output_json='report.json', mode='macro'):
    print("Loading dataset...")
    texts, y, timestamps = load_unified_dataset(csv_file)
    y = np.array(y, dtype=float)
    dataset_description(texts, y, timestamps)

    print("Extracting features (no future leaks, optimised)...")
    X_raw, theoretical = extract_features_no_future(texts, timestamps, window_hmm=100)

    finite = np.isfinite(X_raw).all(axis=1) & np.isfinite(y)
    X_raw = X_raw[finite]
    y = y[finite]
    timestamps = [timestamps[i] for i in range(len(timestamps)) if finite[i]]
    if len(X_raw) < 20:
        raise ValueError("Too few valid samples after cleaning")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    # Build sequences (window=5)
    window = 5
    X_seq, y_seq = [], []
    for i in range(window, len(X_scaled)):
        X_seq.append(X_scaled[i-window:i])
        y_seq.append(y[i])
    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)

    # Spatial weights (based on time distance)
    t_vals = np.array([ts.timestamp() for ts in timestamps])
    n_time = len(t_vals)
    W = np.zeros((n_time, n_time))
    for i in range(n_time):
        for j in range(n_time):
            if i != j:
                dist = abs(t_vals[i] - t_vals[j])
                W[i,j] = np.exp(-0.1 * dist)
    rowsum = W.sum(axis=1, keepdims=True)
    rowsum[rowsum==0] = 1
    W = W / rowsum
    # keep only the part needed for sequences (truncate)
    W_seq = W[:len(X_seq), :len(X_seq)]

    # TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=3)
    all_baseline_metrics = defaultdict(list)
    all_bsgem_metrics = []
    all_cates = []

    best_bsgem_r2 = -np.inf
    best_model = None
    best_train_losses = None
    best_val_losses = None
    best_y_true = None
    best_y_pred = None
    best_lower = None
    best_upper = None
    best_cate = None
    best_avg_cate = None

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X_seq)):
        print(f"\n--- Fold {fold+1}/3 ---")
        X_train = X_seq[train_idx]
        X_test = X_seq[test_idx]
        y_train = y_seq[train_idx]
        y_test = y_seq[test_idx]

        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        X_test_flat = X_test.reshape(X_test.shape[0], -1)

        # Baselines
        lr = LinearRegression().fit(X_train_flat, y_train)
        all_baseline_metrics['LinearRegression'].append({
            'mse': mean_squared_error(y_test, lr.predict(X_test_flat)),
            'mae': mean_absolute_error(y_test, lr.predict(X_test_flat)),
            'r2': r2_score(y_test, lr.predict(X_test_flat))
        })
        rf = RandomForestRegressor(n_estimators=100, random_state=SEED).fit(X_train_flat, y_train)
        y_pred_rf = rf.predict(X_test_flat)
        all_baseline_metrics['RandomForest'].append({
            'mse': mean_squared_error(y_test, y_pred_rf),
            'mae': mean_absolute_error(y_test, y_pred_rf),
            'r2': r2_score(y_test, y_pred_rf)
        })
        lgbm = lgb.LGBMRegressor(n_estimators=100, random_state=SEED, verbosity=-1).fit(X_train_flat, y_train)
        y_pred_lgb = lgbm.predict(X_test_flat)
        all_baseline_metrics['LightGBM'].append({
            'mse': mean_squared_error(y_test, y_pred_lgb),
            'mae': mean_absolute_error(y_test, y_pred_lgb),
            'r2': r2_score(y_test, y_pred_lgb)
        })
        try:
            ar = AutoReg(y_train, lags=2).fit()
            ar_pred = ar.predict(start=len(y_train), end=len(y_train)+len(y_test)-1)
            all_baseline_metrics['VAR(2)'].append({
                'mse': mean_squared_error(y_test, ar_pred),
                'mae': mean_absolute_error(y_test, ar_pred),
                'r2': r2_score(y_test, ar_pred)
            })
        except:
            pass

        # Transformer
        transformer = TransformerForecaster(input_dim=X_train.shape[2])
        transformer.to(device)
        opt = optim.Adam(transformer.parameters(), lr=0.001, weight_decay=1e-4)
        X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
        y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1,1).to(device)
        X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
        for ep in range(50):
            transformer.train()
            opt.zero_grad()
            pred = transformer(X_train_t)
            loss = nn.MSELoss()(pred, y_train_t)
            loss.backward()
            opt.step()
        transformer.eval()
        with torch.no_grad():
            y_pred_tr = transformer(X_test_t).cpu().numpy().flatten()
        all_baseline_metrics['Transformer'].append({
            'mse': mean_squared_error(y_test, y_pred_tr),
            'mae': mean_absolute_error(y_test, y_pred_tr),
            'r2': r2_score(y_test, y_pred_tr)
        })

        # DeepAR
        deepar = DeepAR(input_dim=X_train.shape[2])
        deepar = train_deepar(deepar, X_train, y_train, epochs=30)
        deepar.eval()
        with torch.no_grad():
            mu_test, _ = deepar(X_test_t)
            y_pred_dp = mu_test.cpu().numpy().flatten()
        all_baseline_metrics['DeepAR'].append({
            'mse': mean_squared_error(y_test, y_pred_dp),
            'mae': mean_absolute_error(y_test, y_pred_dp),
            'r2': r2_score(y_test, y_pred_dp)
        })

        # ---- HFE stage ----
        base_model = BSGEM_VAE(input_dim=X_train.shape[2], latent_dim=4, hidden_dim=16,
                               n_agents=20, use_endogenous=False, use_spatial=False,
                               use_dcg=False, use_hmm=False, mode=mode)
        base_model, _, _, _ = train_bsgem_vae(base_model, X_train, y_train, X_test, y_test,
                                              epochs=60, beta_kl=0.01, patience=10, lr=1e-4)
        base_model.eval()
        with torch.no_grad():
            y_pred_base_train = base_model(torch.tensor(X_train, dtype=torch.float32).to(device))[0].cpu().numpy().flatten()
            residuals = y_train - y_pred_base_train
            y_pred_base_test = base_model(torch.tensor(X_test, dtype=torch.float32).to(device))[0].cpu().numpy().flatten()
            residuals_test = y_test - y_pred_base_test

        psi_train = np.zeros_like(residuals)
        for i in range(2, len(residuals)):
            psi_train[i] = 0.5*residuals[i-1] + 0.3*residuals[i-2]
        psi_test = np.zeros_like(residuals_test)
        for i in range(2, len(residuals_test)):
            psi_test[i] = 0.5*residuals_test[i-1] + 0.3*residuals_test[i-2]

        psi_train_aug = psi_train.reshape(-1,1,1).repeat(window, axis=1)
        psi_test_aug = psi_test.reshape(-1,1,1).repeat(window, axis=1)
        X_train_aug = np.concatenate([X_train, psi_train_aug], axis=2)
        X_test_aug = np.concatenate([X_test, psi_test_aug], axis=2)

        # Final model (full VAE with all components, spatial disabled for stability)
        final_model = BSGEM_VAE(input_dim=X_train_aug.shape[2], latent_dim=4, hidden_dim=16,
                                n_agents=20, use_endogenous=True, use_spatial=False,
                                use_dcg=True, use_hmm=True, mode=mode)
        # final_model.set_spatial_weights(W_seq[:X_train_aug.shape[0], :X_train_aug.shape[0]])  # disabled
        final_model, train_losses, val_losses, val_r2s = train_bsgem_vae(
            final_model, X_train_aug, y_train, X_test_aug, y_test,
            epochs=120, beta_kl=0.01, patience=30, lr=1e-4)

        # Interval forecasts via Monte Carlo dropout (simple)
        final_model.eval()
        with torch.no_grad():
            X_test_aug_t = torch.tensor(X_test_aug, dtype=torch.float32).to(device)
            y_pred_samples = []
            for _ in range(30):
                final_model.train()
                y_pred_s, _ = final_model(X_test_aug_t)
                y_pred_samples.append(y_pred_s.cpu().numpy().flatten())
            y_pred_samples = np.array(y_pred_samples)
            y_pred_mean = y_pred_samples.mean(axis=0)
            y_pred_lower = np.percentile(y_pred_samples, 2.5, axis=0)
            y_pred_upper = np.percentile(y_pred_samples, 97.5, axis=0)
        fold_metrics = {
            'mse': mean_squared_error(y_test, y_pred_mean),
            'mae': mean_absolute_error(y_test, y_pred_mean),
            'r2': r2_score(y_test, y_pred_mean)
        }
        all_bsgem_metrics.append(fold_metrics)
        print(f"Fold {fold+1} BSGEM R² = {fold_metrics['r2']:.4f}")

        # CATE
        F_info_abs = np.abs(theoretical['info_gradient'][finite])
        threshold = np.percentile(F_info_abs, 90)
        treatment_all = (F_info_abs > threshold).astype(int)
        treat_test = treatment_all[-len(y_test):]
        cate_est = CausalEffectEstimator()
        cate_est.fit(X_test_flat, treat_test, y_test)
        cate = cate_est.predict_cate(X_test_flat)
        avg_cate = np.mean(cate)
        all_cates.append(avg_cate)
        print(f"Fold {fold+1} avg CATE = {avg_cate:.4f}")

        if fold_metrics['r2'] > best_bsgem_r2:
            best_bsgem_r2 = fold_metrics['r2']
            best_model = final_model
            best_train_losses = train_losses
            best_val_losses = val_losses
            best_y_true = y_test
            best_y_pred = y_pred_mean
            best_lower = y_pred_lower
            best_upper = y_pred_upper
            best_cate = cate
            best_avg_cate = avg_cate

    # Summary
    print("\n" + "="*80)
    print("BASELINE MODELS (time series CV, mean ± std)")
    print("="*80)
    for name, metrics_list in all_baseline_metrics.items():
        if not metrics_list:
            continue
        mse_vals = [m['mse'] for m in metrics_list]
        mae_vals = [m['mae'] for m in metrics_list]
        r2_vals = [m['r2'] for m in metrics_list]
        print(f"{name:15} MSE={np.mean(mse_vals):.4f}±{np.std(mse_vals):.4f}  "
              f"MAE={np.mean(mae_vals):.4f}±{np.std(mae_vals):.4f}  "
              f"R²={np.mean(r2_vals):.4f}±{np.std(r2_vals):.4f}")

    print("\n" + "="*80)
    print("BSGEM-HFE (full VAE + FJ + endogenous + DCG + HMM) results")
    print("="*80)
    mse_vals = [m['mse'] for m in all_bsgem_metrics]
    mae_vals = [m['mae'] for m in all_bsgem_metrics]
    r2_vals = [m['r2'] for m in all_bsgem_metrics]
    print(f"MSE={np.mean(mse_vals):.4f}±{np.std(mse_vals):.4f}  "
          f"MAE={np.mean(mae_vals):.4f}±{np.std(mae_vals):.4f}  "
          f"R²={np.mean(r2_vals):.4f}±{np.std(r2_vals):.4f}")
    print(f"Average CATE (75th percentile info shock): {np.mean(all_cates):.4f}±{np.std(all_cates):.4f}")

    if best_y_pred is not None:
        plt.figure(figsize=(12,5))
        plt.plot(best_y_true, label='True', marker='o')
        plt.plot(best_y_pred, label='BSGEM-HFE', marker='s', linestyle='--')
        plt.fill_between(range(len(best_y_true)), best_lower, best_upper, alpha=0.3, label='95% CI')
        plt.title('BSGEM-HFE: Actual vs Predicted with Uncertainty (best fold)')
        plt.legend()
        plt.grid(True)
        plt.show(block=False)

        plt.figure(figsize=(12,5))
        plt.plot(best_train_losses, label='Train loss (MSE+βKL)')
        plt.plot(best_val_losses, label='Validation loss')
        plt.title('Training curves (best fold)')
        plt.legend()
        plt.grid(True)
        plt.show(block=False)

        plt.figure(figsize=(10,4))
        plt.hist(best_cate, bins=30, alpha=0.7)
        plt.axvline(best_avg_cate, color='red', linestyle='--', label=f'Mean CATE={best_avg_cate:.3f}')
        plt.title('CATE distribution (best fold)')
        plt.legend()
        plt.grid(True)
        plt.show(block=False)

    report = {
        'dataset': {'n': len(y), 'target_mean': float(np.mean(y))},
        'baselines_avg': {
            name: {
                'mse_mean': float(np.mean([m['mse'] for m in metrics_list])),
                'mae_mean': float(np.mean([m['mae'] for m in metrics_list])),
                'r2_mean': float(np.mean([m['r2'] for m in metrics_list]))
            } for name, metrics_list in all_baseline_metrics.items() if metrics_list
        },
        'bsgem_hfe': {
            'mse_mean': float(np.mean(mse_vals)),
            'mae_mean': float(np.mean(mae_vals)),
            'r2_mean': float(np.mean(r2_vals))
        },
        'cate_avg': float(np.mean(all_cates))
    }
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(make_serializable(report), f, indent=2)
    print(f"\nReport saved to {output_json}")
    input("Press Enter to close all plots and exit...")
    plt.close('all')

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python BSGEM_model.py dataset.csv")
        sys.exit(1)
    main(sys.argv[1])