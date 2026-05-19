#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import subprocess
import sys
import os
import random
import re
import warnings
import hashlib
import pickle
from pathlib import Path

warnings.filterwarnings("ignore")

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
    "nltk": "nltk",
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
    "hmmlearn": "hmmlearn",
    "pytorch-forecasting": "pytorch_forecasting"
}

for pkg, imp in IMPORT_MAP.items():
    try:
        __import__(imp)
    except ImportError:
        install(pkg)

import json
import numpy as np
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any, Tuple
from collections import defaultdict
from tqdm import tqdm

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
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
import statsmodels.api as sm
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
import lightgbm as lgb
import shap
import nltk
import scipy.stats as stats
import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 12, 6

# For HMM
from hmmlearn import hmm

# For benchmarks
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.distributions import Normal

# econml
try:
    from econml.dml import CausalForestDML
    CAUSAL_AVAILABLE = True
except ImportError:
    CAUSAL_AVAILABLE = False

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CACHE_DIR = Path("./cache")
CACHE_DIR.mkdir(exist_ok=True)

# ---------- Helper functions ----------
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
    df = pd.read_csv(csv_path)
    required = ['timestamp', 'text', 'label']
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}. Found columns: {list(df.columns)}")
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    initial_len = len(df)
    df = df.dropna(subset=['timestamp'])
    if len(df) < initial_len:
        print(f"Warning: Dropped {initial_len - len(df)} rows with invalid timestamps.")
    texts = df['text'].astype(str).tolist()
    labels = df['label'].astype(float).tolist()
    timestamps = df['timestamp'].tolist()
    return texts, labels, timestamps

def dataset_description(texts, y, timestamps):
    n_messages = len(texts)
    n_labels = len(y)
    pos_ratio = np.mean(y)
    avg_len = np.mean([len(t) for t in texts])
    print("\n=== DATASET DESCRIPTION ===")
    print(f"Number of messages: {n_messages}")
    print(f"Number of labeled instances: {n_labels}")
    print(f"Class balance (positive/irrational ratio): {pos_ratio:.3f}")
    print(f"Average message length (chars): {avg_len:.1f}")
    print(f"Time range: {min(timestamps)} to {max(timestamps)}")

# ---------- Caching for embeddings ----------
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

# ---------- Theoretical components ----------
class InformationField:
    def __init__(self, xi=1.0, rho=0.5, decay_lambda=0.1, kappa=0.05):
        self.xi = xi
        self.rho = rho
        self.decay_lambda = decay_lambda
        self.kappa = kappa

    def potential(self, sources: np.ndarray, distances: np.ndarray, sigma=1.0, emotional_charge=None):
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

class RussianNLI:
    def __init__(self, model_name="cointegrated/rubert-base-cased-nli-threeway"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self.contradiction_idx = 0

    def contradiction_prob_batch(self, pairs):
        inputs = self.tokenizer(pairs, return_tensors="pt", truncation=True, max_length=128, padding=True).to(self.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        return probs[:, self.contradiction_idx]

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

    def persistence_stats(self, topics, timestamps):
        df = pd.DataFrame({'topic': topics, 'ts': timestamps})
        df = df[df.topic != -1]
        if len(df) < 2:
            return {'persistence_length': 0, 'switch_rate': 0, 'lock_in_score': 0}
        changes = (df.topic != df.topic.shift()).cumsum()
        runs = df.groupby(changes).size()
        avg_persistence = runs.mean()
        switch_rate = (df.topic != df.topic.shift()).sum() / len(df)
        top_topic = df.topic.mode().iloc[0]
        lock_in = (df.topic == top_topic).mean()
        return {'persistence_length': avg_persistence, 'switch_rate': switch_rate, 'lock_in_score': lock_in}

class KalmanSmoother:
    def fit_transform(self, obs):
        from pykalman import KalmanFilter
        kf = KalmanFilter(transition_matrices=np.eye(obs.shape[1]),
                          observation_matrices=np.eye(obs.shape[1]),
                          initial_state_mean=obs[0],
                          initial_state_covariance=np.eye(obs.shape[1]),
                          transition_covariance=np.eye(obs.shape[1])*0.01,
                          observation_covariance=np.eye(obs.shape[1])*0.1)
        state_means, _ = kf.filter(obs)
        return state_means

def aggregate_mpc(anxiety, n_agents=5000):
    np.random.seed(SEED)
    alphas = np.random.uniform(0.5, 1.5, n_agents)
    betas  = np.random.uniform(0.5, 1.5, n_agents)
    gammas = np.random.uniform(0.2, 1.0, n_agents)
    def utility(c, s, a, alpha, beta, gamma):
        return alpha * np.log(c+1e-6) + beta * np.log(s+1e-6) - gamma * a
    consumption_shares = []
    for i in range(n_agents):
        best_c = 0.5
        best_u = -np.inf
        for c in np.linspace(0, 1, 20):
            u = utility(c*1.0, (1-c)*1.0, anxiety, alphas[i], betas[i], gammas[i])
            if u > best_u:
                best_u = u
                best_c = c
        consumption_shares.append(best_c)
    return np.mean(consumption_shares)

# =============================================================================
# BENCHMARKS
# =============================================================================
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
        out = self.fc_out(x[:, -1, :])
        return out

class DeepAR(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc_mu = nn.Linear(hidden_dim, 1)
        self.fc_sigma = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        mu = self.fc_mu(out)
        sigma = torch.exp(self.fc_sigma(out)) + 1e-6
        return mu, sigma

def train_deepar(model, X_train, y_train, epochs=50, lr=0.001):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
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

# =============================================================================
# DYNAMIC CAUSAL GRAPH, MARKOV REGIME, VAE
# =============================================================================
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
        k = int(0.2 * seq)
        topk = torch.topk(adj, k, dim=-1).indices
        sparse_adj = torch.zeros_like(adj)
        sparse_adj.scatter_(-1, topk, adj.gather(-1, topk))
        rowsum = sparse_adj.sum(dim=-1, keepdim=True) + 1e-8
        adj_norm = sparse_adj / rowsum
        return adj_norm

class MarkovRegimeSwitching(nn.Module):
    def __init__(self, n_regimes=4, n_features=11, hidden_dim=32):
        super().__init__()
        self.n_regimes = n_regimes
        self.transition = nn.Parameter(torch.ones(n_regimes, n_regimes) / n_regimes)
        self.regime_encoder = nn.Linear(n_features, hidden_dim)
        self.regime_out = nn.Linear(hidden_dim, n_regimes)

    def forward(self, x, prev_regime=None):
        batch, seq, fdim = x.shape
        logits = self.regime_out(F.relu(self.regime_encoder(x)))
        if self.training:
            regime_probs = F.gumbel_softmax(logits, tau=1.0, hard=True, dim=-1)
        else:
            regime_probs = F.softmax(logits, dim=-1)
        if prev_regime is not None:
            transition_t = F.softmax(self.transition, dim=-1)
            regime_probs = torch.einsum('b s i, i j -> b s j', regime_probs, transition_t)
        return regime_probs

class BSGEM_VAE(nn.Module):
    def __init__(self, input_dim, latent_dim=16, hidden_dim=64,
                 n_regimes=4, kappa_self=0.2, omega=0.1,
                 use_endogenous=True, use_regime=True):
        super().__init__()
        self.latent_dim = latent_dim
        self.kappa_self = kappa_self
        self.omega = omega
        self.use_endogenous = use_endogenous
        self.use_regime = use_regime

        self.enc_fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim*2)
        )

        dec_input_dim = latent_dim + input_dim
        if use_endogenous:
            dec_input_dim += 1
        if use_regime:
            dec_input_dim += n_regimes
        self.dec_gru = nn.GRU(dec_input_dim, hidden_dim,
                              batch_first=True, num_layers=2, dropout=0.2)
        self.dec_fc = nn.Linear(hidden_dim, 1)
        self.attn = nn.Linear(hidden_dim, 1)

        self.dcg = DynamicCausalGraph(input_dim, hidden_dim=32)
        if use_regime:
            self.regime_switch = MarkovRegimeSwitching(n_regimes, input_dim, hidden_dim)
        else:
            self.regime_switch = None

    def encode(self, x, adj):
        batch, seq, dim = x.shape
        x_graph = torch.einsum('b s t, b t d -> b s d', adj, x)
        x_comb = x + 0.1 * x_graph
        flat = x_comb.reshape(batch*seq, dim)
        h = self.enc_fc(flat)
        mu = h[:, :self.latent_dim].view(batch, seq, self.latent_dim)
        logvar = h[:, self.latent_dim:].view(batch, seq, self.latent_dim)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, return_attention=False):
        batch, seq, dim = x.shape
        adj = self.dcg(x)
        mu, logvar = self.encode(x, adj)
        z = self.reparameterize(mu, logvar)

        # Self-deception
        z_mean = z.mean(dim=1, keepdim=True)
        confidence = torch.sigmoid(1.0 - logvar.mean(dim=-1, keepdim=True))
        self_deception = self.kappa_self * (z - z_mean) * confidence
        z_aug = z + self_deception

        # Regime probabilities
        if self.use_regime and self.regime_switch is not None:
            regime_probs = self.regime_switch(x)   # (batch, seq, n_regimes)
        else:
            regime_probs = torch.zeros(batch, seq, 1, device=x.device)

        # Endogenous field
        if self.use_endogenous:
            # use first component of regime_probs as collective opinion
            avg_opinion = regime_probs[:, :, 0].mean(dim=1, keepdim=True).unsqueeze(-1)  # (batch,1,1)
            avg_opinion = avg_opinion.expand(-1, seq, -1)  # (batch, seq, 1)
            endog_field = self.omega * avg_opinion
            x_aug = torch.cat([x, endog_field], dim=-1)
        else:
            x_aug = x

        # Build decoder input
        dec_parts = [z_aug, x_aug]
        if self.use_regime:
            dec_parts.append(regime_probs)
        dec_input = torch.cat(dec_parts, dim=-1)

        gru_out, _ = self.dec_gru(dec_input)
        attn_weights = torch.softmax(self.attn(gru_out), dim=1)
        context = torch.sum(attn_weights * gru_out, dim=1)
        y_pred = self.dec_fc(context)

        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=(1,2)).mean()

        if return_attention:
            return y_pred, kl, attn_weights, regime_probs if self.use_regime else None
        return y_pred, kl

def train_vae(model, X_train, y_train, X_val, y_val, epochs=150, lr=0.001, patience=20):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses, val_losses = [], []

    dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                            torch.tensor(y_train, dtype=torch.float32).view(-1,1))
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            y_pred, kl = model(batch_x)
            mse = nn.MSELoss()(y_pred, batch_y)
            loss = mse + 0.01 * kl
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch_x.size(0)
        epoch_loss /= len(X_train)
        train_losses.append(epoch_loss)

        model.eval()
        with torch.no_grad():
            X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
            y_val_t = torch.tensor(y_val, dtype=torch.float32).view(-1,1).to(device)
            y_pred_val, kl_val = model(X_val_t)
            val_loss = nn.MSELoss()(y_pred_val, y_val_t).item() + 0.01 * kl_val.item()
            val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {k:v.cpu().clone() for k,v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        if (epoch+1) % 20 == 0:
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}")

    model.load_state_dict(best_state)
    return model, train_losses, val_losses

# ---------- Spatial weights ----------
def build_spatial_weights(timestamps, decay=0.1):
    t_vals = np.array([ts.timestamp() for ts in timestamps])
    n = len(t_vals)
    W = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                dist = abs(t_vals[i] - t_vals[j])
                W[i,j] = np.exp(-decay * dist)
    rowsum = W.sum(axis=1, keepdims=True)
    rowsum[rowsum==0] = 1
    W = W / rowsum
    return W

# ---------- Feature extraction (batched, cached) ----------
def extract_all_features(texts, timestamps, cache_embeddings_flag=True):
    embedder = SentenceTransformer('cointegrated/rubert-tiny2')
    cache_key = get_cache_key(texts, "rubert_tiny2")
    if cache_embeddings_flag and (CACHE_DIR / cache_key).exists():
        print("Loading cached embeddings...")
        embeddings = cache_embeddings(embedder, texts, cache_key)
    else:
        print("Computing embeddings (batching)...")
        embeddings = embedder.encode(texts, show_progress_bar=True, batch_size=32)
        if cache_embeddings_flag:
            with open(CACHE_DIR / cache_key, 'wb') as f:
                pickle.dump(embeddings, f)

    sentiment_pipeline = pipeline("sentiment-analysis", model="blanchefort/rubert-base-cased-sentiment", device=-1)
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

    nli = RussianNLI()
    pairs = [(texts[i-1], texts[i]) for i in range(1, len(texts))]
    consistency = []
    batch_sz = 32
    for i in tqdm(range(0, len(pairs), batch_sz), desc="NLI"):
        batch_pairs = pairs[i:i+batch_sz]
        contr = nli.contradiction_prob_batch(batch_pairs)
        consistency.extend(1 - contr)
    semantic_consistency = np.array([0.5] + consistency)

    narrative = NarrativeExtractor()
    topics = narrative.fit(texts, embeddings)
    nai_series = narrative.compute_nai_series(topics, timestamps, np.abs(sentiments), window=3)
    narrative_stats = narrative.persistence_stats(topics, timestamps)

    info_field = InformationField()
    time_gaps = np.diff([ts.timestamp() for ts in timestamps])
    time_gaps = np.insert(time_gaps, 0, 1.0)
    distances = np.abs(time_gaps) / (np.max(time_gaps) + 1e-6)
    potentials = []
    for i in range(len(texts)):
        source_strength = np.abs(sentiments[i]) * (1 + 0.5 * anxiety[i])
        pot = info_field.potential(np.array([source_strength]), np.array([distances[i]]), sigma=1.0, emotional_charge=anxiety[i])
        potentials.append(pot)
    potentials = np.array(potentials)
    F_info = info_field.gradient(potentials)
    energy = info_field.energy(F_info, emotional_index=anxiety)
    mem = info_field.cognitive_memory(energy)
    mem_sat = info_field.saturated_memory(mem)

    ssi_calc = SocialSuggestibilityIndex(n_agents=50)
    ssi_series = ssi_calc.compute(potentials)

    def vai_series(sent, conf, unc, nai, window=5):
        vai = []
        for i in range(len(sent)):
            start = max(0, i-window+1)
            vol = np.std(sent[start:i+1]) / 0.5
            unc_mean = np.mean(unc[start:i+1]) * 2
            nai_mean = np.mean(nai[start:i+1])
            conf_inv = 1 - np.mean(conf[start:i+1])
            raw = 0.3*vol + 0.2*unc_mean + 0.2*nai_mean + 0.2*conf_inv + 0.1*0.5
            vai.append(1/(1+np.exp(-3*(raw-0.5))))
        return np.array(vai)

    obs = np.column_stack([anxiety, confidence, uncertainty, sentiments])
    smoother = KalmanSmoother()
    smoothed = smoother.fit_transform(obs)
    smoothed_anxiety = smoothed[:,0]
    smoothed_confidence = smoothed[:,1]
    smoothed_uncertainty = smoothed[:,2]
    smoothed_sentiment = smoothed[:,3]

    # Full HMM with learned transition probabilities
    try:
        model_hmm = MarkovRegression(smoothed_anxiety, k_regimes=2, trend='c', switching_variance=True)
        res_hmm = model_hmm.fit()
        regime_probs = res_hmm.smoothed_marginal_probabilities[:, 0]  # probability of regime 1
    except:
        regime_probs = (smoothed_anxiety > np.median(smoothed_anxiety)).astype(float)

    vai = vai_series(sentiments, confidence, uncertainty, nai_series)

    nai_lag1 = np.roll(nai_series, 1)
    nai_lag2 = np.roll(nai_series, 2)
    nai_lag1[0] = nai_series[0]
    nai_lag2[0] = nai_series[0]
    nai_lag2[1] = nai_series[1]

    avg_anxiety = np.mean(smoothed_anxiety)
    mpc = aggregate_mpc(avg_anxiety, n_agents=2000)
    mpc_series = np.full(len(texts), mpc)

    X = np.column_stack([
        smoothed_anxiety,
        smoothed_confidence,
        smoothed_uncertainty,
        smoothed_sentiment,
        nai_series,
        nai_lag1,
        nai_lag2,
        ssi_series,
        semantic_consistency,
        regime_probs,
        vai,
        mpc_series
    ])
    return X, {'info_gradient': F_info, 'info_energy': energy, 'memory_sat': mem_sat}

# =============================================================================
# MAIN
# =============================================================================
def main(csv_file, output_json='report.json'):
    print("Loading dataset...")
    texts, y, timestamps = load_unified_dataset(csv_file)
    y = np.array(y, dtype=float)
    dataset_description(texts, y, timestamps)

    print("Extracting features...")
    X_raw, theoretical = extract_all_features(texts, timestamps, cache_embeddings_flag=True)

    finite = np.isfinite(X_raw).all(axis=1) & np.isfinite(y)
    X_raw = X_raw[finite]
    y = y[finite]
    timestamps = [timestamps[i] for i in range(len(timestamps)) if finite[i]]
    if len(X_raw) < 10:
        raise ValueError("Too few valid samples after cleaning")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    # Spatial weights (based on time)
    W = build_spatial_weights(timestamps, decay=0.1)
    spatial_lag = W @ X_scaled

    window = 5
    X_seq, y_seq, spatial_seq = [], [], []
    for i in range(window, len(X_scaled)):
        X_seq.append(X_scaled[i-window:i])
        y_seq.append(y[i])
        spatial_seq.append(spatial_lag[i-window:i])
    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)
    spatial_seq = np.array(spatial_seq)

    split = int(0.8 * len(X_seq))
    X_train, X_test = X_seq[:split], X_seq[split:]
    y_train, y_test = y_seq[:split], y_seq[split:]
    spatial_train, spatial_test = spatial_seq[:split], spatial_seq[split:]

    # ---------- Baseline models (including Transformer and DeepAR) ----------
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    baselines = {}

    lr = LinearRegression().fit(X_train_flat, y_train)
    baselines['LinearRegression'] = {'mse': mean_squared_error(y_test, lr.predict(X_test_flat)),
                                     'mae': mean_absolute_error(y_test, lr.predict(X_test_flat)),
                                     'r2': r2_score(y_test, lr.predict(X_test_flat))}
    rf = RandomForestRegressor(n_estimators=100, random_state=SEED).fit(X_train_flat, y_train)
    y_pred_rf = rf.predict(X_test_flat)
    baselines['RandomForest'] = {'mse': mean_squared_error(y_test, y_pred_rf),
                                 'mae': mean_absolute_error(y_test, y_pred_rf),
                                 'r2': r2_score(y_test, y_pred_rf)}
    lgbm = lgb.LGBMRegressor(n_estimators=100, random_state=SEED, verbosity=-1).fit(X_train_flat, y_train)
    y_pred_lgb = lgbm.predict(X_test_flat)
    baselines['LightGBM'] = {'mse': mean_squared_error(y_test, y_pred_lgb),
                             'mae': mean_absolute_error(y_test, y_pred_lgb),
                             'r2': r2_score(y_test, y_pred_lgb)}
    try:
        ar = AutoReg(y_train, lags=2).fit()
        ar_pred = ar.predict(start=len(y_train), end=len(y_train)+len(y_test)-1)
        baselines['VAR(2)'] = {'mse': mean_squared_error(y_test, ar_pred),
                               'mae': mean_absolute_error(y_test, ar_pred),
                               'r2': r2_score(y_test, ar_pred)}
    except:
        baselines['VAR(2)'] = {'mse': np.nan, 'mae': np.nan, 'r2': np.nan}

    # Transformer
    transformer = TransformerForecaster(input_dim=X_train.shape[2])
    transformer.to(device)
    opt = optim.Adam(transformer.parameters(), lr=0.001)
    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1,1).to(device)
    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
    for epoch in range(80):
        transformer.train()
        opt.zero_grad()
        pred = transformer(X_train_t)
        loss = nn.MSELoss()(pred, y_train_t)
        loss.backward()
        opt.step()
    transformer.eval()
    with torch.no_grad():
        y_pred_tr = transformer(X_test_t).cpu().numpy().flatten()
    baselines['Transformer'] = {'mse': mean_squared_error(y_test, y_pred_tr),
                                'mae': mean_absolute_error(y_test, y_pred_tr),
                                'r2': r2_score(y_test, y_pred_tr)}

    # DeepAR
    deepar = DeepAR(input_dim=X_train.shape[2])
    deepar = train_deepar(deepar, X_train, y_train, epochs=50)
    deepar.eval()
    with torch.no_grad():
        mu_test, _ = deepar(X_test_t)
        y_pred_dp = mu_test.cpu().numpy().flatten()
    baselines['DeepAR'] = {'mse': mean_squared_error(y_test, y_pred_dp),
                           'mae': mean_absolute_error(y_test, y_pred_dp),
                           'r2': r2_score(y_test, y_pred_dp)}

    print("\n" + "="*80)
    print("BASELINE MODELS")
    print("="*80)
    for name, met in baselines.items():
        print(f"{name:15} MSE={met['mse']:.4f}  MAE={met['mae']:.4f}  R²={met['r2']:.4f}")

    # ---------- BSGEM-HFE training ----------
    # Base model (without HFE, without endogenous, without regime)
    print("Training base model (without HFE)...")
    base_model = BSGEM_VAE(input_dim=X_train.shape[2], latent_dim=8, hidden_dim=64,
                           use_endogenous=False, use_regime=False, kappa_self=0.0, omega=0.0)
    base_model, _, _ = train_vae(base_model, X_train, y_train, X_test, y_test, epochs=80, patience=10)

    base_model.eval()
    with torch.no_grad():
        X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
        y_pred_train_base, _ = base_model(X_train_t)
        residuals = y_train - y_pred_train_base.cpu().numpy().flatten()
        X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
        y_pred_test_base, _ = base_model(X_test_t)
        residuals_test = y_test - y_pred_test_base.cpu().numpy().flatten()

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
    X_train_aug = np.concatenate([X_train_aug, spatial_train], axis=2)
    X_test_aug = np.concatenate([X_test_aug, spatial_test], axis=2)

    print("Training final BSGEM-HFE (full model)...")
    final_model = BSGEM_VAE(input_dim=X_train_aug.shape[2], latent_dim=8, hidden_dim=64,
                            n_regimes=4, kappa_self=0.3, omega=0.1,
                            use_endogenous=True, use_regime=True)
    final_model, train_losses, val_losses = train_vae(final_model, X_train_aug, y_train,
                                                     X_test_aug, y_test, epochs=200, patience=20)

    # Interval forecasts (sample 30 times from posterior)
    final_model.eval()
    with torch.no_grad():
        X_test_t = torch.tensor(X_test_aug, dtype=torch.float32).to(device)
        n_samples = 30
        y_pred_samples = []
        for _ in range(n_samples):
            y_pred_s, _ = final_model(X_test_t)
            y_pred_samples.append(y_pred_s.cpu().numpy().flatten())
        y_pred_samples = np.array(y_pred_samples)
        y_pred_mean = y_pred_samples.mean(axis=0)
        y_pred_lower = np.percentile(y_pred_samples, 2.5, axis=0)
        y_pred_upper = np.percentile(y_pred_samples, 97.5, axis=0)
    y_true = y_test

    mse = mean_squared_error(y_true, y_pred_mean)
    mae = mean_absolute_error(y_true, y_pred_mean)
    r2 = r2_score(y_true, y_pred_mean)
    print("\n" + "="*80)
    print("FINAL BSGEM-HFE (full model with HMM transitions, interval forecasts)")
    print("="*80)
    print(f"MSE={mse:.4f}  MAE={mae:.4f}  R²={r2:.4f}")

    # CATE with 75th percentile
    F_info_abs = np.abs(theoretical['info_gradient'][finite])
    threshold = np.percentile(F_info_abs, 75)
    treatment = (F_info_abs > threshold).astype(int)
    treat_test = treatment[-len(y_test):]
    cate_est = CausalEffectEstimator()
    X_cov_test = X_scaled[-len(y_test):]
    cate_est.fit(X_cov_test, treat_test, y_test)
    cate = cate_est.predict_cate(X_cov_test)
    avg_cate = np.mean(cate)
    print(f"Average CATE (75th percentile info shock): {avg_cate:.4f}")

    # Plotting
    plt.figure(figsize=(12,5))
    plt.plot(y_true, label='True', marker='o')
    plt.plot(y_pred_mean, label='BSGEM-HFE', marker='s', linestyle='--')
    plt.fill_between(range(len(y_true)), y_pred_lower, y_pred_upper, alpha=0.3, label='95% CI')
    plt.title('BSGEM-HFE: Actual vs Predicted Irrationality with Uncertainty')
    plt.legend()
    plt.grid(True)
    plt.show(block=False)

    plt.figure(figsize=(12,5))
    plt.plot(train_losses, label='Train loss (MSE+KL)')
    plt.plot(val_losses, label='Validation loss')
    plt.title('Training curves')
    plt.legend()
    plt.grid(True)
    plt.show(block=False)

    plt.figure(figsize=(10,4))
    plt.hist(cate, bins=30, alpha=0.7)
    plt.axvline(avg_cate, color='red', linestyle='--', label=f'Mean CATE={avg_cate:.3f}')
    plt.title('CATE distribution (info shock)')
    plt.legend()
    plt.show(block=False)

    report = {
        'dataset': {'n': len(y), 'positive_ratio': float(np.mean(y))},
        'baselines': baselines,
        'bsgem_hfe': {'mse': mse, 'mae': mae, 'r2': r2},
        'cate_avg': float(avg_cate)
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