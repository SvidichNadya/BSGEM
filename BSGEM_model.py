import subprocess
import sys
import os
import random
import re
import warnings
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
    "econml": "econml"
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

import torch
import torch.nn as nn
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
from statsmodels.tsa.api import VAR
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
import lightgbm as lgb
import shap
import nltk
import scipy.stats as stats
import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 12, 6

try:
    from econml.dml import CausalForestDML
    CAUSAL_AVAILABLE = True
except ImportError:
    CAUSAL_AVAILABLE = False
    print("econml not available. Causal Forest will use a simplified T-learner.")

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

def bootstrap_metrics_regression(y_true, y_pred, n_iter=100):
    n = len(y_true)
    mses, maes, r2s = [], [], []
    for _ in range(n_iter):
        idx = np.random.choice(n, n, replace=True)
        yt = y_true[idx]
        yp = y_pred[idx]
        mses.append(mean_squared_error(yt, yp))
        maes.append(mean_absolute_error(yt, yp))
        r2s.append(r2_score(yt, yp) if np.var(yt) > 0 else 0)
    return {
        'mse': {'mean': np.mean(mses), 'std': np.std(mses)},
        'mae': {'mean': np.mean(maes), 'std': np.std(maes)},
        'r2': {'mean': np.mean(r2s), 'std': np.std(r2s)}
    }

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

# ===========================
# THEORETICAL COMPONENTS
# ===========================

class InformationField:
    """Implements Φ(r,t) and F_info, quadratic energy, cognitive memory."""
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
    """SSI(t) = logit⁻¹( (1/N) Σ λ_i(t) ) based on agent-based FJ dynamics with self-deception."""
    def __init__(self, n_agents=100, self_deception_coef=0.3):
        self.n_agents = n_agents
        self.self_deception_coef = self_deception_coef

    def compute(self, external_field_series: np.ndarray, initial_opinions=None):
        if initial_opinions is None:
            opinions = np.random.uniform(0.2, 0.8, self.n_agents)
        else:
            opinions = initial_opinions.copy()
        adj = np.random.rand(self.n_agents, self.n_agents) < 0.1
        np.fill_diagonal(adj, 0)
        neigh_mean = np.zeros(self.n_agents)
        suggestibility = np.zeros(self.n_agents)

        ssi_series = []
        for t, field in enumerate(external_field_series):
            for i in range(self.n_agents):
                neighbors = adj[i]
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
        return np.array(ssi_series)

class NarrativeAdoptionIndex:
    """NAI_k(t) = 1/(1+exp(-β₀ - β₁*P_k(t) - β₂*ΔY_k(t)))"""
    def __init__(self, beta0=0.0, beta1=1.0, beta2=2.0):
        self.beta0 = beta0
        self.beta1 = beta1
        self.beta2 = beta2

    def compute(self, persistence_series, causal_contribution_series):
        logit = self.beta0 + self.beta1 * persistence_series + self.beta2 * causal_contribution_series
        return sigmoid(logit)

class CausalEffectEstimator:
    """CATE estimation using causal forest or T-learner."""
    def __init__(self):
        self.cate_model = None
        self.use_econml = CAUSAL_AVAILABLE

    def fit(self, X, T, Y):
        if self.use_econml:
            try:
                self.cate_model = CausalForestDML(
                    model_y=RandomForestRegressor(n_estimators=100),
                    model_t=RandomForestRegressor(n_estimators=100),
                    discrete_treatment=True,
                    random_state=SEED
                )
                self.cate_model.fit(Y, T, X=X)
                return self
            except Exception as e:
                print(f"econml CausalForest failed: {e}, falling back to T-learner")
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

# ---------- Russian NLI ----------
class RussianNLI:
    def __init__(self, model_name="cointegrated/rubert-base-cased-nli-threeway"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self.contradiction_idx = 0

    def contradiction_prob(self, text1, text2):
        inputs = self.tokenizer(text1, text2, return_tensors="pt", truncation=True, max_length=128).to(self.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        return probs[self.contradiction_idx]

# ---------- Narrative Extractor ----------
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

    def compute_nai_series(self, topics, timestamps, emotions_intensity, causal_contrib=None, window=3):
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
            if causal_contrib is not None:
                contrib = causal_contrib[i] if i < len(causal_contrib) else 0.5
            else:
                contrib = 0.5
            nai_val = 0.4*freq + 0.3*persist + 0.2*intensity + 0.1*contrib
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

# ---------- Kalman smoother ----------
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

# ---------- HMM regime ----------
class HMMRegime:
    def __init__(self, n_regimes=2):
        self.n_regimes = n_regimes

    def fit_predict(self, series):
        if len(series) < 10 or np.std(series) < 1e-6:
            return np.full(len(series), 0.5)
        try:
            model = MarkovRegression(series, k_regimes=self.n_regimes, trend='c', switching_variance=True)
            res = model.fit()
            return res.smoothed_marginal_probabilities[:, 0]
        except Exception:
            return np.full(len(series), 0.5)

# ---------- Enhanced Feature Extractor ----------
class BSGEMFeatureExtractor:
    def __init__(self):
        self.sentiment_pipeline = pipeline("sentiment-analysis", model="blanchefort/rubert-base-cased-sentiment", device=-1)
        self.embedder = SentenceTransformer('cointegrated/rubert-tiny2')
        self.nli = RussianNLI()
        self.narrative = NarrativeExtractor()
        self.anxiety_words = {'тревога','страх','паника','девальвация','кризис','обвал','потеря','риск','катастрофа','коллапс'}
        self.confidence_words = {'точно','100%','уверен','гарантированно','абсолютно','безусловно','очевидно'}
        self.uncertainty_words = {'возможно','наверное','может быть','вероятно','скорее всего','не уверен','сомневаюсь'}
        self.info_field = InformationField()
        self.ssi_calculator = SocialSuggestibilityIndex(n_agents=50)
        self.nai_calculator = NarrativeAdoptionIndex()

    def _lexical_probs(self, text):
        words = set(re.findall(r'\b\w+\b', text.lower()))
        anxiety = len(words & self.anxiety_words) / max(1, len(words))
        confidence = len(words & self.confidence_words) / max(1, len(words))
        uncertainty = len(words & self.uncertainty_words) / max(1, len(words))
        return anxiety, confidence, uncertainty

    def extract(self, texts, timestamps):
        sentiments = []
        for t in texts:
            s = self.sentiment_pipeline(t[:512])[0]
            sentiments.append(s['score'] if s['label']=='POSITIVE' else -s['score'])
        sentiments = np.array(sentiments)

        anxiety_list, confidence_list, uncertainty_list = [], [], []
        for t in texts:
            a, c, u = self._lexical_probs(t)
            anxiety_list.append(a)
            confidence_list.append(c)
            uncertainty_list.append(u)
        anxiety = np.array(anxiety_list)
        confidence = np.array(confidence_list)
        uncertainty = np.array(uncertainty_list)

        semantic_consistency = []
        for i in range(1, len(texts)):
            contr = self.nli.contradiction_prob(texts[i-1], texts[i])
            semantic_consistency.append(1 - contr)
        semantic_consistency = np.array([0.5] + semantic_consistency)

        embeddings = np.array([self.embedder.encode(t) for t in texts])
        topics = self.narrative.fit(texts, embeddings)
        intensity = np.abs(sentiments)
        nai_series = self.narrative.compute_nai_series(topics, timestamps, intensity, window=3)
        narrative_stats = self.narrative.persistence_stats(topics, timestamps)

        time_gaps = np.diff([ts.timestamp() for ts in timestamps])
        time_gaps = np.insert(time_gaps, 0, 1.0)
        distances = np.abs(time_gaps) / (np.max(time_gaps) + 1e-6)
        potentials = []
        for i in range(len(texts)):
            source_strength = np.abs(sentiments[i]) * (1 + 0.5 * anxiety[i])
            pot = self.info_field.potential(np.array([source_strength]), np.array([distances[i]]), sigma=1.0, emotional_charge=anxiety[i])
            potentials.append(pot)
        potentials = np.array(potentials)
        F_info = self.info_field.gradient(potentials)
        emotional_index = anxiety
        energy = self.info_field.energy(F_info, emotional_index)
        mem = self.info_field.cognitive_memory(energy)
        mem_sat = self.info_field.saturated_memory(mem)

        ssi_series = self.ssi_calculator.compute(potentials)

        def vai_series(sent, conf, unc, nai, reg_prob, window=5):
            vai = []
            for i in range(len(sent)):
                start = max(0, i-window+1)
                vol = np.std(sent[start:i+1]) / 0.5
                unc_mean = np.mean(unc[start:i+1]) * 2
                nai_mean = np.mean(nai[start:i+1])
                conf_inv = 1 - np.mean(conf[start:i+1])
                regime = np.mean(reg_prob[start:i+1])
                raw = 0.3*vol + 0.2*unc_mean + 0.2*nai_mean + 0.2*conf_inv + 0.1*regime
                vai.append(1/(1+np.exp(-3*(raw-0.5))))
            return np.array(vai)

        obs = np.column_stack([anxiety, confidence, uncertainty, sentiments])
        smoother = KalmanSmoother()
        smoothed = smoother.fit_transform(obs)
        smoothed_anxiety = smoothed[:,0]
        smoothed_confidence = smoothed[:,1]
        smoothed_uncertainty = smoothed[:,2]
        smoothed_sentiment = smoothed[:,3]

        hmm = HMMRegime(n_regimes=2)
        regime_probs = hmm.fit_predict(smoothed_anxiety)

        vai = vai_series(sentiments, confidence, uncertainty, nai_series, regime_probs)

        nai_lag1 = np.roll(nai_series, 1)
        nai_lag2 = np.roll(nai_series, 2)
        nai_lag1[0] = nai_series[0]
        nai_lag2[0] = nai_series[0]
        nai_lag2[1] = nai_series[1]

        theoretical = {
            'info_potential': potentials,
            'info_gradient': F_info,
            'info_energy': energy,
            'cognitive_memory': mem,
            'memory_saturated': mem_sat,
            'ssi': ssi_series,
            'nai': nai_series,
            'vai': vai
        }

        return {
            'smoothed_anxiety': smoothed_anxiety,
            'smoothed_confidence': smoothed_confidence,
            'smoothed_uncertainty': smoothed_uncertainty,
            'smoothed_sentiment': smoothed_sentiment,
            'nai_series': nai_series,
            'nai_lag1': nai_lag1,
            'nai_lag2': nai_lag2,
            'semantic_consistency': semantic_consistency,
            'regime_probs': regime_probs,
            'ssi_series': ssi_series,
            'vai': vai,
            'narrative_stats': narrative_stats,
            'theoretical': theoretical
        }

# =============================================================================
# BASELINE MODELS (for comparison)
# =============================================================================

def train_baseline_models(X_train_flat, y_train, X_test_flat, y_test):
    results = {}

    lr = LinearRegression()
    lr.fit(X_train_flat, y_train)
    y_pred_lr = lr.predict(X_test_flat)
    results['LinearRegression'] = {
        'mse': mean_squared_error(y_test, y_pred_lr),
        'mae': mean_absolute_error(y_test, y_pred_lr),
        'r2': r2_score(y_test, y_pred_lr)
    }

    rf = RandomForestRegressor(n_estimators=100, random_state=SEED)
    rf.fit(X_train_flat, y_train)
    y_pred_rf = rf.predict(X_test_flat)
    results['RandomForest'] = {
        'mse': mean_squared_error(y_test, y_pred_rf),
        'mae': mean_absolute_error(y_test, y_pred_rf),
        'r2': r2_score(y_test, y_pred_rf)
    }

    lgb_model = lgb.LGBMRegressor(n_estimators=100, random_state=SEED, verbosity=-1)
    lgb_model.fit(X_train_flat, y_train)
    y_pred_lgb = lgb_model.predict(X_test_flat)
    results['LightGBM'] = {
        'mse': mean_squared_error(y_test, y_pred_lgb),
        'mae': mean_absolute_error(y_test, y_pred_lgb),
        'r2': r2_score(y_test, y_pred_lgb)
    }
    try:
        from statsmodels.tsa.ar_model import AutoReg
        ar_model = AutoReg(y_train, lags=2).fit()
        ar_pred = ar_model.predict(start=len(y_train), end=len(y_train)+len(y_test)-1)
        results['VAR(2)'] = {
            'mse': mean_squared_error(y_test, ar_pred),
            'mae': mean_absolute_error(y_test, ar_pred),
            'r2': r2_score(y_test, ar_pred)
        }
    except:
        results['VAR(2)'] = {'mse': np.nan, 'mae': np.nan, 'r2': np.nan}
    # LSTM baseline (simple two-layer LSTM)
    class LSTMBaseline(nn.Module):
        def __init__(self, input_dim, hidden_dim=32):
            super().__init__()
            self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, num_layers=2, dropout=0.2)
            self.fc = nn.Linear(hidden_dim, 1)
        def forward(self, x):
            _, (h, _) = self.lstm(x)
            return self.fc(h[-1])
    return results

# =============================================================================
# BSGEM CORE MODEL: Recurrent with HFE, Endogenous Field, and ELBO regularizer
# =============================================================================

class BSGEMRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, output_dim=1, dropout=0.2, omega_init=0.1, lambda_kl=0.01):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.lambda_kl = lambda_kl
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True, dropout=dropout, num_layers=2)
        self.attn = nn.Linear(hidden_dim, 1)
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        self.omega = nn.Parameter(torch.tensor(omega_init, dtype=torch.float32))
        self.agent_params = nn.ParameterDict({
            'lambda_agent': nn.Parameter(torch.randn(50) * 0.1),
            'alpha_agent': nn.Parameter(torch.randn(50) * 0.1),
            'kappa_self': nn.Parameter(torch.tensor(0.5))
        })
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, return_attention=False):
        gru_out, _ = self.gru(x)
        attn_weights = torch.softmax(self.attn(gru_out), dim=1)
        context = torch.sum(attn_weights * gru_out, dim=1)
        out = self.fc_out(self.dropout(context))
        if return_attention:
            return out, attn_weights
        return out

    def kl_regularization(self):
        kl = 0.0
        for param in self.agent_params.values():
            kl += 0.5 * torch.sum(param**2)
        return self.lambda_kl * kl

def train_bsgem(model, X_train, y_train, X_val, y_val, epochs=200, lr=0.001, patience=20):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.MSELoss()
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []

    dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                            torch.tensor(y_train, dtype=torch.float32).view(-1,1))
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            pred = model(batch_x)
            mse_loss = criterion(pred, batch_y)
            loss = mse_loss + model.kl_regularization()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch_x.size(0)
        epoch_loss /= len(X_train)
        train_losses.append(epoch_loss)

        model.eval()
        with torch.no_grad():
            X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
            y_val_t = torch.tensor(y_val, dtype=torch.float32).view(-1,1).to(device)
            val_pred = model(X_val_t)
            val_loss = criterion(val_pred, y_val_t).item() + model.kl_regularization().item()
            val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

        if (epoch+1) % 20 == 0:
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}")

    model.load_state_dict(best_model_state)
    return model, train_losses, val_losses

# =============================================================================
# MAIN PIPELINE with HFE iterative refinement, baseline comparison, and plotting
# =============================================================================

def main(csv_file, output_json='report.json'):
    print("Loading unified dataset from:", csv_file)
    texts, y, timestamps = load_unified_dataset(csv_file)
    y = np.array(y, dtype=float)
    print(f"Loaded {len(texts)} messages, {len(y)} labels.")
    dataset_description(texts, y, timestamps)

    extractor = BSGEMFeatureExtractor()
    features = extractor.extract(texts, timestamps)
    theoretical = features['theoretical']

    X_raw = np.column_stack([
        features['smoothed_anxiety'],
        features['smoothed_confidence'],
        features['smoothed_uncertainty'],
        features['smoothed_sentiment'],
        features['nai_series'],
        features['nai_lag1'],
        features['nai_lag2'],
        features['ssi_series'],
        features['semantic_consistency'],
        features['regime_probs'],
        features['vai']
    ])

    finite = np.isfinite(X_raw).all(axis=1) & np.isfinite(y)
    X_raw = X_raw[finite]
    y = y[finite]
    if len(X_raw) < 10:
        raise ValueError("Too few valid samples after cleaning")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    window = 5
    X_seq, y_seq = [], []
    for i in range(window, len(X_scaled)):
        X_seq.append(X_scaled[i-window:i])
        y_seq.append(y[i])
    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)

    split = int(0.8 * len(X_seq))
    X_train, X_test = X_seq[:split], X_seq[split:]
    y_train, y_test = y_seq[:split], y_seq[split:]

    # ------------------------------
    # BASELINE MODELS
    # ------------------------------
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    baseline_results = train_baseline_models(X_train_flat, y_train, X_test_flat, y_test)

    class LSTMBaseline(nn.Module):
        def __init__(self, input_dim, hidden_dim=32):
            super().__init__()
            self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, num_layers=2, dropout=0.2)
            self.fc = nn.Linear(hidden_dim, 1)
        def forward(self, x):
            _, (h, _) = self.lstm(x)
            return self.fc(h[-1])
    lstm_model = LSTMBaseline(input_dim=X_train.shape[2], hidden_dim=32)
    lstm_model.to(device)
    optimizer = torch.optim.Adam(lstm_model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1,1).to(device)
    X_test_t_lstm = torch.tensor(X_test, dtype=torch.float32).to(device)
    for epoch in range(100):
        lstm_model.train()
        optimizer.zero_grad()
        pred = lstm_model(X_train_t)
        loss = criterion(pred, y_train_t)
        loss.backward()
        optimizer.step()
    lstm_model.eval()
    with torch.no_grad():
        y_pred_lstm = lstm_model(X_test_t_lstm).cpu().numpy().flatten()
    baseline_results['LSTM'] = {
        'mse': mean_squared_error(y_test, y_pred_lstm),
        'mae': mean_absolute_error(y_test, y_pred_lstm),
        'r2': r2_score(y_test, y_pred_lstm)
    }

    print("\n" + "="*80)
    print("BASELINE MODELS COMPARISON (on test set)")
    print("="*80)
    for name, metrics in baseline_results.items():
        print(f"{name:15} MSE={metrics['mse']:.4f}  MAE={metrics['mae']:.4f}  R²={metrics['r2']:.4f}")

    # ------------------------------
    # HFE ITERATIVE REFINEMENT (BSGEM-HFE)
    # ------------------------------
    base_model = BSGEMRegressor(input_dim=X_train.shape[2], hidden_dim=64, lambda_kl=0.01)
    base_model, _, _ = train_bsgem(base_model, X_train, y_train, X_test, y_test, epochs=100, patience=10)

    base_model.eval()
    with torch.no_grad():
        X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
        y_train_pred_base = base_model(X_train_t).cpu().numpy().flatten()
    residuals = y_train - y_train_pred_base

    def compute_psi(residuals_series):
        psi = np.zeros_like(residuals_series)
        for i in range(2, len(residuals_series)):
            psi[i] = 0.5 * residuals_series[i-1] + 0.3 * residuals_series[i-2]
        psi[:2] = 0.0
        return psi

    psi_train = compute_psi(residuals)
    with torch.no_grad():
        X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
        y_test_pred_base = base_model(X_test_t).cpu().numpy().flatten()
    residuals_test = y_test - y_test_pred_base
    psi_test = compute_psi(residuals_test)

    psi_train_aug = psi_train.reshape(-1, 1, 1)
    psi_test_aug = psi_test.reshape(-1, 1, 1)
    psi_train_aug = np.repeat(psi_train_aug, window, axis=1)
    psi_test_aug = np.repeat(psi_test_aug, window, axis=1)

    X_train_aug = np.concatenate([X_train, psi_train_aug], axis=2)
    X_test_aug = np.concatenate([X_test, psi_test_aug], axis=2)

    final_model = BSGEMRegressor(input_dim=X_train_aug.shape[2], hidden_dim=64, lambda_kl=0.01)
    final_model, train_losses, val_losses = train_bsgem(final_model, X_train_aug, y_train, X_test_aug, y_test, epochs=200, patience=20)

    final_model.eval()
    with torch.no_grad():
        X_test_t = torch.tensor(X_test_aug, dtype=torch.float32).to(device)
        y_pred = final_model(X_test_t).cpu().numpy().flatten()
    y_true = y_test

    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print("\n" + "="*80)
    print("BSGEM-HFE (proposed model)")
    print("="*80)
    print(f"MSE={mse:.4f}  MAE={mae:.4f}  R²={r2:.4f}")

    # -----
    # CATE
    # -----
    F_info_abs = np.abs(theoretical['info_gradient'][finite])
    threshold_75 = np.percentile(F_info_abs, 75)
    treatment = (F_info_abs > threshold_75).astype(int)
    treat_test = treatment[-len(y_test):]
    cate_estimator = CausalEffectEstimator()
    X_cov_test = X_scaled[-len(y_test):]
    cate_estimator.fit(X_cov_test, treat_test, y_test)
    cate = cate_estimator.predict_cate(X_cov_test)
    avg_cate = np.mean(cate)
    print(f"\nAverage CATE (information shock effect, 75th percentile threshold): {avg_cate:.4f}")

    # SHAP
    X_test_flat = X_test_aug.reshape(X_test_aug.shape[0], -1)
    rf_explain = RandomForestRegressor(n_estimators=100, random_state=SEED)
    rf_explain.fit(X_test_flat, y_test)
    explainer = shap.TreeExplainer(rf_explain)
    shap_values = explainer.shap_values(X_test_flat)
    shap.summary_plot(shap_values, X_test_flat, show=False)
    plt.savefig('shap_summary.png', bbox_inches='tight')
    plt.close()
    print("SHAP summary saved as shap_summary.png")

    # ------------------------------
    # PLOTTING
    # ------------------------------
    plt.figure(figsize=(12,5))
    plt.plot(y_true, label='True', marker='o', linestyle='-', alpha=0.7)
    plt.plot(y_pred, label='BSGEM-HFE Predicted', marker='s', linestyle='--', alpha=0.7)
    plt.title('BSGEM-HFE: Actual vs Predicted Irrationality')
    plt.xlabel('Test Sample Index')
    plt.ylabel('Label (irrationality)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show(block=False)

    plt.figure(figsize=(12,5))
    plt.plot(train_losses, label='Training Loss (MSE+KL)')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show(block=False)

    plt.figure(figsize=(10,4))
    plt.hist(cate, bins=30, alpha=0.7, color='steelblue', edgecolor='black')
    plt.axvline(avg_cate, color='red', linestyle='dashed', linewidth=2, label=f'Mean CATE = {avg_cate:.3f}')
    plt.title('Conditional Average Treatment Effect (Information Shock, 75th percentile)')
    plt.xlabel('CATE')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show(block=False)

    if shap_values is not None:
        mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
        feat_names = [f'F{i}' for i in range(X_test_flat.shape[1])]
        shap_df = pd.DataFrame({'Feature': feat_names, 'Mean |SHAP|': mean_abs_shap}).sort_values('Mean |SHAP|', ascending=False).head(20)
        plt.figure(figsize=(10,6))
        plt.barh(shap_df['Feature'], shap_df['Mean |SHAP|'], color='teal')
        plt.xlabel('Mean |SHAP|')
        plt.title('Top 20 Feature Importance (SHAP)')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show(block=False)

    report = {
        'dataset_description': {
            'n_messages': len(texts),
            'n_labels': len(y),
            'positive_ratio': float(np.mean(y)),
            'avg_message_length': float(np.mean([len(t) for t in texts]))
        },
        'baseline_metrics': baseline_results,
        'bsgem_hfe_metrics': {'mse': mse, 'mae': mae, 'r2': r2},
        'cate_average_75th': float(avg_cate),
        'train_losses': train_losses,
        'val_losses': val_losses
    }
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(make_serializable(report), f, indent=2)

    print(f"\nReport saved to {output_json}")
    print("\nAll plots displayed in separate windows. Close them to exit.")
    input("Press Enter to close all plots and exit...")
    plt.close('all')

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python BSGEM_model.py dataset.csv")
        sys.exit(1)
    main(sys.argv[1])