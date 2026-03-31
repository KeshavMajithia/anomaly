"""
scorer.py — Load trained models and score new ZTF light curves for anomalousness.
Uses the exported TransformerAE weights + reference embeddings from Kaggle.
"""
import json
import math
import os

import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import (
    TRANSFORMER_WEIGHTS, MASKED_AE_WEIGHTS, SCALER_FILE,
    REF_EMBEDDINGS, CLASS_CENTROIDS, CONFIG_FILE,
    NOISE_LIST, MODELS_DIR
)

# ── Model hyperparams (overridden by config.json at load time) ─────────
N_BINS      = 50
N_FEAT      = 12
BOTTLENECK  = 64


# ── Architecture (must match Kaggle training code exactly) ─────────────

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[:d_model // 2]) if d_model % 2 else torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class TransformerAE(nn.Module):
    def __init__(self, n_feat=N_FEAT, n_bins=N_BINS, bottleneck=BOTTLENECK):
        super().__init__()
        self.n_bins  = n_bins
        d = 128
        self.inp     = nn.Linear(n_feat, d)
        self.pos     = PositionalEncoding(d, n_bins)
        self.enc     = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d, 4, 256, 0.1, activation='gelu', batch_first=True), 4)
        self.bn      = nn.Linear(d, bottleneck)
        self.dec_proj = nn.Linear(bottleneck, d)
        self.dec     = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d, 4, 256, 0.1, activation='gelu', batch_first=True), 2)
        self.out     = nn.Linear(d, n_feat)

    def encode(self, x):
        return self.bn(self.enc(self.pos(self.inp(x))).mean(dim=1))

    def decode(self, z):
        h = self.dec_proj(z).unsqueeze(1).expand(-1, self.n_bins, -1)
        return self.out(self.dec(self.pos(h)))

    def forward(self, x, mask_ratio=0.3):
        B, T, _ = x.shape
        noise = torch.rand(B, T, device=x.device)
        mask = (noise.argsort(dim=1) >= int(T * mask_ratio)).float()
        z = self.encode(x * mask.unsqueeze(-1))
        return self.decode(z), z, mask


def _normalize(x):
    mn, mx = x.min(), x.max()
    return (x - mn) / (mx - mn + 1e-10)


class AnomalyScorer:
    """
    Loads trained model weights and scores objects by anomalousness.
    Scores range [0, 1]. Higher = more anomalous.
    """

    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.feedback_clf = None  # Set by retrain.py
        print(f"[Scorer] Device: {self.device}")
        self._load()

    def _load(self):
        # Load config
        cfg = json.load(open(CONFIG_FILE))
        global N_BINS, N_FEAT, BOTTLENECK
        N_BINS     = cfg.get('N_BINS', N_BINS)
        N_FEAT     = cfg.get('N_FEAT', N_FEAT)
        BOTTLENECK = cfg.get('BOTTLENECK', BOTTLENECK)
        self.classes = cfg.get('classes', [])

        # Load scaler
        self.scaler = joblib.load(SCALER_FILE)

        # Load TransformerAE
        self.model = TransformerAE(N_FEAT, N_BINS, BOTTLENECK).to(self.device)
        self.model.load_state_dict(
            torch.load(TRANSFORMER_WEIGHTS, map_location=self.device))
        self.model.eval()
        print(f"[Scorer] TransformerAE loaded ({sum(p.numel() for p in self.model.parameters()):,} params)")

        # Load reference embeddings (kNN baseline)
        self.ref_emb = np.load(REF_EMBEDDINGS)
        print(f"[Scorer] Reference embeddings: {self.ref_emb.shape}")

        # Load class centroids (optional — if missing, auto-classification is disabled)
        if os.path.exists(CLASS_CENTROIDS):
            self.class_centroids = json.load(open(CLASS_CENTROIDS))
            self.centroid_names  = list(self.class_centroids.keys())
            self.centroid_matrix = np.array([self.class_centroids[k]
                                              for k in self.centroid_names], dtype=np.float32)
            print(f"[Scorer] Class centroids: {self.centroid_names}")
        else:
            self.class_centroids = {}
            self.centroid_names  = []
            self.centroid_matrix = np.empty((0, BOTTLENECK), dtype=np.float32)
            print("[Scorer] ⚠️  class_centroids.json not found — auto-classification disabled")
            print("             Add the export cell to Kaggle and re-download, or centroids")
            print("             will be built automatically from labeled feedback over time.")

        # Load noise list (grows with feedback)
        if os.path.exists(NOISE_LIST):
            self.noise_emb = np.load(NOISE_LIST)
            print(f"[Scorer] Noise embeddings: {len(self.noise_emb)}")
        else:
            self.noise_emb = np.empty((0, BOTTLENECK), dtype=np.float32)
            print(f"[Scorer] No noise list yet")

    def encode(self, X_tensor):
        """Extract embedding from preprocessed tensor. X: (N, N_BINS, N_FEAT)"""
        self.model.eval()
        with torch.no_grad():
            x = torch.FloatTensor(X_tensor).to(self.device)
            emb = self.model.encode(x)
            recon = self.model.decode(emb)
            rec_err = ((recon - x) ** 2).mean(dim=(1, 2))
        return emb.cpu().numpy(), rec_err.cpu().numpy()

    def score_batch(self, X_arr, M_arr):
        """
        Score a batch of preprocessed objects.
        Blends: TransformerAE (rec+knn) + IsolationForest + FeedbackClassifier

        Args:
            X_arr: (N, N_BINS, N_FEAT) float32
            M_arr: (N, N_BINS) float32 mask

        Returns:
            dict with keys: scores, embeddings, rec_errors, knn_dists,
                            if_scores, feedback_scores
        """
        if len(X_arr) == 0:
            return {'scores': np.array([]), 'embeddings': np.empty((0, BOTTLENECK)),
                    'rec_errors': np.array([]), 'knn_dists': np.array([]),
                    'if_scores': np.array([]), 'feedback_scores': np.array([])}

        emb, rec_err = self.encode(X_arr)

        # kNN distance to reference embeddings
        ref_norm = self.ref_emb / (np.linalg.norm(self.ref_emb, axis=1, keepdims=True) + 1e-10)
        emb_norm = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-10)
        sim = emb_norm @ ref_norm.T
        top11 = np.sort(sim, axis=1)[:, -11:]
        knn_dist = 1 - top11[:, 1:].mean(axis=1)

        # Noise similarity penalty
        noise_penalty = np.zeros(len(emb))
        if len(self.noise_emb) > 0:
            noise_norm = self.noise_emb / (
                np.linalg.norm(self.noise_emb, axis=1, keepdims=True) + 1e-10)
            noise_sim = emb_norm @ noise_norm.T
            noise_penalty = noise_sim.max(axis=1)

        # Base anomaly score (TransformerAE)
        raw_ae = _normalize(rec_err) + _normalize(knn_dist) - 0.5 * noise_penalty
        ae_score = _normalize(np.clip(raw_ae, 0, None))

        # Isolation Forest score (if trained)
        if_scores = np.full(len(emb), 0.5)
        if self.feedback_clf is not None:
            if_scores = self.feedback_clf.predict_if_score(emb)

        # Feedback classifier score (if trained)
        fb_scores = np.full(len(emb), 0.5)
        if self.feedback_clf is not None and self.feedback_clf.is_trained:
            fb_scores = self.feedback_clf.predict_feedback_score(emb)

        # Blended final score
        if self.feedback_clf is not None and self.feedback_clf.is_trained:
            # Full blend: 0.4 × AE + 0.3 × IF + 0.3 × feedback
            scores = 0.4 * ae_score + 0.3 * if_scores + 0.3 * fb_scores
            print(f"   [Score] Blended: AE + IF + Feedback")
        elif self.feedback_clf is not None and self.feedback_clf.iso_forest is not None:
            # IF only: 0.6 × AE + 0.4 × IF
            scores = 0.6 * ae_score + 0.4 * if_scores
            print(f"   [Score] Blended: AE + IF")
        else:
            # Original: AE only
            scores = ae_score

        return {
            'scores':          scores,
            'embeddings':      emb,
            'rec_errors':      rec_err,
            'knn_dists':       knn_dist,
            'if_scores':       if_scores,
            'feedback_scores': fb_scores,
        }

    def nearest_class(self, embedding):
        """
        Find the nearest known class centroid.

        Returns:
            (class_name, cosine_distance)
        """
        if len(self.centroid_matrix) == 0:
            return 'unknown', 1.0
        emb_n = embedding / (np.linalg.norm(embedding) + 1e-10)
        cent_n = self.centroid_matrix / (
            np.linalg.norm(self.centroid_matrix, axis=1, keepdims=True) + 1e-10)
        sims = cent_n @ emb_n
        best_idx = sims.argmax()
        return self.centroid_names[best_idx], float(1 - sims[best_idx])

    def add_noise(self, embedding):
        """Add an embedding to the noise list (from 👎 feedback)."""
        self.noise_emb = np.vstack([self.noise_emb, embedding.reshape(1, -1)])
        np.save(NOISE_LIST, self.noise_emb)

    def reload_noise(self):
        """Reload noise list from disk (after external update)."""
        if os.path.exists(NOISE_LIST):
            self.noise_emb = np.load(NOISE_LIST)
