"""
retrain.py — 3-source self-improving AI for the Stellar Anomaly Hunter.

Sources of training data:
  1. Human feedback (weight=1.0)
  2. Self-validation (weight=0.5)
  3. Self-consistency pseudo-labels (weight=0.3)
  4. LLM reviewer (weight=0.7 * confidence)

Trains a FeedbackClassifier + IsolationForest on top of TransformerAE embeddings.
"""
import os
import json
import time
import numpy as np
import joblib
from datetime import datetime

from config import MODELS_DIR

FEEDBACK_CLF_PATH = os.path.join(MODELS_DIR, 'feedback_classifier.pkl')
ISO_FOREST_PATH   = os.path.join(MODELS_DIR, 'isolation_forest.pkl')
TRAIN_LOG_PATH    = os.path.join(MODELS_DIR, 'training_log.json')
MIN_LABELS        = 15   # Minimum labels before training feedback classifier


class FeedbackClassifier:
    """Learns what humans find interesting vs noise from 3 sources."""

    def __init__(self):
        self.model = None
        self.iso_forest = None
        self.is_trained = False
        self.training_stats = {}

    def load(self):
        """Load trained models from disk if available."""
        if os.path.exists(FEEDBACK_CLF_PATH):
            self.model = joblib.load(FEEDBACK_CLF_PATH)
            self.is_trained = True
            print(f"[Retrain] ✅ Feedback classifier loaded")
        if os.path.exists(ISO_FOREST_PATH):
            self.iso_forest = joblib.load(ISO_FOREST_PATH)
            print(f"[Retrain] ✅ Isolation Forest loaded")

    def train(self, embeddings, labels, weights=None):
        """
        Train the feedback classifier.

        Args:
            embeddings: (N, 64) array of TransformerAE embeddings
            labels:     (N,) array of targets (1.0=interesting, 0.0=noise)
            weights:    (N,) optional sample weights
        """
        from sklearn.ensemble import GradientBoostingClassifier

        if len(embeddings) < MIN_LABELS:
            print(f"[Retrain] ⚠️ Only {len(embeddings)} labels, need ≥{MIN_LABELS}")
            return False

        # Convert to binary (threshold at 0.5)
        y_binary = (np.array(labels) >= 0.5).astype(int)

        # Check if we have both classes
        if len(np.unique(y_binary)) < 2:
            print(f"[Retrain] ⚠️ Need both positive and negative examples")
            return False

        print(f"[Retrain] 🧠 Training feedback classifier on {len(embeddings)} samples...")
        print(f"   Positives: {y_binary.sum()} | Negatives: {(1 - y_binary).sum()}")

        self.model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42
        )
        self.model.fit(embeddings, y_binary, sample_weight=weights)
        self.is_trained = True

        # Save
        joblib.dump(self.model, FEEDBACK_CLF_PATH)

        # Log
        self.training_stats = {
            'n_samples': len(embeddings),
            'n_positive': int(y_binary.sum()),
            'n_negative': int((1 - y_binary).sum()),
            'timestamp': datetime.now().isoformat(),
        }
        _save_training_log(self.training_stats)

        print(f"[Retrain] ✅ Feedback classifier trained and saved")
        return True

    def train_isolation_forest(self, all_embeddings):
        """
        Train Isolation Forest on ALL embeddings (unsupervised).
        Outliers = high anomaly score.
        """
        from sklearn.ensemble import IsolationForest

        print(f"[Retrain] 🌲 Training Isolation Forest on {len(all_embeddings)} embeddings...")

        self.iso_forest = IsolationForest(
            n_estimators=200,
            contamination=0.05,
            random_state=42,
            n_jobs=-1
        )
        self.iso_forest.fit(all_embeddings)

        joblib.dump(self.iso_forest, ISO_FOREST_PATH)
        print(f"[Retrain] ✅ Isolation Forest trained and saved")

    def predict_feedback_score(self, embeddings):
        """Predict how interesting each object is (0-1)."""
        if self.model is None or not self.is_trained:
            return np.full(len(embeddings), 0.5)
        return self.model.predict_proba(embeddings)[:, 1]

    def predict_if_score(self, embeddings):
        """Get Isolation Forest anomaly scores (0-1, higher = more anomalous)."""
        if self.iso_forest is None:
            return np.full(len(embeddings), 0.5)
        # IF returns negative scores for outliers, positive for inliers
        raw = self.iso_forest.decision_function(embeddings)
        # Normalize to 0-1, flip so outliers = high
        return 1.0 - _normalize(raw)


def collect_training_data(db_module, emb_store_path):
    """
    Collect training data from all 3 sources.

    Returns:
        embeddings, labels, weights, sources
    """
    all_emb, all_labels, all_weights, all_sources = [], [], [], []

    # Load embedding store
    emb_dict = _load_emb_store(emb_store_path)
    if not emb_dict:
        print("[Retrain] ⚠️ No embeddings found")
        return np.array([]), np.array([]), np.array([]), []

    # ── Source 1: Human feedback ──────────────────────────────────────
    conn = db_module.get_conn()
    rows = conn.execute(
        "SELECT oid, action, new_label FROM feedback_log"
    ).fetchall()
    conn.close()

    for row in rows:
        oid = row['oid']
        if oid not in emb_dict:
            continue
        action = row['action']
        if action == 'interesting':
            all_emb.append(emb_dict[oid])
            all_labels.append(1.0)
            all_weights.append(1.0)   # High weight: human said so
            all_sources.append('human')
        elif action == 'noise':
            all_emb.append(emb_dict[oid])
            all_labels.append(0.0)
            all_weights.append(1.0)
            all_sources.append('human')
        elif action == 'classify':
            all_emb.append(emb_dict[oid])
            all_labels.append(0.3)   # Known type = less anomalous
            all_weights.append(0.7)
            all_sources.append('human')

    n_human = len(all_emb)
    print(f"[Retrain] Source 1 (human): {n_human} labels")

    # ── Source 2: Self-validation ─────────────────────────────────────
    # Objects we flagged that later got auto-classified → false positives
    conn = db_module.get_conn()
    flagged_rows = conn.execute(
        "SELECT oid, score FROM objects WHERE triage='flagged'"
    ).fetchall()
    classified_rows = conn.execute(
        "SELECT oid, auto_class FROM objects WHERE triage='classified' AND auto_class IS NOT NULL"
    ).fetchall()
    conn.close()

    # Objects in classified that we can learn from
    for row in classified_rows:
        oid = row['oid']
        if oid in emb_dict:
            all_emb.append(emb_dict[oid])
            all_labels.append(0.2)   # Auto-classified = not anomalous
            all_weights.append(0.5)  # Medium weight: AI's own judgment
            all_sources.append('self_validation')

    n_selfval = len(all_emb) - n_human
    print(f"[Retrain] Source 2 (self-validation): {n_selfval} labels")

    # ── Source 3: Self-consistency pseudo-labels ──────────────────────
    conn = db_module.get_conn()
    all_objects = conn.execute(
        "SELECT oid, score FROM objects"
    ).fetchall()
    conn.close()

    n_pseudo = 0
    for row in all_objects:
        oid = row['oid']
        score = row['score']
        if oid not in emb_dict:
            continue

        # Very high confidence predictions as pseudo-labels
        if score >= 0.95:
            all_emb.append(emb_dict[oid])
            all_labels.append(1.0)
            all_weights.append(0.3)  # Low weight: pseudo-label
            all_sources.append('pseudo')
            n_pseudo += 1
        elif score <= 0.05:
            all_emb.append(emb_dict[oid])
            all_labels.append(0.0)
            all_weights.append(0.3)
            all_sources.append('pseudo')
            n_pseudo += 1

    print(f"[Retrain] Source 3 (pseudo-labels): {n_pseudo} labels")

    # ── Source 4: LLM feedback ────────────────────────────────────────
    try:
        from llm_interpreter import feed_llm_to_retrain
        llm_emb, llm_labels, llm_weights = feed_llm_to_retrain(db_module, emb_store_path)
        n_llm = len(llm_emb)
        if n_llm > 0:
            all_emb.extend(llm_emb.tolist())
            all_labels.extend(llm_labels.tolist())
            all_weights.extend(llm_weights.tolist())
            all_sources.extend(['llm'] * n_llm)
    except Exception:
        n_llm = 0
    print(f"[Retrain] Source 4 (LLM): {n_llm} labels")

    print(f"[Retrain] Total: {len(all_emb)} training samples")

    if not all_emb:
        return np.array([]), np.array([]), np.array([]), []

    return (np.array(all_emb), np.array(all_labels),
            np.array(all_weights), all_sources)


def self_validate(db_module):
    """
    Cross-check previously flagged objects against ALeRCE's classifier.
    If ALeRCE now classifies a flagged object → it was a false positive.
    """
    try:
        from alerce.core import Alerce
        alerce = Alerce()
    except Exception:
        print("[Retrain] ⚠️ ALeRCE not available for self-validation")
        return 0

    conn = db_module.get_conn()
    flagged = conn.execute(
        "SELECT oid FROM objects WHERE triage='flagged'"
    ).fetchall()
    conn.close()

    if not flagged:
        return 0

    n_corrected = 0
    for row in flagged:
        oid = row['oid']
        try:
            obj = alerce.query_objects(oid=oid, format='pandas')
            if obj is not None and len(obj) > 0:
                # Check if ALeRCE has classified it
                cls = obj.iloc[0].get('class', None)
                if cls and str(cls) not in ['Unknown', 'None', 'nan', '']:
                    # ALeRCE classified it — our flagging was a false positive
                    conn = db_module.get_conn()
                    conn.execute(
                        "INSERT OR REPLACE INTO feedback_log (oid, action, old_triage, label, timestamp) "
                        "VALUES (?, 'self_validate', 'flagged', ?, ?)",
                        (oid, f'ALeRCE:{cls}', datetime.now().isoformat())
                    )
                    conn.commit()
                    conn.close()
                    n_corrected += 1
                    print(f"[Self-Val] {oid}: flagged → ALeRCE says {cls} (false positive)")
            time.sleep(0.3)
        except Exception:
            continue

    print(f"[Self-Val] {n_corrected} false positives identified")
    return n_corrected


def run_retraining(scorer, db_module):
    """
    Full retraining pipeline:
    1. Collect labels from 3 sources
    2. Train feedback classifier
    3. Train/update Isolation Forest
    4. Return updated FeedbackClassifier
    """
    from feedback import EMB_STORE

    print(f"\n{'='*50}")
    print(f"  🧠 AI RETRAINING")
    print(f"{'='*50}\n")

    # Collect data
    emb, labels, weights, sources = collect_training_data(db_module, EMB_STORE)

    fc = FeedbackClassifier()

    if len(emb) >= MIN_LABELS:
        # Train feedback classifier
        fc.train(emb, labels, weights)
    else:
        print(f"[Retrain] ⏳ Not enough labels yet ({len(emb)}/{MIN_LABELS})")

    # Always train Isolation Forest on all available embeddings
    emb_dict = _load_emb_store(EMB_STORE)
    if emb_dict and len(emb_dict) >= 20:
        all_emb = np.array(list(emb_dict.values()))
        fc.train_isolation_forest(all_emb)

    print(f"\n{'='*50}")
    print(f"  ✅ RETRAINING COMPLETE")
    print(f"{'='*50}\n")

    return fc


# ── Helpers ────────────────────────────────────────────────────────────

def _normalize(x):
    mn, mx = x.min(), x.max()
    return (x - mn) / (mx - mn + 1e-10)


def _load_emb_store(path):
    """Load embedding store into dict."""
    if not os.path.exists(path):
        return {}
    try:
        data = np.load(path, allow_pickle=True)
        keys = list(data['keys'])
        vecs = data['vecs']
        return {k: vecs[i] for i, k in enumerate(keys)}
    except Exception:
        return {}


def _save_training_log(stats):
    """Append to training log."""
    log = []
    if os.path.exists(TRAIN_LOG_PATH):
        try:
            with open(TRAIN_LOG_PATH) as f:
                log = json.load(f)
        except Exception:
            pass
    log.append(stats)
    with open(TRAIN_LOG_PATH, 'w') as f:
        json.dump(log, f, indent=2)
