"""
feedback.py — Process human feedback and update the model's noise/class knowledge.

Feedback actions:
  👍 'interesting' → log as discovery
  👎 'noise'       → add embedding to noise filter, move to dismissed
  🏷️ 'classify'    → update class label
"""
import os
import numpy as np

import database as db
from config import NOISE_LIST


# ── Embedding store ────────────────────────────────────────────────────
# Embeddings are saved to disk so feedback can retrieve them after scoring
EMB_STORE = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'database', 'emb_store.npz')
_emb_cache = {}   # in-memory: {oid: embedding}


def save_embeddings(oids, embeddings):
    """Save embeddings for all scored objects, flush immediately."""
    global _emb_cache
    for oid, emb in zip(oids, embeddings):
        _emb_cache[oid] = emb
    # Always flush after saving - critical for feedback to work
    _flush_embeddings()
    print(f"   [Embeddings] Saved {len(_emb_cache)} embeddings to disk")


def _flush_embeddings():
    """Write all cached embeddings to disk."""
    global _emb_cache
    if not _emb_cache:
        return
    # Load existing if present
    existing_keys, existing_vecs = [], []
    if os.path.exists(EMB_STORE):
        try:
            data = np.load(EMB_STORE, allow_pickle=True)
            existing_keys = list(data['keys'])
            existing_vecs = list(data['vecs'])
        except Exception:
            pass

    # Merge: existing + new (new overwrites duplicates)
    merged = {}
    for k, v in zip(existing_keys, existing_vecs):
        merged[k] = v
    for k, v in _emb_cache.items():
        merged[k] = v

    keys = list(merged.keys())
    vecs = np.stack(list(merged.values()))
    np.savez_compressed(EMB_STORE, keys=np.array(keys), vecs=vecs)


def _get_embedding(oid):
    """Retrieve embedding for a given OID."""
    # Check in-memory cache first
    if oid in _emb_cache:
        return _emb_cache[oid]
    # Check disk
    if os.path.exists(EMB_STORE):
        try:
            data = np.load(EMB_STORE, allow_pickle=True)
            keys = list(data['keys'])
            if oid in keys:
                idx = keys.index(oid)
                return data['vecs'][idx]
        except Exception:
            pass
    return None


# ── Feedback handlers ──────────────────────────────────────────────────

def handle_feedback(oid, action, scorer, label=None):
    """
    Process a single feedback action from the dashboard.

    Args:
        oid:     ZTF object ID
        action:  'interesting', 'noise', or 'classify'
        scorer:  AnomalyScorer instance (to update noise list)
        label:   class name if action == 'classify'
    """
    # Get object's current triage from DB
    conn = db.get_conn()
    row = conn.execute("SELECT * FROM objects WHERE oid=?", (oid,)).fetchone()
    conn.close()

    if row is None:
        print(f"[Feedback] ⚠️ OID {oid} not found in DB")
        return {'status': 'error', 'message': 'Object not found'}

    old_triage = row['triage']

    if action == 'interesting':
        # Add to discoveries table
        db.add_discovery(
            oid=oid,
            score=row['score'],
            ra=row['ra'],
            dec=row['dec'],
            simbad_match=row['simbad_match']
        )
        db.save_feedback(oid, 'interesting', old_triage, 'confirmed_interesting')
        # Move out of flagged so it leaves the Review tab
        conn = db.get_conn()
        conn.execute(
            "UPDATE objects SET triage='confirmed', triage_reason='Human: confirmed interesting' WHERE oid=?",
            (oid,))
        conn.commit()
        conn.close()
        print(f"[Feedback] 🌟 {oid} → DISCOVERY logged")
        return {'status': 'ok', 'message': 'Logged as discovery'}

    elif action == 'noise':
        # Get object's embedding and add to noise list
        emb = _get_embedding(oid)
        if emb is not None:
            scorer.add_noise(emb)
            print(f"[Feedback] 🚫 {oid} → noise list updated (total: {len(scorer.noise_emb)})")
        else:
            print(f"[Feedback] ⚠️ {oid} → no embedding found, noise list NOT updated")

        db.save_feedback(oid, 'noise', old_triage, 'noise')
        # Move to dismissed
        conn = db.get_conn()
        conn.execute(
            "UPDATE objects SET triage='dismissed', triage_reason='Human: noise' WHERE oid=?",
            (oid,))
        conn.commit()
        conn.close()
        return {'status': 'ok', 'message': 'Marked as noise, embedding added to filter'}

    elif action == 'classify':
        if not label:
            return {'status': 'error', 'message': 'Label required for classify action'}

        db.save_feedback(oid, 'classify', old_triage, label)
        conn = db.get_conn()
        conn.execute(
            "UPDATE objects SET triage='classified', auto_class=?, triage_reason=? WHERE oid=?",
            (label, f'Human: classified as {label}', oid)
        )
        conn.commit()
        conn.close()

        # Update class centroid if we have the embedding
        emb = _get_embedding(oid)
        if emb is not None:
            _update_centroid(scorer, label, emb)

        print(f"[Feedback] 🏷️ {oid} → classified as {label}")
        return {'status': 'ok', 'message': f'Classified as {label}'}

    return {'status': 'error', 'message': f'Unknown action: {action}'}


def _update_centroid(scorer, class_name, new_embedding):
    """
    Update a class centroid incrementally with a new labeled example.
    Uses running average: new_centroid = (old * n + new) / (n + 1)
    """
    import json
    from config import CLASS_CENTROIDS

    if class_name in scorer.class_centroids:
        old = np.array(scorer.class_centroids[class_name])
        # Approximate running average (assume each class had ~50 examples)
        n_est = 50
        updated = (old * n_est + new_embedding) / (n_est + 1)
        scorer.class_centroids[class_name] = updated.tolist()
    else:
        scorer.class_centroids[class_name] = new_embedding.tolist()

    # Update centroid matrix
    scorer.centroid_names = list(scorer.class_centroids.keys())
    scorer.centroid_matrix = np.array(
        [scorer.class_centroids[k] for k in scorer.centroid_names], dtype=np.float32)

    # Persist to disk
    try:
        with open(CLASS_CENTROIDS, 'w') as f:
            json.dump(scorer.class_centroids, f)
        print(f"[Feedback] 📊 Centroid updated for class '{class_name}'")
    except Exception as e:
        print(f"[Feedback] ⚠️ Could not save centroids: {e}")
