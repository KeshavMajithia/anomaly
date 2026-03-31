"""
rescan.py — Retrospective re-evaluation of auto-dismissed objects.

Runs the current (possibly updated) model over everything in Bucket C,
and promotes objects that now score above the anomaly threshold.
"""
import numpy as np

import database as db
from preprocessor import batch_preprocess
from triage import triage_object
from config import ANOMALY_THRESHOLD


def run_rescan(scorer, lc_df, verbose=True):
    """
    Re-score all dismissed objects with the current (smarter) model.
    Promotes anything that now looks interesting to 'flagged' status.

    Args:
        scorer:  AnomalyScorer instance (updated since last triage)
        lc_df:   DataFrame with light curves for dismissed objects
                 (loaded from the original IRSA/ALeRCE cache)
        verbose: print progress

    Returns:
        n_promoted: number of objects promoted from dismissed → flagged
    """
    dismissed = db.get_dismissed()
    if not dismissed:
        if verbose:
            print("📦 No dismissed objects to rescan.")
        return 0

    if verbose:
        print(f"🔄 Rescanning {len(dismissed):,} dismissed objects with updated model...")

    oids = [o['oid'] for o in dismissed]
    meta = {o['oid']: o for o in dismissed}

    # Preprocess available light curves
    X_arr, M_arr, P_arr, valid_oids = batch_preprocess(lc_df, oids, scorer.scaler)
    if len(X_arr) == 0:
        if verbose:
            print("   ⚠️ No light curves available for dismissed objects.")
        return 0

    # Score
    scored = scorer.score_batch(X_arr, M_arr)
    promoted = 0

    conn = db.get_conn()

    for i, oid in enumerate(valid_oids):
        new_score = float(scored['scores'][i])
        emb = scored['embeddings'][i]
        m = meta.get(oid, {})

        nearest_cls, cls_dist = scorer.nearest_class(emb)

        # Noise similarity
        noise_emb = scorer.noise_emb
        if len(noise_emb) > 0:
            emb_n = emb / (np.linalg.norm(emb) + 1e-10)
            noise_n = noise_emb / (np.linalg.norm(noise_emb, axis=1, keepdims=True) + 1e-10)
            noise_max = float((noise_n @ emb_n).max())
        else:
            noise_max = 0.0

        bucket, reason = triage_object(
            score=new_score,
            embedding=emb,
            nearest_class=nearest_cls,
            class_distance=cls_dist,
            n_detections=int(m.get('n_detections', 999)),
            mag_err_mean=float(m.get('mag_err_mean', 0.0)),
            simbad_matched=False,
            simbad_otype=None,
            noise_max_sim=noise_max,
        )

        if bucket == 'flagged':
            conn.execute("""
                UPDATE objects
                SET triage='flagged', score=?, triage_reason=?, auto_class=?
                WHERE oid=?
            """, (new_score, f'Rescan: {reason}', nearest_cls, oid))
            promoted += 1
            if verbose:
                print(f"   🔄 Promoted: {oid} (score={new_score:.4f})")

    conn.commit()
    conn.close()

    if verbose:
        print(f"\n✅ Rescan complete: {promoted} objects promoted from dismissed → flagged")

    return promoted
