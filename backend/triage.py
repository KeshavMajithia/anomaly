"""
triage.py — AI auto-triage engine. Sorts objects into 3 buckets:
  - 'flagged'    → Show to human (high anomaly score + unknown)
  - 'classified' → Auto-classified as a known type
  - 'dismissed'  → Noise / low quality / matches known noise patterns
"""
import numpy as np

from config import (
    NOISE_MIN_DETECTIONS, NOISE_MAX_MAG_ERR,
    CLASSIFY_THRESHOLD, ANOMALY_THRESHOLD, NOISE_SIMILARITY_CUTOFF
)


def triage_object(score, embedding, nearest_class, class_distance,
                  n_detections, mag_err_mean, simbad_matched,
                  simbad_otype, noise_max_sim):
    """
    Determine which bucket an object belongs to.

    Args:
        score:           float [0,1] anomaly score
        embedding:       np.array (BOTTLENECK,)
        nearest_class:   str — name of nearest known class centroid
        class_distance:  float — cosine distance to nearest centroid
        n_detections:    int
        mag_err_mean:    float — mean magnitude error
        simbad_matched:  bool
        simbad_otype:    str or None
        noise_max_sim:   float — max cosine similarity to noise list

    Returns:
        (bucket, reason) where bucket is 'flagged' / 'classified' / 'dismissed'
    """

    # ── BUCKET C: Auto-dismiss (noise/low quality) ─────────────────────
    if n_detections < NOISE_MIN_DETECTIONS:
        return 'dismissed', f'Too few detections ({n_detections} < {NOISE_MIN_DETECTIONS})'

    if mag_err_mean > NOISE_MAX_MAG_ERR:
        return 'dismissed', f'High mag error ({mag_err_mean:.3f} > {NOISE_MAX_MAG_ERR})'

    if noise_max_sim > NOISE_SIMILARITY_CUTOFF:
        return 'dismissed', f'Matches known noise (sim={noise_max_sim:.3f})'

    # ── BUCKET B: Auto-classify (matches a known class centroid) ───────
    if class_distance < CLASSIFY_THRESHOLD:
        reason = f'Auto: {nearest_class} (d={class_distance:.3f})'
        if simbad_matched and simbad_otype:
            reason += f', SIMBAD={simbad_otype}'
        return 'classified', reason

    # ── BUCKET A: Flag for human (genuinely anomalous + unknown) ────────
    if score >= ANOMALY_THRESHOLD:
        reason = f'Score={score:.4f}'
        if not simbad_matched:
            reason += ' + No SIMBAD match'
        else:
            reason += f' + SIMBAD={simbad_otype}'
        return 'flagged', reason

    # ── Default: low-confidence classification ──────────────────────────
    return 'classified', f'Low-conf: {nearest_class} (d={class_distance:.3f}, s={score:.3f})'


def triage_batch(results, scorer):
    """
    Triage a list of scored objects.

    Args:
        results: list of dicts with keys from scorer + metadata
        scorer:  AnomalyScorer instance (for noise similarity)

    Returns:
        list of dicts with added 'triage' and 'triage_reason' keys
    """
    noise_emb = scorer.noise_emb
    has_noise = len(noise_emb) > 0

    triaged = []
    counts = {'flagged': 0, 'classified': 0, 'dismissed': 0}

    for obj in results:
        emb = obj['embedding']
        emb_n = emb / (np.linalg.norm(emb) + 1e-10)

        # Noise similarity
        if has_noise:
            noise_n = noise_emb / (np.linalg.norm(noise_emb, axis=1, keepdims=True) + 1e-10)
            noise_sims = noise_n @ emb_n
            noise_max = float(noise_sims.max())
        else:
            noise_max = 0.0

        # Nearest class
        nearest_cls, cls_dist = scorer.nearest_class(emb)

        bucket, reason = triage_object(
            score=obj['score'],
            embedding=emb,
            nearest_class=nearest_cls,
            class_distance=cls_dist,
            n_detections=int(obj.get('n_detections', 999)),
            mag_err_mean=float(obj.get('mag_err_mean', 0.0)),
            simbad_matched=obj.get('simbad_matched', False),
            simbad_otype=obj.get('simbad_otype', None),
            noise_max_sim=noise_max,
        )

        counts[bucket] += 1
        triaged.append({**obj,
                        'triage': bucket,
                        'triage_reason': reason,
                        'auto_class': nearest_cls,
                        'class_distance': cls_dist})

    print(f"\n🤖 Triage results:")
    print(f"   🚨 Flagged:      {counts['flagged']:>5,}")
    print(f"   🏷️  Classified:   {counts['classified']:>5,}")
    print(f"   📦 Dismissed:    {counts['dismissed']:>5,}")

    return triaged
