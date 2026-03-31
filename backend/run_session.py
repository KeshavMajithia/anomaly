"""
run_session.py — Main entry point for a daily Stellar Anomaly Hunter session.

Usage:
    cd anomaly_hunter/backend
    python run_session.py          # Full session (1000 objects, ~15 min)
    python run_session.py --demo   # Quick test (50 objects, ~2 min)
"""
import os
import sys
import time
import argparse

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from config import SIMBAD_CHECK_TOP_N, LOOKBACK_HOURS, MAX_ALERTS, DEMO_ALERTS, LLM_ENABLED
from scorer import AnomalyScorer
from preprocessor import batch_preprocess
from poller import fetch_recent_alerts
from simbad_checker import check_simbad
from triage import triage_batch
from feedback import save_embeddings
from retrain import FeedbackClassifier, run_retraining, self_validate
import database as db
import app as flask_app


def run_session(demo=False, args=None):
    t0 = time.time()
    n_alerts = DEMO_ALERTS if demo else MAX_ALERTS
    mode_str = "DEMO (fast)" if demo else "FULL"

    print("=" * 60)
    print(f"  🌌 STELLAR ANOMALY HUNTER — {mode_str} SESSION")
    print("=" * 60)
    if demo:
        print(f"  ⚡ Demo mode: {n_alerts} objects, no SIMBAD check")

    # ── Step 1: Load model + AI improvements ───────────────────────────
    print("\n[1/8] Loading trained model...")
    db.init_db()
    scorer = AnomalyScorer()

    # Load feedback classifier + isolation forest (if previously trained)
    fc = FeedbackClassifier()
    fc.load()
    scorer.feedback_clf = fc
    print(f"   ✅ Model ready with {len(scorer.noise_emb)} noise embeddings")
    if fc.is_trained:
        print(f"   🧠 Feedback classifier ACTIVE (learned from previous sessions)")
    if fc.iso_forest is not None:
        print(f"   🌲 Isolation Forest ACTIVE")

    # ── Step 1.5: Self-validation ──────────────────────────────────────
    if not demo:
        print(f"\n[1.5/8] Self-validating previous predictions...")
        n_corrected = self_validate(db)
        if n_corrected > 0:
            print(f"   ✅ Found {n_corrected} false positives via self-validation")
        else:
            print(f"   ✅ No corrections needed")

    # ── Step 2: Fetch alerts ───────────────────────────────────────────
    print(f"\n[2/8] Fetching last {LOOKBACK_HOURS}h of ZTF alerts...")
    meta_df, lc_df = fetch_recent_alerts(
        hours_back=LOOKBACK_HOURS, max_alerts=n_alerts)

    if len(meta_df) == 0:
        print("   ⚠️ No alerts fetched. Exiting.")
        return

    print(f"   ✅ {len(meta_df):,} objects | {lc_df['oid'].nunique() if len(lc_df) else 0:,} with LCs")

    # ── Step 3: Preprocess + Score ─────────────────────────────────────
    print(f"\n[3/8] Preprocessing and scoring...")
    oids = lc_df['oid'].unique().tolist() if len(lc_df) else []
    X_arr, M_arr, P_arr, valid_oids = batch_preprocess(lc_df, oids, scorer.scaler)
    print(f"   Valid: {len(valid_oids):,} | Dropped: {len(oids) - len(valid_oids):,}")

    if len(X_arr) == 0:
        print("   ⚠️ No valid light curves to score. Exiting.")
        return

    scored = scorer.score_batch(X_arr, M_arr)
    save_embeddings(valid_oids, scored['embeddings'])
    print(f"   ✅ Scored {len(valid_oids):,} objects")

    # Build per-object metadata dict
    meta_dict = {}
    if 'oid' in meta_df.columns:
        for _, row in meta_df.iterrows():
            oid = str(row['oid'])
            meta_dict[oid] = {
                'ra': row.get('meanra', row.get('ra', np.nan)),
                'dec': row.get('meandec', row.get('dec', np.nan)),
                'n_detections': int(row.get('ndet', row.get('n_detections', 0))),
                'mag_mean': float(row.get('magmean', row.get('mag_mean', np.nan))),
                'mag_err_mean': float(row.get('sigmamean', row.get('mag_err_mean', 0.1))),
            }

    # Assemble results list
    results = []
    for i, oid in enumerate(valid_oids):
        m = meta_dict.get(oid, {})
        results.append({
            'oid': oid,
            'score': float(scored['scores'][i]),
            'rec_error': float(scored['rec_errors'][i]),
            'knn_dist': float(scored['knn_dists'][i]),
            'embedding': scored['embeddings'][i],
            'ra': m.get('ra', np.nan),
            'dec': m.get('dec', np.nan),
            'n_detections': m.get('n_detections', 0),
            'mag_mean': m.get('mag_mean', np.nan),
            'mag_err_mean': m.get('mag_err_mean', np.nan),
            'simbad_matched': False,
            'simbad_match': None,
            'simbad_otype': None,
        })

    # Sort by score descending
    results.sort(key=lambda x: x['score'], reverse=True)

    # ── Step 4: SIMBAD check top objects ───────────────────────────────
    if demo:
        print(f"\n[4/8] Skipping SIMBAD check (demo mode)")
    else:
        n_check = min(SIMBAD_CHECK_TOP_N, len(results))
        print(f"\n[4/8] SIMBAD checking top {n_check} objects...")
        top = results[:n_check]
        simbad_results = check_simbad(
            oids=[r['oid'] for r in top],
            ras=[r['ra'] for r in top],
            decs=[r['dec'] for r in top],
        )
        for r in results[:n_check]:
            s = simbad_results.get(r['oid'], {})
            r['simbad_matched'] = s.get('matched', False)
            r['simbad_match']   = s.get('match', None)
            r['simbad_otype']   = s.get('otype', None)

    # ── Step 5: AI Triage ──────────────────────────────────────────────
    print(f"\n[5/8] 🤖 Running AI triage...")
    triaged = triage_batch(results, scorer)

    n_flagged     = sum(1 for o in triaged if o['triage'] == 'flagged')
    n_classified  = sum(1 for o in triaged if o['triage'] == 'classified')
    n_dismissed   = sum(1 for o in triaged if o['triage'] == 'dismissed')

    # ── Step 6: LLM Review ────────────────────────────────────────────
    llm_results = []
    use_llm = LLM_ENABLED and not getattr(args, 'no_llm', False)
    if use_llm:
        print(f"\n[6/8] 🧠 LLM astronomer reviewing flagged objects...")
        try:
            from llm_interpreter import batch_review
            flagged_objs = [o for o in triaged if o['triage'] == 'flagged']
            if flagged_objs:
                llm_results = batch_review(flagged_objs, db)
            else:
                print("   No flagged objects to review")
        except Exception as e:
            print(f"   ⚠️ LLM review failed: {e}")
    else:
        print(f"\n[6/8] 🧠 LLM review: {'disabled' if not LLM_ENABLED else 'skipped (--no-llm)'}")

    # ── Step 7: Save to DB ─────────────────────────────────────────────
    print(f"\n[7/8] 💾 Saving to database...")
    for obj in triaged:
        obj.pop('embedding', None)
    db.upsert_objects(triaged)

    # ── Step 8: Auto-retrain if enough feedback ────────────────────────
    print(f"\n[8/8] 🧠 Checking if retraining is needed...")
    fc_updated = run_retraining(scorer, db)
    scorer.feedback_clf = fc_updated

    elapsed = time.time() - t0
    n_candidates = sum(1 for r in llm_results if r.get('is_candidate'))
    print(f"\n{'='*60}")
    print(f"  ✅ SESSION COMPLETE ({elapsed:.0f}s)")
    print(f"{'='*60}")
    print(f"  Objects scored:   {len(triaged):,}")
    print(f"  🚨 Flagged:       {n_flagged:,}")
    print(f"  🏷️  Classified:    {n_classified:,}")
    print(f"  📦 Dismissed:     {n_dismissed:,}")
    if llm_results:
        n_interesting = sum(1 for r in llm_results if r['verdict'] == 'interesting')
        print(f"  🧠 LLM reviewed:  {len(llm_results)} ({n_interesting} interesting, {n_candidates} candidates)")
    if fc_updated.is_trained:
        print(f"  🧠 AI retrained:  YES (feedback classifier active)")
    if fc_updated.iso_forest is not None:
        print(f"  🌲 Isolation Forest: ACTIVE")
    print(f"{'='*60}\n")

    # ── Open dashboard ─────────────────────────────────────────────────
    flask_app.run(scorer=scorer, lc_df=lc_df)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Stellar Anomaly Hunter Session')
    parser.add_argument('--demo', action='store_true',
                        help=f'Demo mode: fetch only {DEMO_ALERTS} objects (~2 min)')
    parser.add_argument('--no-llm', action='store_true',
                        help='Skip LLM review step')
    args = parser.parse_args()
    run_session(demo=args.demo, args=args)


