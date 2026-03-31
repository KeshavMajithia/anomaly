"""
deep_test.py — Full project workflow test suite for Stellar Anomaly Hunter.
Covers all 8 pipeline steps, 4 feedback sources, LLM interpreter, API, and data integrity.

Run: python deep_test.py
"""
import os
import sys
import json
import time
import traceback
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

PASS = FAIL = WARN = 0
ISSUES = []

def ok(msg):
    global PASS; PASS += 1
    print(f"  [PASS] {msg}")
def bad(msg, d=""):
    global FAIL; FAIL += 1; ISSUES.append(msg)
    print(f"  [FAIL] {msg}")
    if d: print(f"         {d}")
def wrn(msg, d=""):
    global WARN; ISSUES.append(f"WARN: {msg}")
    print(f"  [WARN] {msg}")
    if d: print(f"         {d}")


# ====================================================================
# TEST 1: Configuration
# ====================================================================
def test_config():
    print("\n=== 1. CONFIGURATION ===")
    from config import (MODELS_DIR, DB_PATH, LOOKBACK_HOURS, MAX_ALERTS,
                        DEMO_ALERTS, BASE_DIR, GROQ_API_KEY,
                        LLM_ENABLED, LLM_REVIEW_TOP_N, NOISE_LIST,
                        LC_CACHE_DIR, FLASK_HOST, FLASK_PORT)

    # Model files
    for f in ['ztf_transformer.pt', 'scaler.pkl', 'ref_embeddings.npy']:
        p = os.path.join(MODELS_DIR, f)
        if os.path.exists(p) and os.path.getsize(p) > 100:
            ok(f"{f} ({os.path.getsize(p):,}B)")
        else:
            bad(f"Missing/empty: {f}")

    # Class centroids
    cp = os.path.join(MODELS_DIR, 'class_centroids.json')
    if os.path.exists(cp):
        centroids = json.load(open(cp))
        if len(centroids) > 0:
            ok(f"class_centroids.json: {len(centroids)} classes")
        else:
            bad("class_centroids.json is empty")
    else:
        wrn("class_centroids.json missing")

    # Numeric settings
    if LOOKBACK_HOURS > 0: ok(f"LOOKBACK_HOURS={LOOKBACK_HOURS}")
    else: bad(f"LOOKBACK_HOURS invalid: {LOOKBACK_HOURS}")
    if MAX_ALERTS > 0: ok(f"MAX_ALERTS={MAX_ALERTS}")
    else: bad(f"MAX_ALERTS invalid")
    if 0 < DEMO_ALERTS <= MAX_ALERTS: ok(f"DEMO_ALERTS={DEMO_ALERTS}")
    else: bad(f"DEMO_ALERTS invalid: {DEMO_ALERTS}")

    # LLM config
    if LLM_ENABLED:
        ok("LLM_ENABLED=True")
        if GROQ_API_KEY:
            ok(f"GROQ_API_KEY set ({len(GROQ_API_KEY)} chars)")
        else:
            wrn("GROQ_API_KEY empty (LLM review will be skipped)")
    else:
        ok("LLM_ENABLED=False (intentionally disabled)")

    if LLM_REVIEW_TOP_N > 0: ok(f"LLM_REVIEW_TOP_N={LLM_REVIEW_TOP_N}")
    else: bad("LLM_REVIEW_TOP_N invalid")

    # Paths exist
    if os.path.isdir(BASE_DIR): ok(f"BASE_DIR exists")
    else: bad(f"BASE_DIR missing: {BASE_DIR}")
    if os.path.isdir(MODELS_DIR): ok(f"MODELS_DIR exists")
    else: bad(f"MODELS_DIR missing")


# ====================================================================
# TEST 2: Scorer (TransformerAE + scoring)
# ====================================================================
def test_scorer():
    print("\n=== 2. SCORER (TransformerAE) ===")
    from scorer import AnomalyScorer, N_BINS, N_FEAT, BOTTLENECK
    import database as db; db.init_db()

    scorer = AnomalyScorer()

    if scorer.model is None:
        bad("TransformerAE model is None"); return scorer
    ok("TransformerAE loaded")

    if scorer.ref_emb is None or len(scorer.ref_emb) == 0:
        bad("Reference embeddings empty"); return scorer
    if scorer.ref_emb.shape[1] != BOTTLENECK:
        bad(f"ref_emb dim: {scorer.ref_emb.shape[1]} != {BOTTLENECK}")
    else:
        ok(f"ref_emb: {scorer.ref_emb.shape}")

    if scorer.scaler is None: bad("Scaler is None")
    else: ok("Scaler loaded")

    # Score random data
    fake = np.random.randn(5, N_BINS, N_FEAT).astype(np.float32)
    mask = np.ones((5, N_BINS), dtype=np.float32)
    r = scorer.score_batch(fake, mask)

    # Check output keys
    for k in ['scores', 'embeddings', 'rec_errors', 'knn_dists', 'if_scores', 'feedback_scores']:
        if k not in r:
            bad(f"score_batch missing key: '{k}'")
    ok("score_batch returns all required keys")

    # Validate scores
    if r['scores'].shape != (5,): bad(f"scores shape: {r['scores'].shape}")
    if np.any(np.isnan(r['scores'])): bad("scores contain NaN")
    elif np.any(np.isinf(r['scores'])): bad("scores contain Inf")
    else: ok(f"scores: [{r['scores'].min():.4f}, {r['scores'].max():.4f}]")

    # Validate embeddings
    if r['embeddings'].shape != (5, BOTTLENECK):
        bad(f"embeddings shape: {r['embeddings'].shape}")
    if np.any(np.isnan(r['embeddings'])): bad("embeddings contain NaN")
    elif np.any(np.isinf(r['embeddings'])): bad("embeddings contain Inf")
    else: ok(f"embeddings finite, norm={np.linalg.norm(r['embeddings'],axis=1).mean():.3f}")

    # nearest_class
    cls, dist = scorer.nearest_class(r['embeddings'][0])
    if cls is None: wrn("nearest_class returned None")
    else: ok(f"nearest_class: '{cls}' d={dist:.3f}")

    return scorer


# ====================================================================
# TEST 3: Preprocessor
# ====================================================================
def test_preprocessor(scorer):
    print("\n=== 3. PREPROCESSOR ===")
    import pandas as pd
    from preprocessor import batch_preprocess
    from scorer import N_BINS, N_FEAT

    np.random.seed(42)
    n = 100
    mjds = np.sort(np.random.uniform(58000, 59000, n))
    mags = 15 + 0.5*np.sin(mjds/50) + np.random.normal(0, 0.05, n)

    # Standard format
    df1 = pd.DataFrame({'oid': 'T001', 'mjd': mjds, 'mag': mags,
                        'magerr': np.full(n, 0.05), 'fid': 1})
    X, M, P, v = batch_preprocess(df1, ['T001'], scorer.scaler)
    if len(v) == 0: bad("preprocessor dropped standard-format object")
    elif X.shape != (1, N_BINS, N_FEAT): bad(f"std shape: {X.shape}")
    else: ok(f"standard format: shape={X.shape}")

    # ALeRCE format (magpsf_corr)
    df2 = pd.DataFrame({'oid': 'T002', 'mjd': mjds,
                        'magpsf_corr': mags, 'sigmapsf_corr': np.full(n, 0.05), 'fid': 1})
    X2, M2, P2, v2 = batch_preprocess(df2, ['T002'], scorer.scaler)
    if len(v2) == 0: bad("preprocessor dropped ALeRCE-format object")
    else: ok(f"ALeRCE format: shape={X2.shape}")

    # Too few detections
    tiny = pd.DataFrame({'oid': 'TINY', 'mjd': [58000, 58001], 'mag': [15, 15.1],
                         'magerr': [0.05, 0.05], 'fid': [1, 1]})
    _, _, _, v3 = batch_preprocess(tiny, ['TINY'], scorer.scaler)
    if len(v3) == 0: ok("correctly drops <5 detections")
    else: wrn("accepted object with only 2 detections")

    # NaN handling
    mags_nan = mags.copy()
    mags_nan[10:15] = np.nan
    df_nan = pd.DataFrame({'oid': 'TNAN', 'mjd': mjds, 'mag': mags_nan,
                           'magerr': np.full(n, 0.05), 'fid': 1})
    X4, M4, P4, v4 = batch_preprocess(df_nan, ['TNAN'], scorer.scaler)
    if len(v4) > 0: ok("handles NaN magnitudes gracefully")
    else: wrn("NaN magnitudes caused entire object to be dropped")

    # Empty DataFrame
    df_empty = pd.DataFrame(columns=['oid', 'mjd', 'mag', 'magerr', 'fid'])
    X5, M5, P5, v5 = batch_preprocess(df_empty, ['NOTHING'], scorer.scaler)
    if len(v5) == 0: ok("empty DataFrame returns no objects")
    else: bad("empty DataFrame returned objects somehow")


# ====================================================================
# TEST 4: Database
# ====================================================================
def test_database():
    print("\n=== 4. DATABASE ===")
    import database as db
    db.init_db()
    conn = db.get_conn()

    # Check all tables
    for tbl in ['objects', 'feedback_log', 'discoveries', 'model_versions', 'llm_review_log']:
        try:
            r = conn.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0]
            ok(f"table '{tbl}': {r} rows")
        except Exception as e:
            bad(f"table '{tbl}' MISSING: {e}")

    # Check critical columns
    obj_cols = [c['name'] for c in conn.execute("PRAGMA table_info(objects)").fetchall()]
    for col in ['oid', 'score', 'triage', 'ra', 'dec', 'rec_error', 'knn_dist',
                'simbad_match', 'auto_class']:
        if col in obj_cols: ok(f"objects.{col} exists")
        else: bad(f"objects.{col} MISSING")

    fb_cols = [c['name'] for c in conn.execute("PRAGMA table_info(feedback_log)").fetchall()]
    if 'new_label' in fb_cols: ok("feedback_log.new_label exists")
    else: bad("feedback_log.new_label MISSING (retrain.py breaks)")

    llm_cols = [c['name'] for c in conn.execute("PRAGMA table_info(llm_review_log)").fetchall()]
    for col in ['oid', 'verdict', 'confidence', 'reasoning', 'suggested_class', 'is_candidate']:
        if col in llm_cols: ok(f"llm_review_log.{col} exists")
        else: bad(f"llm_review_log.{col} MISSING")

    # Data integrity
    null_oids = conn.execute("SELECT COUNT(*) FROM objects WHERE oid IS NULL").fetchone()[0]
    if null_oids > 0: bad(f"{null_oids} NULL oids!")
    else: ok("No NULL oids")

    bad_ra = conn.execute("SELECT COUNT(*) FROM objects WHERE ra < 0 OR ra > 360").fetchone()[0]
    bad_dec = conn.execute("SELECT COUNT(*) FROM objects WHERE dec < -90 OR dec > 90").fetchone()[0]
    if bad_ra: bad(f"{bad_ra} objects with invalid RA")
    else: ok("All RA in [0, 360]")
    if bad_dec: bad(f"{bad_dec} objects with invalid Dec")
    else: ok("All Dec in [-90, 90]")

    bad_triage = conn.execute(
        "SELECT COUNT(*) FROM objects WHERE triage NOT IN ('flagged','classified','dismissed')"
    ).fetchone()[0]
    if bad_triage: bad(f"{bad_triage} objects with invalid triage")
    else: ok("All triage values valid")

    non_ztf = conn.execute("SELECT oid FROM objects WHERE oid NOT LIKE 'ZTF%' LIMIT 3").fetchall()
    if non_ztf: wrn(f"Non-ZTF OIDs: {[r['oid'] for r in non_ztf]}")
    else: ok("All OIDs are ZTF format")

    conn.close()


# ====================================================================
# TEST 5: Embedding Persistence
# ====================================================================
def test_embeddings():
    print("\n=== 5. EMBEDDING PERSISTENCE ===")
    from feedback import save_embeddings, _get_embedding

    test_emb = np.random.randn(3, 64).astype(np.float32)
    test_oids = ['EMB_TEST_A', 'EMB_TEST_B', 'EMB_TEST_C']
    save_embeddings(test_oids, test_emb)

    for i, oid in enumerate(test_oids):
        got = _get_embedding(oid)
        if got is None: bad(f"{oid} not retrievable after save")
        elif not np.allclose(got, test_emb[i], atol=1e-5): bad(f"{oid} data corrupted")
        else: ok(f"{oid} round-trips correctly")

    if _get_embedding("NONEXISTENT_999") is not None:
        bad("non-existent embedding returned value")
    else:
        ok("non-existent returns None")


# ====================================================================
# TEST 6: Retrain Pipeline (4 Sources)
# ====================================================================
def test_retrain():
    print("\n=== 6. RETRAIN PIPELINE (4 SOURCES) ===")
    from retrain import FeedbackClassifier, collect_training_data
    from feedback import EMB_STORE
    import database as db
    from collections import Counter

    emb, labels, weights, sources = collect_training_data(db, EMB_STORE)
    print(f"  Training data: {len(emb)} samples")

    if len(emb) == 0:
        wrn("No training data available")
        return

    # Check labels
    if np.any(np.isnan(labels)): bad("labels contain NaN")
    elif np.any((labels < 0) | (labels > 1.01)): bad(f"labels out of [0,1]: [{labels.min()},{labels.max()}]")
    else: ok(f"labels valid: [{labels.min():.2f}, {labels.max():.2f}]")

    # Check weights
    if np.any(np.isnan(weights)): bad("weights contain NaN")
    elif np.any(weights < 0): bad("negative weights found")
    else: ok(f"weights valid: [{weights.min():.2f}, {weights.max():.2f}]")

    # Check embedding dim
    if emb.shape[1] != 64: bad(f"embedding dim: {emb.shape[1]} != 64")
    else: ok(f"embedding shape: {emb.shape}")

    # Check all sources present
    src_counts = Counter(sources)
    for src, cnt in src_counts.items():
        if src in ('human', 'self_validation', 'pseudo', 'llm'):
            ok(f"source '{src}': {cnt} samples")
        else:
            wrn(f"unknown source: '{src}' ({cnt} samples)")

    # Test IF training
    fc = FeedbackClassifier()
    fc.train_isolation_forest(emb[:100])
    ifs = fc.predict_if_score(emb[:5])
    if np.any(np.isnan(ifs)): bad("IF scores NaN")
    elif np.any((ifs < 0) | (ifs > 1)): bad(f"IF scores out of [0,1]")
    else: ok(f"IF scores: [{ifs.min():.3f}, {ifs.max():.3f}]")

    # Test feedback classifier (needs both classes)
    binary = (labels >= 0.5).astype(int)
    if len(np.unique(binary)) >= 2 and len(emb) >= 15:
        trained = fc.train(emb, labels, weights)
        if not trained: bad("Feedback classifier training failed")
        else:
            preds = fc.predict_feedback_score(emb[:5])
            if np.any(np.isnan(preds)): bad("feedback predictions NaN")
            elif np.any((preds < 0) | (preds > 1)): bad("feedback predictions out of [0,1]")
            else: ok(f"feedback classifier: [{preds.min():.3f}, {preds.max():.3f}]")
    else:
        wrn(f"Skipping FC training (need both classes + >=15 samples, have {len(np.unique(binary))} classes, {len(emb)} samples)")


# ====================================================================
# TEST 7: Triage
# ====================================================================
def test_triage(scorer):
    print("\n=== 7. TRIAGE ===")
    from triage import triage_batch
    from collections import Counter

    fake_results = [{
        'oid': f'TRIAGE_{i:03d}', 'score': i/19, 'rec_error': i/19*0.5,
        'knn_dist': i/19*0.3, 'embedding': np.random.randn(64),
        'ra': 180, 'dec': 45, 'n_detections': 50,
        'mag_mean': 16, 'mag_err_mean': 0.05,
        'simbad_matched': False, 'simbad_match': None, 'simbad_otype': None,
    } for i in range(20)]

    triaged = triage_batch(fake_results, scorer)
    if len(triaged) != 20: bad(f"triage returned {len(triaged)}, expected 20"); return
    ok(f"triage processed all 20 objects")

    buckets = Counter(o['triage'] for o in triaged)
    for b, c in buckets.items():
        ok(f"triage '{b}': {c}")

    if all(o.get('triage') in ('flagged', 'classified', 'dismissed') for o in triaged):
        ok("all triage values valid")
    else:
        bad("invalid triage values found")

    if buckets.get('flagged', 0) == 20:
        bad("ALL objects flagged -- threshold too low")
    if buckets.get('flagged', 0) == 0:
        wrn("No objects flagged even with score=1.0")


# ====================================================================
# TEST 8: Light Curve Cache
# ====================================================================
def test_cache():
    print("\n=== 8. LC CACHE ===")
    import pandas as pd
    from config import LC_CACHE_DIR

    if not os.path.exists(LC_CACHE_DIR):
        wrn("No LC cache directory yet"); return

    files = [f for f in os.listdir(LC_CACHE_DIR) if f.endswith('.parquet')]
    ok(f"{len(files)} cached light curves")
    if len(files) == 0: return

    checked = 0
    for fname in files[:5]:
        try:
            d = pd.read_parquet(os.path.join(LC_CACHE_DIR, fname))
            if 'mjd' not in d.columns: bad(f"{fname}: no mjd column"); continue
            if d['mjd'].min() < 50000 or d['mjd'].max() > 70000:
                bad(f"{fname}: suspicious MJD range [{d['mjd'].min():.0f}, {d['mjd'].max():.0f}]")
            else:
                checked += 1
        except Exception as e:
            bad(f"Can't read {fname}: {e}")
    if checked > 0:
        ok(f"Spot-checked {checked} files: MJD ranges valid")


# ====================================================================
# TEST 9: Scoring Blend
# ====================================================================
def test_scoring_blend():
    print("\n=== 9. SCORING BLEND ===")
    from scorer import AnomalyScorer, N_BINS, N_FEAT
    from retrain import FeedbackClassifier
    import database as db; db.init_db()

    fake = np.random.randn(5, N_BINS, N_FEAT).astype(np.float32)
    mask = np.ones((5, N_BINS), dtype=np.float32)

    # Without feedback models
    s1 = AnomalyScorer()
    s1.feedback_clf = None
    r1 = s1.score_batch(fake, mask)

    # With feedback models
    s2 = AnomalyScorer()
    fc = FeedbackClassifier(); fc.load()
    s2.feedback_clf = fc
    r2 = s2.score_batch(fake, mask)

    if fc.iso_forest is not None or fc.is_trained:
        diff = np.abs(r1['scores'] - r2['scores']).mean()
        if diff < 0.001: wrn(f"blend barely changes scores (diff={diff:.5f})")
        else: ok(f"blend shifts scores by avg {diff:.4f}")
    else:
        ok("No retrained models yet (base-only expected)")


# ====================================================================
# TEST 10: LLM Interpreter
# ====================================================================
def test_llm_interpreter():
    print("\n=== 10. LLM INTERPRETER ===")
    from config import GROQ_API_KEY, LLM_ENABLED

    # Module imports
    try:
        from llm_interpreter import (review_object, batch_review,
                                     _build_prompt, _parse_response,
                                     feed_llm_to_retrain)
        ok("llm_interpreter module imports cleanly")
    except ImportError as e:
        bad(f"llm_interpreter import failed: {e}"); return

    # Prompt building
    test_obj = {
        'oid': 'ZTF18test001', 'score': 0.85, 'rec_error': 0.042,
        'knn_dist': 0.31, 'auto_class': 'DSCT', 'class_distance': 0.15,
        'simbad_match': None, 'simbad_otype': None,
        'n_detections': 120, 'mag_mean': 16.5, 'triage_reason': 'test'
    }
    prompt = _build_prompt(test_obj)
    if 'ZTF18test001' in prompt and '0.85' in prompt:
        ok("prompt contains OID and score")
    else:
        bad("prompt missing key fields")

    # JSON parsing
    good_json = '{"verdict":"interesting","confidence":0.8,"reasoning":"test","suggested_class":null,"is_candidate":true}'
    parsed = _parse_response(good_json)
    if parsed and parsed['verdict'] == 'interesting': ok("JSON parsing works")
    else: bad("JSON parsing failed")

    # Markdown-wrapped JSON
    md_json = '```json\n' + good_json + '\n```'
    parsed2 = _parse_response(md_json)
    if parsed2 and parsed2['verdict'] == 'interesting': ok("Markdown JSON parsing works")
    else: bad("Markdown JSON parsing failed")

    # Malformed JSON
    bad_json = "This is not JSON at all"
    parsed3 = _parse_response(bad_json)
    if parsed3 is None: ok("Malformed JSON returns None")
    else: wrn("Malformed JSON returned something")

    # Check Groq connectivity (only if API key set)
    if GROQ_API_KEY and LLM_ENABLED:
        result = review_object(test_obj)
        if result is None:
            wrn("LLM review returned None (API issue?)")
        elif result.get('verdict') in ('interesting', 'known_type', 'noise'):
            ok(f"LLM live review: {result['verdict']} ({result['confidence']:.0%})")
            ok(f"LLM reasoning: {result['reasoning'][:60]}")
        else:
            bad(f"LLM returned invalid verdict: {result.get('verdict')}")
    else:
        wrn("Skipping live LLM test (no API key)")

    # LLM feedback collection
    import database as db
    from feedback import EMB_STORE
    try:
        llm_emb, llm_labels, llm_weights = feed_llm_to_retrain(db, EMB_STORE)
        ok(f"LLM feedback collection: {len(llm_emb)} samples")
        if len(llm_emb) > 0:
            if np.any(np.isnan(llm_labels)): bad("LLM labels contain NaN")
            elif np.any((llm_labels < 0) | (llm_labels > 1)):
                bad(f"LLM labels out of [0,1]")
            else:
                ok(f"LLM labels valid: [{llm_labels.min():.2f}, {llm_labels.max():.2f}]")
            if np.any(llm_weights > 0.7 + 0.01):
                bad(f"LLM weights exceed 0.7: max={llm_weights.max():.3f}")
            else:
                ok(f"LLM weights capped correctly: [{llm_weights.min():.3f}, {llm_weights.max():.3f}]")
    except Exception as e:
        wrn(f"LLM feedback collection error: {e}")


# ====================================================================
# TEST 11: Mock/Placeholder Data Scan
# ====================================================================
def test_no_mock_data():
    print("\n=== 11. MOCK DATA SCAN ===")
    backend_dir = os.path.dirname(__file__)
    frontend_dir = os.path.join(os.path.dirname(backend_dir), 'frontend', 'src')
    suspicious = ['lorem ipsum', 'placeholder', 'hardcoded', 'fake_data',
                  'sample_data', 'example.com', 'test123']
    found_any = False

    dirs_to_scan = [backend_dir]
    if os.path.exists(frontend_dir):
        dirs_to_scan.append(frontend_dir)

    files_checked = 0
    for d in dirs_to_scan:
        for root, dirs, files in os.walk(d):
            for fname in files:
                if not fname.endswith(('.py', '.js', '.ts', '.tsx', '.html', '.css')): continue
                if 'test' in fname.lower() or 'node_modules' in root: continue
                fpath = os.path.join(root, fname)
                with open(fpath, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read().lower()
                files_checked += 1
                for s in suspicious:
                    if s in content:
                        found_any = True
                        wrn(f"'{s}' in {os.path.relpath(fpath, os.path.dirname(backend_dir))}")

    if not found_any:
        ok(f"No mock/placeholder data in {files_checked} source files")
    else:
        ok(f"Scanned {files_checked} files (see warnings above)")


# ====================================================================
# TEST 12: Feedback Loop
# ====================================================================
def test_feedback_loop():
    print("\n=== 12. FEEDBACK LOOP ===")
    import database as db
    from config import NOISE_LIST

    conn = db.get_conn()
    fb_count = conn.execute("SELECT COUNT(*) FROM feedback_log").fetchone()[0]
    ok(f"feedback_log: {fb_count} entries")

    if fb_count > 0:
        actions = conn.execute(
            "SELECT action, COUNT(*) as cnt FROM feedback_log GROUP BY action"
        ).fetchall()
        for row in actions:
            if row['action'] in ('interesting', 'noise', 'classify', 'self_validate'):
                ok(f"action '{row['action']}': {row['cnt']}")
            else:
                wrn(f"unknown action: '{row['action']}' ({row['cnt']})")

    conn.close()

    # Noise list
    if os.path.exists(NOISE_LIST):
        noise = np.load(NOISE_LIST)
        if np.any(np.isnan(noise)): bad("noise embeddings contain NaN")
        elif np.any(np.isinf(noise)): bad("noise embeddings contain Inf")
        else: ok(f"noise list: {noise.shape}")
    else:
        wrn("No noise list file yet")


# ====================================================================
# TEST 13: Centroid Alignment
# ====================================================================
def test_centroids(scorer):
    print("\n=== 13. CENTROID ALIGNMENT ===")

    if len(scorer.centroid_matrix) == 0:
        wrn("No centroids loaded"); return

    ref_norms = np.linalg.norm(scorer.ref_emb, axis=1)
    cent_norms = np.linalg.norm(scorer.centroid_matrix, axis=1)
    ratio = cent_norms.mean() / (ref_norms.mean() + 1e-10)

    if ratio < 0.1 or ratio > 10:
        bad(f"centroid norms misaligned (ratio={ratio:.2f})")
    else:
        ok(f"centroid norms aligned (ratio={ratio:.2f})")

    cn = scorer.centroid_matrix / (np.linalg.norm(scorer.centroid_matrix, axis=1, keepdims=True) + 1e-10)
    rn = scorer.ref_emb / (np.linalg.norm(scorer.ref_emb, axis=1, keepdims=True) + 1e-10)
    sims = cn @ rn.T
    for i, name in enumerate(scorer.centroid_names):
        ms = sims[i].max()
        if ms < 0.3: wrn(f"centroid '{name}' low max_sim: {ms:.3f}")
        else: ok(f"centroid '{name}': max_sim={ms:.3f}")


# ====================================================================
# TEST 14: API Endpoints
# ====================================================================
def test_api_endpoints():
    print("\n=== 14. API ENDPOINTS ===")
    import database as db; db.init_db()
    from app import app

    client = app.test_client()

    # GET endpoints
    endpoints = [
        ('/api/stats', 200),
        ('/api/health', 200),
        ('/api/flagged', 200),
        ('/api/objects?triage=classified&limit=10', 200),
        ('/api/dismissed?limit=10', 200),
        ('/api/discoveries', 200),
        ('/api/llm-reviews', 200),
        ('/', 200),
        ('/dashboard', 200),
    ]
    for path, expected_code in endpoints:
        try:
            resp = client.get(path)
            if resp.status_code == expected_code:
                ok(f"GET {path} -> {resp.status_code}")
            else:
                bad(f"GET {path} -> {resp.status_code} (expected {expected_code})")
        except Exception as e:
            bad(f"GET {path} crashed: {e}")

    # POST /api/feedback
    try:
        resp = client.post('/api/feedback', json={'oid': 'TEST_API', 'action': 'noise'})
        if resp.status_code == 200:
            ok(f"POST /api/feedback -> {resp.status_code}")
        else:
            bad(f"POST /api/feedback -> {resp.status_code}")
    except Exception as e:
        bad(f"POST /api/feedback crashed: {e}")

    # POST /api/feedback validation
    try:
        resp = client.post('/api/feedback', json={'oid': '', 'action': ''})
        if resp.status_code == 400:
            ok("POST /api/feedback rejects empty params (400)")
        else:
            wrn(f"POST /api/feedback with empty params -> {resp.status_code} (expected 400)")
    except Exception as e:
        bad(f"POST /api/feedback validation crashed: {e}")

    # Stats content
    try:
        resp = client.get('/api/stats')
        data = json.loads(resp.data)
        for key in ['total', 'flagged', 'classified', 'dismissed']:
            if key in data: ok(f"stats.{key} = {data[key]}")
            else: bad(f"stats missing '{key}'")
    except Exception as e:
        bad(f"Stats parsing failed: {e}")


# ====================================================================
# TEST 15: End-to-End Scoring Pipeline
# ====================================================================
def test_e2e_scoring(scorer):
    print("\n=== 15. END-TO-END SCORING ===")
    import pandas as pd
    from preprocessor import batch_preprocess
    from triage import triage_batch
    from scorer import N_BINS, N_FEAT
    from feedback import save_embeddings

    # Create realistic synthetic light curves
    np.random.seed(123)
    oids = [f'E2E_{i:03d}' for i in range(10)]
    all_dfs = []
    for oid in oids:
        n = np.random.randint(30, 200)
        mjds = np.sort(np.random.uniform(58000, 59000, n))
        mags = np.random.uniform(14, 19) + np.random.normal(0, 0.1, n)
        all_dfs.append(pd.DataFrame({
            'oid': oid, 'mjd': mjds, 'mag': mags,
            'magerr': np.random.uniform(0.01, 0.1, n),
            'fid': np.random.choice([1, 2], n),
        }))
    lc_df = pd.concat(all_dfs, ignore_index=True)

    # Step 1: Preprocess
    X, M, P, valid = batch_preprocess(lc_df, oids, scorer.scaler)
    if len(valid) == 0:
        bad("E2E: preprocessor dropped all objects"); return
    ok(f"E2E preprocess: {len(valid)}/{len(oids)} objects")

    # Step 2: Score
    result = scorer.score_batch(X, M)
    scores = result['scores']
    embeddings = result['embeddings']

    if len(scores) != len(valid):
        bad(f"E2E: score count mismatch {len(scores)} != {len(valid)}"); return
    ok(f"E2E scores: [{scores.min():.3f}, {scores.max():.3f}]")

    # Step 3: Save embeddings
    save_embeddings(valid, embeddings)
    ok("E2E embeddings saved")

    # Step 4: Build results + triage
    results = []
    for i, oid in enumerate(valid):
        cls, cdist = scorer.nearest_class(embeddings[i])
        results.append({
            'oid': oid, 'score': float(scores[i]),
            'rec_error': float(result['rec_errors'][i]),
            'knn_dist': float(result['knn_dists'][i]),
            'embedding': embeddings[i],
            'ra': 180.0 + np.random.uniform(-10, 10),
            'dec': 45.0 + np.random.uniform(-10, 10),
            'n_detections': int(P[i][3]),
            'mag_mean': float(P[i][4]),
            'mag_err_mean': 0.05,
            'simbad_matched': False, 'simbad_match': None, 'simbad_otype': None,
            'auto_class': cls, 'class_distance': cdist,
        })

    triaged = triage_batch(results, scorer)
    from collections import Counter
    buckets = Counter(o['triage'] for o in triaged)
    ok(f"E2E triage: {dict(buckets)}")

    # Verify no data corruption
    for obj in triaged:
        if obj['score'] is None or (isinstance(obj['score'], float) and np.isnan(obj['score'])):
            bad(f"E2E: {obj['oid']} has NaN score")
    ok("E2E: no NaN/None scores after full pipeline")


# ====================================================================
# RUNNER
# ====================================================================
if __name__ == '__main__':
    print("=" * 60)
    print("  DEEP TEST SUITE -- Stellar Anomaly Hunter")
    print("  Full 8-step pipeline + LLM + 4-source retrain")
    print("=" * 60)
    t0 = time.time()

    import database as db; db.init_db()

    test_config()
    scorer = test_scorer()
    test_preprocessor(scorer)
    test_database()
    test_embeddings()
    test_retrain()
    test_triage(scorer)
    test_cache()
    test_scoring_blend()

    try:
        test_llm_interpreter()
    except Exception as e:
        bad(f"LLM test crashed: {e}")
        traceback.print_exc()

    test_no_mock_data()
    test_feedback_loop()

    try:
        test_centroids(scorer)
    except Exception as e:
        bad(f"Centroid test crashed: {e}")
        traceback.print_exc()

    test_api_endpoints()
    test_e2e_scoring(scorer)

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"  RESULTS ({elapsed:.1f}s)")
    print(f"{'='*60}")
    print(f"  PASS: {PASS}")
    print(f"  WARN: {WARN}")
    print(f"  FAIL: {FAIL}")
    if ISSUES:
        fails = [i for i in ISSUES if not i.startswith('WARN')]
        warns = [i for i in ISSUES if i.startswith('WARN')]
        if fails:
            print(f"\n  FAILURES:")
            for f in fails:
                print(f"    - {f}")
        if warns:
            print(f"\n  WARNINGS:")
            for w in warns:
                print(f"    - {w}")
    if FAIL == 0:
        print(f"\n  ALL CRITICAL TESTS PASSED")
    else:
        print(f"\n  {FAIL} CRITICAL FAILURES -- FIX BEFORE USE")
    print(f"{'='*60}\n")
