"""
app.py — Flask API server for the Stellar Anomaly Hunter dashboard.
"""
import os
import sys

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

# Add backend dir to path
sys.path.insert(0, os.path.dirname(__file__))

import database as db
import feedback as fb
from config import FLASK_HOST, FLASK_PORT, FLASK_DEBUG

# Serve from React build (frontend/dist/)
FRONTEND_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'frontend', 'dist')

app = Flask(__name__, static_folder=FRONTEND_DIR, static_url_path='')
CORS(app)

# Global scorer reference (set by run_session.py after model loads)
_scorer = None

def set_scorer(scorer):
    global _scorer
    _scorer = scorer


# ── Frontend (React SPA) ────────────────────────────────────────────────
@app.route('/')
def index():
    return send_from_directory(FRONTEND_DIR, 'index.html')


@app.route('/dashboard')
def dashboard():
    return send_from_directory(FRONTEND_DIR, 'index.html')


# ── Stats ───────────────────────────────────────────────────────────────
@app.route('/api/stats')
def api_stats():
    return jsonify(db.get_stats())


# ── Objects ─────────────────────────────────────────────────────────────
@app.route('/api/objects')
def api_objects():
    triage  = request.args.get('triage', None)
    limit   = int(request.args.get('limit', 100))
    objects = db.get_all_objects(limit=limit, triage=triage)
    return jsonify(objects)


@app.route('/api/flagged')
def api_flagged():
    limit = int(request.args.get('limit', 50))
    return jsonify(db.get_flagged(limit))


@app.route('/api/dismissed')
def api_dismissed():
    limit = int(request.args.get('limit', 200))
    return jsonify(db.get_dismissed(limit))


@app.route('/api/discoveries')
def api_discoveries():
    return jsonify(db.get_discoveries())


# ── Feedback ────────────────────────────────────────────────────────────
@app.route('/api/feedback', methods=['POST'])
def api_feedback():
    data   = request.json
    oid    = data.get('oid')
    action = data.get('action')   # 'interesting', 'noise', 'classify'
    label  = data.get('label')    # optional, for 'classify'

    if not oid or not action:
        return jsonify({'status': 'error', 'message': 'oid and action required'}), 400

    if _scorer is None:
        # Still save to DB even if scorer not available
        db.save_feedback(oid, action, 'unknown', label)
        return jsonify({'status': 'ok', 'message': 'Saved (model not loaded)'})

    result = fb.handle_feedback(oid, action, _scorer, label=label)
    return jsonify(result)


# ── Rescan trigger ───────────────────────────────────────────────────────
@app.route('/api/rescan', methods=['POST'])
def api_rescan():
    if _scorer is None:
        return jsonify({'status': 'error', 'message': 'Model not loaded'}), 503
    from rescan import run_rescan
    # Note: lc_df needs to be in memory — set by run_session.py
    lc_df = app.config.get('LC_DF')
    if lc_df is None:
        return jsonify({'status': 'error', 'message': 'No light curve data in memory'}), 503
    n_promoted = run_rescan(_scorer, lc_df)
    return jsonify({'status': 'ok', 'promoted': n_promoted})


# ── Retrain trigger ──────────────────────────────────────────────────────
@app.route('/api/retrain', methods=['POST'])
def api_retrain():
    if _scorer is None:
        return jsonify({'status': 'error', 'message': 'Model not loaded'}), 503
    from retrain import run_retraining
    fc = run_retraining(_scorer, db)
    _scorer.feedback_clf = fc
    return jsonify({
        'status': 'ok',
        'feedback_clf_trained': fc.is_trained,
        'iso_forest_trained': fc.iso_forest is not None,
        'stats': fc.training_stats
    })


# ── LLM Reviews ─────────────────────────────────────────────────────────
@app.route('/api/llm-reviews')
def api_llm_reviews():
    conn = db.get_conn()
    rows = conn.execute(
        "SELECT * FROM llm_review_log ORDER BY timestamp DESC LIMIT 200"
    ).fetchall()
    conn.close()
    return jsonify([dict(r) for r in rows])


@app.route('/api/llm-review/<oid>')
def api_llm_review_detail(oid):
    conn = db.get_conn()
    rows = conn.execute(
        "SELECT * FROM llm_review_log WHERE oid=? ORDER BY timestamp DESC",
        (oid,)
    ).fetchall()
    conn.close()
    return jsonify([dict(r) for r in rows])


# ── Score Distribution (for charts) ─────────────────────────────────────
@app.route('/api/score-distribution')
def api_score_distribution():
    conn = db.get_conn()
    rows = conn.execute("SELECT score FROM objects WHERE score IS NOT NULL").fetchall()
    conn.close()
    scores = [r['score'] for r in rows]
    # Bucket into 10 bins
    bins = [0]*10
    for s in scores:
        idx = min(int(s * 10), 9)
        bins[idx] += 1
    return jsonify([
        {"range": f"{i/10:.1f}-{(i+1)/10:.1f}", "count": bins[i]}
        for i in range(10)
    ])


@app.route('/api/llm-stats')
def api_llm_stats():
    conn = db.get_conn()
    rows = conn.execute(
        "SELECT verdict, COUNT(*) as count FROM llm_review_log GROUP BY verdict"
    ).fetchall()
    total = conn.execute("SELECT COUNT(*) FROM llm_review_log").fetchone()[0]
    candidates = conn.execute(
        "SELECT COUNT(*) FROM llm_review_log WHERE is_candidate=1"
    ).fetchone()[0]
    conn.close()
    return jsonify({
        "verdicts": {r['verdict']: r['count'] for r in rows},
        "total": total,
        "candidates": candidates,
    })


# ── Health ───────────────────────────────────────────────────────────────
@app.route('/api/health')
def api_health():
    has_fc = _scorer is not None and _scorer.feedback_clf is not None
    return jsonify({
        'status': 'ok',
        'model_loaded': _scorer is not None,
        'feedback_clf': has_fc and _scorer.feedback_clf.is_trained,
        'isolation_forest': has_fc and _scorer.feedback_clf.iso_forest is not None,
        'db': os.path.exists(db.DB_PATH)
    })


def run(scorer, lc_df=None):
    set_scorer(scorer)
    app.config['LC_DF'] = lc_df
    print(f"\n🌌 Dashboard ready at http://{FLASK_HOST}:{FLASK_PORT}")
    print(f"   Press Ctrl+C when done reviewing.\n")
    app.run(host=FLASK_HOST, port=FLASK_PORT, debug=FLASK_DEBUG)
