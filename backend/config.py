"""
config.py — Central configuration for the Stellar Anomaly Hunter PoC
"""
import os

# Load .env file directly (NOT from terminal environment)
_ENV_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env')
_env_vars = {}
try:
    from dotenv import dotenv_values
    _env_vars = dotenv_values(_ENV_PATH)
except ImportError:
    pass

# ── Paths ──────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR  = os.path.join(BASE_DIR, 'models')
DB_PATH     = os.path.join(BASE_DIR, 'database', 'anomalies.db')
NOISE_LIST  = os.path.join(BASE_DIR, 'database', 'noise_embeddings.npy')

# ── Model files ────────────────────────────────────────────────────────
TRANSFORMER_WEIGHTS = os.path.join(MODELS_DIR, 'ztf_transformer.pt')
MASKED_AE_WEIGHTS   = os.path.join(MODELS_DIR, 'ztf_masked_ae.pt')
SCALER_FILE         = os.path.join(MODELS_DIR, 'scaler.pkl')
REF_EMBEDDINGS      = os.path.join(MODELS_DIR, 'ref_embeddings.npy')
CLASS_CENTROIDS     = os.path.join(MODELS_DIR, 'class_centroids.json')
CONFIG_FILE         = os.path.join(MODELS_DIR, 'config.json')

# ── Triage thresholds ──────────────────────────────────────────────────
NOISE_MIN_DETECTIONS    = 10      # Objects with fewer detections → auto-dismissed
NOISE_MAX_MAG_ERR       = 0.3     # Objects with high mag error → auto-dismissed
CLASSIFY_THRESHOLD      = 0.4     # kNN distance to nearest class centroid → auto-classified
ANOMALY_THRESHOLD       = 0.6     # Score above this → flagged for human
NOISE_SIMILARITY_CUTOFF = 0.90    # Cosine similarity to noise list → auto-dismissed

# ── Session settings ───────────────────────────────────────────────────
LOOKBACK_HOURS     = 72    # Fetch alerts from past N hours (72h catches quiet nights)
MAX_ALERTS         = 5000  # Max objects to fetch per session (production-scale for paper)
DEMO_ALERTS        = 50    # Demo mode: fast (~2 min)
LC_CACHE_DIR       = os.path.join(BASE_DIR, 'database', 'lc_cache')
SIMBAD_CHECK_TOP_N = 50    # SIMBAD-check the top N flagged objects
SIMBAD_RADIUS_ARCSEC = 5.0

# ── LLM Interpreter (Groq — Llama 3.3 70B) ───────────────────────────
GROQ_API_KEY       = _env_vars.get('GROQ_API_KEY', '')     # Get free key: https://console.groq.com/keys
LLM_ENABLED        = True     # Set False to skip LLM review
LLM_REVIEW_TOP_N   = 20       # Review top N flagged objects per session

# ── Flask ──────────────────────────────────────────────────────────────
FLASK_HOST = '127.0.0.1'
FLASK_PORT = 5000
FLASK_DEBUG = True