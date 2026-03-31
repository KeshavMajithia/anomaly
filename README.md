# 🌌 Stellar Anomaly Hunter

A **self-improving, LLM-assisted** anomaly detection system for astronomical transient surveys. It processes ZTF light curves in real-time, flags genuinely unknown objects, and learns from human feedback, its own mistakes, and an AI astronomer.

---

## What This Does

Telescopes like ZTF generate **millions of alerts per night**. Most are known variable stars. A tiny fraction are genuinely new — supernovae, cataclysmic variables, or completely unknown objects. This system finds the unknown ones automatically.

**The pipeline:** Fetch live data → Score anomalousness → AI triage → LLM astronomer review → Human override → Learn → Repeat

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    8-STEP SESSION PIPELINE                   │
│                                                             │
│  1. Load models (TransformerAE + IF + FeedbackClassifier)   │
│  2. Self-validate past predictions against ALeRCE           │
│  3. Fetch live ZTF objects from ALeRCE broker               │
│  4. Preprocess → Score with blended model                   │
│  5. AI Triage: flagged / classified / dismissed             │
│  6. LLM Astronomer reviews flagged objects (Groq)   │
│  7. Save to database                                        │
│  8. Auto-retrain from 4 feedback sources                    │
└─────────────────────────────────────────────────────────────┘

Scoring = 0.4 × TransformerAE + 0.3 × IsolationForest + 0.3 × FeedbackClassifier

4-Source Self-Improvement:
  Source 1: Human feedback     (weight 1.0)  — gold standard
  Source 2: LLM astronomer     (weight 0.7)  — automated expert
  Source 3: Self-validation    (weight 0.5)  — learns from own mistakes
  Source 4: Pseudo-labels      (weight 0.3)  — high-confidence auto
```

## Quick Start

```bash
# 1. Set Groq API key (for LLM astronomer)
set GROQ_API_KEY=your-key-here

# 2. Install dependencies
cd anomaly_hunter
pip install -r requirements.txt
pip install google-generativeai

# 3. Demo run (~2 min, 50 objects)
cd backend
python run_session.py --demo

# 4. Full run (~15 min, 1000 objects)
python run_session.py

# 5. Skip LLM review
python run_session.py --demo --no-llm
```

Dashboard: **http://127.0.0.1:5000**

## Project Structure

```
anomaly_hunter/
├── backend/
│   ├── config.py            # All settings (API keys, thresholds, paths)
│   ├── run_session.py       # Main entry (8-step pipeline)
│   ├── poller.py            # Fetch ZTF alerts from ALeRCE
│   ├── preprocessor.py      # Light curve → (50,12) tensor
│   ├── scorer.py            # Blended anomaly scoring
│   ├── triage.py            # 3-bucket triage
│   ├── llm_interpreter.py   # Groq automated astronomer
│   ├── simbad_checker.py    # SIMBAD cross-matching
│   ├── database.py          # SQLite management
│   ├── feedback.py          # Human feedback + embedding persistence
│   ├── retrain.py           # 4-source self-improvement
│   ├── rescan.py            # Retrospective re-evaluation
│   └── app.py               # Flask API (12 endpoints)
├── frontend/                # React + Tailwind dashboard (Lovable)
│   ├── src/                 # React source
│   └── dist/                # Production build (served by Flask)
├── models/                  # Trained models
│   ├── ztf_transformer.pt   # TransformerAE weights (814K params)
│   ├── scaler.pkl           # Feature scaler
│   ├── ref_embeddings.npy   # 74,831 reference embeddings
│   ├── class_centroids.json # 6-class centroids
│   ├── feedback_classifier.pkl  # (auto-generated)
│   └── isolation_forest.pkl     # (auto-generated)
├── database/
│   ├── anomalies.db         # SQLite (objects, feedback, LLM reviews)
│   └── lc_cache/            # Cached light curves (.parquet)
├── README.md
└── requirements.txt
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/stats` | Session statistics |
| GET | `/api/health` | Model + DB status |
| GET | `/api/flagged` | Flagged objects |
| GET | `/api/objects?triage=X` | Filter by triage |
| GET | `/api/dismissed` | Dismissed objects |
| GET | `/api/discoveries` | Confirmed discoveries |
| GET | `/api/llm-reviews` | LLM astronomer verdicts |
| GET | `/api/llm-review/<oid>` | LLM reasoning for one object |
| POST | `/api/feedback` | Submit human feedback |
| POST | `/api/rescan` | Retrospective rescan |
| POST | `/api/retrain` | Trigger retraining |

## Tech Stack

**Model**: TransformerAE (814K params, 64-dim embeddings) trained on 74,831 ZTF light curves
**Data**: ALeRCE broker API (real-time ZTF alerts), SIMBAD catalog
**AI**: Groq (LLM astronomer), scikit-learn (IF + GBClassifier)
**Backend**: Python, Flask, PyTorch
**Frontend**: React, Tailwind, shadcn/ui
**Database**: SQLite

## Documentation

See [`PROJECT_DOCUMENT.md`](./PROJECT_DOCUMENT.md) for the full technical narrative: research gap, methodology, Kaggle training, streaming pipeline, self-improvement, LLM integration, and cross-domain applications.
