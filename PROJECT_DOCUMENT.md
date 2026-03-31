# Stellar Anomaly Hunter — Project Document

## 1. The Idea

Every night, telescopes like the Zwicky Transient Facility (ZTF) scan the sky and generate **millions of alerts**. Each alert says "something changed here." Most of these are known variable stars doing their usual thing. But buried in that flood are genuinely new objects — supernovae catching fire, stars being eaten by black holes, or things nobody has ever seen before.

The problem: **no human can review millions of alerts.** Existing systems like ALeRCE, ANTARES, and Fink classify alerts into known categories. But they're designed to sort objects into boxes, not to find things that don't fit any box. They're stamp sorters, not treasure hunters.

**Our idea:** Build a system that specifically hunts for the anomalies — the objects that DON'T fit. And make it learn from every interaction so it gets smarter over time, without needing an expert astronomer babysitting it 24/7.

---

## 2. The Research Gap

| What exists | What's missing |
|-------------|----------------|
| ALeRCE classifies alerts into known types | Doesn't flag genuinely unknown objects |
| Isolation Forest finds statistical outliers | No domain knowledge — flags noise too |
| Deep learning autoencoders learn patterns | Static — doesn't improve after deployment |
| Human-in-the-loop systems exist | Requires expert astronomer availability |

**Nobody has built a system that:**
1. Combines deep learning embeddings with anomaly detection for **domain-aware** outlier detection
2. **Self-improves** from multiple feedback sources (not just human labeling)
3. Uses an **LLM as an automated astronomical expert** in the feedback loop
4. Works **autonomously** while still accepting human oversight

**That's our contribution.**

---

## 3. What We Propose

A **4-tier autonomous anomaly detection system** with hierarchical self-improvement:

```
Tier 1: TransformerAE + Isolation Forest (trained on Kaggle)
        → "This object has an unusual light curve shape"

Tier 2: Self-Validation + Pseudo-Labels (automated)
        → "We flagged this before but ALeRCE classified it — learn from mistake"

Tier 3: LLM Astronomer (Groq)
        → "This looks like a CV in outburst based on rapid brightening + no SIMBAD match"

Tier 4: Human Override (optional)
        → "I agree with the LLM" or "No, this is actually just noise"
```

Each tier feeds back into the training pipeline with decreasing trust weights: Human (1.0) > LLM (0.7) > Self-validation (0.5) > Pseudo-labels (0.3). The system gets smarter from all four simultaneously.

---

## 4. How Things Were Done Before (Traditional Approaches)

**Classification-based systems** (ALeRCE, RAPID):
- Train a classifier on known types (RR Lyrae, eclipsing binaries, etc.)
- New alert comes in → "This is probably an RR Lyrae (92% confidence)"
- **Problem:** If the object is genuinely new, the classifier forces it into the closest known box. It can't say "I don't know."

**Simple anomaly detection** (Isolation Forest, DBSCAN):
- Statistical methods that find points far from the crowd
- **Problem:** No domain knowledge. Flags noisy data, bad photometry, and instrument artifacts alongside real anomalies. High false positive rate.

**Human review systems:**
- Expert astronomers manually review candidates
- **Problem:** Doesn't scale. ZTF produces 300,000+ alerts per night. LSST (starting 2025) will produce 10 million.

---

## 5. What We Did on Kaggle (Phase 1: Training)

We trained multiple anomaly detection models on **74,831 real ZTF light curves** spanning 6 known variable star classes.

### What we trained:
- **TransformerAE** (our primary model): A transformer-based autoencoder with a 64-dimensional bottleneck. It learns to compress light curves into dense embeddings. Objects that compress poorly (high reconstruction error) are anomalous.
- **Masked Autoencoder**: Similar but masks random time bins during training (data augmentation).
- **Isolation Forest**: Classic statistical anomaly detector, run on the 64-dim embeddings.
- **kNN Detector**: Compares each object's embedding to the nearest reference objects. Far = anomalous.

### Key results from Kaggle:
- TransformerAE: **69.2% probe accuracy** on 6 classes (vs 17% random baseline) — proves the embeddings capture meaningful structure
- Isolation Forest: **ρ = +0.14 correlation** between IF score and actual rarity — it genuinely separates anomalies
- **Insight that shaped the whole project:** TransformerAE is best at creating embeddings, IF is best at scoring. Use both.

### What we extracted from Kaggle:
- `ztf_transformer.pt` — Trained model weights (814K parameters)
- `scaler.pkl` — Feature normalization parameters
- `ref_embeddings.npy` — 74,831 reference embeddings (the "normal" population)
- `class_centroids.json` — Average embeddings for each of 6 known classes

These files ARE the foundation of the streaming tool. The Kaggle notebook is the training ground; the tool is the deployment.

---

## 6. How We Used Kaggle's Output (The Bridge)

The flow from training to deployment:

```
Kaggle Notebook (training)
    │
    ├── ztf_transformer.pt ──→ scorer.py loads this model
    │                          Encodes new objects into 64-dim embeddings
    │                          Measures reconstruction error
    │
    ├── ref_embeddings.npy ──→ scorer.py uses these as "normal" population
    │                          kNN distance = how far from known objects
    │
    ├── scaler.pkl ──────────→ preprocessor.py normalizes new light curves
    │                          Same normalization as training = consistent
    │
    └── class_centroids.json → triage.py uses these to auto-classify
                               "This embedding is close to RR Lyrae centroid"
```

The key design decision: **the Kaggle model is frozen.** We never retrain the TransformerAE (that would require GPU + full dataset). Instead, we train lightweight models ON TOP of its embeddings — the Feedback Classifier and Isolation Forest. This is computationally cheap (runs on CPU in seconds) but adds supervised signal from real-world usage.

---

## 7. The Streaming Tool (What It Actually Does)

### 8-Step Pipeline per Session:

**Step 1: Load Models**
Load the TransformerAE (from Kaggle) + Feedback Classifier + Isolation Forest (from previous retraining).

**Step 2: Self-Validate**
Check if any previously flagged objects have since been classified by ALeRCE. If yes → our flag was a false positive. Log this as a training signal.

**Step 3: Fetch Live Data**
Query ALeRCE's API for recent ZTF alerts. Download their light curves. Cache locally as `.parquet` files (subsequent runs are instant for cached objects).

**Step 4: Preprocess + Score**
Convert each light curve into a (50, 12) tensor (50 time bins × 12 features per bin: 3 bands × 4 features). Run through TransformerAE. Compute blended score:

```
final_score = 0.4 × AE_score + 0.3 × IF_score + 0.3 × feedback_score
```

- **AE score** = reconstruction error (how poorly the model predicts the light curve) + kNN distance (how far from known objects) − noise penalty
- **IF score** = Isolation Forest anomaly score on the 64-dim embedding
- **Feedback score** = GradientBoosting classifier trained from human + LLM + self-validation feedback

**Step 5: AI Triage**
Sort objects into three buckets:
- 🚨 **Flagged**: High score + no SIMBAD match + no close class centroid → needs review
- 🏷️ **Classified**: Close to a known class centroid → auto-classified (e.g., "likely RR Lyrae")
- 📦 **Dismissed**: Low score, high noise, or matches known noise patterns

**Step 6: LLM Astronomer Review**
For each flagged object, send its metadata to Groq with an expert astronomy prompt. The LLM provides:
- **Verdict**: interesting / known_type / noise
- **Confidence**: 0.0–1.0
- **Reasoning**: "Rapid 3-mag brightening with no SIMBAD match suggests nova candidate"
- **Is candidate?**: Could this be a genuinely unclassified object?

**Step 7: Save to Database**
All results, LLM reviews, and scores persist in SQLite.

**Step 8: Auto-Retrain**
Collect training data from all 4 feedback sources and retrain the Feedback Classifier + Isolation Forest. The next session's scores will reflect everything learned.

---

## 8. The Scoring Formula (Why It Works)

```
final = 0.4 × AE + 0.3 × IF + 0.3 × Feedback
```

**Why these three, not just one?**

- **AE alone** catches objects with unusual light curve shapes but doesn't know what humans find interesting vs what's just noisy data.
- **IF alone** catches statistical outliers in embedding space but doesn't leverage the autoencoder's domain-specific representations.
- **Feedback alone** would only reflect past preferences with no anomaly detection capability.

Together: the AE provides the domain-specific representation, the IF provides unsupervised anomaly detection, and the Feedback Classifier provides learned human/LLM preferences. Each compensates for the others' weaknesses.

---

## 9. The 4-Source Self-Improvement (Why It's Novel)

Most anomaly detection systems in astronomy are **static** — train once, deploy forever. Ours improves continuously from four sources:

### Source 1: Human Feedback (weight = 1.0)
The astronomer clicks 👍 Interesting, 👎 Noise, or 🏷️ Classify on the dashboard. This is the highest-quality signal.

### Source 2: LLM Astronomer (weight = 0.7 × confidence)
Groq reviews flagged objects with an expert astronomy prompt. Its decisions feed directly into retraining. Weight is scaled by the LLM's own confidence — uncertain LLM predictions have less influence.

### Source 3: Self-Validation (weight = 0.5)
After each session, the system checks: "Did any objects I flagged before get classified by ALeRCE since then?" If yes → our flag was wrong. The system **learns from its own mistakes** without any human involvement.

### Source 4: Pseudo-Labels (weight = 0.3)
Objects scored with very high confidence (>0.95 anomaly or <0.05 normal) become pseudo-labeled training data. The system bootstraps itself.

**Why this hierarchy matters:** The human is always right (weight 1.0). The LLM is usually right (0.7). Self-validation is mechanically correct but limited (0.5). Pseudo-labels are probabilistic (0.3). If these conflict, the higher-weight source wins during retraining.

---

## 10. The LLM-in-the-Loop (Why It Changes Everything)

Traditional human-in-the-loop: **Requires an expert. Doesn't scale. Expensive.**

Our LLM-in-the-loop:
- Reviews 50 objects in 30 seconds (flat cost)
- Runs 24/7 without fatigue
- Provides **reasoning** (not just a label — explains WHY)
- The human reviews the LLM's reasoning, not raw data. Much faster.

The hierarchy enables a new workflow:

```
Night 1: System scans 1000 objects. LLM reviews 20 flagged.
         LLM marks 3 as candidates. Retrains overnight.
Night 2: System is smarter. Scans 1000 more. Better precision.
         LLM marks 2 as candidates.
Night 3: Astronomer logs in, sees 5 LLM candidates from 2 nights.
         Reviews them in 10 minutes (with LLM reasoning visible).
         Confirms 2 as genuinely interesting.
         → System learns from all of it.
```

**The astronomer's job goes from "review 1000 objects" to "review 5 LLM-vetted candidates."** That's a 200x reduction in workload.

---

## 11. How It's Better Than Everything Else

| System | Detects anomalies? | Self-improves? | LLM review? | Fully autonomous? |
|--------|-------------------|---------------|-------------|-------------------|
| ALeRCE | ❌ (classifies only) | ❌ | ❌ | ✅ |
| ANTARES | ⚠️ (rule-based) | ❌ | ❌ | ✅ |
| Fink | ⚠️ (module-based) | ❌ | ❌ | ✅ |
| Isolation Forest | ✅ | ❌ | ❌ | ✅ |
| **Stellar Anomaly Hunter** | **✅** | **✅ (4 sources)** | **✅** | **✅** |

The combination of deep learning embeddings + anomaly detection + multi-source self-improvement + LLM interpretation does not exist in any published system for astronomy.

---

## 12. How the Entire Combination Works Together

```
                    ┌─────────────────┐
                    │  ZTF Telescope   │
                    │  (real-time)     │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  ALeRCE Broker   │
                    │  (data source)   │
                    └────────┬────────┘
                             │
            ┌────────────────▼───────────────┐
            │    SCORING ENGINE               │
            │    TransformerAE (Kaggle)        │
            │    + Isolation Forest            │
            │    + Feedback Classifier         │
            │    = Blended anomaly score       │
            └────────────────┬───────────────┘
                             │
                    ┌────────▼────────┐
                    │    AI TRIAGE     │
                    │  Flag/Class/Dismiss│
                    └────────┬────────┘
                             │
              ┌──────────────▼─────────────┐
              │     LLM ASTRONOMER          │
              │  (Groq)             │
              │  Reviews flagged objects    │
              │  Provides expert reasoning  │
              └──────────────┬─────────────┘
                             │
              ┌──────────────▼─────────────┐
              │      DASHBOARD              │
              │  Human reviews LLM opinions │
              │  Confirms / Overrides       │
              └──────────────┬─────────────┘
                             │
         ┌───────────────────▼──────────────────┐
         │        4-SOURCE RETRAINING            │
         │  Human(1.0) + LLM(0.7) +             │
         │  Self-Val(0.5) + Pseudo(0.3)          │
         │  → Better IF + Better Classifier       │
         └───────────────────┬──────────────────┘
                             │
                     ┌───────▼───────┐
                     │   NEXT SESSION │
                     │  (smarter)     │
                     └───────────────┘
```

---

## 13. Summary

Built a system that goes from raw telescope data to ranked anomaly candidates with **zero human intervention required**, while still allowing expert oversight. The 4-tier feedback hierarchy means it gets smarter every night. The LLM bridges the gap between machine anomaly scores and human scientific reasoning.
