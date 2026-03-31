"""
llm_interpreter.py — LLM-in-the-Loop Automated Astronomer.

Uses Groq (Llama 3.3 70B) to review flagged astronomical objects,
provide expert reasoning, and feed decisions into the self-improvement loop.
"""
import os
import sys
import json
import time
import re
import requests
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))
from config import GROQ_API_KEY, LLM_REVIEW_TOP_N

# ── Groq API ─────────────────────────────────────────────────────────
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama-3.3-70b-versatile"

# ── Prompt ───────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an expert astronomer specialized in ZTF variable star classification and transient detection. You are reviewing objects flagged by an anomaly detection AI pipeline.

Your task: determine if each flagged object is genuinely interesting (potentially unclassified/new), a known variable star type, or noise/artifact.

Context about the scoring system:
- Anomaly Score (0-1): higher = more anomalous. Combines reconstruction error, kNN distance, and isolation forest score.
- Reconstruction Error: how poorly the TransformerAE model reconstructs this light curve. High = unusual shape.
- kNN Distance: how far this object's embedding is from known reference objects. High = unlike anything seen before.
- Nearest Class: the AI's best guess of what known class this resembles, with cosine distance.
- SIMBAD Match: whether this object appears in the SIMBAD astronomical database.

You must respond in valid JSON with exactly these fields:
{
  "verdict": "interesting" | "known_type" | "noise",
  "confidence": 0.0 to 1.0,
  "reasoning": "1-2 sentence explanation",
  "suggested_class": "class name or null",
  "is_candidate": true or false
}"""


def _build_prompt(obj):
    """Build review prompt for a single object."""
    simbad = obj.get('simbad_match') or 'NO MATCH'
    simbad_type = obj.get('simbad_otype') or 'unknown'
    auto_class = obj.get('auto_class') or 'unclassified'
    class_dist = obj.get('class_distance')
    class_dist_str = f"{class_dist:.3f}" if class_dist else 'N/A'

    return f"""Review this ZTF object:

Object ID: {obj['oid']}
Anomaly Score: {obj.get('score', 0):.4f}
Reconstruction Error: {obj.get('rec_error', 0):.5f}
kNN Distance: {obj.get('knn_dist', 0):.5f}
Nearest Known Class: {auto_class} (distance: {class_dist_str})
SIMBAD: {simbad} (type: {simbad_type})
Number of Detections: {obj.get('n_detections', 0)}
Mean Magnitude: {obj.get('mag_mean', 0):.2f}
Triage Reason: {obj.get('triage_reason', 'N/A')}

Provide your assessment as JSON."""


def _parse_response(text):
    """Extract JSON from LLM response, handling markdown code blocks."""
    text = text.strip()
    if text.startswith('```'):
        text = re.sub(r'^```(?:json)?\s*', '', text)
        text = re.sub(r'\s*```$', '', text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
    return None


def _call_groq(prompt, _retries=2):
    """Call Groq REST API with retry logic."""
    if not GROQ_API_KEY:
        return None

    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
        "max_tokens": 300,
    }

    for attempt in range(_retries):
        try:
            resp = requests.post(
                GROQ_URL,
                json=payload,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {GROQ_API_KEY}",
                },
                timeout=30,
            )
            if resp.status_code == 200:
                return resp.json()['choices'][0]['message']['content']
            elif resp.status_code == 429:
                wait = 5 * (attempt + 1)
                print(f"[LLM] Rate limited, waiting {wait}s (attempt {attempt+1}/{_retries})...")
                time.sleep(wait)
            else:
                print(f"[LLM] Groq error {resp.status_code}: {resp.text[:150]}")
                return None
        except requests.exceptions.Timeout:
            print(f"[LLM] Timeout (attempt {attempt+1}/{_retries})")
            time.sleep(3)
        except Exception as e:
            print(f"[LLM] Error: {e}")
            return None

    print(f"[LLM] Failed after {_retries} retries")
    return None


# ── Core review functions ────────────────────────────────────────────

def review_object(obj, _retries=2):
    """
    Review a single flagged object using Groq (Llama 3.3).

    Returns:
        dict with verdict, confidence, reasoning, suggested_class, is_candidate
        or None if LLM unavailable
    """
    if not GROQ_API_KEY:
        print("[LLM] No GROQ_API_KEY set. LLM review disabled.")
        return None

    prompt = _build_prompt(obj)
    text = _call_groq(prompt, _retries)
    if text is None:
        return None

    result = _parse_response(text)
    if result is None:
        print(f"[LLM] Warning: Could not parse response for {obj['oid']}: {text[:100]}")
        return None

    # Validate and normalize
    result['verdict'] = result.get('verdict', 'noise')
    result['confidence'] = float(result.get('confidence', 0.5))
    result['reasoning'] = result.get('reasoning', '')
    result['suggested_class'] = result.get('suggested_class')
    result['is_candidate'] = bool(result.get('is_candidate', False))
    result['oid'] = obj['oid']
    result['timestamp'] = datetime.utcnow().isoformat()

    return result


def batch_review(flagged_objects, db_module):
    """
    Review top N flagged objects using Groq.

    Args:
        flagged_objects: list of object dicts, sorted by score desc
        db_module: database module for logging

    Returns:
        list of review results
    """
    if not GROQ_API_KEY:
        print("[LLM] Skipping LLM review (no GROQ_API_KEY)")
        return []

    # ── Dedup: skip objects already reviewed by LLM ─────────────────────
    conn = db_module.get_conn()
    already_reviewed = set(
        r['oid'] for r in conn.execute(
            "SELECT DISTINCT oid FROM llm_review_log"
        ).fetchall()
    )
    conn.close()

    to_review = [o for o in flagged_objects[:LLM_REVIEW_TOP_N]
                 if o['oid'] not in already_reviewed]

    if len(to_review) < len(flagged_objects[:LLM_REVIEW_TOP_N]):
        skipped = len(flagged_objects[:LLM_REVIEW_TOP_N]) - len(to_review)
        print(f"[LLM] Skipping {skipped} already-reviewed objects")

    if not to_review:
        print("[LLM] All flagged objects already reviewed")
        return []

    print(f"[LLM] Reviewing {len(to_review)} flagged objects with Llama 3.3 (Groq)...")

    results = []
    for i, obj in enumerate(to_review):
        result = review_object(obj)
        if result:
            results.append(result)
            _log_review(db_module, result)

            candidate = ' ** CANDIDATE **' if result['is_candidate'] else ''
            print(f"   [{i+1}/{len(to_review)}] {obj['oid']}: "
                  f"{result['verdict']} ({result['confidence']:.0%}){candidate}")
            if result['reasoning']:
                print(f"         > {result['reasoning'][:80]}")

        # Rate limit: stay under 30 RPM free tier
        time.sleep(2)

    n_interesting = sum(1 for r in results if r['verdict'] == 'interesting')
    n_candidates = sum(1 for r in results if r.get('is_candidate'))
    print(f"\n[LLM] Review complete: {n_interesting} interesting, "
          f"{n_candidates} candidates out of {len(results)} reviewed")

    return results


def _log_review(db_module, result):
    """Save LLM review to database."""
    conn = db_module.get_conn()
    conn.execute("""
        INSERT INTO llm_review_log
        (oid, verdict, confidence, reasoning, suggested_class, is_candidate, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        result['oid'], result['verdict'], result['confidence'],
        result['reasoning'], result.get('suggested_class'),
        1 if result.get('is_candidate') else 0,
        result['timestamp']
    ))
    conn.commit()
    conn.close()


def feed_llm_to_retrain(db_module, emb_store_path):
    """
    Collect LLM reviews as training data for the feedback classifier.

    Returns:
        embeddings, labels, weights (for LLM source only)
    """
    import numpy as np

    conn = db_module.get_conn()
    reviews = conn.execute(
        "SELECT oid, verdict, confidence FROM llm_review_log"
    ).fetchall()
    conn.close()

    if not reviews:
        return np.array([]), np.array([]), np.array([])

    from retrain import _load_emb_store
    emb_dict = _load_emb_store(emb_store_path)
    if not emb_dict:
        return np.array([]), np.array([]), np.array([])

    emb_list, label_list, weight_list = [], [], []
    for row in reviews:
        oid = row['oid']
        if oid not in emb_dict:
            continue

        verdict = row['verdict']
        confidence = row['confidence']

        if verdict == 'interesting':
            label = 1.0
        elif verdict == 'noise':
            label = 0.0
        elif verdict == 'known_type':
            label = 0.2
        else:
            continue

        emb_list.append(emb_dict[oid])
        label_list.append(label)
        weight_list.append(0.7 * confidence)

    if not emb_list:
        return np.array([]), np.array([]), np.array([])

    return np.array(emb_list), np.array(label_list), np.array(weight_list)
