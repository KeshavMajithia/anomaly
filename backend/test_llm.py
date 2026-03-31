"""
test_llm.py — Tests for the AI Astronomer (Groq / Llama 3.3 70B).
Tests Groq connectivity, prompt building, JSON parsing, live reviews,
verdict correctness, and database logging.

Run: python test_llm.py
"""
import sys, os, time, json
sys.path.insert(0, os.path.dirname(__file__))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from config import GROQ_API_KEY, LLM_ENABLED
import database as db
db.init_db()

PASS = FAIL = 0

def ok(msg):
    global PASS; PASS += 1; print(f"  [PASS] {msg}")
def bad(msg):
    global FAIL; FAIL += 1; print(f"  [FAIL] {msg}")


# ── Test 1: API Key & Config ────────────────────────────────────────
print("\n=== 1. CONFIG ===")
if GROQ_API_KEY:
    ok(f"GROQ_API_KEY loaded from .env ({len(GROQ_API_KEY)} chars)")
else:
    bad("GROQ_API_KEY empty — add it to .env"); sys.exit(1)
if LLM_ENABLED:
    ok("LLM_ENABLED = True")
else:
    bad("LLM_ENABLED = False")


# ── Test 2: Module Imports ──────────────────────────────────────────
print("\n=== 2. MODULE IMPORTS ===")
try:
    from llm_interpreter import (review_object, batch_review,
                                 _build_prompt, _parse_response,
                                 _call_groq, feed_llm_to_retrain,
                                 SYSTEM_PROMPT, GROQ_MODEL)
    ok("llm_interpreter imports clean")
    ok(f"Model: {GROQ_MODEL}")
except Exception as e:
    bad(f"Import failed: {e}"); sys.exit(1)


# ── Test 3: Prompt Building ─────────────────────────────────────────
print("\n=== 3. PROMPT BUILDING ===")
test_obj_flagged = {
    "oid": "ZTF18acfldkj", "score": 0.92, "rec_error": 0.058,
    "knn_dist": 0.41, "auto_class": "DSCT", "class_distance": 0.28,
    "simbad_match": None, "simbad_otype": None,
    "n_detections": 85, "mag_mean": 17.3,
    "triage_reason": "High anomaly score, far from all centroids, no SIMBAD match",
}
prompt = _build_prompt(test_obj_flagged)
for field in ["ZTF18acfldkj", "0.9200", "0.05800", "0.41000", "DSCT", "NO MATCH", "85", "17.30"]:
    if field in prompt:
        ok(f"Prompt contains '{field}'")
    else:
        bad(f"Prompt missing '{field}'")


# ── Test 4: JSON Parsing ────────────────────────────────────────────
print("\n=== 4. JSON PARSING ===")
# Clean JSON
r1 = _parse_response('{"verdict":"interesting","confidence":0.85,"reasoning":"test","suggested_class":null,"is_candidate":true}')
if r1 and r1['verdict'] == 'interesting': ok("Clean JSON parsed")
else: bad("Clean JSON failed")

# Markdown-wrapped
r2 = _parse_response('```json\n{"verdict":"noise","confidence":0.9,"reasoning":"known","suggested_class":"RRL","is_candidate":false}\n```')
if r2 and r2['verdict'] == 'noise': ok("Markdown JSON parsed")
else: bad("Markdown JSON failed")

# Extra text around JSON
r3 = _parse_response('Here is my analysis:\n{"verdict":"known_type","confidence":0.7,"reasoning":"test","suggested_class":"EB","is_candidate":false}\nEnd.')
if r3 and r3['verdict'] == 'known_type': ok("Embedded JSON parsed")
else: bad("Embedded JSON failed")

# Garbage
r4 = _parse_response("I am not JSON at all")
if r4 is None: ok("Garbage returns None")
else: bad("Garbage didn't return None")


# ── Test 5: Groq API Connectivity ───────────────────────────────────
print("\n=== 5. GROQ API CONNECTIVITY ===")
simple = _call_groq("Respond with exactly: OK")
if simple and len(simple) > 0:
    ok(f"Groq API responds: '{simple.strip()[:30]}'")
else:
    bad("Groq API returned None"); sys.exit(1)

time.sleep(2)


# ── Test 6: Live Review — Anomalous Object ──────────────────────────
print("\n=== 6. LIVE REVIEW: ANOMALOUS OBJECT ===")
result1 = review_object(test_obj_flagged)
if result1 is None:
    bad("Review returned None"); sys.exit(1)

ok(f"Verdict: {result1['verdict']} ({result1['confidence']:.0%})")
ok(f"Reasoning: {result1['reasoning']}")
ok(f"Candidate: {result1['is_candidate']}")
ok(f"Suggested class: {result1.get('suggested_class')}")

if result1['verdict'] in ('interesting', 'known_type', 'noise'):
    ok("Verdict is valid enum value")
else:
    bad(f"Invalid verdict: {result1['verdict']}")

if 0 <= result1['confidence'] <= 1:
    ok("Confidence in [0, 1]")
else:
    bad(f"Confidence out of range: {result1['confidence']}")

if len(result1['reasoning']) > 10:
    ok("Reasoning is substantive")
else:
    bad("Reasoning too short")

time.sleep(2)


# ── Test 7: Live Review — Known Object (should NOT be interesting) ──
print("\n=== 7. LIVE REVIEW: KNOWN OBJECT ===")
test_obj_known = {
    "oid": "ZTF19known01", "score": 0.15, "rec_error": 0.005,
    "knn_dist": 0.02, "auto_class": "RRL", "class_distance": 0.01,
    "simbad_match": "V* RR Lyr", "simbad_otype": "RRLyr",
    "n_detections": 500, "mag_mean": 12.8,
    "triage_reason": "Low score, SIMBAD matched as RR Lyrae",
}
result2 = review_object(test_obj_known)
if result2 is None:
    bad("Known object review returned None")
else:
    ok(f"Verdict: {result2['verdict']} ({result2['confidence']:.0%})")
    ok(f"Reasoning: {result2['reasoning']}")
    if result2['verdict'] in ('known_type', 'noise'):
        ok("Correctly identified known object as NOT interesting")
    else:
        bad(f"Misclassified known RR Lyr as '{result2['verdict']}'")
    if not result2['is_candidate']:
        ok("Correctly NOT flagged as candidate")
    else:
        bad("Incorrectly flagged known object as candidate")

time.sleep(2)


# ── Test 8: Live Review — Noisy Object ──────────────────────────────
print("\n=== 8. LIVE REVIEW: NOISY OBJECT ===")
test_obj_noise = {
    "oid": "ZTF20noise99", "score": 0.08, "rec_error": 0.002,
    "knn_dist": 0.01, "auto_class": "LPV", "class_distance": 0.005,
    "simbad_match": "V* Mira", "simbad_otype": "Mira",
    "n_detections": 3, "mag_mean": 11.2,
    "triage_reason": "Very low score, few detections, SIMBAD match",
}
result3 = review_object(test_obj_noise)
if result3 is None:
    bad("Noise review returned None")
else:
    ok(f"Verdict: {result3['verdict']} ({result3['confidence']:.0%})")
    if result3['verdict'] in ('known_type', 'noise'):
        ok("Correctly dismissed noisy/known object")
    else:
        bad(f"Noisy object flagged as '{result3['verdict']}'")

time.sleep(2)


# ── Test 9: Live Review — Genuinely Weird Object ────────────────────
print("\n=== 9. LIVE REVIEW: GENUINELY WEIRD OBJECT ===")
test_obj_weird = {
    "oid": "ZTF21weird42", "score": 0.99, "rec_error": 0.120,
    "knn_dist": 0.85, "auto_class": "unclassified", "class_distance": 0.95,
    "simbad_match": None, "simbad_otype": None,
    "n_detections": 45, "mag_mean": 19.1,
    "triage_reason": "Extreme anomaly score, no SIMBAD match, far from all classes, faint",
}
result4 = review_object(test_obj_weird)
if result4 is None:
    bad("Weird object review returned None")
else:
    ok(f"Verdict: {result4['verdict']} ({result4['confidence']:.0%})")
    ok(f"Reasoning: {result4['reasoning']}")
    if result4['verdict'] == 'interesting':
        ok("Correctly flagged genuinely weird object as interesting")
    else:
        bad(f"Genuinely weird object classified as '{result4['verdict']}' (expected 'interesting')")
    if result4['is_candidate']:
        ok("Flagged as candidate")
    else:
        bad("Not flagged as candidate (expected true for extreme anomaly)")

time.sleep(2)


# ── Test 10: Database Logging ───────────────────────────────────────
print("\n=== 10. DATABASE LOGGING ===")
from llm_interpreter import _log_review
test_result = {
    "oid": "ZTF_DBTEST_01", "verdict": "interesting", "confidence": 0.88,
    "reasoning": "Test entry for DB logging", "suggested_class": None,
    "is_candidate": True, "timestamp": "2026-03-05T00:00:00"
}
_log_review(db, test_result)

conn = db.get_conn()
row = conn.execute("SELECT * FROM llm_review_log WHERE oid='ZTF_DBTEST_01' ORDER BY rowid DESC LIMIT 1").fetchone()
conn.close()

if row is None:
    bad("LLM review not found in database")
else:
    if row['verdict'] == 'interesting': ok("DB verdict correct")
    else: bad(f"DB verdict: {row['verdict']}")
    if abs(row['confidence'] - 0.88) < 0.01: ok("DB confidence correct")
    else: bad(f"DB confidence: {row['confidence']}")
    if row['is_candidate'] == 1: ok("DB is_candidate correct")
    else: bad(f"DB is_candidate: {row['is_candidate']}")
    if 'Test entry' in row['reasoning']: ok("DB reasoning correct")
    else: bad(f"DB reasoning: {row['reasoning']}")


# ── Test 11: Feed to Retrain ────────────────────────────────────────
print("\n=== 11. FEED LLM TO RETRAIN ===")
from feedback import EMB_STORE
llm_emb, llm_labels, llm_weights = feed_llm_to_retrain(db, EMB_STORE)
ok(f"LLM feedback: {len(llm_emb)} samples")
if len(llm_emb) > 0:
    if all(0 <= l <= 1 for l in llm_labels): ok("Labels in [0, 1]")
    else: bad("Labels out of range")
    if all(0 <= w <= 0.7 for w in llm_weights): ok("Weights capped at 0.7")
    else: bad(f"Weights exceed 0.7: max={max(llm_weights):.3f}")


# ── Summary ─────────────────────────────────────────────────────────
print(f"\n{'='*50}")
print(f"  AI ASTRONOMER TEST RESULTS")
print(f"{'='*50}")
print(f"  PASS: {PASS}")
print(f"  FAIL: {FAIL}")
if FAIL == 0:
    print(f"  ALL TESTS PASSED")
else:
    print(f"  {FAIL} FAILURES — see above")
print(f"{'='*50}\n")