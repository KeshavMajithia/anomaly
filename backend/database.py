"""
database.py — SQLite helpers for the anomaly_hunter PoC.
"""
import sqlite3
from datetime import datetime

from config import DB_PATH


def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create all tables if they don't already exist."""
    conn = get_conn()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS objects (
            oid             TEXT PRIMARY KEY,
            ra              REAL,
            dec             REAL,
            score           REAL,
            rec_error       REAL,
            knn_dist        REAL,
            n_detections    INTEGER,
            mag_mean        REAL,
            mag_err_mean    REAL,
            triage          TEXT,
            triage_reason   TEXT,
            auto_class      TEXT,
            class_distance  REAL,
            simbad_match    TEXT,
            simbad_otype    TEXT,
            human_feedback  TEXT,
            model_version   INTEGER DEFAULT 1,
            scored_at       TEXT,
            feedback_at     TEXT,
            flag_count      INTEGER DEFAULT 0,
            last_flagged_at TEXT
        );

        CREATE TABLE IF NOT EXISTS feedback_log (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            oid         TEXT,
            action      TEXT,
            old_triage  TEXT,
            new_label   TEXT,
            timestamp   TEXT
        );

        CREATE TABLE IF NOT EXISTS discoveries (
            oid         TEXT PRIMARY KEY,
            score       REAL,
            ra          REAL,
            dec         REAL,
            simbad_match TEXT,
            flagged_at  TEXT,
            confirmed_by TEXT DEFAULT 'human',
            notes       TEXT
        );

        CREATE TABLE IF NOT EXISTS model_versions (
            version      INTEGER PRIMARY KEY,
            trained_at   TEXT,
            noise_size   INTEGER,
            n_feedback   INTEGER,
            notes        TEXT
        );

        CREATE TABLE IF NOT EXISTS llm_review_log (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            oid             TEXT,
            verdict         TEXT,
            confidence      REAL,
            reasoning       TEXT,
            suggested_class TEXT,
            is_candidate    INTEGER DEFAULT 0,
            timestamp       TEXT
        );
    """)
    # ── Migration: add flag_count columns to existing DBs ────────────────
    for col, definition in [
        ('flag_count',      'INTEGER DEFAULT 0'),
        ('last_flagged_at', 'TEXT'),
    ]:
        try:
            conn.execute(f'ALTER TABLE objects ADD COLUMN {col} {definition}')
            conn.commit()
        except Exception:
            pass  # Column already exists — fine
    conn.close()
    print(f"✅ Database ready: {DB_PATH}")


def upsert_objects(objects):
    """
    Insert or update a list of scored+triaged objects.
    Preserves human_feedback, flag_count, and last_flagged_at from previous sessions.
    Increments flag_count each time an object is scored as 'flagged'.
    """
    conn = get_conn()
    now = datetime.utcnow().isoformat()
    for o in objects:
        triage = o.get('triage')
        is_flagged = 1 if triage == 'flagged' else 0
        conn.execute("""
            INSERT INTO objects
            (oid, ra, dec, score, rec_error, knn_dist,
             n_detections, mag_mean, mag_err_mean,
             triage, triage_reason, auto_class, class_distance,
             simbad_match, simbad_otype, human_feedback,
             model_version, scored_at, feedback_at,
             flag_count, last_flagged_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(oid) DO UPDATE SET
                ra              = excluded.ra,
                dec             = excluded.dec,
                score           = excluded.score,
                rec_error       = excluded.rec_error,
                knn_dist        = excluded.knn_dist,
                n_detections    = excluded.n_detections,
                mag_mean        = excluded.mag_mean,
                mag_err_mean    = excluded.mag_err_mean,
                triage          = excluded.triage,
                triage_reason   = excluded.triage_reason,
                auto_class      = excluded.auto_class,
                class_distance  = excluded.class_distance,
                simbad_match    = excluded.simbad_match,
                simbad_otype    = excluded.simbad_otype,
                model_version   = excluded.model_version,
                scored_at       = excluded.scored_at,
                flag_count      = objects.flag_count + excluded.flag_count,
                last_flagged_at = CASE WHEN excluded.flag_count > 0
                                       THEN excluded.last_flagged_at
                                       ELSE objects.last_flagged_at END
                -- human_feedback and feedback_at are intentionally NOT updated
                -- so that human labels survive re-scoring
        """, (
            o.get('oid'), o.get('ra'), o.get('dec'),
            o.get('score'), o.get('rec_error'), o.get('knn_dist'),
            o.get('n_detections'), o.get('mag_mean'), o.get('mag_err_mean'),
            triage, o.get('triage_reason'), o.get('auto_class'),
            o.get('class_distance'), o.get('simbad_match'), o.get('simbad_otype'),
            None, 1, now, None,
            is_flagged, now if is_flagged else None
        ))
    conn.commit()
    conn.close()


def save_feedback(oid, action, old_triage, new_label=None):
    """Log human feedback + update the object record."""
    conn = get_conn()
    now = datetime.utcnow().isoformat()
    conn.execute("""
        INSERT INTO feedback_log (oid, action, old_triage, new_label, timestamp)
        VALUES (?, ?, ?, ?, ?)
    """, (oid, action, old_triage, new_label, now))
    conn.execute("""
        UPDATE objects SET human_feedback=?, feedback_at=? WHERE oid=?
    """, (new_label or action, now, oid))
    conn.commit()
    conn.close()


def add_discovery(oid, score, ra, dec, simbad_match=None):
    """Record a confirmed interesting object."""
    conn = get_conn()
    conn.execute("""
        INSERT OR REPLACE INTO discoveries
        (oid, score, ra, dec, simbad_match, flagged_at)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (oid, score, ra, dec, simbad_match, datetime.utcnow().isoformat()))
    conn.commit()
    conn.close()


def get_flagged(limit=100):
    conn = get_conn()
    rows = conn.execute("""
        SELECT * FROM objects
        WHERE triage='flagged'
          AND (human_feedback IS NULL OR human_feedback = '')
        ORDER BY score DESC LIMIT ?
    """, (limit,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_dismissed(limit=5000):
    conn = get_conn()
    rows = conn.execute("""
        SELECT oid, ra, dec, score, rec_error, knn_dist,
               n_detections, mag_err_mean, model_version
        FROM objects WHERE triage='dismissed'
        ORDER BY score DESC LIMIT ?
    """, (limit,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_stats():
    conn = get_conn()
    stats = {}
    stats['total'] = conn.execute("SELECT COUNT(*) FROM objects").fetchone()[0]
    for triage in ['flagged', 'classified', 'dismissed']:
        stats[triage] = conn.execute(
            "SELECT COUNT(*) FROM objects WHERE triage=?", (triage,)).fetchone()[0]
    stats['discoveries'] = conn.execute("SELECT COUNT(*) FROM discoveries").fetchone()[0]
    stats['feedback'] = conn.execute("SELECT COUNT(*) FROM feedback_log").fetchone()[0]
    conn.close()
    return stats


def get_all_objects(limit=200, triage=None):
    conn = get_conn()
    if triage:
        rows = conn.execute(
            "SELECT * FROM objects WHERE triage=? ORDER BY score DESC LIMIT ?",
            (triage, limit)).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM objects ORDER BY score DESC LIMIT ?", (limit,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_discoveries():
    conn = get_conn()
    rows = conn.execute("""
        SELECT d.*, o.flag_count, o.last_flagged_at
        FROM discoveries d
        LEFT JOIN objects o ON d.oid = o.oid
        ORDER BY d.score DESC
    """).fetchall()
    conn.close()
    return [dict(r) for r in rows]
