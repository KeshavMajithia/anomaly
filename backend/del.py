"""
del.py — Reset the Stellar Anomaly Hunter database for a fresh start.

Usage:
    python del.py          # Reset everything (DB + trained models)
    python del.py --db     # Reset only the database
    python del.py --models # Delete only the trained models
"""
import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(__file__))
from config import DB_PATH, MODELS_DIR, BASE_DIR

EMB_STORE = os.path.join(BASE_DIR, 'database', 'emb_store.npz')


def reset_db():
    """Clear all tables in the database."""
    import database as db
    db.init_db()
    conn = db.get_conn()

    tables = ['objects', 'feedback_log', 'discoveries', 'model_versions', 'llm_review_log']
    print("\n  Current state:")
    for tbl in tables:
        cnt = conn.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0]
        print(f"    {tbl}: {cnt} rows")

    for tbl in tables:
        conn.execute(f"DELETE FROM {tbl}")
    conn.commit()
    conn.close()
    print("\n  ✅ All tables cleared")


def reset_models():
    """Delete trained feedback models (not Kaggle-trained base models)."""
    deleted = []
    for f in ['feedback_classifier.pkl', 'isolation_forest.pkl']:
        p = os.path.join(MODELS_DIR, f)
        if os.path.exists(p):
            os.remove(p)
            deleted.append(f)

    if os.path.exists(EMB_STORE):
        os.remove(EMB_STORE)
        deleted.append('emb_store.npz')

    if deleted:
        print(f"  ✅ Deleted: {', '.join(deleted)}")
    else:
        print("  ✅ No trained models to delete (already clean)")

    # Keep these (Kaggle base model artifacts):
    kept = ['ztf_transformer.pt', 'scaler.pkl', 'ref_embeddings.npy',
            'class_centroids.json', 'noise_embeddings.npy']
    print(f"  ℹ️  Kept base model files: {', '.join(kept)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Reset Stellar Anomaly Hunter data')
    parser.add_argument('--db', action='store_true', help='Reset only the database')
    parser.add_argument('--models', action='store_true', help='Delete only trained models')
    args = parser.parse_args()

    print("=" * 50)
    print("  🗑️  STELLAR ANOMALY HUNTER — RESET")
    print("=" * 50)

    if args.db:
        reset_db()
    elif args.models:
        reset_models()
    else:
        # Reset everything
        reset_db()
        reset_models()

    print(f"\n{'=' * 50}")
    print("  ✅ Reset complete. Ready for fresh discovery runs.")
    print(f"{'=' * 50}\n")
