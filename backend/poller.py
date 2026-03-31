"""
poller.py — Fetch recent ZTF alerts from the ALeRCE broker.
Features: lastmjd query, ndet filter, local LC cache, demo mode.
"""
import os
import time
from datetime import datetime, timezone, timedelta

import pandas as pd

from config import LOOKBACK_HOURS, MAX_ALERTS, LC_CACHE_DIR

# Minimum detections for a useful light curve
MIN_DETECTIONS = 20


def fetch_recent_alerts(hours_back=LOOKBACK_HOURS, max_alerts=MAX_ALERTS, verbose=True):
    """
    Fetch ZTF objects that had recent activity AND have enough observations.
    Uses local cache for light curves to avoid re-downloading.
    """
    try:
        from alerce.core import Alerce
    except ImportError:
        raise ImportError("Install alerce: pip install alerce")

    alerce = Alerce()
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours_back)
    cutoff_mjd = _to_mjd(cutoff)
    cutoff_str = cutoff.strftime('%Y-%m-%d %H:%M:%S')

    if verbose:
        print(f"📡 Fetching ZTF objects active since {cutoff_str} ({hours_back}h ago)...")
        print(f"   Min detections: {MIN_DETECTIONS} | Max objects: {max_alerts}")

    all_meta = []
    page = 1
    fetched = 0
    errors = 0

    while fetched < max_alerts:
        try:
            batch = alerce.query_objects(
                format='pandas',
                page_size=min(500, max_alerts - fetched),
                page=page,
                lastmjd=cutoff_mjd,
                ndet=MIN_DETECTIONS,
            )
        except Exception as e:
            errors += 1
            if verbose:
                print(f"   ⚠️ Page {page} error: {e}")
            if errors > 5:
                break
            time.sleep(3)
            continue

        if batch is None or len(batch) == 0:
            break

        all_meta.append(batch)
        fetched += len(batch)

        if verbose and page % 5 == 0:
            print(f"   {fetched:,} objects fetched so far...")

        page += 1
        time.sleep(0.5)

    if not all_meta:
        if verbose:
            print("   ⚠️ No objects returned from ALeRCE")
        return pd.DataFrame(), pd.DataFrame()

    meta_df = pd.concat(all_meta, ignore_index=True)
    if 'oid' not in meta_df.columns and meta_df.index.name == 'oid':
        meta_df = meta_df.reset_index()
    meta_df['oid'] = meta_df['oid'].astype(str)

    if verbose:
        print(f"   ✅ {len(meta_df):,} objects fetched (all ≥{MIN_DETECTIONS} detections)")
        if 'ndet' in meta_df.columns:
            print(f"   Detections: min={meta_df['ndet'].min()}, "
                  f"median={meta_df['ndet'].median():.0f}, "
                  f"max={meta_df['ndet'].max()}")

    # Fetch light curves (with caching)
    lc_df = _fetch_lightcurves(alerce, meta_df['oid'].tolist(), verbose=verbose)
    return meta_df, lc_df


def _fetch_lightcurves(alerce, oids, verbose=True):
    """Fetch detection light curves with local caching."""
    os.makedirs(LC_CACHE_DIR, exist_ok=True)

    all_lc = []
    errors = 0
    cached = 0
    downloaded = 0

    if verbose:
        print(f"   📡 Fetching light curves for {len(oids):,} objects...")

    for i, oid in enumerate(oids):
        cache_path = os.path.join(LC_CACHE_DIR, f'{oid}.parquet')

        # Check cache first
        if os.path.exists(cache_path):
            try:
                df = pd.read_parquet(cache_path)
                df['oid'] = oid
                all_lc.append(df)
                cached += 1
                continue
            except Exception:
                pass  # corrupt cache, re-download

        # Download from ALeRCE
        try:
            dets = alerce.query_detections(oid, format='pandas')
            if dets is not None and len(dets) > 0:
                dets['oid'] = oid
                all_lc.append(dets)
                downloaded += 1
                # Cache it
                try:
                    dets.drop(columns=['oid'], errors='ignore').to_parquet(cache_path)
                except Exception:
                    pass
        except Exception:
            errors += 1
            if errors > len(oids) * 0.3:
                if verbose:
                    print(f"   ⚠️ Too many errors ({errors}), stopping")
                break

        if (i + 1) % 50 == 0:
            time.sleep(1)
            if verbose:
                print(f"   {i + 1:,}/{len(oids):,}... "
                      f"(cached={cached}, downloaded={downloaded}, errors={errors})")

    if not all_lc:
        return pd.DataFrame()

    lc = pd.concat(all_lc, ignore_index=True)
    lc['oid'] = lc['oid'].astype(str)

    if verbose:
        total = lc['oid'].nunique()
        avg = len(lc) / total if total > 0 else 0
        print(f"   ✅ {total:,} light curves ({len(lc):,} detections)")
        print(f"      {cached} from cache, {downloaded} downloaded, {errors} errors")
        print(f"      Average {avg:.0f} detections per object")

    return lc


def _to_mjd(dt):
    """Convert datetime to Modified Julian Date."""
    j2000 = datetime(2000, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    return 51544.5 + (dt - j2000).total_seconds() / 86400.0
