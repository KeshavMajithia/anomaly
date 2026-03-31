"""
preprocessor.py — Convert raw ZTF light curve data into model-ready tensors.
Handles both IRSA (Kaggle training) and ALeRCE (live session) column formats.
"""
import numpy as np
import pandas as pd


N_BINS = 50
N_BANDS = 3     # ZTF: g(1), r(2), i(3)
N_FPB = 4       # mean, std, count, time_delta
N_FEAT = N_BANDS * N_FPB  # 12
BAND_MAP = {1: 0, 2: 1, 3: 2}

# ALeRCE → standard column name mapping
COL_MAP = {
    # Time
    'mjd': 'mjd', 'jd': 'mjd',
    # Magnitude
    'magpsf': 'mag', 'mag': 'mag', 'magpsf_corr': 'mag',
    'magap': 'mag', 'magap_corr': 'mag',
    # Magnitude error
    'sigmapsf': 'magerr', 'magerr': 'magerr', 'sigmapsf_corr': 'magerr',
    'sigmagap': 'magerr', 'sigmagap_corr': 'magerr',
    # Band
    'fid': 'fid', 'filterid': 'fid',
    # Object ID
    'oid': 'oid', 'objectId': 'oid', 'objectid': 'oid',
    # Position
    'ra': 'ra', 'dec': 'dec',
}


def normalize_columns(df):
    """Rename ALeRCE columns to standard names used by preprocessor."""
    rename = {}
    have = set()
    for col in df.columns:
        std = COL_MAP.get(col)
        if std and std not in have:
            rename[col] = std
            have.add(std)
    return df.rename(columns=rename)


def preprocess_single(lc_df, scaler):
    """
    Convert one light curve DataFrame to a normalized (N_BINS, N_FEAT) tensor.

    Args:
        lc_df: DataFrame with time + mag + fid columns (any naming convention)
        scaler: fitted sklearn StandardScaler

    Returns:
        X: np.float32 array of shape (N_BINS, N_FEAT), or None if invalid
        M: np.float32 mask of shape (N_BINS,)
        phys: np.float32 array of shape (5,) — physical metadata
    """
    X = np.zeros((N_BINS, N_FEAT), dtype=np.float32)
    M = np.zeros(N_BINS, dtype=np.float32)
    phys = np.zeros(5, dtype=np.float32)

    # Find time column
    t_col = None
    for c in ['mjd', 'jd']:
        if c in lc_df.columns:
            t_col = c
            break
    if t_col is None:
        return None, None, None

    # Find mag column
    mag_col = None
    for c in ['mag', 'magpsf', 'magpsf_corr', 'magap']:
        if c in lc_df.columns:
            mag_col = c
            break
    if mag_col is None:
        return None, None, None

    t   = pd.to_numeric(lc_df[t_col].values, errors='coerce')
    mag = pd.to_numeric(lc_df[mag_col].values, errors='coerce')

    # Band filter
    if 'fid' in lc_df.columns:
        bands = pd.to_numeric(lc_df['fid'].values, errors='coerce')
        bands = np.where(np.isfinite(bands), bands, 1).astype(int)
    else:
        bands = np.ones(len(t), dtype=int)

    ok = np.isfinite(t) & np.isfinite(mag) & (mag > 0) & (mag < 30)
    t, mag, bands = t[ok], mag[ok], bands[ok]

    if len(t) < 5:
        return None, None, None

    # Physical metadata
    phys[0] = np.min(mag)
    phys[1] = t.max() - t.min()
    phys[2] = np.max(mag) - np.min(mag)
    phys[3] = len(t)
    phys[4] = np.mean(mag)

    # Convert to flux-like
    flux = 10 ** ((mag.mean() - mag) / 2.5)

    # Quantile binning
    edges = np.unique(np.percentile(t, np.linspace(0, 100, N_BINS + 1)))
    if len(edges) < 3:
        return None, None, None

    nb = min(len(edges) - 1, N_BINS)
    bi = np.clip(np.digitize(t, edges[:-1]) - 1, 0, nb - 1)

    for bb in range(nb):
        ib = bi == bb
        if not ib.any():
            continue
        M[bb] = 1.0
        for band_fid in [1, 2, 3]:
            band_idx = BAND_MAP.get(band_fid, -1)
            if band_idx < 0:
                continue
            bm = (bands == band_fid) & ib
            if not bm.any():
                continue
            bf = flux[bm]
            off = band_idx * N_FPB
            X[bb, off]     = np.mean(bf)
            X[bb, off + 1] = np.std(bf) if len(bf) > 1 else 0
            X[bb, off + 2] = len(bf)
        if bb > 0:
            c0 = (edges[max(0, bb - 1)] + edges[bb]) / 2
            c1 = (edges[bb] + edges[min(bb + 1, len(edges) - 1)]) / 2
            for band_idx in range(N_BANDS):
                X[bb, band_idx * N_FPB + 3] = c1 - c0

    # Normalize with fitted scaler
    nz = X.sum(1) != 0
    if nz.sum() > 0:
        try:
            X[nz] = scaler.transform(X[nz])
        except Exception:
            pass

    return X, M, phys


def batch_preprocess(lc_df, oid_list, scaler):
    """
    Preprocess a batch of light curves from a grouped DataFrame.

    Args:
        lc_df: DataFrame with oid column + light curve columns
        oid_list: list of object IDs to process
        scaler: fitted StandardScaler

    Returns:
        X_arr, M_arr, P_arr, valid_oids
    """
    # Normalize column names first
    lc_df = normalize_columns(lc_df)

    # Debug: print columns and sample
    print(f"   [Debug] LC columns: {list(lc_df.columns)}")
    print(f"   [Debug] LC shape: {lc_df.shape}")
    if len(lc_df) > 0:
        print(f"   [Debug] Sample row: {dict(lc_df.iloc[0])}")

    # Ensure oid column exists and is string
    if 'oid' not in lc_df.columns:
        if lc_df.index.name == 'oid':
            lc_df = lc_df.reset_index()
        elif 'objectId' in lc_df.columns:
            lc_df = lc_df.rename(columns={'objectId': 'oid'})
        else:
            print("   ⚠️ No 'oid' column found in light curve data")
            return np.empty((0, N_BINS, N_FEAT)), np.empty((0, N_BINS)), np.empty((0, 5)), []

    lc_df['oid'] = lc_df['oid'].astype(str)
    oid_list = [str(o) for o in oid_list]

    grp = lc_df.groupby('oid')
    X_list, M_list, P_list, valid = [], [], [], []

    reasons = {'not_in_df': 0, 'preprocess_none': 0, 'few_bins': 0, 'ok': 0}

    for oid in oid_list:
        if oid not in grp.groups:
            reasons['not_in_df'] += 1
            continue
        obj_df = grp.get_group(oid)
        X, M, phys = preprocess_single(obj_df, scaler)
        if X is None:
            reasons['preprocess_none'] += 1
            continue
        if M.sum() < 3:
            reasons['few_bins'] += 1
            continue
        X_list.append(X)
        M_list.append(M)
        P_list.append(phys)
        valid.append(oid)
        reasons['ok'] += 1

    # Debug summary
    if reasons['ok'] == 0:
        print(f"   [Debug] Drop reasons: {reasons}")
        # Show a sample object to diagnose
        sample_oid = oid_list[0] if oid_list else None
        if sample_oid and sample_oid in grp.groups:
            s = grp.get_group(sample_oid)
            print(f"   [Debug] Sample OID {sample_oid}: {len(s)} rows")
            print(f"   [Debug] Columns: {list(s.columns)}")
            print(f"   [Debug] Head:\n{s.head(3).to_string()}")

    if not X_list:
        return np.empty((0, N_BINS, N_FEAT)), np.empty((0, N_BINS)), np.empty((0, 5)), []

    return (np.stack(X_list).astype(np.float32),
            np.stack(M_list).astype(np.float32),
            np.stack(P_list).astype(np.float32),
            valid)
