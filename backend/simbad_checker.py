"""
simbad_checker.py — Cross-match objects with the SIMBAD astronomical database.
"""
import time

import numpy as np


def check_simbad(oids, ras, decs, radius_arcsec=5.0, verbose=True):
    """
    Query SIMBAD for each object by RA/Dec.

    Returns:
        dict: {oid: {'match': str or None, 'otype': str or None}}
    """
    try:
        from astroquery.simbad import Simbad
        from astropy.coordinates import SkyCoord
        import astropy.units as u
    except ImportError:
        raise ImportError("Install astroquery: pip install astroquery")

    simbad = Simbad()
    simbad.add_votable_fields('otype')
    results = {}

    if verbose:
        print(f"🔭 SIMBAD checking {len(oids)} objects ({radius_arcsec}\" radius)...")

    for i, (oid, ra, dec) in enumerate(zip(oids, ras, decs)):
        if not (np.isfinite(ra) and np.isfinite(dec)):
            results[oid] = {'match': None, 'otype': None, 'matched': False}
            continue

        try:
            from astropy.coordinates import SkyCoord
            import astropy.units as u
            coord = SkyCoord(ra=ra * u.degree, dec=dec * u.degree)
            result = simbad.query_region(coord, radius=radius_arcsec * u.arcsec)

            if result is None or len(result) == 0:
                results[oid] = {'match': None, 'otype': None, 'matched': False}
                if verbose:
                    print(f"   #{i+1:>4} {oid} → ❌ NO MATCH")
            else:
                main_id = str(result['MAIN_ID'][0])
                otype   = str(result['OTYPE'][0]) if 'OTYPE' in result.columns else '?'
                results[oid] = {'match': main_id, 'otype': otype, 'matched': True}
                if verbose:
                    print(f"   #{i+1:>4} {oid} → ✅ {main_id} ({otype})")

        except Exception as e:
            results[oid] = {'match': f'ERROR: {e}', 'otype': None, 'matched': False}

        if (i + 1) % 15 == 0:
            time.sleep(2)  # respect SIMBAD rate limits

    unmatched = sum(1 for v in results.values() if not v['matched'])
    if verbose:
        print(f"\n   SIMBAD: {len(results) - unmatched} matched, {unmatched} unmatched")

    return results
