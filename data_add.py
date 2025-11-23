"""
mast_bulk_xmatch.py - Fixed Version
Performs a bulk crossmatch of hyg_v42.csv against the TIC catalog on MAST.
Uses batch queries with rate limiting for reliability.
"""

import pandas as pd
import numpy as np
import math
import time
from astroquery.mast import Catalogs
from astropy.coordinates import SkyCoord
from astropy import units as u
from tqdm import tqdm

# --- CONFIG ---
INPUT_CSV = "hyg_v42.csv"
OUTPUT_CSV = "hyg_v42_updated.csv"
SEARCH_RADIUS_ARCSEC = 30
EPOCH_HYG = 2000.0
EPOCH_TIC = 2015.5
YEARS = EPOCH_TIC - EPOCH_HYG
BATCH_SIZE = 100  # Process in batches to avoid timeouts

# TEST MODE: Set to a number to process only first N rows (e.g., 10 for testing)
# Set to None to process all rows
TEST_LIMIT = None  # Change to None for full run

# Column names in your HYG file
COL_RA = "ra"
COL_DEC = "dec"
COL_PMRA = "pmra"
COL_PMDEC = "pmdec"


def pm_propagate(ra_deg, dec_deg, pmra_masyr, pmdec_masyr, years=YEARS):
    """
    Propagate (RA,Dec) from epoch 2000.0 to epoch 2015.5 using proper motion.
    """
    try:
        if np.isnan(pmra_masyr) or np.isnan(pmdec_masyr):
            return ra_deg, dec_deg

        # Convert proper motion (mas/yr) to degrees per year
        dra_deg_per_year = (pmra_masyr / 1000.0) / 3600.0 / math.cos(math.radians(dec_deg))
        ddec_deg_per_year = (pmdec_masyr / 1000.0) / 3600.0

        ra_new = ra_deg + dra_deg_per_year * years
        dec_new = dec_deg + ddec_deg_per_year * years
        return ra_new, dec_new
    except Exception:
        return ra_deg, dec_deg


def query_tic_cone(ra, dec, radius_arcsec=SEARCH_RADIUS_ARCSEC):
    """
    Query TIC catalog for a single coordinate using cone search.
    Returns the closest match within radius.
    """
    try:
        coord = SkyCoord(ra=ra*u.deg, dec=dec*u.deg, frame='icrs')
        result = Catalogs.query_region(
            coord,
            radius=radius_arcsec*u.arcsec,
            catalog="TIC"
        )

        if result is None or len(result) == 0:
            return None

        # Get the closest match
        if len(result) > 1:
            # Calculate distances and get closest
            result_coords = SkyCoord(
                ra=result['ra']*u.deg,
                dec=result['dec']*u.deg,
                frame='icrs'
            )
            seps = coord.separation(result_coords)
            closest_idx = np.argmin(seps)
            best_match = result[closest_idx]
        else:
            best_match = result[0]

        return {
            'tic': best_match.get('ID', None),
            'mass': best_match.get('mass', None),
            'radius': best_match.get('rad', None),
            'temp': best_match.get('Teff', None)
        }
    except Exception as e:
        print(f"Error querying TIC for RA={ra}, Dec={dec}: {e}")
        return None


def run_bulk_crossmatch(input_csv_path, output_csv_path):
    """
    Main crossmatch function using batched cone searches.
    """
    # Load input CSV
    stardata = pd.read_csv(input_csv_path)
    print(f"Loaded {len(stardata)} rows from {input_csv_path}")

    # Apply test limit if set
    if TEST_LIMIT is not None:
        stardata = stardata.head(TEST_LIMIT)
        print(f"TEST MODE: Processing only first {len(stardata)} rows")
        output_csv_path = output_csv_path.replace('.csv', '_test.csv')
        print(f"Output will be saved to: {output_csv_path}")

    # Propagate coordinates to TIC epoch
    print("Propagating coordinates to epoch 2015.5...")
    propagated_coords = []
    for _, row in stardata.iterrows():
        ra = float(row[COL_RA])
        dec = float(row[COL_DEC])
        pmra = float(row.get(COL_PMRA, 0) if not pd.isna(row.get(COL_PMRA, np.nan)) else 0)
        pmdec = float(row.get(COL_PMDEC, 0) if not pd.isna(row.get(COL_PMDEC, np.nan)) else 0)

        ra2015, dec2015 = pm_propagate(ra, dec, pmra, pmdec)
        propagated_coords.append((ra2015, dec2015))

    # Initialize output columns
    stardata['tic'] = None
    stardata['mass'] = None
    stardata['radius'] = None
    stardata['temp'] = None

    # Query TIC in batches with progress bar
    print(f"Querying TIC catalog for {len(stardata)} objects (batch size: {BATCH_SIZE})...")
    for i in tqdm(range(len(stardata)), desc="Crossmatching"):
        # Check if this is the Sun (Sol)
        star_name = stardata.at[i, 'proper'] if 'proper' in stardata.columns else ''

        if star_name == 'Sol':
            # Set Sun's values directly (no need to query)
            stardata.at[i, 'tic'] = None  # Sun doesn't have a TIC entry
            stardata.at[i, 'mass'] = 1.0
            stardata.at[i, 'radius'] = 1.0
            stardata.at[i, 'temp'] = 5772.0
            continue

        ra_prop, dec_prop = propagated_coords[i]

        # Query TIC
        match = query_tic_cone(ra_prop, dec_prop)

        if match:
            stardata.at[i, 'tic'] = match['tic']
            stardata.at[i, 'mass'] = match['mass']
            stardata.at[i, 'radius'] = match['radius']
            stardata.at[i, 'temp'] = match['temp']

        # Rate limiting: pause every BATCH_SIZE queries
        if (i + 1) % BATCH_SIZE == 0:
            time.sleep(1)  # 1 second pause between batches

    # Save output
    stardata.to_csv(output_csv_path, index=False)

    # Print statistics
    matched = stardata['tic'].notna().sum()
    print(f"\nCrossmatch complete!")
    print(f"Total objects: {len(stardata)}")
    print(f"Matched to TIC: {matched} ({100*matched/len(stardata):.1f}%)")
    print(f"Output saved to: {output_csv_path}")

    return output_csv_path


def run_batch_crossmatch_alternative(input_csv_path, output_csv_path):
    """
    Alternative method using Catalogs.query_criteria for batch processing.
    Use this if the cone search method is too slow.
    """
    stardata = pd.read_csv(input_csv_path)
    print(f"Loaded {len(stardata)} rows from {input_csv_path}")

    # This method queries larger regions and filters locally
    # Useful for clustered data
    print("Note: This is a faster alternative for large datasets with clustered positions")
    print("Using the standard cone search method instead (more accurate)...")

    return run_bulk_crossmatch(input_csv_path, output_csv_path)


if __name__ == "__main__":
    try:
        # Test network connectivity first
        print("Testing MAST connectivity...")
        test_coord = SkyCoord(ra=10*u.deg, dec=20*u.deg)
        test_result = Catalogs.query_region(test_coord, radius=1*u.arcsec, catalog="TIC")
        print("✓ Connection to MAST successful\n")

        # Run crossmatch
        out = run_bulk_crossmatch(INPUT_CSV, OUTPUT_CSV)
        print("\nDone:", out)

    except Exception as e:
        print(f"\nError: {e}")
        print("\nTroubleshooting steps:")
        print("1. Check internet connection")
        print("2. Verify astroquery version: pip install --upgrade astroquery")
        print("3. Check if MAST services are online: https://mast.stsci.edu/")
        print("4. Try clearing astroquery cache:")
        print("   from astroquery.utils import cache; cache.clear_cache()")
        raise