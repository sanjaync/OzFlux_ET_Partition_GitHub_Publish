#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run_TEA_ozflux.py
-----------------
Runs the Transpiration Estimation Algorithm (TEA) on all OzFlux L6 NetCDF
files listed in path_of_L6.txt, handling the variable name/unit mapping
between OzFlux conventions and TEA expectations.

Variable mapping (OzFlux L6  →  TEA):
  ET       : L6 ET  [kg/m²/s]   → mm/hh  (× 1800)
  GPP      : L6 GPP_SOLO [umol/m²/s]  (best gap-fill, no conversion)
  Tair     : L6 Ta  [degC]             (direct)
  RH       : L6 RH  [percent]          (direct)
  VPD      : L6 VPD [kPa]       → hPa  (× 10)
  precip   : L6 Precip [mm/hh]         (direct, already per half-hour sum)
  Rg       : L6 Fsd [W/m²]            (down-welling shortwave, direct)
  Rg_pot   : computed from PotRad.py   (no Rg_pot in L6 files)
  u        : L6 Ws  [m/s]             (wind speed, direct)
  qualityFlag: (Fe_QCFlag == 0) & (Fco2_QCFlag == 0)

nStepsPerDay is read from the global attribute 'time_step'
  (30 min → 48 steps/day, 60 min → 24 steps/day).

Output: one NetCDF per site saved to TEA_output/<sitename>_TEA.nc
        with TEA_T, TEA_E, TEA_WUE variables added.

Usage:
    conda activate ecosystem-transpiration
    python run_TEA_ozflux.py [--n_jobs 4] [--site SiteName]

Author: Antigravity (auto-generated for Sanjay)
"""

import sys
import os
import argparse
import warnings
import traceback
from pathlib import Path

import numpy as np
import xarray as xr

# ── TEA imports (from TEA_original installed as editable package) ──────────
from TEA.PotRad import CalcPotRadiation
from TEA.PreProc import build_dataset, preprocess
from TEA.TEA import partition

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = Path(__file__).resolve().parent
L6_DIR       = Path("/home/sanjays/et97_scratch2/oldscratch/Ozflux_data_full/L6")
OUTPUT_DIR   = SCRIPT_DIR / "TEA_output"
LOG_DIR      = SCRIPT_DIR / "TEA_logs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# TEA RF settings (following tutorial defaults)
RF_KWARGS = {
    "n_estimators": 100,
    "oob_score":    True,
    "max_features": "n/3",
    "verbose":      0,
    "warm_start":   False,
    "n_jobs":       1,          # overridden by --n_jobs
}

RF_MOD_VARS = [
    "Rg", "Tair", "RH", "u",
    "Rg_pot_daily", "Rgpotgrad",
    "year", "GPPgrad", "DWCI", "C_Rg_ET", "CSWI",
]

PERCS    = np.array([75])          # 75th percentile (standard for TEA)
CSWILIMS = np.array([-0.5])       # CSWI limit (tutorial default)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_all_l6_files():
    """Return sorted list of all *_L6.nc files in L6_DIR."""
    files = sorted(L6_DIR.glob("*_L6.nc"))
    if not files:
        raise FileNotFoundError(f"No *_L6.nc files found in {L6_DIR}")
    return files


def site_name_from_path(nc_path: Path) -> str:
    """Extract site name from filename: AdelaideRiver_L6.nc → AdelaideRiver"""
    return nc_path.name.replace("_L6.nc", "")


def compute_rg_pot(timestamp_dt64, lat, lon, timezone_h):
    """
    Compute potential radiation for each half-hour timestamp using PotRad.py.

    Parameters
    ----------
    timestamp_dt64 : numpy array of datetime64[ns]
    lat, lon       : float  (degrees)
    timezone_h     : float  (hours east of UTC, e.g. +9.5 for ACST)

    Returns
    -------
    rg_pot : numpy array (W/m²)
    """
    import pandas as pd
    ts = pd.to_datetime(timestamp_dt64)
    doy  = ts.day_of_year.values.astype(float)
    hour = ts.hour.values + ts.minute.values / 60.0 + ts.second.values / 3600.0
    rg_pot = CalcPotRadiation(doy, hour, lat, lon, timezone_h, useSolartime=True)
    return rg_pot.astype(np.float64)


def timezone_from_tz_string(tz_str):
    """
    Convert OzFlux time_zone attribute (e.g. 'Australia/Darwin') to a UTC
    offset float in hours.  Uses a simple lookup for common Australian zones;
    falls back to pytz if available.
    """
    _LOOKUP = {
        "Australia/Darwin":   9.5,
        "Australia/Brisbane": 10.0,
        "Australia/Sydney":   10.0,
        "Australia/Melbourne":10.0,
        "Australia/Adelaide":  9.5,
        "Australia/Perth":     8.0,
        "Australia/Hobart":   10.0,
        "Australia/Lord_Howe":10.5,
        "UTC":                 0.0,
    }
    if tz_str in _LOOKUP:
        return _LOOKUP[tz_str]
    try:
        import pytz, datetime
        tz  = pytz.timezone(tz_str)
        now = datetime.datetime(2008, 6, 15)          # mid-year, avoid DST issues
        off = tz.utcoffset(now).total_seconds() / 3600
        return off
    except Exception:
        warnings.warn(f"Unknown timezone '{tz_str}', assuming UTC+10")
        return 10.0


def check_variable(ds, *candidates):
    """
    Return the first variable name from *candidates that exists in ds.
    Raises KeyError if none found.
    """
    for name in candidates:
        if name in ds.data_vars or name in ds.coords:
            return name
    raise KeyError(f"None of {candidates} found in dataset. "
                   f"Available: {list(ds.data_vars)}")


def squeeze2d(arr):
    """Squeeze (time, lat, lon) → (time,) 1-D array."""
    return np.asarray(arr).reshape(-1)


# ─────────────────────────────────────────────────────────────────────────────
# Main processing function for a single site
# ─────────────────────────────────────────────────────────────────────────────

def process_site(nc_path: Path, n_jobs: int = 1, overwrite: bool = False):
    """
    Load one OzFlux L6 NetCDF, map variables to TEA convention,
    run TEA partitioning, and save output NetCDF.

    Returns
    -------
    str : 'ok', 'skip', or 'fail:<reason>'
    """
    site = site_name_from_path(nc_path)
    out_path = OUTPUT_DIR / f"{site}_TEA.nc"
    log_path = LOG_DIR / f"{site}.log"

    if out_path.exists() and not overwrite:
        print(f"  [SKIP] {site} — output already exists")
        return "skip"

    print(f"  [START] {site}")

    try:
        # ── 1. Load L6 file ──────────────────────────────────────────────
        ds_raw = xr.open_dataset(nc_path, decode_times=True)

        # ── 2. Read site metadata ─────────────────────────────────────────
        lat       = float(ds_raw.attrs.get("latitude",  ds_raw.latitude.values.ravel()[0]))
        lon       = float(ds_raw.attrs.get("longitude", ds_raw.longitude.values.ravel()[0]))
        time_step = int(ds_raw.attrs.get("time_step", 30))    # minutes
        tz_str    = ds_raw.attrs.get("time_zone", "UTC")
        timezone_h = timezone_from_tz_string(tz_str)
        nStepsPerDay = 1440 // time_step     # 48 for 30-min, 24 for 60-min

        # ── 3. Extract timestamp ──────────────────────────────────────────
        timestamp = ds_raw.time.values.astype("datetime64[ns]")
        n = len(timestamp)

        # Truncate to whole days (TEA requires n % nStepsPerDay == 0)
        remainder = n % nStepsPerDay
        if remainder != 0:
            warnings.warn(f"{site}: trimming {remainder} trailing timesteps "
                          f"to make length divisible by {nStepsPerDay}")
            timestamp = timestamp[:-remainder]
            ds_raw = ds_raw.isel(time=slice(None, n - remainder))
            n = len(timestamp)

        def get1d(varname, fallback=None):
            """Squeeze to 1-D and return as float64 array with robust cleaning."""
            if varname not in ds_raw.data_vars:
                if fallback is not None:
                    return fallback
                raise KeyError(f"{varname} not in dataset")
            arr = squeeze2d(ds_raw[varname].values[:n])
            arr = arr.astype(np.float64)
            
            # ── Robust Cleaning ──
            # 1. Handle common OzFlux fill values (-9999, -6999, etc.)
            arr[np.isclose(arr, -9999.0, atol=0.1)] = np.nan
            arr[np.isclose(arr, 9999.0,  atol=0.1)] = np.nan
            
            # 2. Handle extreme NetCDF fill values
            arr[arr > 1e30]  = np.nan
            arr[arr < -1e30] = np.nan
            
            # 3. Variable-specific physical range filtering
            if varname == "ET":
                # ET in kg/m2/s. 1e-3 kg/m2/s is ~3.6 mm/h (extreme)
                arr[arr > 0.002]  = np.nan
                arr[arr < -0.0005] = np.nan
            elif varname in ("GPP_SOLO", "GPP_LL", "GPP_LT"):
                # GPP in umol/m2/s.
                arr[arr > 100]  = np.nan
                arr[arr < -10]  = np.nan
            elif varname == "VPD":
                # VPD in kPa.
                arr[arr < 0]    = np.nan
                arr[arr > 15]   = np.nan
            elif varname == "Ta":
                # Air temp in degC.
                arr[arr < -20]  = np.nan
                arr[arr > 60]   = np.nan
                
            return arr

        # ── 4. Map & convert variables ────────────────────────────────────

        # ET: kg/m²/s → mm/hh  (×1800 s/hh)
        ET_raw = get1d("ET")
        ET = ET_raw * 1800.0

        # GPP: prefer SOLO (best gap fill), fallback to LL then LT
        for gpp_var in ("GPP_SOLO", "GPP_LL", "GPP_LT"):
            if gpp_var in ds_raw.data_vars:
                GPP = get1d(gpp_var)
                break
        else:
            raise KeyError("No GPP variable (GPP_SOLO/LL/LT) found")

        # Tair: degC
        Tair = get1d("Ta")

        # RH: percent (already 0-100, TEA expects %)
        RH = get1d("RH")

        # VPD: kPa → hPa (×10)
        VPD = get1d("VPD") * 10.0

        # Precip: mm per timestep (already correct)
        Precip = get1d("Precip")
        Precip[Precip < 0] = 0.0    # no negative precip

        # Rg: down-welling shortwave W/m²
        Rg = get1d("Fsd")

        # Rg_pot: compute from solar geometry
        print(f"    → Computing Rg_pot (lat={lat:.3f}, lon={lon:.3f}, tz={timezone_h}h)")
        Rg_pot = compute_rg_pot(timestamp, lat, lon, timezone_h)

        # u: wind speed m/s
        u = get1d("Ws")

        # Quality flag: good = Fe_QCFlag == 0  AND  Fco2_QCFlag == 0
        # (OzFlux QCFlag=0 → good measured data; >0 → gap-filled or poor)
        fe_qc    = squeeze2d(ds_raw["Fe_QCFlag"].values[:n]).astype(int)
        fco2_qc  = squeeze2d(ds_raw["Fco2_QCFlag"].values[:n]).astype(int)
        qualityFlag = ((fe_qc == 0) & (fco2_qc == 0)).astype(bool)

        # ── 5. Build TEA dataset ──────────────────────────────────────────
        OtherVars = {
            "Fe":   squeeze2d(ds_raw["Fe"].values[:n]).astype(np.float64),   # latent heat W/m²
            "Fco2": squeeze2d(ds_raw["Fco2"].values[:n]).astype(np.float64), # CO2 flux
        }

        ds = build_dataset(
            timestamp, ET, GPP, RH, Rg, Rg_pot, Tair, VPD, Precip, u,
            qualityFlag=qualityFlag,
            OtherVars=OtherVars,
        )

        # ── 6. Preprocess ─────────────────────────────────────────────────
        print(f"    → Preprocessing (nStepsPerDay={nStepsPerDay})")
        ds = preprocess(ds, nStepsPerDay=nStepsPerDay)

        # ── 7. Partition ──────────────────────────────────────────────────
        rf_kwargs = dict(RF_KWARGS)
        rf_kwargs["n_jobs"] = n_jobs

        print(f"    → Running TEA RF partitioning...")
        ds = partition(
            ds,
            percs=PERCS,
            CSWIlims=CSWILIMS,
            RFmod_vars=RF_MOD_VARS,
            RandomForestRegressor_kwargs=rf_kwargs,
        )

        n_training = int(ds["NumForestPoints"].values[0])
        oob        = float(ds["oob_scores"].values[0]) if n_training > 240 else np.nan
        print(f"    → Training pts: {n_training}, OOB R²: {oob:.3f}" if not np.isnan(oob)
              else f"    → Training pts: {n_training} (< 240, TEA not run)")

        # ── 8. Add site metadata back ──────────────────────────────────────
        ds.attrs["site"]           = site
        ds.attrs["latitude"]       = lat
        ds.attrs["longitude"]      = lon
        ds.attrs["time_step_min"]  = time_step
        ds.attrs["nStepsPerDay"]   = nStepsPerDay
        ds.attrs["GPP_source"]     = gpp_var
        ds.attrs["TEA_version"]    = "TEA_original (jnelson18/TranspirationEstimationAlgorithm)"
        ds.attrs["processing_note"] = (
            f"ET from OzFlux L6 ET[kg/m2/s]*1800; GPP={gpp_var}; "
            f"VPD kPa*10=hPa; Rg=Fsd; Rg_pot computed; qualityFlag=(Fe_QCFlag==0)&(Fco2_QCFlag==0)"
        )

        # ── 9. Save output ────────────────────────────────────────────────
        # Drop coords that can't be serialised as plain netcdf
        ds_out = ds.drop_vars(
            [v for v in ["RFmod_vars"] if v in ds.coords], errors="ignore"
        )
        # Convert object coords to string if needed
        encoding = {}
        ds_out.to_netcdf(out_path, encoding=encoding)
        print(f"    ✓  Saved → {out_path.name}")

        with open(log_path, "w") as flog:
            flog.write(f"site={site}\n")
            flog.write(f"n_timesteps={n}\n")
            flog.write(f"nStepsPerDay={nStepsPerDay}\n")
            flog.write(f"GPP_source={gpp_var}\n")
            flog.write(f"n_training_pts={n_training}\n")
            flog.write(f"oob_score={oob}\n")
            flog.write(f"output={out_path}\n")

        return "ok"

    except Exception as exc:
        msg = traceback.format_exc()
        print(f"    ✗  FAILED: {exc}")
        with open(log_path, "w") as flog:
            flog.write(f"FAILED: {site}\n{msg}\n")
        return f"fail:{exc}"


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Run TEA partitioning on all OzFlux L6 NetCDF files"
    )
    parser.add_argument("--n_jobs",   type=int,  default=1,
                        help="Parallel jobs inside each RF fit (default: 1)")
    parser.add_argument("--site",     type=str,  default=None,
                        help="Run only this site name (e.g. Tumbarumba)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-run sites that already have output")
    args = parser.parse_args()

    all_files = get_all_l6_files()

    if args.site:
        all_files = [f for f in all_files if args.site in f.name]
        if not all_files:
            sys.exit(f"ERROR: no file matching site '{args.site}' found in {L6_DIR}")

    print(f"\n{'='*60}")
    print(f" TEA OzFlux Runner")
    print(f"  Sites to process : {len(all_files)}")
    print(f"  Output dir       : {OUTPUT_DIR}")
    print(f"  n_jobs (RF)      : {args.n_jobs}")
    print(f"{'='*60}\n")

    results = {"ok": [], "skip": [], "fail": []}

    for i, nc_path in enumerate(all_files, 1):
        site = site_name_from_path(nc_path)
        print(f"[{i:>2}/{len(all_files)}] {site}")
        status = process_site(nc_path, n_jobs=args.n_jobs, overwrite=args.overwrite)
        key = "ok" if status == "ok" else ("skip" if status == "skip" else "fail")
        results[key].append(site)

    print(f"\n{'='*60}")
    print(f" Summary")
    print(f"  OK    : {len(results['ok'])}")
    print(f"  Skipped: {len(results['skip'])}")
    print(f"  Failed : {len(results['fail'])}")
    if results["fail"]:
        print(f"  Failed sites: {results['fail']}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
