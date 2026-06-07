#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run_TEA_original.py
-------------------
Re-runs TEA partitioning using the UNMODIFIED original code from
TEA_original/ (jnelson18/TranspirationEstimationAlgorithm).

This script is identical to run_TEA_ozflux.py EXCEPT:
  - It imports from TEA_original.TEA instead of TEA (the modified version)
  - Output goes to TEA_output_original/ (preserving old results)
  - It uses the original CSWIlims default of -0.5

Author: Antigravity (for sanjays)
"""

import sys
import os
import argparse
import warnings
import traceback
from pathlib import Path

import numpy as np
import xarray as xr
import pandas as pd

# ── Import from the ORIGINAL, UNMODIFIED TEA package ─────────────────────
# We need to make sure we import from TEA_original, not TEA
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR / "TEA_original"))

from TEA.PotRad import CalcPotRadiation
from TEA.PreProc import build_dataset, preprocess
from TEA.TEA import partition

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
L6_DIR       = Path("/home/sanjays/et97_scratch2/oldscratch/Ozflux_data_full/L6")
OUTPUT_DIR   = SCRIPT_DIR / "TEA_output_original"
CSV_DIR      = OUTPUT_DIR / "csv"
LOG_DIR      = SCRIPT_DIR / "TEA_logs_original"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CSV_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# TEA RF settings (following tutorial defaults exactly)
RF_KWARGS = {
    "n_estimators": 100,
    "oob_score":    True,
    "max_features": "n/3",
    "verbose":      0,
    "warm_start":   False,
    "n_jobs":       1,
}

RF_MOD_VARS = [
    "Rg", "Tair", "RH", "u",
    "Rg_pot_daily", "Rgpotgrad",
    "year", "GPPgrad", "DWCI", "C_Rg_ET", "CSWI",
]

PERCS    = np.array([75])          # 75th percentile (standard for TEA)
CSWILIMS = np.array([-0.5])       # ORIGINAL default from simplePartition


# ─────────────────────────────────────────────────────────────────────────────
# Helpers (same as run_TEA_ozflux.py)
# ─────────────────────────────────────────────────────────────────────────────

def get_all_l6_files():
    files = sorted(L6_DIR.glob("*_L6.nc"))
    if not files:
        raise FileNotFoundError(f"No *_L6.nc files found in {L6_DIR}")
    return files


def site_name_from_path(nc_path):
    return nc_path.name.replace("_L6.nc", "")


def compute_rg_pot(timestamp_dt64, lat, lon, timezone_h):
    ts = pd.to_datetime(timestamp_dt64)
    doy  = ts.day_of_year.values.astype(float)
    hour = ts.hour.values + ts.minute.values / 60.0 + ts.second.values / 3600.0
    rg_pot = CalcPotRadiation(doy, hour, lat, lon, timezone_h, useSolartime=True)
    return rg_pot.astype(np.float64)


def timezone_from_tz_string(tz_str):
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
        now = datetime.datetime(2008, 6, 15)
        off = tz.utcoffset(now).total_seconds() / 3600
        return off
    except Exception:
        warnings.warn(f"Unknown timezone '{tz_str}', assuming UTC+10")
        return 10.0


def squeeze2d(arr):
    return np.asarray(arr).reshape(-1)


# ─────────────────────────────────────────────────────────────────────────────
# Process one site
# ─────────────────────────────────────────────────────────────────────────────

def process_site(nc_path, n_jobs=1, overwrite=False):
    site = site_name_from_path(nc_path)
    out_path = OUTPUT_DIR / f"{site}_TEA_original.nc"
    log_path = LOG_DIR / f"{site}.log"

    if out_path.exists() and not overwrite:
        print(f"  [SKIP] {site} — output already exists")
        return "skip"

    print(f"  [START] {site}")

    try:
        ds_raw = xr.open_dataset(nc_path, decode_times=True)

        lat       = float(ds_raw.attrs.get("latitude",  ds_raw.latitude.values.ravel()[0]))
        lon       = float(ds_raw.attrs.get("longitude", ds_raw.longitude.values.ravel()[0]))
        time_step = int(ds_raw.attrs.get("time_step", 30))
        tz_str    = ds_raw.attrs.get("time_zone", "UTC")
        timezone_h = timezone_from_tz_string(tz_str)
        nStepsPerDay = 1440 // time_step

        timestamp = ds_raw.time.values.astype("datetime64[ns]")
        n = len(timestamp)

        # Truncate to whole days
        remainder = n % nStepsPerDay
        if remainder != 0:
            timestamp = timestamp[:-remainder]
            ds_raw = ds_raw.isel(time=slice(None, n - remainder))
            n = len(timestamp)

        def get1d(varname, fallback=None):
            if varname not in ds_raw.data_vars:
                if fallback is not None:
                    return fallback
                raise KeyError(f"{varname} not in dataset")
            arr = squeeze2d(ds_raw[varname].values[:n]).astype(np.float64)
            arr[np.isclose(arr, -9999.0, atol=0.1)] = np.nan
            arr[np.isclose(arr, 9999.0,  atol=0.1)] = np.nan
            arr[arr > 1e30]  = np.nan
            arr[arr < -1e30] = np.nan
            return arr

        # ── Map & convert variables ────────────────────────────────────
        ET = get1d("ET") * 1800.0          # kg/m²/s → mm/hh

        for gpp_var in ("GPP_SOLO", "GPP_LL", "GPP_LT"):
            if gpp_var in ds_raw.data_vars:
                GPP = get1d(gpp_var)
                break
        else:
            raise KeyError("No GPP variable found")

        Tair   = get1d("Ta")
        RH     = get1d("RH")
        VPD    = get1d("VPD") * 10.0       # kPa → hPa
        Precip = get1d("Precip")
        Precip[Precip < 0] = 0.0
        Rg     = get1d("Fsd")
        Rg_pot = compute_rg_pot(timestamp, lat, lon, timezone_h)
        u      = get1d("Ws")

        # Quality flag
        fe_qc    = squeeze2d(ds_raw["Fe_QCFlag"].values[:n]).astype(int)
        fco2_qc  = squeeze2d(ds_raw["Fco2_QCFlag"].values[:n]).astype(int)
        qualityFlag = ((fe_qc == 0) & (fco2_qc == 0)).astype(bool)

        # ── Build TEA dataset ──────────────────────────────────────────
        ds = build_dataset(
            timestamp, ET, GPP, RH, Rg, Rg_pot, Tair, VPD, Precip, u,
            qualityFlag=qualityFlag,
        )

        # ── Preprocess (ORIGINAL code: hardcoded 1800s, no adaptive fallback) ──
        print(f"    → Preprocessing (nStepsPerDay={nStepsPerDay})")
        ds = preprocess(ds, nStepsPerDay=nStepsPerDay)

        # ── Partition (ORIGINAL code: strict CSWI, no fallback) ──
        rf_kwargs = dict(RF_KWARGS)
        rf_kwargs["n_jobs"] = n_jobs

        print(f"    → Running TEA RF partitioning (ORIGINAL code, CSWIlim={CSWILIMS[0]})...")
        ds = partition(
            ds,
            percs=PERCS,
            CSWIlims=CSWILIMS,
            RFmod_vars=RF_MOD_VARS,
            RandomForestRegressor_kwargs=rf_kwargs,
        )

        n_training = int(ds["NumForestPoints"].values[0])
        oob        = float(ds["oob_scores"].values[0]) if n_training > 240 else np.nan

        if n_training <= 240:
            print(f"    ⚠ Only {n_training} training points (< 240). TEA not applicable.")
        else:
            print(f"    → Training pts: {n_training}, OOB R²: {oob:.3f}")

        # ── Add metadata ──────────────────────────────────────────────
        ds.attrs["site"]            = site
        ds.attrs["latitude"]        = lat
        ds.attrs["longitude"]       = lon
        ds.attrs["time_step_min"]   = time_step
        ds.attrs["nStepsPerDay"]    = nStepsPerDay
        ds.attrs["GPP_source"]      = gpp_var
        ds.attrs["CSWIlim_used"]    = float(CSWILIMS[0])
        ds.attrs["TEA_version"]     = "ORIGINAL (jnelson18/TranspirationEstimationAlgorithm, unmodified)"

        # ── Save NetCDF ───────────────────────────────────────────────
        ds_out = ds.drop_vars(
            [v for v in ["RFmod_vars"] if v in ds.coords], errors="ignore"
        )
        ds_out.to_netcdf(out_path)
        print(f"    ✓  Saved → {out_path.name}")

        # ── Generate daily CSV ─────────────────────────────────────────
        tea_T = ds.TEA_T.sel(percentiles=75).values.ravel()
        tea_E = ds.TEA_E.sel(percentiles=75).values.ravel()
        et_vals = ds.ET.values.ravel()

        df = pd.DataFrame({
            'timestamp': ds.timestamp.values,
            'ET_mm': et_vals,
            'TEA_T_75_mm': tea_T,
            'TEA_E_75_mm': tea_E,
        })
        df.loc[df['ET_mm'] < -9000, 'ET_mm'] = np.nan
        df.loc[df['TEA_T_75_mm'] < -9000, 'TEA_T_75_mm'] = np.nan
        df.loc[df['TEA_E_75_mm'] < -9000, 'TEA_E_75_mm'] = np.nan

        try:
            df['date'] = pd.to_datetime(df['timestamp']).dt.date
        except Exception:
            df['date'] = [pd.Timestamp(str(d)).date() for d in df['timestamp']]

        daily = df.groupby('date').agg({
            'ET_mm': 'sum',
            'TEA_T_75_mm': 'sum',
            'TEA_E_75_mm': 'sum',
        }).reset_index()

        daily['T_ET_ratio'] = np.where(
            daily['ET_mm'] > 0,
            daily['TEA_T_75_mm'] / daily['ET_mm'],
            np.nan
        )
        daily.columns = ['date', 'ET_daily_mm', 'T_TEA_daily_mm', 'E_TEA_daily_mm', 'T_ET_ratio']
        daily.insert(0, 'site', site)

        csv_path = CSV_DIR / f"{site}_TEA_original_daily.csv"
        daily.to_csv(csv_path, index=False)
        print(f"    ✓  Saved CSV → {csv_path.name}")

        # ── Log ───────────────────────────────────────────────────────
        valid_ratio = daily['T_ET_ratio'].dropna()
        valid_ratio = valid_ratio[(valid_ratio >= 0) & (valid_ratio <= 1)]

        with open(log_path, "w") as flog:
            flog.write(f"site={site}\n")
            flog.write(f"n_timesteps={n}\n")
            flog.write(f"nStepsPerDay={nStepsPerDay}\n")
            flog.write(f"GPP_source={gpp_var}\n")
            flog.write(f"n_training_pts={n_training}\n")
            flog.write(f"oob_score={oob}\n")
            flog.write(f"CSWIlim={CSWILIMS[0]}\n")
            flog.write(f"T_ET_mean={valid_ratio.mean():.4f}\n")
            flog.write(f"T_ET_median={valid_ratio.median():.4f}\n")
            flog.write(f"n_valid_days={len(valid_ratio)}\n")
            flog.write(f"status={'OK' if n_training > 240 else 'INSUFFICIENT_DATA'}\n")

        return "ok"

    except Exception as exc:
        msg = traceback.format_exc()
        print(f"    ✗  FAILED: {exc}")
        with open(log_path, "w") as flog:
            flog.write(f"FAILED: {site}\n{msg}\n")
        return f"fail:{exc}"


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Run TEA partitioning using ORIGINAL unmodified code"
    )
    parser.add_argument("--n_jobs",   type=int,  default=4)
    parser.add_argument("--site",     type=str,  default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    all_files = get_all_l6_files()

    if args.site:
        all_files = [f for f in all_files if args.site in f.name]

    print(f"\n{'='*60}")
    print(f" TEA OzFlux Runner — ORIGINAL CODE (unmodified)")
    print(f"  Sites to process : {len(all_files)}")
    print(f"  Output dir       : {OUTPUT_DIR}")
    print(f"  CSWIlim          : {CSWILIMS[0]}")
    print(f"  n_jobs (RF)      : {args.n_jobs}")
    print(f"{'='*60}\n")

    results = {"ok": [], "skip": [], "fail": []}

    for i, nc_path in enumerate(all_files, 1):
        site = site_name_from_path(nc_path)
        print(f"[{i:>2}/{len(all_files)}] {site}")
        status = process_site(nc_path, n_jobs=args.n_jobs, overwrite=args.overwrite)
        key = "ok" if status == "ok" else ("skip" if status == "skip" else "fail")
        results[key].append(site)

    # ── Summary CSV ───────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f" Summary")
    print(f"  OK     : {len(results['ok'])}")
    print(f"  Skipped: {len(results['skip'])}")
    print(f"  Failed : {len(results['fail'])}")
    if results["fail"]:
        print(f"  Failed sites: {results['fail']}")
    print(f"{'='*60}\n")

    # Build summary from log files
    summary_rows = []
    for log_file in sorted(LOG_DIR.glob("*.log")):
        info = {}
        with open(log_file) as f:
            for line in f:
                if "=" in line and not line.startswith("FAILED"):
                    k, v = line.strip().split("=", 1)
                    info[k] = v
        if 'site' in info:
            summary_rows.append(info)

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_path = CSV_DIR / "TEA_original_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"Summary saved: {summary_path}")


if __name__ == "__main__":
    main()
