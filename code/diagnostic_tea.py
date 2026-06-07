import numpy as np
import xarray as xr
import pandas as pd
import warnings
import sys

# Load TEA and Preprocess
from TEA import TEA
from TEA import PreProc

import ozflux_preprocess

def run_diagnostics(site_name):
    print(f"==================================================")
    print(f"TEA DIAGNOSTICS FOR: {site_name}")
    print(f"==================================================")
    
    nc_file = f'/home/sanjays/et97_scratch2/oldscratch/Ozflux_data_full/L6/{site_name}_L6.nc'
    ds = ozflux_preprocess.build_ozflux_dataset(nc_file)
    
    nStepsPerDay = 48
    
    # Extract variables
    timestamp = ds.time.values
    ET = ds.ET.values
    GPP = ds.GPP_NT.values
    Tair = ds.TA.values
    VPD = ds.VPD.values  # Already in hPa after preprocessing
    precip = ds.P.values
    Rg = ds.SW_IN.values
    Rg_pot = ds.SW_IN_POT.values
    u = ds.WS.values
    nee_qc = ds.NEE_QC.values
    le_qc = ds.LE_QC.values
    
    # Clean missing values immediately
    ET = np.where(ET < -9000, np.nan, ET)
    GPP = np.where(GPP < -9000, np.nan, GPP)
    Tair = np.where(Tair < -9000, np.nan, Tair)
    VPD = np.where(VPD < -9000, np.nan, VPD)
    precip = np.where(precip < -9000, np.nan, precip)
    Rg = np.where(Rg < -9000, np.nan, Rg)
    Rg_pot = np.where(Rg_pot < -9000, np.nan, Rg_pot)
    u = np.where(u < -9000, np.nan, u)
    
    # Calculate RH if not present
    if 'RH' in ds:
        RH = ds.RH.values
        print("Using RH from dataset")
    else:
        import bigleaf
        # VPD is in hPa, bigleaf needs kPa
        RH = bigleaf.VPD_to_RH(VPD / 10.0, Tair)
        print("Calculated RH from VPD and Tair")
    
    # ===================================================================
    # STEP 0: CHECK RAW INPUT DATA
    # ===================================================================
    print("\n--- 0. RAW INPUT DATA STATISTICS ---")
    print(f"Total timesteps: {len(timestamp)}")
    print(f"Date range: {pd.Timestamp(timestamp[0]).date()} to {pd.Timestamp(timestamp[-1]).date()}")
    print(f"\nMean values:")
    print(f"  ET:     {np.nanmean(ET):.4f} mm/timestep (range: {np.nanmin(ET):.4f} to {np.nanmax(ET):.4f})")
    print(f"  GPP:    {np.nanmean(GPP):.2f} umol/m2/s (range: {np.nanmin(GPP):.2f} to {np.nanmax(GPP):.2f})")
    print(f"  VPD:    {np.nanmean(VPD):.2f} hPa (range: {np.nanmin(VPD):.2f} to {np.nanmax(VPD):.2f})")
    print(f"  Tair:   {np.nanmean(Tair):.1f} °C (range: {np.nanmin(Tair):.1f} to {np.nanmax(Tair):.1f})")
    print(f"  Precip: {np.nanmean(precip):.4f} mm/timestep (total: {np.nansum(precip):.1f} mm)")
    print(f"  Rg:     {np.nanmean(Rg):.1f} W/m2")
    
    print(f"\nNaN percentages:")
    print(f"  ET: {100*np.isnan(ET).sum()/len(ET):.1f}%")
    print(f"  GPP: {100*np.isnan(GPP).sum()/len(GPP):.1f}%")
    print(f"  Precip: {100*np.isnan(precip).sum()/len(precip):.1f}%")
    
    # Sanity checks
    if np.nanmean(ET) > 1.0:
        print("⚠️  WARNING: ET values seem very high for 30-min timestep")
    if np.nanmean(VPD) < 1.0:
        print("⚠️  WARNING: VPD values seem low - check units (should be hPa)")
    if np.nansum(precip) < 100:
        print("⚠️  WARNING: Very low total precipitation")
    
    # Trim to complete days
    n_total = len(timestamp)
    n_complete = (n_total // nStepsPerDay) * nStepsPerDay
    if n_total != n_complete:
        print(f"\nTrimming from {n_total} to {n_complete} timesteps for complete days")
    
    timestamp = timestamp[:n_complete]
    ET = ET[:n_complete]
    GPP = GPP[:n_complete]
    Tair = Tair[:n_complete]
    RH = RH[:n_complete]
    VPD = VPD[:n_complete]
    precip = precip[:n_complete]
    Rg = Rg[:n_complete]
    Rg_pot = Rg_pot[:n_complete]
    u = u[:n_complete]
    nee_qc = nee_qc[:n_complete]
    le_qc = le_qc[:n_complete]
    
    # Build dataset with quality flags
    OtherVars = {'qualityFlag': (nee_qc <= 1) & (le_qc <= 1)}
    tea_ds = PreProc.build_dataset(timestamp, ET, GPP, RH, Rg, Rg_pot, 
                                   Tair, VPD, precip, u, OtherVars=OtherVars)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tea_ds = PreProc.preprocess(tea_ds, nStepsPerDay=nStepsPerDay)
    
    # ===================================================================
    # STEP 1: CHECK CSWI STATISTICS
    # ===================================================================
    print("\n--- 1. CSWI STATISTICS ---")
    cswi = tea_ds.CSWI.values
    valid_cswi = cswi[np.isfinite(cswi)]
    
    if len(valid_cswi) == 0:
        print("❌ CRITICAL: CSWI is all NaN! Check precipitation data.")
        return
    
    print(f"CSWI min:    {np.nanmin(valid_cswi):.2f} mm")
    print(f"CSWI max:    {np.nanmax(valid_cswi):.2f} mm")
    print(f"CSWI mean:   {np.nanmean(valid_cswi):.2f} mm")
    print(f"CSWI median: {np.nanmedian(valid_cswi):.2f} mm")
    
    # Distribution of CSWI values
    print(f"\nCSWI distribution (percentage of dry points):")
    print(f"  CSWI < -3.0: {100*(cswi < -3.0).sum()/len(cswi):.1f}%")
    print(f"  CSWI < -2.0: {100*(cswi < -2.0).sum()/len(cswi):.1f}%")
    print(f"  CSWI < -1.0: {100*(cswi < -1.0).sum()/len(cswi):.1f}%")
    print(f"  CSWI < -0.5: {100*(cswi < -0.5).sum()/len(cswi):.1f}%")
    print(f"  CSWI >  0.0: {100*(cswi > 0.0).sum()/len(cswi):.1f}%")
    
    # ===================================================================
    # STEP 2: CHECK TRAINING DATA RETENTION
    # ===================================================================
    print("\n--- 2. TRAINING DATA RETENTION ---")
    
    # Replicate TEA's exact filtering logic
    is_daytime = Rg > 0  # FIX: Use Rg, not non-existent DayNightFlag
    is_warm = Tair > 5
    is_valid_gpp_half_hourly = GPP > 0.05
    is_valid_et_half_hourly = ET > 0.02
    is_quality = (nee_qc <= 1) & (le_qc <= 1)
    is_finite = np.isfinite(ET) & np.isfinite(GPP)
    
    # Calculate WUE for filtering
    temp_wue = np.zeros_like(ET)
    temp_wue[ET > 0.02] = GPP[ET > 0.02] / ET[ET > 0.02] * 21.6
    is_valid_wue = temp_wue < 15000
    
    # Daily GPP filter (FIX: Add this - it's in the actual TEA code)
    gpp_daily = GPP.reshape(-1, nStepsPerDay).sum(axis=1) * 12 * 1e-6 * 1800  # Convert to gC/m2/day
    gpp_daily_flag = gpp_daily > 0.5
    gpp_daily_flag_expanded = np.repeat(gpp_daily_flag, nStepsPerDay)
    
    # Base filter (everything except CSWI)
    base_filter = (is_daytime & is_warm & is_valid_gpp_half_hourly & is_valid_et_half_hourly & 
                   is_valid_wue & gpp_daily_flag_expanded & is_quality & is_finite)
    
    # Training filters with different CSWI thresholds
    train_minus3 = base_filter & (cswi < -3.0)
    train_minus2 = base_filter & (cswi < -2.0)
    train_minus1 = base_filter & (cswi < -1.0)
    train_minus05 = base_filter & (cswi < -0.5)
    
    total_points = len(timestamp)
    print(f"Total points: {total_points}")
    print(f"After base filters: {base_filter.sum()} ({100*base_filter.sum()/total_points:.1f}%)")
    print(f"\nTraining data with different CSWI thresholds:")
    print(f"  CSWI < -3.0: {train_minus3.sum()} points ({100*train_minus3.sum()/total_points:.2f}%)")
    print(f"  CSWI < -2.0: {train_minus2.sum()} points ({100*train_minus2.sum()/total_points:.2f}%)")
    print(f"  CSWI < -1.0: {train_minus1.sum()} points ({100*train_minus1.sum()/total_points:.2f}%)")
    print(f"  CSWI < -0.5: {train_minus05.sum()} points ({100*train_minus05.sum()/total_points:.2f}%)")
    
    if train_minus1.sum() < 500:
        print("\n⚠️  WARNING: Very few training points at CSWI < -1.0!")
    if train_minus05.sum() < 500:
        print("⚠️  CRITICAL: Very few training points even at CSWI < -0.5!")
    
    # ===================================================================
    # STEP 3: CHECK WUE DISTRIBUTION
    # ===================================================================
    print("\n--- 3. WUE DISTRIBUTION ---")
    
    wue = tea_ds.inst_WUE.values
    
    wue_all = wue[base_filter & np.isfinite(wue)]
    wue_train_05 = wue[train_minus05 & np.isfinite(wue)]
    wue_train_1 = wue[train_minus1 & np.isfinite(wue)]
    
    if len(wue_train_05) > 0:
        print(f"WUE (All valid data, n={len(wue_all)}):")
        print(f"  Mean={wue_all.mean():.3f}, Median={np.median(wue_all):.3f}, 95th={np.percentile(wue_all, 95):.3f} gC/kgH2O")
        
        print(f"WUE (Training CSWI < -0.5, n={len(wue_train_05)}):")
        print(f"  Mean={wue_train_05.mean():.3f}, Median={np.median(wue_train_05):.3f}, 95th={np.percentile(wue_train_05, 95):.3f} gC/kgH2O")
        
        if len(wue_train_1) > 0:
            print(f"WUE (Training CSWI < -1.0, n={len(wue_train_1)}):")
            print(f"  Mean={wue_train_1.mean():.3f}, Median={np.median(wue_train_1):.3f}, 95th={np.percentile(wue_train_1, 95):.3f} gC/kgH2O")
        
        print(f"\nRatio (Training / All):")
        print(f"  Mean: {wue_train_05.mean() / wue_all.mean():.2f}x")
        print(f"  Median: {np.median(wue_train_05) / np.median(wue_all):.2f}x")
        
        if wue_train_05.mean() / wue_all.mean() < 0.8:
            print("⚠️  Training WUE is LOWER than overall WUE - unexpected!")
        elif wue_train_05.mean() / wue_all.mean() > 1.2:
            print("✓  Training WUE is higher than overall WUE (expected)")
    else:
        print("❌ CRITICAL: NO TRAINING DATA AVAILABLE!")
    
    # ===================================================================
    # STEP 4: CHECK PARTITIONING OUTPUT
    # ===================================================================
    print("\n--- 4. PARTITIONING OUTPUT ---")
    
    tea_ds = TEA.partition(
        tea_ds,
        percs=np.array([50, 75, 90]),
        CSWIlims=np.array([-1.0]),
        RandomForestRegressor_kwargs={'n_jobs': 1}
    )
    
    tea_t_50 = tea_ds.TEA_T.sel(percentiles=50, CSWIlims=-1.0).values
    tea_t_75 = tea_ds.TEA_T.sel(percentiles=75, CSWIlims=-1.0).values
    tea_t_90 = tea_ds.TEA_T.sel(percentiles=90, CSWIlims=-1.0).values
    tea_wue_75 = tea_ds.TEA_WUE.sel(percentiles=75, CSWIlims=-1.0).values
    et = tea_ds.ET.values
    
    # TEA uses -9999 as missing value flag
    valid_t_50 = (tea_t_50 > -9000) & np.isfinite(tea_t_50)
    valid_t_75 = (tea_t_75 > -9000) & np.isfinite(tea_t_75)
    valid_t_90 = (tea_t_90 > -9000) & np.isfinite(tea_t_90)
    valid_et = np.isfinite(et) & (et > -9000)
    
    print(f"Valid points:")
    print(f"  ET: {valid_et.sum()} ({100*valid_et.sum()/total_points:.1f}%)")
    print(f"  TEA_T (P75): {valid_t_75.sum()} ({100*valid_t_75.sum()/total_points:.1f}%)")
    
    valid_wue = (tea_wue_75 > -9000) & np.isfinite(tea_wue_75)
    print(f"\nPredicted TEA_WUE (P75):")
    print(f"  Mean: {np.nanmean(tea_wue_75[valid_wue]):.1f}")
    print(f"  Median: {np.nanmedian(tea_wue_75[valid_wue]):.1f}")
    print(f"  Max: {np.nanmax(tea_wue_75[valid_wue]):.1f}")
    
    # Calculate T/ET ratios
    et_sum = np.nansum(et[valid_et])
    
    # NEW GUARD: Only sum T where GPP is positive to avoid "negative transpiration" errors
    gpp_positive = (tea_ds.GPP.values > 0)
    
    t_sum_50 = np.nansum(tea_t_50[valid_t_50 & gpp_positive])
    t_sum_75 = np.nansum(tea_t_75[valid_t_75 & gpp_positive])
    t_sum_90 = np.nansum(tea_t_90[valid_t_90 & gpp_positive])
    
    gpp_sum_valid_t = np.nansum(tea_ds.GPP.values[valid_t_75 & (tea_t_75 > 0) & gpp_positive])
    print(f"\nManual Math Check:")
    print(f"  Sum of GPP where TEA_T > 0: {gpp_sum_valid_t:.1f} umol/m2/s")
    median_wue = np.nanmedian(wue_train_05) if len(wue_train_05) > 0 else 1800
    expected_t = gpp_sum_valid_t / (median_wue / 21.6)
    print(f"  Expected TEA_T (if WUE is constant {median_wue:.1f}): {expected_t:.1f} mm")
    
    print(f"\nSums:")
    print(f"  Total ET: {et_sum:.1f} mm")
    print(f"  Total T (P50): {t_sum_50:.1f} mm")
    print(f"  Total T (P75): {t_sum_75:.1f} mm")
    print(f"  Total T (P90): {t_sum_90:.1f} mm")
    
    print(f"\nT/ET Ratios:")
    print(f"  P50: {t_sum_50 / et_sum:.4f}")
    print(f"  P75: {t_sum_75 / et_sum:.4f}")
    print(f"  P90: {t_sum_90 / et_sum:.4f}")
    
    print("\n==================================================")

if __name__ == '__main__':
    run_diagnostics('Tumbarumba')
