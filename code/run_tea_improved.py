"""
run_tea_improved.py
===================
Improved version of run_tea with better unit validation and diagnostics.

Key improvements:
1. Explicit unit checks and validation
2. Better error messages
3. Diagnostic output at each step
4. Unit conversion verification
5. FIXED: Added positivity guard for negative GPP noise
6. FIXED: Safe datetime extraction
7. FIXED: Safe division to prevent zero-denominator crashes
"""

import numpy as np
import pandas as pd
import warnings
import traceback


def run_tea_improved(ds, site_name, output_dir, n_jobs=4, verbose=True):
    """Run the TEA partitioning method with improved diagnostics.
    
    Parameters
    ----------
    ds : xr.Dataset
        Preprocessed OzFlux dataset (FLUXNET convention)
    site_name : str
        Site name for labeling output
    output_dir : str
        Directory to save output files
    n_jobs : int
        Number of parallel jobs for RandomForestRegressor
    verbose : bool
        Print detailed diagnostic information
        
    Returns
    -------
    result : dict
        Dictionary with TEA results, or None if failed
    """
    import TEA
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"[TEA] Running for {site_name} with diagnostics...")
        print(f"{'='*70}")
    
    nStepsPerDay = ds.attrs.get('nStepsPerDay', 48)
    
    try:
        # ===================================================================
        # STEP 1: EXTRACT AND VALIDATE VARIABLES
        # ===================================================================
        if verbose:
            print(f"\nSTEP 1: Extracting variables...")
        
        timestamp = ds.time.values
        
        # ET - CRITICAL: Must be in mm per timestep
        ET = ds.ET.values
        if verbose:
            print(f"  ET: mean={np.nanmean(ET):.4f}, range=[{np.nanmin(ET):.4f}, {np.nanmax(ET):.4f}]")
            expected_et_range = (0.001, 1.0)  # typical range for 30-min timesteps
            if np.nanmean(ET) < expected_et_range[0]:
                print(f"  ⚠️  WARNING: ET values seem too low (expected {expected_et_range})")
                print(f"     Check if ET is in mm per timestep")
            elif np.nanmean(ET) > expected_et_range[1]:
                print(f"  ⚠️  WARNING: ET values seem too high (expected {expected_et_range})")
                print(f"     Check if ET is in mm per timestep (not mm per day)")
        
        # GPP - Must be in umol m-2 s-1
        GPP = ds.GPP_NT.values
        
        # FIX 1: Positivity guard! Force negative GPP noise to 0 to prevent negative summation
        GPP = np.where(GPP < 0, 0, GPP)
        
        if verbose:
            gpp_valid = GPP[np.isfinite(GPP) & (GPP > 0)]
            if len(gpp_valid) > 0:
                print(f"  GPP: mean={np.nanmean(gpp_valid):.2f}, range=[{gpp_valid.min():.2f}, {gpp_valid.max():.2f}]")
                if np.nanmean(gpp_valid) < 1:
                    print(f"  ⚠️  WARNING: GPP values seem low (expected 5-30 for daytime)")
                elif np.nanmean(gpp_valid) > 50:
                    print(f"  ⚠️  WARNING: GPP values seem high (check units: should be umol m-2 s-1)")
        
        # Tair - Must be in deg C
        Tair = ds.TA.values
        if verbose:
            print(f"  Tair: mean={np.nanmean(Tair):.1f}°C, range=[{np.nanmin(Tair):.1f}, {np.nanmax(Tair):.1f}]")
        
        # VPD - Must be in hPa (not kPa!)
        VPD = ds.VPD.values
        if verbose:
            print(f"  VPD: mean={np.nanmean(VPD):.2f}, range=[{np.nanmin(VPD):.2f}, {np.nanmax(VPD):.2f}]")
            if np.nanmean(VPD) < 1:
                print(f"  ⚠️  WARNING: VPD values seem low (expected 5-30 hPa)")
                print(f"     Check if VPD is in hPa (multiply by 10 if in kPa)")
            elif np.nanmean(VPD) > 100:
                print(f"  ⚠️  WARNING: VPD values seem high")
                print(f"     Check if VPD is in hPa (divide by 100 if in Pa)")
        
        precip = ds.P.values
        Rg = ds.SW_IN.values
        Rg_pot = ds.SW_IN_POT.values
        u = ds.WS.values
        
        if verbose:
            print(f"  Precip: mean={np.nanmean(precip):.3f} mm/timestep")
            print(f"  Rg: mean={np.nanmean(Rg):.1f} W/m2")
            print(f"  Rg_pot: mean={np.nanmean(Rg_pot):.1f} W/m2")
            print(f"  Wind speed: mean={np.nanmean(u):.2f} m/s")
        
        # ===================================================================
        # STEP 2: TRIM TO COMPLETE DAYS
        # ===================================================================
        n_total = len(timestamp)
        n_complete = (n_total // nStepsPerDay) * nStepsPerDay
        
        if n_total != n_complete:
            if verbose:
                print(f"\nSTEP 2: Trimming from {n_total} to {n_complete} timesteps")
                print(f"        (removing {n_total - n_complete} partial-day records)")
            timestamp = timestamp[:n_complete]
            ET = ET[:n_complete]
            GPP = GPP[:n_complete]
            Tair = Tair[:n_complete]
            VPD = VPD[:n_complete]
            precip = precip[:n_complete]
            Rg = Rg[:n_complete]
            Rg_pot = Rg_pot[:n_complete]
            u = u[:n_complete]
        
        # ===================================================================
        # STEP 3: CALCULATE RH
        # ===================================================================
        if verbose:
            print(f"\nSTEP 3: Calculating relative humidity...")
        
        if 'RH' in ds:
            RH = ds.RH.values[:n_complete]
            if verbose:
                print(f"  Using RH from dataset: mean={np.nanmean(RH):.1f}%")
        else:
            import bigleaf
            # bigleaf expects VPD in kPa, so convert
            RH = bigleaf.VPD_to_RH(VPD / 10.0, Tair)
            if verbose:
                print(f"  Calculated RH from VPD and Tair: mean={np.nanmean(RH):.1f}%")
        
        # ===================================================================
        # STEP 4: QUALITY FLAGS
        # ===================================================================
        if verbose:
            print(f"\nSTEP 4: Building quality flags...")
        
        nee_qc = ds.NEE_QC.values[:n_complete]
        le_qc = ds.LE_QC.values[:n_complete]
        
        qualityFlag = (nee_qc <= 1) & (le_qc <= 1)
        qualityFlag = qualityFlag & np.isfinite(ET) & np.isfinite(GPP)
        
        if verbose:
            n_good = qualityFlag.sum()
            pct_good = 100 * n_good / len(qualityFlag)
            print(f"  Good quality data: {n_good}/{len(qualityFlag)} ({pct_good:.1f}%)")
            
            if pct_good < 20:
                print(f"  ⚠️  WARNING: Less than 20% of data has good quality!")
                print(f"     TEA may struggle with limited high-quality data")
        
        OtherVars = {'qualityFlag': qualityFlag}
        
        # ===================================================================
        # STEP 5: BUILD TEA DATASET
        # ===================================================================
        if verbose:
            print(f"\nSTEP 5: Building TEA dataset...")
        
        tea_ds = TEA.build_dataset(timestamp, ET, GPP, RH, Rg, Rg_pot,
                                   Tair, VPD, precip, u,
                                   OtherVars=OtherVars)
        
        if verbose:
            print(f"  TEA dataset created with {len(tea_ds.timestamp)} timesteps")
        
        # ===================================================================
        # STEP 6: PREPROCESS
        # ===================================================================
        if verbose:
            print(f"\nSTEP 6: Running TEA preprocessing...")
        
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore',
                r'invalid value encountered in (less|less_equal|greater|true_divide)')
            tea_ds = TEA.preprocess(tea_ds, nStepsPerDay=nStepsPerDay)
        
        # ===================================================================
        # STEP 7: CHECK CSWI
        # ===================================================================
        if verbose:
            print(f"\nSTEP 7: Checking CSWI (Cumulative Shortwave Water Index)...")
        
        cswi_valid = np.isfinite(tea_ds.CSWI.values).sum()
        cswi_total = len(tea_ds.CSWI)
        pct_cswi = 100 * cswi_valid / cswi_total
        
        if verbose:
            print(f"  CSWI valid points: {cswi_valid}/{cswi_total} ({pct_cswi:.1f}%)")
            print(f"  CSWI range: [{np.nanmin(tea_ds.CSWI.values):.2f}, {np.nanmax(tea_ds.CSWI.values):.2f}]")
        
        if cswi_valid == 0:
            print(f"  ❌ ERROR: No valid CSWI for {site_name}")
            print(f"     CSWI is required for TEA partitioning")
            print(f"     Possible causes:")
            print(f"       - Missing or all-NaN precipitation data")
            print(f"       - Missing Rg_pot data")
            print(f"       - Insufficient data span")
            return None
        
        if pct_cswi < 50:
            print(f"  ⚠️  WARNING: Only {pct_cswi:.1f}% of CSWI is valid")
            print(f"     TEA results may be unreliable")
        
        # Check other important variables
        if verbose:
            for var in ['DWCI', 'C_Rg_ET', 'GPPgrad']:
                if var in tea_ds:
                    valid = np.isfinite(tea_ds[var].values).sum()
                    total = len(tea_ds[var].values)
                    print(f"  {var} valid: {valid}/{total} ({100*valid/total:.1f}%)")
        
        # ===================================================================
        # STEP 8: PARTITION
        # ===================================================================
        if verbose:
            print(f"\nSTEP 8: Running TEA partitioning...")
        
        CSWIlims = [-1]  # Default threshold
        percs = [50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
        
        tea_ds = TEA.partition(
            tea_ds,
            percs=np.array(percs),
            CSWIlims=np.array(CSWIlims),
            RandomForestRegressor_kwargs={'n_jobs': n_jobs}
        )
        
        tea_ds = tea_ds.sel(CSWIlims=CSWIlims[0])
        
        # ===================================================================
        # STEP 9: CHECK RESULTS
        # ===================================================================
        if verbose:
            print(f"\nSTEP 9: Checking TEA results...")
        
        if np.all(np.isnan(tea_ds.TEA_T)):
            print(f"  ❌ ERROR: All TEA_T values are NaN")
            
            # Diagnostic checks
            checkvars = ['Rg', 'Tair', 'RH', 'u', 'Rg_pot_daily', 'Rgpotgrad',
                        'year', 'GPPgrad', 'DWCI', 'C_Rg_ET', 'CSWI', 'precip']
            missingVars = [v for v in checkvars 
                          if v in tea_ds and np.all(~np.isfinite(tea_ds[v].values))]
            if missingVars:
                print(f"     Missing/invalid variables: {missingVars}")
            else:
                print(f"     All input variables appear valid")
                print(f"     This may indicate a problem with the Random Forest regression")
            return None
        
        # Check distribution of results
        tea_T_75 = tea_ds.TEA_T.sel(percentiles=75).values
        tea_E_75 = tea_ds.TEA_E.sel(percentiles=75).values
        et_vals = tea_ds.ET.values
        
        valid_T = np.isfinite(tea_T_75) & (tea_T_75 > -9000)
        valid_E = np.isfinite(tea_E_75) & (tea_E_75 > -9000)
        valid_ET = np.isfinite(et_vals) & (et_vals > -9000)
        
        if verbose:
            print(f"  TEA_T valid: {valid_T.sum()}/{len(tea_T_75)} ({100*valid_T.sum()/len(tea_T_75):.1f}%)")
            print(f"  TEA_E valid: {valid_E.sum()}/{len(tea_E_75)} ({100*valid_E.sum()/len(tea_E_75):.1f}%)")
            print(f"  TEA_T range: [{tea_T_75[valid_T].min():.4f}, {tea_T_75[valid_T].max():.4f}] mm/timestep")
            print(f"  TEA_E range: [{tea_E_75[valid_E].min():.4f}, {tea_E_75[valid_E].max():.4f}] mm/timestep")
            print(f"  ET range: [{et_vals[valid_ET].min():.4f}, {et_vals[valid_ET].max():.4f}] mm/timestep")
        
        # ===================================================================
        # STEP 10: SAVE OUTPUTS
        # ===================================================================
        if verbose:
            print(f"\nSTEP 10: Saving outputs...")
        
        # Save TEA output as NetCDF
        import os
        tea_outfile = os.path.join(output_dir, f'{site_name}_TEA_output.nc')
        tea_ds.to_netcdf(tea_outfile)
        if verbose:
            print(f"  Saved NetCDF: {tea_outfile}")
        
        # Build daily summary CSV
        df = pd.DataFrame({
            'timestamp': tea_ds.timestamp.values,
            'ET_mm': et_vals,
            'TEA_T_75_mm': tea_T_75,
            'TEA_E_75_mm': tea_E_75,
        })
        
        # Mask invalid values
        df.loc[df['ET_mm'] < -9000, 'ET_mm'] = np.nan
        df.loc[df['TEA_T_75_mm'] < -9000, 'TEA_T_75_mm'] = np.nan
        df.loc[df['TEA_E_75_mm'] < -9000, 'TEA_E_75_mm'] = np.nan
        
        # Add daily sums
        # FIX 2: Safely extract dates to prevent Pandas crashing on non-standard calendars (like cftime)
        try:
            df['date'] = pd.to_datetime(df['timestamp']).dt.date
        except Exception:
            # Fallback if xarray passed a cftime object instead of standard datetime
            df['date'] = [d.date() if hasattr(d, 'date') else pd.Timestamp(str(d)).date() for d in df['timestamp']]
            
        daily = df.groupby('date').agg({
            'ET_mm': 'sum',
            'TEA_T_75_mm': 'sum',
            'TEA_E_75_mm': 'sum',
        }).reset_index()
        
        # FIX 3: Safe division using numpy to avoid ZeroDivisionError when ET == 0
        daily['T_ET_ratio'] = np.divide(
            daily['TEA_T_75_mm'].to_numpy(dtype=float), 
            daily['ET_mm'].to_numpy(dtype=float), 
            out=np.full_like(daily['TEA_T_75_mm'].values, np.nan, dtype=float), 
            where=(daily['ET_mm'].to_numpy(dtype=float) != 0)
        )
        
        daily.columns = ['date', 'ET_daily_mm', 'T_TEA_daily_mm', 'E_TEA_daily_mm', 'T_ET_ratio']
        
        csv_file = os.path.join(output_dir, f'{site_name}_TEA_daily.csv')
        daily.to_csv(csv_file, index=False)
        if verbose:
            print(f"  Saved daily CSV: {csv_file}")
        
        # ===================================================================
        # STEP 11: SUMMARY STATISTICS
        # ===================================================================
        valid_ratio = daily['T_ET_ratio'].dropna()
        valid_ratio = valid_ratio[(valid_ratio >= 0) & (valid_ratio <= 1)]
        
        result = {
            'site': site_name,
            'method': 'TEA',
            'T_ET_mean': valid_ratio.mean() if len(valid_ratio) > 0 else np.nan,
            'T_ET_median': valid_ratio.median() if len(valid_ratio) > 0 else np.nan,
            'T_annual_mm': daily['T_TEA_daily_mm'].sum(),
            'ET_annual_mm': daily['ET_daily_mm'].sum(),
            'n_days': len(daily),
            'n_valid_days': len(valid_ratio),
            'status': 'OK'
        }
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"TEA RESULTS FOR {site_name}:")
            print(f"{'='*70}")
            print(f"  T/ET mean: {result['T_ET_mean']:.3f}")
            print(f"  T/ET median: {result['T_ET_median']:.3f}")
            print(f"  Annual T: {result['T_annual_mm']:.1f} mm")
            print(f"  Annual ET: {result['ET_annual_mm']:.1f} mm")
            print(f"  Valid days: {result['n_valid_days']}/{result['n_days']}")
            print(f"{'='*70}\n")
            
            # Interpretation
            if result['T_ET_mean'] < 0.3:
                print(f"  ⚠️  WARNING: T/ET ratio seems low (< 0.3)")
                print(f"     Check input data units and preprocessing")
            elif result['T_ET_mean'] > 0.9:
                print(f"  ⚠️  WARNING: T/ET ratio seems high (> 0.9)")
                print(f"     This is unusual for most ecosystems")
        
        return result
        
    except Exception as e:
        print(f"\n  ❌ ERROR for {site_name}: {e}")
        if verbose:
            traceback.print_exc()
        return {
            'site': site_name, 'method': 'TEA', 
            'status': f'ERROR: {str(e)}',
            'T_ET_mean': np.nan, 'T_ET_median': np.nan,
            'T_annual_mm': np.nan, 'ET_annual_mm': np.nan,
            'n_days': 0, 'n_valid_days': 0
        }


if __name__ == '__main__':
    # Example usage
    import sys
    import xarray as xr
    from ozflux_preprocess import build_ozflux_dataset
    
    if len(sys.argv) < 2:
        print("Usage: python run_tea_improved.py <L6_file.nc> [output_dir]")
        sys.exit(1)
    
    nc_file = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else './output'
    
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    result = run_tea_improved(nc_file, output_dir, verbose=True)
    
    if result:
        print(f"\n✓ Success!")
    else:
        print(f"\n✗ Failed")