"""
run_all_partitioning.py
=======================
Batch processing script for running TEA (Nelson 2018) and Zhou/uWUE (Zhou 2016)
ET partitioning methods on all OzFlux L6 sites.

Usage:
    python run_all_partitioning.py [--sites SITE1 SITE2 ...] [--methods tea zhou]

If no --sites argument is given, all L6 files are processed.

Author: Generated for sanjays OzFlux analysis
"""

import os
import sys
import argparse
import warnings
import traceback
import numpy as np
import xarray as xr
import pandas as pd
from datetime import datetime

# Add paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.join(SCRIPT_DIR, 'ecosystem-transpiration')
sys.path.insert(0, REPO_DIR)
sys.path.insert(0, SCRIPT_DIR)

from ozflux_preprocess import build_ozflux_dataset, get_site_list
import bigleaf
from run_tea_improved import run_tea_improved

# Default paths
L6_DIR = '/home/sanjays/et97_scratch2/oldscratch/Ozflux_data_full/L6'
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'output')


def run_tea(ds, site_name, output_dir, n_jobs=4):
    """Run the TEA partitioning method.
    
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
        
    Returns
    -------
    result : dict
        Dictionary with TEA results, or None if failed
    """
    import TEA
    
    print(f"\n  [TEA] Running for {site_name}...")
    
    nStepsPerDay = ds.attrs.get('nStepsPerDay', 48)
    
    try:
        # Extract variables for TEA
        timestamp = ds.time.values
        ET        = ds.ET.values
        GPP       = ds.GPP_NT.values
        Tair      = ds.TA.values
        VPD       = ds.VPD.values  # in hPa
        precip    = ds.P.values
        Rg        = ds.SW_IN.values
        Rg_pot    = ds.SW_IN_POT.values
        u         = ds.WS.values
        
        # Trim to exact multiple of nStepsPerDay (TEA reshapes to -1,48)
        n_total = len(timestamp)
        n_complete = (n_total // nStepsPerDay) * nStepsPerDay
        if n_total != n_complete:
            print(f"  [TEA] Trimming from {n_total} to {n_complete} timesteps "
                  f"(removing {n_total - n_complete} partial-day records)")
            timestamp = timestamp[:n_complete]
            ET = ET[:n_complete]
            GPP = GPP[:n_complete]
            Tair = Tair[:n_complete]
            VPD = VPD[:n_complete]
            precip = precip[:n_complete]
            Rg = Rg[:n_complete]
            Rg_pot = Rg_pot[:n_complete]
            u = u[:n_complete]
        
        # RH: use from dataset
        if 'RH' in ds:
            RH = ds.RH.values[:n_complete]
        else:
            RH = bigleaf.VPD_to_RH(VPD / 10.0, Tair)  # convert hPa to kPa for bigleaf
        
        # Quality flag: accept measured (0) and good gap-fill (1)
        # This matches the FLUXNET convention where QC < 2 is acceptable
        nee_qc = ds.NEE_QC.values[:n_complete]
        le_qc = ds.LE_QC.values[:n_complete]
        qualityFlag = (nee_qc <= 1) & (le_qc <= 1)
        qualityFlag = qualityFlag & np.isfinite(ET) & np.isfinite(GPP)
        
        OtherVars = {'qualityFlag': qualityFlag}
        
        # Build TEA dataset
        tea_ds = TEA.build_dataset(timestamp, ET, GPP, RH, Rg, Rg_pot,
                                   Tair, VPD, precip, u,
                                   OtherVars=OtherVars)
        
        # Preprocess
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore',
                r'invalid value encountered in (less|less_equal|greater|true_divide)')
            tea_ds = TEA.preprocess(tea_ds, nStepsPerDay=nStepsPerDay)
        
        # Check CSWI
        cswi_valid = np.isfinite(tea_ds.CSWI.values).sum()
        print(f"  [TEA] CSWI valid points: {cswi_valid}/{len(tea_ds.CSWI)}")
        
        if cswi_valid == 0:
            print(f"  [TEA] WARNING: No valid CSWI for {site_name}, skipping TEA")
            return None
        
        # Partition
        # CSWIlim = -1 is the TEA default; -0.5 is more restrictive
        CSWIlims = [-1]
        percs = [50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
        
        tea_ds = TEA.partition(
            tea_ds,
            percs=np.array(percs),
            CSWIlims=np.array(CSWIlims),
            RandomForestRegressor_kwargs={'n_jobs': n_jobs}
        )
        
        tea_ds = tea_ds.sel(CSWIlims=CSWIlims[0])
        
        # Check if partitioning succeeded
        if np.all(np.isnan(tea_ds.TEA_T)):
            checkvars = ['Rg', 'Tair', 'RH', 'u', 'Rg_pot_daily', 'Rgpotgrad',
                        'year', 'GPPgrad', 'DWCI', 'C_Rg_ET', 'CSWI', 'precip']
            missingVars = [v for v in checkvars 
                          if v in tea_ds and np.all(~np.isfinite(tea_ds[v].values))]
            if missingVars:
                print(f"  [TEA] FAILED for {site_name}: missing vars = {missingVars}")
            else:
                print(f"  [TEA] FAILED for {site_name}: unknown error")
            return None
        
        # Save TEA output as NetCDF
        tea_outfile = os.path.join(output_dir, f'{site_name}_TEA_output.nc')
        tea_ds.to_netcdf(tea_outfile)
        print(f"  [TEA] Saved: {tea_outfile}")
        
        # Also save daily summary CSV
        tea_T_75 = tea_ds.TEA_T.sel(percentiles=75).values
        tea_E_75 = tea_ds.TEA_E.sel(percentiles=75).values
        et_vals = tea_ds.ET.values
        
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

        # Filter to overlapping valid points for the ratio
        mask = np.isfinite(df['TEA_T_75_mm']) & np.isfinite(df['ET_mm'])
        df_valid = df[mask].copy()
        
        # Add daily sums from valid-only points
        df_valid['date'] = pd.to_datetime(df_valid['timestamp']).dt.date
        daily = df_valid.groupby('date').agg({
            'ET_mm': 'sum',
            'TEA_T_75_mm': 'sum',
            'TEA_E_75_mm': 'sum',
        }).reset_index()
        daily['T_ET_ratio'] = daily['TEA_T_75_mm'] / daily['ET_mm']
        daily.columns = ['date', 'ET_daily_mm', 'T_TEA_daily_mm', 'E_TEA_daily_mm', 'T_ET_ratio']
        
        csv_file = os.path.join(output_dir, f'{site_name}_TEA_daily.csv')
        daily.to_csv(csv_file, index=False)
        print(f"  [TEA] Saved daily CSV: {csv_file}")
        
        # Summary stats
        valid_ratio = daily['T_ET_ratio'].dropna()
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
        print(f"  [TEA] {site_name}: T/ET = {result['T_ET_mean']:.3f} "
              f"(median {result['T_ET_median']:.3f})")
        return result
        
    except Exception as e:
        print(f"  [TEA] ERROR for {site_name}: {e}")
        traceback.print_exc()
        return {'site': site_name, 'method': 'TEA', 'status': f'ERROR: {str(e)}',
                'T_ET_mean': np.nan, 'T_ET_median': np.nan,
                'T_annual_mm': np.nan, 'ET_annual_mm': np.nan,
                'n_days': 0, 'n_valid_days': 0}


def run_zhou(ds, site_name, output_dir):
    """Run the Zhou/uWUE partitioning method.
    
    Parameters
    ----------
    ds : xr.Dataset
        Preprocessed OzFlux dataset (FLUXNET convention)
    site_name : str
        Site name for labeling output
    output_dir : str
        Directory to save output files
        
    Returns
    -------
    result : dict
        Dictionary with Zhou results, or None if failed
    """
    import zhou
    
    print(f"\n  [Zhou] Running for {site_name}...")
    
    nStepsPerDay = ds.attrs.get('nStepsPerDay', 48)
    
    try:
        # Zhou method requires specific variable names in the dataset
        # Build a temporary dataset with the right names for zhouFlags
        zhou_ds = ds.copy()
        
        # Ensure PET exists
        if 'PET' not in zhou_ds or np.all(np.isnan(zhou_ds.PET.values)):
            print(f"  [Zhou] WARNING: PET not available for {site_name}, skipping Zhou")
            return None
        
        # Set up hourly mask
        hourlyMask = np.ones(zhou_ds.LE.shape).astype(bool)
        
        # Calculate GxV = GPP * VPD^0.5 (VPD needs to be in hPa for Zhou)
        GPP = zhou_ds.GPP_NT.values
        VPD = zhou_ds.VPD.values  # already in hPa
        ET = zhou_ds.ET.values
        
        # GPP needs to be in gC m-2 d-1 for Zhou
        # OzFlux GPP is umol m-2 s-1; convert:
        # umol/m2/s * 12e-6 g/umol * 86400 s/d * (1/nStepsPerDay) = gC/m2/d per timestep
        # Actually, the Zhou method works per timestep, and GxV is computed per timestep
        # Looking at zhou.py, it sums ET and computes daily, so let's keep per-timestep
        GxV = GPP * np.sqrt(np.maximum(VPD, 0))
        
        # Get Zhou flags
        try:
            uWUEa_Mask, uWUEp_Mask = zhou.zhouFlags(
                zhou_ds, nStepsPerDay=nStepsPerDay,
                hourlyMask=hourlyMask, GPPvariant='GPP_NT'
            )
        except Exception as e:
            print(f"  [Zhou] Could not compute Zhou flags: {e}")
            # Build flags manually
            qualityMask = np.ones(ET.shape).astype(bool)
            for var in ['NEE', 'LE', 'TA', 'VPD']:
                qc_var = var + '_QC'
                if qc_var in zhou_ds:
                    QC = zhou_ds[qc_var].values.copy()
                    QC[QC < 0] = 3
                    QC[~np.isfinite(QC)] = 3
                    qualityMask &= QC < 2
            for var in ['GPP_NT', 'ET', 'TA', 'VPD', 'NETRAD']:
                if var in zhou_ds:
                    qualityMask &= np.isfinite(zhou_ds[var].values)
                    qualityMask &= zhou_ds[var].values > -9000
            
            zeroMask = np.ones(ET.shape).astype(bool)
            for var in ['GPP_NT', 'ET', 'NETRAD', 'VPD']:
                if var in zhou_ds:
                    zeroMask &= zhou_ds[var].values > 0
            
            uWUEa_Mask = zeroMask & qualityMask
            uWUEp_Mask = uWUEa_Mask.copy()
        
        # Check we have enough valid data
        n_valid = uWUEp_Mask.sum()
        print(f"  [Zhou] Valid data points: uWUEa={uWUEa_Mask.sum()}, uWUEp={n_valid}")
        
        if n_valid < 100:
            print(f"  [Zhou] Not enough valid data for {site_name}, skipping Zhou")
            return None
        
        # Ensure data length is divisible by 48
        n_total = len(ET)
        n_complete = (n_total // nStepsPerDay) * nStepsPerDay
        if n_total != n_complete:
            print(f"  [Zhou] Trimming data from {n_total} to {n_complete} timesteps")
            ET = ET[:n_complete]
            GxV = GxV[:n_complete]
            uWUEa_Mask = uWUEa_Mask[:n_complete]
            uWUEp_Mask = uWUEp_Mask[:n_complete]
        
        # Run Zhou partitioning
        uWUEp, zhou_T, zhou_T_8day = zhou.zhou_part(
            ET, GxV, uWUEa_Mask, uWUEp_Mask,
            nStepsPerDay=nStepsPerDay, hourlyMask=hourlyMask[:n_complete]
        )
        
        print(f"  [Zhou] uWUEp = {uWUEp:.4f}")
        
        # Build daily output
        n_days = n_complete // nStepsPerDay
        
        # Mask invalid values
        ET_valid = ET.copy()
        ET_valid[ET_valid < -9000] = np.nan
        ET_daily = ET_valid.reshape(-1, nStepsPerDay).sum(axis=1)
        
        zhou_T_valid = zhou_T.copy()
        zhou_T_valid[zhou_T_valid < -9000] = np.nan
        
        zhou_T_8day_valid = zhou_T_8day.copy()
        if zhou_T_8day_valid is not None and not np.isscalar(zhou_T_8day_valid):
            zhou_T_8day_valid[zhou_T_8day_valid < -9000] = np.nan
            
        # Timestamps for daily data
        time_vals = ds.time.values[:n_complete]
        daily_times = time_vals.reshape(-1, nStepsPerDay)[:, 0]
        
        df = pd.DataFrame({
            'date': pd.to_datetime(daily_times).date,
            'ET_daily_mm': ET_daily,
            'T_Zhou_daily_mm': zhou_T_valid,
            'T_Zhou_8day_mm': zhou_T_8day_valid,
        })
        df['T_ET_ratio'] = df['T_Zhou_daily_mm'] / df['ET_daily_mm']
        df['T_ET_ratio_8day'] = df['T_Zhou_8day_mm'] / df['ET_daily_mm']
        
        csv_file = os.path.join(output_dir, f'{site_name}_Zhou_daily.csv')
        df.to_csv(csv_file, index=False)
        print(f"  [Zhou] Saved: {csv_file}")
        
        # Summary stats - use filtered data for annual sums too
        df_valid = df[df['T_ET_ratio'].notna() & (df['T_ET_ratio'] >= 0) & (df['T_ET_ratio'] <= 1)].copy()
        
        result = {
            'site': site_name,
            'method': 'Zhou_uWUE',
            'T_ET_mean': df_valid['T_ET_ratio'].mean() if len(df_valid) > 0 else np.nan,
            'T_ET_median': df_valid['T_ET_ratio'].median() if len(df_valid) > 0 else np.nan,
            'T_annual_mm': df_valid['T_Zhou_daily_mm'].sum(),
            'ET_annual_mm': df_valid['ET_daily_mm'].sum(),
            'n_days': len(df),
            'n_valid_days': len(df_valid),
            'uWUEp': uWUEp,
            'status': 'OK'
        }
        print(f"  [Zhou] {site_name}: T/ET = {result['T_ET_mean']:.3f} "
              f"(median {result['T_ET_median']:.3f}), uWUEp = {uWUEp:.4f}")
        return result
        
    except Exception as e:
        print(f"  [Zhou] ERROR for {site_name}: {e}")
        traceback.print_exc()
        return {'site': site_name, 'method': 'Zhou_uWUE', 'status': f'ERROR: {str(e)}',
                'T_ET_mean': np.nan, 'T_ET_median': np.nan,
                'T_annual_mm': np.nan, 'ET_annual_mm': np.nan,
                'n_days': 0, 'n_valid_days': 0, 'uWUEp': np.nan}


def main():
    parser = argparse.ArgumentParser(
        description='Run TEA and Zhou ET partitioning on OzFlux L6 data')
    parser.add_argument('--l6_dir', default=L6_DIR,
                       help='Directory containing L6 NetCDF files')
    parser.add_argument('--output_dir', default=OUTPUT_DIR,
                       help='Output directory for results')
    parser.add_argument('--sites', nargs='*', default=None,
                       help='Specific site names to process (default: all)')
    parser.add_argument('--methods', nargs='*', default=['tea', 'zhou'],
                       choices=['tea', 'zhou'],
                       help='Methods to run (default: tea zhou)')
    parser.add_argument('--gpp_variant', default='GPP_SOLO',
                       choices=['GPP_SOLO', 'GPP_LL', 'GPP_LT'],
                       help='GPP variant to use (default: GPP_SOLO)')
    parser.add_argument('--er_variant', default='ER_SOLO',
                       choices=['ER_SOLO', 'ER_LL', 'ER_LT'],
                       help='ER variant to use (default: ER_SOLO)')
    parser.add_argument('--n_jobs', type=int, default=4,
                       help='Number of parallel jobs for TEA (default: 4)')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get file list
    all_files = get_site_list(args.l6_dir)
    print(f"Found {len(all_files)} L6 files in {args.l6_dir}")
    
    # Filter by site if specified
    if args.sites:
        files = [f for f in all_files
                 if any(s in os.path.basename(f) for s in args.sites)]
        print(f"Filtered to {len(files)} sites: {args.sites}")
    else:
        files = all_files
    
    if not files:
        print("ERROR: No files found to process!")
        sys.exit(1)
    
    # Process each site
    all_results = []
    start_time = datetime.now()
    
    for i, nc_file in enumerate(files):
        site_name = os.path.basename(nc_file).replace('_L6.nc', '')
        print(f"\n{'='*70}")
        print(f"[{i+1}/{len(files)}] Processing: {site_name}")
        print(f"{'='*70}")
        
        try:
            # Preprocess
            ds = build_ozflux_dataset(nc_file,
                                      gpp_variant=args.gpp_variant,
                                      er_variant=args.er_variant)
            
            # Also save the preprocessed dataset for Pérez-Priego R method
            preproc_file = os.path.join(args.output_dir,
                                        f'{site_name}_preprocessed.nc')
            ds.to_netcdf(preproc_file)
            
            # Run TEA
            if 'tea' in args.methods:
                result = run_tea_improved(ds, site_name, args.output_dir,
                               n_jobs=args.n_jobs, verbose=True)
                if result:
                    all_results.append(result)
            
            # Run Zhou
            if 'zhou' in args.methods:
                result = run_zhou(ds, site_name, args.output_dir)
                if result:
                    all_results.append(result)
            
            ds.close()
            
        except Exception as e:
            print(f"ERROR processing {site_name}: {e}")
            traceback.print_exc()
            all_results.append({
                'site': site_name, 'method': 'preprocessing',
                'status': f'ERROR: {str(e)}',
                'T_ET_mean': np.nan, 'T_ET_median': np.nan,
                'T_annual_mm': np.nan, 'ET_annual_mm': np.nan,
                'n_days': 0, 'n_valid_days': 0
            })
    
    # Save summary
    elapsed = datetime.now() - start_time
    print(f"\n{'='*70}")
    print(f"COMPLETED: {len(files)} sites in {elapsed}")
    print(f"{'='*70}")
    
    if all_results:
        summary_df = pd.DataFrame(all_results)
        summary_file = os.path.join(args.output_dir, 'all_sites_summary.csv')
        summary_df.to_csv(summary_file, index=False)
        print(f"\nSummary saved: {summary_file}")
        print(f"\nResults overview:")
        print(summary_df[['site', 'method', 'T_ET_mean', 'T_ET_median', 'status']].to_string())
    
    # Print failed sites
    failed = [r for r in all_results if r.get('status', '').startswith('ERROR')]
    if failed:
        print(f"\n⚠ {len(failed)} failures:")
        for r in failed:
            print(f"  {r['site']} ({r['method']}): {r['status']}")


if __name__ == '__main__':
    main()
