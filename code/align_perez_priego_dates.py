import os
import glob
import pandas as pd
import numpy as np
import xarray as xr

work_dir = "/home/sanjays/et97_scratch2/oldscratch/Ozflux_data_full/TEA_partition"
output_dir = os.path.join(work_dir, "output")

# Find all PerezPriego daily CSV files (excluding year-specific splits)
pp_files = glob.glob(os.path.join(output_dir, "*_PerezPriego_daily.csv"))

print("Starting alignment of Perez-Priego dates...")

for file_path in pp_files:
    filename = os.path.basename(file_path)
    parts = filename.split('_')
    if len(parts) > 3:  # e.g., Calperum_2010_PerezPriego_daily.csv (4 parts)
        continue
    
    site = parts[0]
    print(f"\nProcessing site: {site}")
    
    # Read daily CSV
    df = pd.read_csv(file_path)
    
    # Open preprocessed NetCDF
    nc_path = os.path.join(output_dir, f"{site}_preprocessed.nc")
    if not os.path.exists(nc_path):
        print(f"  Warning: NetCDF not found for {site} at {nc_path}. Skipping.")
        continue
        
    try:
        ds = xr.open_dataset(nc_path)
        times = pd.to_datetime(ds.time.values)
        ds.close()
        
        # Map loops to dates (assuming 48 timesteps per day)
        dates = []
        for loop_val in df['loop']:
            idx = int((loop_val - 1) * 48)
            if idx < len(times):
                dates.append(times[idx].strftime('%Y-%m-%d'))
            else:
                dates.append(None)
                
        df['date'] = dates
        
        # Reorder columns to put date first
        cols = ['date'] + [col for col in df.columns if col != 'date']
        df = df[cols]
        
        # Save back the daily CSV
        df.to_csv(file_path, index=False)
        print(f"  Successfully aligned dates for {site}. Total days: {len(df)}")
        
        # Verify length alignment
        n_days_nc = int(np.ceil(len(times) / 48))
        if len(df) != n_days_nc:
            print(f"  [Notice] CSV row count ({len(df)}) differs from NC day count ({n_days_nc})")
            
    except Exception as e:
        print(f"  Error processing {site}: {e}")

print("\nDate alignment complete!")
