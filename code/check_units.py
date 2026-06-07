import xarray as xr

# Open the L6 NetCDF file
nc_file = '/home/sanjays/et97_scratch2/oldscratch/Ozflux_data_full/L6/AdelaideRiver_L6.nc'
ds = xr.open_dataset(nc_file)

print("=== ORIGINAL L6 UNITS ===")
for var in ['VPD', 'GPP_NT', 'Fe', 'Ta', 'Precip']:
    if var in ds:
        attrs = ds[var].attrs
        units = attrs.get('units', 'UNKNOWN')
        long_name = attrs.get('long_name', 'UNKNOWN')
        print(f"{var}: {units} ({long_name})")
    else:
        print(f"{var}: NOT FOUND")
