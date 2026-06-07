"""
ozflux_preprocess.py
====================
Preprocessing adapter for OzFlux L6 NetCDF files.

Converts OzFlux L6 NetCDF data into the xarray Dataset format expected
by the TEA (Nelson 2018) and Zhou/uWUE (Zhou 2016) ET partitioning methods.

Key tasks:
  - Renames OzFlux variables to FLUXNET2015 convention
  - Converts units (VPD kPa→hPa, RH %→fraction, Sws m3/m3→%)
  - Maps QC flags to 0-3 scheme
  - Calculates derived variables (ET, RH, NIGHT, SW_IN_POT, PET)

Author: Generated for sanjays OzFlux analysis
"""

import numpy as np
import xarray as xr
import warnings
import sys
import os

# Add the ecosystem-transpiration directory to path for imports
REPO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'ecosystem-transpiration')
sys.path.insert(0, REPO_DIR)

import bigleaf


# =============================================================================
# Variable mapping: OzFlux L6 → FLUXNET2015/TEA convention
# =============================================================================
OZFLUX_TO_FLUXNET = {
    # Fluxes
    'Fe':       'LE',        # Latent heat flux (W m-2)
    'Fh':       'H',         # Sensible heat flux (W m-2)
    'Fco2':     'NEE',       # Net ecosystem exchange (umol m-2 s-1)
    # Radiation
    'Fn':       'NETRAD',    # Net radiation (W m-2)
    'Fsd':      'SW_IN',     # Downwelling shortwave (W m-2)
    'Fsu':      'SW_OUT',    # Upwelling shortwave (W m-2)
    'Fld':      'LW_IN',     # Downwelling longwave (W m-2)
    'Flu':      'LW_OUT',    # Upwelling longwave (W m-2)
    # Meteorology
    'Ta':       'TA',        # Air temperature (degC)
    'ps':       'PA',        # Surface pressure (kPa)
    'Precip':   'P',         # Precipitation (mm)
    'Ws':       'WS',        # Wind speed (m s-1)
    'ustar':    'USTAR',     # Friction velocity (m s-1)
    # Soil
    'Fg':       'G',         # Ground heat flux (W m-2)
    'Ts':       'TS_1',      # Soil temperature (degC)
    'Sws':      'SWC_1',     # Soil water content (m3/m3 → %)
    # Carbon partitioning (SOLO variant by default)
    'GPP_SOLO': 'GPP_NT',   # Gross primary productivity
    'ER_SOLO':  'RECO_NT',  # Ecosystem respiration
    'GPP_LT':  'GPP_DT',   # Alternative GPP
    # QC flags
    'Fe_QCFlag':   'LE_QC',
    'Fh_QCFlag':   'H_QC',
    'Fco2_QCFlag': 'NEE_QC',
    'Fsd_QCFlag':  'SW_IN_QC',
    'Ta_QCFlag':   'TA_QC',
    'Fg_QCFlag':   'G_QC',
}


def map_qc_flags(qc_values):
    """Map OzFlux QC flags to FLUXNET 0-3 scheme.
    
    OzFlux QC flags:
        0  = observed, good quality
        10 = gap filled from alternate source
        20 = gap filled from ERA5
        30 = gap filled from climatology
        
    FLUXNET QC flags:
        0 = measured
        1 = good quality gap fill
        2 = medium quality gap fill
        3 = poor quality gap fill
    """
    result = np.full_like(qc_values, 3, dtype=float)
    result[qc_values == 0] = 0     # observed
    result[qc_values == 10] = 1    # alternate source gap fill
    result[qc_values == 20] = 2    # ERA5 gap fill
    result[qc_values == 30] = 3    # climatology gap fill
    # Also handle 1 (QC'd, good quality)
    result[qc_values == 1] = 0
    return result


def calc_potential_radiation(timestamps, latitude, longitude):
    """Calculate potential (top-of-atmosphere) shortwave radiation.
    
    Uses solar geometry to estimate clear-sky potential radiation.
    
    Parameters
    ----------
    timestamps : array of datetime64
        Time values
    latitude : float
        Site latitude in degrees
    longitude : float
        Site longitude in degrees
        
    Returns
    -------
    sw_in_pot : array
        Potential incoming shortwave radiation (W m-2)
    """
    solar_constant = 1366.1  # W m-2
    
    # Convert timestamps to day of year and hour
    times = np.array(timestamps, dtype='datetime64[ns]')
    
    # Day of year
    year_start = times.astype('datetime64[Y]')
    doy = (times - year_start).astype('timedelta64[D]').astype(float) + 1
    
    # Fractional hour (UTC)
    day_start = times.astype('datetime64[D]')
    hour_utc = (times - day_start).astype('timedelta64[m]').astype(float) / 60.0
    
    # Solar declination (radians)
    gamma = 2 * np.pi * (doy - 1) / 365.0
    decl = (0.006918 - 0.399912 * np.cos(gamma) + 0.070257 * np.sin(gamma)
            - 0.006758 * np.cos(2 * gamma) + 0.000907 * np.sin(2 * gamma)
            - 0.002697 * np.cos(3 * gamma) + 0.00148 * np.sin(3 * gamma))
    
    # Equation of time (hours)
    eqtime = (229.18 * (0.000075 + 0.001868 * np.cos(gamma)
              - 0.032077 * np.sin(gamma) - 0.014615 * np.cos(2 * gamma)
              - 0.04089 * np.sin(2 * gamma))) / 60.0
    
    # Solar hour angle
    solar_time = hour_utc + longitude / 15.0 + eqtime
    hour_angle = np.radians((solar_time - 12.0) * 15.0)
    
    # Solar zenith angle
    lat_rad = np.radians(latitude)
    cos_zenith = (np.sin(lat_rad) * np.sin(decl) +
                  np.cos(lat_rad) * np.cos(decl) * np.cos(hour_angle))
    cos_zenith = np.maximum(cos_zenith, 0)
    
    # Earth-sun distance correction
    dist_corr = 1.00011 + 0.034221 * np.cos(gamma) + 0.00128 * np.sin(gamma) + \
                0.000719 * np.cos(2 * gamma) + 0.000077 * np.sin(2 * gamma)
    
    sw_in_pot = solar_constant * dist_corr * cos_zenith
    
    return sw_in_pot


def build_ozflux_dataset(nc_file, gpp_variant='GPP_SOLO', er_variant='ER_SOLO'):
    """Build an xarray Dataset from an OzFlux L6 NetCDF file.
    
    Converts variable names and units to match the FLUXNET2015 convention
    used by the TEA and Zhou ET partitioning methods.
    
    Parameters
    ----------
    nc_file : str
        Path to an OzFlux L6 NetCDF file
    gpp_variant : str
        Which GPP variant to use: 'GPP_SOLO', 'GPP_LL', or 'GPP_LT'
    er_variant : str
        Which ER variant to use: 'ER_SOLO', 'ER_LL', or 'ER_LT'
        
    Returns
    -------
    ds : xarray.Dataset
        Dataset with FLUXNET2015-convention variable names and units,
        ready for TEA/Zhou partitioning
    """
    print(f"  Loading: {nc_file}")
    raw = xr.open_dataset(nc_file)
    
    # Global missing value cleaning for numeric variables
    for var in raw.data_vars:
        if np.issubdtype(raw[var].dtype, np.number):
            # Ensure it is a float array before assigning NaN
            if not np.issubdtype(raw[var].dtype, np.floating):
                raw[var] = raw[var].astype(float)
            raw[var].values[raw[var].values < -9000] = np.nan
    
    # Squeeze lat/lon singleton dimensions
    if 'latitude' in raw.dims:
        raw = raw.squeeze('latitude', drop=True)
    if 'longitude' in raw.dims:
        raw = raw.squeeze('longitude', drop=True)
    
    # Extract site metadata
    site_name = raw.attrs.get('site_name', os.path.basename(nc_file).replace('_L6.nc', ''))
    latitude = float(raw.attrs.get('latitude', raw['latitude'].values if 'latitude' in raw else 0))
    longitude = float(raw.attrs.get('longitude', raw['longitude'].values if 'longitude' in raw else 0))
    time_step = int(raw.attrs.get('time_step', 30))  # minutes
    
    # Build variable mapping based on user-selected GPP/ER variants
    var_map = OZFLUX_TO_FLUXNET.copy()
    var_map[gpp_variant] = 'GPP_NT'
    var_map[er_variant] = 'RECO_NT'
    
    # Create new dataset with renamed variables
    ds_vars = {}
    
    for ozflux_name, fluxnet_name in var_map.items():
        if ozflux_name in raw:
            ds_vars[fluxnet_name] = raw[ozflux_name]
        else:
            warnings.warn(f"  Variable {ozflux_name} not found in {nc_file}")
    
    ds = xr.Dataset(ds_vars, coords={'time': raw.time})
    
    # =========================================================================
    # Unit conversions
    # =========================================================================
    
    # VPD: OzFlux stores in kPa, TEA/Zhou expect hPa (1 kPa = 10 hPa)
    if 'VPD' in raw:
        ds['VPD'] = raw['VPD'] * 10.0  # kPa → hPa
        ds['VPD'].attrs = {'units': 'hPa', 'long_name': 'Vapor pressure deficit'}
    elif 'TA' in ds and 'RH' not in raw:
        warnings.warn("  VPD not found in file, will need to compute from other variables")
    
    # Also store VPD in kPa for bigleaf functions
    if 'VPD' in raw:
        ds['VPD_kPa'] = raw['VPD']  # keep original kPa version
    
    # VPD QC flag
    if 'VPD_QCFlag' in raw:
        ds['VPD_QC'] = xr.DataArray(
            map_qc_flags(raw['VPD_QCFlag'].values),
            coords=[raw.time], dims=['time']
        )
    elif 'AH_QCFlag' in raw:
        # Use AH QC as proxy for VPD QC
        ds['VPD_QC'] = xr.DataArray(
            map_qc_flags(raw['AH_QCFlag'].values),
            coords=[raw.time], dims=['time']
        )
    
    # RH: OzFlux stores as percent (0-100), TEA expects percent (0-100)
    if 'RH' in raw:
        ds['RH'] = raw['RH']
        ds['RH'].attrs = {'units': '%', 'long_name': 'Relative humidity'}
    
    # SWC: OzFlux stores as m3/m3, FLUXNET expects %
    if 'SWC_1' in ds:
        ds['SWC_1'] = ds['SWC_1'] * 100.0
        ds['SWC_1'].attrs = {'units': '%', 'long_name': 'Soil water content'}
    
    # GPP Positivity Guard: Mask negative values from raw GPP variables
    for gpp_var in ['GPP_NT', 'GPP_SOLO', 'GPP_LL', 'GPP_LT']:
        if gpp_var in ds:
            ds[gpp_var] = xr.where(ds[gpp_var] < 0, 0, ds[gpp_var])
            ds[gpp_var].attrs = ds[gpp_var].attrs # Preserve attributes
    
    # Map QC flags
    for ozflux_qc, fluxnet_qc in var_map.items():
        if ozflux_qc.endswith('_QCFlag') and fluxnet_qc in ds:
            if ozflux_qc in raw:
                ds[fluxnet_qc] = xr.DataArray(
                    map_qc_flags(raw[ozflux_qc].values),
                    coords=[raw.time], dims=['time']
                )
    
    # =========================================================================
    # Derived variables
    # =========================================================================
    
    # Determine timestep
    if time_step == 30:
        nStepsPerDay = 48
        agg_code = 'HH'
    else:
        nStepsPerDay = 24
        agg_code = 'HR'
    
    # ET: Convert LE (W m-2) to ET (mm per timestep)
    if 'LE' in ds and 'TA' in ds:
        # Use OzFlux ET if available, otherwise compute from LE
        if 'ET' in raw:
            # OzFlux ET is in kg/m2/s, convert to mm per timestep
            seconds_per_step = time_step * 60
            ds['ET'] = raw['ET'] * seconds_per_step  # kg/m2/s → mm/timestep
        else:
            ds['ET'] = bigleaf.LE_to_ET(ds['LE'].values, ds['TA'].values) * 60 * 60 * (24 / nStepsPerDay)
        ds['ET'] = ds['ET'].assign_attrs(
            long_name='evapotranspiration', units='mm per timestep'
        )
    
    # RH from VPD if not already present
    if 'RH' not in ds and 'VPD_kPa' in ds and 'TA' in ds:
        ds['RH'] = bigleaf.VPD_to_RH(ds['VPD_kPa'].values, ds['TA'].values) * 100.0
        ds['RH'] = ds['RH'].assign_attrs(long_name='relative humidity', units='%')
    
    # NIGHT flag: based on Fsd threshold (10 W m-2)
    if 'SW_IN' in ds:
        ds['NIGHT'] = xr.DataArray(
            (ds['SW_IN'].values < 10).astype(float),
            coords=[ds.time], dims=['time']
        )
        ds['NIGHT'].attrs = {'units': 'adimensional', 'long_name': 'Nighttime flag'}
    
    # SW_IN_POT: Potential shortwave radiation
    ds['SW_IN_POT'] = xr.DataArray(
        calc_potential_radiation(ds.time.values, latitude, longitude),
        coords=[ds.time], dims=['time']
    )
    ds['SW_IN_POT'].attrs = {'units': 'W m-2', 'long_name': 'Potential incoming shortwave radiation'}
    
    # PET: Priestley-Taylor potential ET (needed for Zhou method)
    if 'TA' in ds and 'PA' in ds and 'NETRAD' in ds:
        try:
            G_vals = ds['G'].values if 'G' in ds else None
            ET_pot, LE_pot = bigleaf.PET(
                ds['TA'].values, ds['PA'].values, ds['NETRAD'].values,
                G=G_vals, formula='Priestley-Taylor'
            )
            # Convert to mm per timestep
            ds['PET'] = xr.DataArray(
                ET_pot * 60 * 60 * (24 / nStepsPerDay),
                coords=[ds.time], dims=['time']
            )
            ds['PET'].attrs = {'units': 'mm per timestep',
                               'long_name': 'Potential evapotranspiration (Priestley-Taylor)'}
        except Exception as e:
            warnings.warn(f"  Could not calculate PET: {e}")
            ds['PET'] = ds['TA'] * np.nan
    
    # =========================================================================
    # Metadata
    # =========================================================================
    ds.attrs['site_name'] = site_name
    ds.attrs['latitude'] = latitude
    ds.attrs['longitude'] = longitude
    ds.attrs['agg_code'] = agg_code
    ds.attrs['nStepsPerDay'] = nStepsPerDay
    ds.attrs['time_step_minutes'] = time_step
    ds.attrs['source_file'] = nc_file
    ds.attrs['gpp_variant'] = gpp_variant
    ds.attrs['er_variant'] = er_variant
    ds.attrs['vegetation'] = raw.attrs.get('vegetation', 'Unknown')
    ds.attrs['time_zone'] = raw.attrs.get('time_zone', 'Unknown')
    
    raw.close()
    
    print(f"  Site: {site_name} | {latitude:.2f}°, {longitude:.2f}° | "
          f"{len(ds.time)} timesteps | {agg_code} | {ds.attrs['vegetation']}")
    
    return ds


def get_site_list(l6_dir):
    """Get list of OzFlux L6 NetCDF files (half-hourly, not Daily/Monthly/etc).
    
    Parameters
    ----------
    l6_dir : str
        Path to directory containing L6 NetCDF files
        
    Returns
    -------
    files : list of str
        Sorted list of L6 file paths
    """
    import glob
    files = sorted(glob.glob(os.path.join(l6_dir, '*_L6.nc')))
    # Exclude aggregated files (Daily, Monthly, Annual, etc.)
    files = [f for f in files if not any(x in os.path.basename(f)
             for x in ['_Daily', '_Monthly', '_Annual', '_Cumulative', '_Summary'])]
    return files


if __name__ == '__main__':
    # Quick test with AdelaideRiver
    test_file = '/home/sanjays/et97_scratch2/oldscratch/Ozflux_data_full/L6/AdelaideRiver_L6.nc'
    if os.path.exists(test_file):
        ds = build_ozflux_dataset(test_file)
        print("\nDataset variables:")
        for var in sorted(ds.data_vars):
            vals = ds[var].values
            valid = np.isfinite(vals).sum()
            print(f"  {var:15s} | shape={vals.shape} | valid={valid}/{len(vals)} "
                  f"| range=[{np.nanmin(vals):.3f}, {np.nanmax(vals):.3f}]")
    else:
        print(f"Test file not found: {test_file}")
