# TODO: This is entirely AI generated and unchecked.
# I just wanted something to quickly demonstrate what units might looks like.
# Should be carefully reviewed.

import numpy as np

def apply_cf_compliant_attrs(ds):
    """
    Apply CF-compliant units and comments to radar echogram dataset variables.

    Parameters
    ----------
    ds : xarray.Dataset
        The input radar echogram dataset.

    Returns
    -------
    xarray.Dataset
        Dataset with CF-compliant attributes applied.
    """

    # Create a copy to avoid modifying the original dataset
    ds_cf = ds.copy()

    # Define CF-compliant attributes for coordinates
    coordinate_attrs = {
        'slow_time': {
            #'units': 'seconds since 1970-01-01T00:00:00Z',
            'standard_name': 'time',
            'long_name': 'slow time',
            'comment': 'Time coordinate for radar pulse transmission along flight track'
        },
        'twtt': {
            'units': 's',
            'standard_name': 'time',
            'long_name': 'two-way travel time',
            'comment': 'Two-way travel time from radar to target and back'
        }
    }

    # Define CF-compliant attributes for data variables
    data_var_attrs = {
        'Bottom': {
            'units': 's',
            'long_name': 'bottom surface two-way travel time',
            'comment': 'Two-way travel time to detected bottom surface. NaN where bottom not detected.',
            '_FillValue': np.nan
        },
        'Data': {
            'units': '1',  # TODO: Appropriate units for radar data -- can we calibrate to watts?
            'long_name': 'radar echo power',
            'comment': 'Radar echo power in linear scale',
            'coordinates': 'slow_time twtt'
        },
        'Elevation': {
            'units': 'm',
            'standard_name': 'height_above_reference_ellipsoid',
            'long_name': 'platform elevation above WGS84 ellipsoid',
            'comment': 'GPS-derived elevation of radar platform above WGS84 reference ellipsoid'
        },
        'Heading': {
            'units': 'radians',
            'standard_name': 'platform_orientation',
            'long_name': 'platform heading angle',
            'comment': 'Platform heading angle in radians from north, clockwise positive',
            'valid_min': -np.pi,
            'valid_max': np.pi,
            'valid_range': [-np.pi, np.pi],
        },
        'Latitude': {
            'units': 'degrees_north',
            'standard_name': 'latitude',
            'long_name': 'platform latitude',
            'comment': 'GPS-derived latitude of radar platform in WGS84 coordinate system',
            'valid_min': -90.0,
            'valid_max': 90.0,
            'valid_range': [-90.0, 90.0]
        },
        'Longitude': {
            'units': 'degrees_east',
            'standard_name': 'longitude',
            'long_name': 'platform longitude',
            'comment': 'GPS-derived longitude of radar platform in WGS84 coordinate system',
            'valid_min': -180.0,
            'valid_max': 180.0,
            'valid_range': [-180.0, 180.0]
        },
        'Pitch': {
            'units': 'radians',
            'standard_name': 'platform_pitch_angle',
            'long_name': 'platform pitch angle',
            'comment': 'Platform pitch angle in radians, positive nose up',
            'valid_min': -np.pi/2,
            'valid_max': np.pi/2,
            'valid_range': [-np.pi/2, np.pi/2]
        },
        'Roll': {
            'units': 'radians',
            'standard_name': 'platform_roll_angle',
            'long_name': 'platform roll angle',
            'comment': 'Platform roll angle in radians, positive right wing down',
            'valid_min': -np.pi,
            'valid_max': np.pi,
            'valid_range': [-np.pi, np.pi]
        },
        'Surface': {
            'units': 's',
            'long_name': 'surface two-way travel time',
            'comment': 'Two-way travel time to detected surface. Zero indicates surface at platform level.',
            'valid_min': 0.0
        }
    }

    # Apply coordinate attributes
    for coord_name, attrs in coordinate_attrs.items():
        if coord_name in ds_cf.coords:
            ds_cf[coord_name].attrs.update(attrs)

    # Apply data variable attributes
    for var_name, attrs in data_var_attrs.items():
        if var_name in ds_cf.data_vars:
            ds_cf[var_name].attrs.update(attrs)

    # Add global attributes for CF compliance
    global_attrs = {
        'Conventions': 'CF-1.8',
        'title': 'Radar Echogram Data',
        'institution': 'Open Polar Radar (OPR)',
        'source': 'Airborne/ground-based radar sounder',
        'history': f'Converted to CF-compliant format on {np.datetime64("now").astype(str)}',
        'references': 'https://gitlab.com/openpolarradar/opr',
        'comment': 'Polar radar echogram data with CF-compliant metadata',
        'geospatial_lat_min': float(ds_cf.Latitude.min()) if 'Latitude' in ds_cf else None,
        'geospatial_lat_max': float(ds_cf.Latitude.max()) if 'Latitude' in ds_cf else None,
        'geospatial_lon_min': float(ds_cf.Longitude.min()) if 'Longitude' in ds_cf else None,
        'geospatial_lon_max': float(ds_cf.Longitude.max()) if 'Longitude' in ds_cf else None,
        'time_coverage_start': str(ds_cf.slow_time.min().values) if 'slow_time' in ds_cf else None,
        'time_coverage_end': str(ds_cf.slow_time.max().values) if 'slow_time' in ds_cf else None
    }

    # Remove None values from global attributes
    global_attrs = {k: v for k, v in global_attrs.items() if v is not None}

    # Update global attributes
    ds_cf.attrs.update(global_attrs)

    return ds_cf
