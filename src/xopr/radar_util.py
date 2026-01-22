"""
Processing and transformation utilities for radar datasets.

This module provides functions for common radar data processing tasks including
coordinate transformations, along-track distance calculations, and vertical
coordinate conversions between two-way travel time (TWTT), range, and elevation.

"""

import xarray as xr
import numpy as np
import scipy.constants
import xarray as xr

from xopr.geometry import project_dataset


def add_along_track(ds: xr.Dataset, projection: str = None) -> xr.Dataset:
    """
    Add cumulative along-track distance coordinate.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset with Latitude and Longitude coordinates.
    projection : str, optional
        CRS for distance calculation. If None, uses EPSG:3031 (Antarctic)
        or EPSG:3413 (Arctic) based on mean latitude.

    Returns
    -------
    xr.Dataset
        Dataset with added along_track coordinate in meters.
    """

    if 'Latitude' not in ds or 'Longitude' not in ds:
        if 'lat' in ds and 'lon' in ds:
            ds = ds.rename({'lat': 'Latitude', 'lon': 'Longitude'})
        else:
            raise ValueError("Dataset must contain 'Latitude' and 'Longitude' or 'lat' and 'lon' coordinates.")

    # Project the dataset to the specified projection
    if projection is None:
        if ds['Latitude'].mean() < 0:
            projection = "EPSG:3031"  # Antarctic Polar Stereographic
        else:
            projection = "EPSG:3413"  # Arctic Polar Stereographic
    projected_ds = project_dataset(ds, target_crs=projection)

    # Calculate differences between consecutive points
    dx = projected_ds['x'].diff(dim='slow_time', label='upper').to_numpy()
    dy = projected_ds['y'].diff(dim='slow_time', label='upper').to_numpy()

    # Calculate incremental distances
    distances = (dx**2 + dy**2)**0.5
    # Add a zero at the start to align with slow_time
    distances = np.insert(distances, 0, 0)

    # Calculate cumulative distance along track
    along_track = np.cumsum(distances)

    # Add the along-track coordinate to the original dataset
    ds = ds.assign_coords(along_track=('slow_time', along_track))
    ds['along_track'].attrs['units'] = 'meters'
    ds['along_track'].attrs['description'] = 'Cumulative distance along the radar track'

    return ds

def estimate_vertical_distances(ds: xr.Dataset, epsilon_ice: float = 3.15) -> xr.Dataset:
    """
    Convert TWTT (two-way travel time) to vertical distances accounting for propagation speed.

    Uses speed of light in air above surface and speed of light in ice below.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset with Data, twtt, and Surface variables.
    epsilon_ice : float, optional
        Relative permittivity of ice for subsurface propagation.

    Returns
    -------
    xr.DataArray
        Vertical distance from aircraft in meters.
    """

    v_ice = scipy.constants.c / np.sqrt(epsilon_ice)  # Speed of light in ice (m/s)

    # Initialize local_speed with dimensions (slow_time, twtt) to always be scipy.constants.c
    local_speed = xr.full_like(ds['Data'], scipy.constants.c)

    # Where twtt (a 1D dimension) > ds['Surface'] (data variable with dimension slow_time), set local_speed to v_ice
    # Broadcast comparison: expand Surface to match Data dimensions
    surface_broadcast = ds['Surface'].broadcast_like(ds['Data'])
    twtt_broadcast = ds['twtt'].broadcast_like(ds['Data'])
    local_speed = xr.where(twtt_broadcast > surface_broadcast, v_ice, scipy.constants.c)

    # Multiply against the differences in the twtt dimension to get the distance intervals
    twtt_intervals = np.diff(ds['twtt'])
    twtt_intervals = np.insert(twtt_intervals, 0, ds['twtt'].isel(twtt=0))  # Add the first interval
    twtt_intervals = xr.DataArray(twtt_intervals, dims=['twtt'], coords={'twtt': ds['twtt']})

    # Calculate distance for each interval (one-way distance = speed * time / 2)
    distance_intervals = local_speed * twtt_intervals / 2

    # Cumulatively sum the distance intervals to get the vertical distance
    vertical_distance = distance_intervals.cumsum(dim='twtt')
    vertical_distance.name = 'vertical_distance'
    vertical_distance.attrs['units'] = 'meters'
    vertical_distance.attrs['description'] = 'Vertical distance from aircraft calculated from TWTT'

    return vertical_distance


def interpolate_to_vertical_grid(ds: xr.Dataset,
                                  vertical_coordinate: str = 'range',
                                  vert_min: float = None,
                                  vert_max: float = None,
                                  vert_min: float = None,
                                  vert_max: float = None,
                                  vert_spacing: float = 10.0,
                                  epsilon_ice: float = 3.15) -> xr.Dataset:
    """
    Interpolate radar data to regular vertical grid.

    Converts from irregular TWTT spacing to uniform range or elevation grid.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset with Data, Surface, and optionally Elevation variables.
    vertical_coordinate : {'range', 'wgs84'}, optional
        Target vertical coordinate system.
    vert_min : float, optional
        Minimum vertical distance in meters. If None, uses data minimum.
    vert_max : float, optional
        Maximum vertical distance in meters. If None, uses data maximum.
    vert_spacing : float, optional
        Vertical grid spacing in meters.
    epsilon_ice : float, optional
        Relative permittivity of ice.

    Returns
    -------
    xr.Dataset
        Dataset with Data interpolated to regular vertical grid with coordinate
        'range' or 'wgs84'.
    """

    # Calculate vertical distances
    vert_dist = estimate_vertical_distances(ds, epsilon_ice)

    vert_coord_name = 'range'

    if vertical_coordinate == 'wgs84':
        if 'Elevation' not in ds:
            raise ValueError("Dataset must contain 'Elevation' variable to use elevation as vertical coordinate.")
        vert_dist = ds['Elevation'].broadcast_like(vert_dist) - vert_dist
        vert_coord_name = 'wgs84'
    elif vertical_coordinate != 'range':
        raise ValueError("vertical_coordinate must be either 'range' or 'wgs84'")

    if vert_min is None:
        vert_min = float(vert_dist.min().values)
    if vert_max is None:
        vert_max = float(vert_dist.max().values)

    # Create regular vertical distance grid
    regular_vert = np.arange(vert_min, vert_max, vert_spacing)

    # Use 1D interpolation along each trace (much faster than 2D griddata)

    n_traces = len(ds['slow_time'])
    n_vert = len(regular_vert)
    data_regular = np.full((n_traces, n_vert), np.nan, dtype=np.float32)

    # Interpolate each trace individually
    for i in range(n_traces):
        trace_data = ds['Data'].isel(slow_time=i).values
        trace_vert = vert_dist.isel(slow_time=i).values

        if vertical_coordinate == 'wgs84':
            trace_data = trace_data[::-1]
            trace_vert = trace_vert[::-1]

        # Remove NaN values for this trace
        valid_idx = ~(np.isnan(trace_data) | np.isnan(trace_vert))

        if not np.all(np.diff(trace_vert[valid_idx]) > 0):
            raise ValueError("Vertical distances must be strictly increasing for interpolation.")

        if np.sum(valid_idx) > 1:  # Need at least 2 points for interpolation
            data_regular[i, :] = np.interp(regular_vert, trace_vert[valid_idx],
                                                    trace_data[valid_idx],
                                                    left=-1, right=-2)

    # Create new dataset
    ds_regular = xr.Dataset(
        {
            'Data': (('slow_time', vert_coord_name), data_regular),
        },
        coords={
            'slow_time': ds['slow_time'],
            vert_coord_name: regular_vert,
        }
    )

    if 'along_track' in ds:
        along_track = ds['along_track'].values
        ds_regular = ds_regular.assign_coords(along_track=('slow_time', along_track))

    for data_var in ds.data_vars:
        if data_var not in ['Data']:
            ds_regular[data_var] = ds[data_var]

    # Copy relevant attributes
    ds_regular.attrs = ds.attrs.copy()
    ds_regular[vert_coord_name].attrs['units'] = 'meters'
    if vertical_coordinate == 'range':
        ds_regular[vert_coord_name].attrs['description'] = 'Vertical distance from aircraft (positive down)'
    else:
        ds_regular[vert_coord_name].attrs['description'] = 'WGS84 Elevation (meters)'

    return ds_regular

def layer_twtt_to_range(layer_ds, surface_layer_ds, vertical_coordinate='range', subsurface_dielectric_permittivity=3.15):
    """
    Convert layer TWTT to range or elevation coordinates.

    Parameters
    ----------
    layer_ds : xr.Dataset
        Layer dataset with twtt variable.
    surface_layer_ds : xr.Dataset
        Surface layer dataset with twtt and optionally elev variables.
    vertical_coordinate : {'range', 'elevation', 'wgs84'}, optional
        Target coordinate system.
    subsurface_dielectric_permittivity : float, optional
        Dielectric permittivity for subsurface propagation.

    Returns
    -------
    xr.Dataset
        Copy of layer_ds with added 'range' or 'wgs84' variable.
    """
    # Create a copy of the layer dataset
    result_ds = layer_ds.copy()

    # Calculate speed of light in the subsurface medium
    speed_in_medium = scipy.constants.c / np.sqrt(subsurface_dielectric_permittivity)

    # Get TWTT values
    layer_twtt = layer_ds['twtt']
    surface_twtt = surface_layer_ds['twtt']

    # Calculate surface range (distance from aircraft to surface)
    surface_range = surface_twtt * (scipy.constants.c / 2)

    # Calculate TWTT difference from surface to layer
    twtt_from_surface = layer_twtt - surface_twtt

    # Calculate range from aircraft to layer
    layer_range = surface_range + (twtt_from_surface * (speed_in_medium / 2))

    if vertical_coordinate == 'range':
        result_ds['range'] = layer_range
        result_ds['range'].attrs['units'] = 'meters'
        result_ds['range'].attrs['description'] = 'Range from aircraft to layer'
    elif vertical_coordinate == 'elevation' or vertical_coordinate == 'wgs84':
        # Calculate WGS84 elevation
        # Surface elevation = aircraft elevation - surface range
        if 'elev' in surface_layer_ds:
            surface_elev = surface_layer_ds['elev']
        else:
            raise ValueError("Surface elevation data ('elev') required for elevation coordinate conversion")

        surface_wgs84 = surface_elev - surface_range

        # Layer elevation = surface elevation - distance from surface to layer
        layer_wgs84 = surface_wgs84 - (twtt_from_surface * (speed_in_medium / 2))

        result_ds['wgs84'] = layer_wgs84
        result_ds['wgs84'].attrs['units'] = 'meters'
        result_ds['wgs84'].attrs['description'] = 'WGS84 elevation of layer'
    else:
        raise ValueError(f"Unknown vertical coordinate: {vertical_coordinate}. Use 'range', 'elevation', or 'wgs84'.")

    return result_ds
