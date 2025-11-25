import xarray as xr
import shapely
import numpy as np
import scipy.constants

from xopr.geometry import project_dataset

def add_along_track(ds: xr.Dataset, projection: str = None) -> xr.Dataset:
    """
    Add an along-track distance coordinate to the dataset based on the latitude and longitude coordinates.
    
    Parameters:
    ds (xr.Dataset): Input xarray Dataset with 'Latitude' and 'Longitude' coordinates.
    
    Returns:
    xr.Dataset: Dataset with an added 'along_track' coordinate.
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
    Estimate vertical distances from two-way travel time (TWTT) using the speed of light in ice.
    
    Parameters:
    ds (xr.Dataset): Input xarray Dataset with TWTT layers as variables.
    epsilon_ice (float): Relative permittivity of ice. Default is 3.15.
    
    Returns:
    xr.Dataset: Dataset with added vertical distance variables for each TWTT layer.
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
                                  vert_spacing: float = 10.0,
                                  epsilon_ice: float = 3.15) -> xr.Dataset:
    """
    Interpolate radar data from TWTT coordinates to regular vertical distance coordinates.
    
    Parameters:
    -----------
    ds : xr.Dataset
        Input dataset with 'Data' variable, 'along_track' coordinate, and 'Surface' variable
    vertical_coordinate : str
        The vertical coordinate to use for interpolation. 'range' will interpolate to the
        vertical range from the instrument. 'wgs84' will interpolate to WGS84 elevation
        using the 'Elevation' variable in the dataset. Default is 'range'.
    vert_min : float
        Minimum vertical distance in meters, if None, uses minimum from data
    vert_max : float
        Maximum vertical distance in meters, if None, uses maximum from data
    vert_spacing : float
        Vertical spacing in meters
    epsilon_ice : float
        Relative permittivity of ice (default 3.15)
    
    Returns:
    --------
    xr.Dataset
        Dataset with data interpolated to regular vertical distance grid
    """
    from scipy.interpolate import griddata
    
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
    from scipy.interpolate import interp1d
    
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
    Convert layer two-way travel time (TWTT) to range or elevation coordinates.
    
    Parameters:
    -----------
    layer_ds : xr.Dataset
        Dataset containing layer TWTT values
    surface_layer_ds : xr.Dataset
        Dataset containing surface layer TWTT values (typically layer 1)
    vertical_coordinate : str
        'range' for distance from aircraft or 'elevation'/'wgs84' for WGS84 elevation
    subsurface_dielectric_permittivity : float
        Dielectric permittivity for subsurface propagation (default 3.15 for ice)
    
    Returns:
    --------
    xr.Dataset
        Copy of layer_ds with added 'range' or 'wgs84' field containing layer positions
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