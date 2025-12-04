"""
Comparison functions for bedmap vs OPR layer data.

This module provides functions to match and compare bedmap ice thickness
measurements with OPR (Open Polar Radar) surface and bed layer picks.
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
from scipy.spatial import cKDTree
from typing import Optional, Tuple, Dict, List, Union
from datetime import datetime
import warnings


def _get_lon_lat_columns(df: pd.DataFrame) -> Tuple[str, str]:
    """
    Get the longitude and latitude column names from a DataFrame.

    Handles both old format ('longitude (degree_east)', 'latitude (degree_north)')
    and new GeoParquet format ('lon', 'lat').

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame to check

    Returns
    -------
    tuple
        (lon_col, lat_col) column names
    """
    if 'lon' in df.columns and 'lat' in df.columns:
        return 'lon', 'lat'
    elif 'longitude (degree_east)' in df.columns:
        return 'longitude (degree_east)', 'latitude (degree_north)'
    else:
        raise ValueError("Could not find longitude/latitude columns")


def match_bedmap_to_opr(
    bedmap_data: gpd.GeoDataFrame,
    opr_dataset: xr.Dataset,
    max_distance_m: float = 100.0,
    time_tolerance_days: Optional[float] = None
) -> pd.DataFrame:
    """
    Match bedmap points to nearest OPR measurements.

    Parameters
    ----------
    bedmap_data : geopandas.GeoDataFrame
        Bedmap data with lat/lon coordinates
    opr_dataset : xarray.Dataset
        OPR dataset with Longitude, Latitude coordinates
    max_distance_m : float
        Maximum matching distance in meters
    time_tolerance_days : float, optional
        Maximum time difference in days for matching

    Returns
    -------
    pandas.DataFrame
        Matched data with bedmap and OPR measurements
    """
    # Extract coordinates from bedmap (handle both column naming conventions)
    lon_col, lat_col = _get_lon_lat_columns(bedmap_data)
    bedmap_lons = bedmap_data[lon_col].values
    bedmap_lats = bedmap_data[lat_col].values

    # Extract coordinates from OPR
    opr_lons = opr_dataset['Longitude'].values
    opr_lats = opr_dataset['Latitude'].values

    # Convert to radians for distance calculation
    bedmap_lons_rad = np.radians(bedmap_lons)
    bedmap_lats_rad = np.radians(bedmap_lats)
    opr_lons_rad = np.radians(opr_lons)
    opr_lats_rad = np.radians(opr_lats)

    # Convert to Cartesian coordinates for KDTree
    # Using Earth radius in meters
    R = 6371000  # Earth radius in meters

    # Bedmap points in Cartesian
    bedmap_x = R * np.cos(bedmap_lats_rad) * np.cos(bedmap_lons_rad)
    bedmap_y = R * np.cos(bedmap_lats_rad) * np.sin(bedmap_lons_rad)
    bedmap_z = R * np.sin(bedmap_lats_rad)
    bedmap_xyz = np.vstack([bedmap_x, bedmap_y, bedmap_z]).T

    # OPR points in Cartesian
    opr_x = R * np.cos(opr_lats_rad) * np.cos(opr_lons_rad)
    opr_y = R * np.cos(opr_lats_rad) * np.sin(opr_lons_rad)
    opr_z = R * np.sin(opr_lats_rad)
    opr_xyz = np.vstack([opr_x, opr_y, opr_z]).T

    # Build KDTree for OPR points
    tree = cKDTree(opr_xyz)

    # Find nearest neighbors
    distances, indices = tree.query(bedmap_xyz, k=1)

    # Convert Cartesian distances to arc distances
    # Arc distance = 2 * R * arcsin(cartesian_distance / (2 * R))
    arc_distances = 2 * R * np.arcsin(np.minimum(distances / (2 * R), 1.0))

    # Create matched dataframe
    matched_data = bedmap_data.copy()

    # Add OPR matches
    matched_data['opr_match_index'] = indices
    matched_data['opr_match_distance_m'] = arc_distances
    matched_data['opr_longitude'] = opr_lons[indices]
    matched_data['opr_latitude'] = opr_lats[indices]

    # Extract OPR layer values if available
    if 'Surface' in opr_dataset:
        matched_data['opr_surface'] = opr_dataset['Surface'].values[indices]
    if 'Bottom' in opr_dataset:
        matched_data['opr_bottom'] = opr_dataset['Bottom'].values[indices]

    # Apply distance filter
    matched_data['is_matched'] = arc_distances <= max_distance_m

    # Apply time filter if specified
    if time_tolerance_days and 'timestamp' in bedmap_data.columns:
        if 'time' in opr_dataset.coords:
            opr_times = pd.to_datetime(opr_dataset['time'].values)
            bedmap_times = pd.to_datetime(bedmap_data['timestamp'])

            time_diffs = np.abs(
                (bedmap_times - opr_times[indices]).dt.total_seconds() / 86400
            )
            matched_data['time_diff_days'] = time_diffs
            matched_data['is_matched'] &= (time_diffs <= time_tolerance_days)

    return matched_data


def compare_with_opr(
    bedmap_data: gpd.GeoDataFrame,
    opr_surface: Optional[xr.DataArray] = None,
    opr_bed: Optional[xr.DataArray] = None,
    tolerance_m: float = 100.0,
    compute_statistics: bool = True
) -> Dict:
    """
    Compare bedmap measurements with OPR layer picks.

    Parameters
    ----------
    bedmap_data : geopandas.GeoDataFrame
        Bedmap data with surface and bed elevations
    opr_surface : xarray.DataArray, optional
        OPR surface elevation picks
    opr_bed : xarray.DataArray, optional
        OPR bed elevation picks
    tolerance_m : float
        Tolerance for matching in meters
    compute_statistics : bool
        Whether to compute comparison statistics

    Returns
    -------
    dict
        Dictionary containing comparison results and statistics
    """
    results = {
        'matched_data': None,
        'statistics': {},
        'differences': {}
    }

    # Validate input data
    if 'surface_altitude (m)' not in bedmap_data.columns:
        warnings.warn("Surface altitude not found in bedmap data")
    if 'bedrock_altitude (m)' not in bedmap_data.columns:
        warnings.warn("Bedrock altitude not found in bedmap data")

    # Calculate ice thickness from bedmap if not present
    if 'land_ice_thickness (m)' not in bedmap_data.columns:
        if ('surface_altitude (m)' in bedmap_data.columns and
            'bedrock_altitude (m)' in bedmap_data.columns):
            bedmap_data['land_ice_thickness (m)'] = (
                bedmap_data['surface_altitude (m)'] -
                bedmap_data['bedrock_altitude (m)']
            )

    # Get coordinate columns
    lon_col, lat_col = _get_lon_lat_columns(bedmap_data)

    # Compare with OPR surface if provided
    if opr_surface is not None:
        bedmap_surface = bedmap_data['surface_altitude (m)'].values
        opr_surface_matched = _interpolate_to_points(
            opr_surface,
            bedmap_data[lon_col].values,
            bedmap_data[lat_col].values
        )

        surface_diff = bedmap_surface - opr_surface_matched
        results['differences']['surface'] = surface_diff

        if compute_statistics:
            results['statistics']['surface'] = {
                'mean_diff': np.nanmean(surface_diff),
                'std_diff': np.nanstd(surface_diff),
                'median_diff': np.nanmedian(surface_diff),
                'rmse': np.sqrt(np.nanmean(surface_diff**2)),
                'n_valid': np.sum(~np.isnan(surface_diff))
            }

    # Compare with OPR bed if provided
    if opr_bed is not None:
        bedmap_bed = bedmap_data['bedrock_altitude (m)'].values
        opr_bed_matched = _interpolate_to_points(
            opr_bed,
            bedmap_data[lon_col].values,
            bedmap_data[lat_col].values
        )

        bed_diff = bedmap_bed - opr_bed_matched
        results['differences']['bed'] = bed_diff

        if compute_statistics:
            results['statistics']['bed'] = {
                'mean_diff': np.nanmean(bed_diff),
                'std_diff': np.nanstd(bed_diff),
                'median_diff': np.nanmedian(bed_diff),
                'rmse': np.sqrt(np.nanmean(bed_diff**2)),
                'n_valid': np.sum(~np.isnan(bed_diff))
            }

    # Compare ice thickness if both surface and bed are available
    if opr_surface is not None and opr_bed is not None:
        opr_thickness = opr_surface_matched - opr_bed_matched
        bedmap_thickness = bedmap_data['land_ice_thickness (m)'].values

        thickness_diff = bedmap_thickness - opr_thickness
        results['differences']['thickness'] = thickness_diff

        if compute_statistics:
            results['statistics']['thickness'] = {
                'mean_diff': np.nanmean(thickness_diff),
                'std_diff': np.nanstd(thickness_diff),
                'median_diff': np.nanmedian(thickness_diff),
                'rmse': np.sqrt(np.nanmean(thickness_diff**2)),
                'n_valid': np.sum(~np.isnan(thickness_diff)),
                'correlation': np.corrcoef(
                    bedmap_thickness[~np.isnan(thickness_diff)],
                    opr_thickness[~np.isnan(thickness_diff)]
                )[0, 1] if np.sum(~np.isnan(thickness_diff)) > 1 else np.nan
            }

    # Store matched data
    matched_df = bedmap_data.copy()
    if 'surface' in results['differences']:
        matched_df['surface_diff_m'] = results['differences']['surface']
    if 'bed' in results['differences']:
        matched_df['bed_diff_m'] = results['differences']['bed']
    if 'thickness' in results['differences']:
        matched_df['thickness_diff_m'] = results['differences']['thickness']

    results['matched_data'] = matched_df

    return results


def _interpolate_to_points(
    data_array: xr.DataArray,
    lons: np.ndarray,
    lats: np.ndarray,
    method: str = 'nearest'
) -> np.ndarray:
    """
    Interpolate DataArray values to point locations.

    Parameters
    ----------
    data_array : xarray.DataArray
        Data to interpolate from
    lons : numpy.ndarray
        Longitude coordinates
    lats : numpy.ndarray
        Latitude coordinates
    method : str
        Interpolation method ('nearest', 'linear')

    Returns
    -------
    numpy.ndarray
        Interpolated values at point locations
    """
    # Create dataset with target points
    target_ds = xr.Dataset({
        'lon': xr.DataArray(lons, dims='points'),
        'lat': xr.DataArray(lats, dims='points')
    })

    # Interpolate
    if 'Longitude' in data_array.dims and 'Latitude' in data_array.dims:
        interp_coords = {
            'Longitude': target_ds['lon'],
            'Latitude': target_ds['lat']
        }
    elif 'longitude' in data_array.dims and 'latitude' in data_array.dims:
        interp_coords = {
            'longitude': target_ds['lon'],
            'latitude': target_ds['lat']
        }
    else:
        warnings.warn("Could not find longitude/latitude dimensions")
        return np.full_like(lons, np.nan)

    try:
        interpolated = data_array.interp(interp_coords, method=method)
        return interpolated.values
    except Exception as e:
        warnings.warn(f"Interpolation failed: {e}")
        return np.full_like(lons, np.nan)


def aggregate_comparisons_by_region(
    comparison_results: Dict,
    regions: gpd.GeoDataFrame,
    region_name_col: str = 'name'
) -> pd.DataFrame:
    """
    Aggregate comparison statistics by geographic regions.

    Parameters
    ----------
    comparison_results : dict
        Results from compare_with_opr()
    regions : geopandas.GeoDataFrame
        Regions for aggregation
    region_name_col : str
        Column name for region identifier

    Returns
    -------
    pandas.DataFrame
        Statistics aggregated by region
    """
    matched_data = comparison_results['matched_data']

    if matched_data is None or matched_data.empty:
        return pd.DataFrame()

    # Ensure matched_data is a GeoDataFrame
    if not isinstance(matched_data, gpd.GeoDataFrame):
        if 'geometry' not in matched_data.columns:
            # Create point geometries (handle both column naming conventions)
            lon_col, lat_col = _get_lon_lat_columns(matched_data)
            matched_data = gpd.GeoDataFrame(
                matched_data,
                geometry=gpd.points_from_xy(
                    matched_data[lon_col],
                    matched_data[lat_col]
                ),
                crs='EPSG:4326'
            )

    # Spatial join with regions
    joined = gpd.sjoin(matched_data, regions, how='left', predicate='within')

    # Aggregate by region
    aggregated = []
    for region_name in regions[region_name_col].unique():
        region_data = joined[joined[region_name_col] == region_name]

        if region_data.empty:
            continue

        stats = {'region': region_name, 'n_points': len(region_data)}

        # Aggregate differences
        for diff_type in ['surface_diff_m', 'bed_diff_m', 'thickness_diff_m']:
            if diff_type in region_data.columns:
                valid_data = region_data[diff_type].dropna()
                if not valid_data.empty:
                    stats[f'{diff_type}_mean'] = valid_data.mean()
                    stats[f'{diff_type}_std'] = valid_data.std()
                    stats[f'{diff_type}_median'] = valid_data.median()
                    stats[f'{diff_type}_rmse'] = np.sqrt((valid_data**2).mean())

        aggregated.append(stats)

    return pd.DataFrame(aggregated)


def create_crossover_analysis(
    bedmap_data: gpd.GeoDataFrame,
    opr_tracks: List[xr.Dataset],
    crossover_threshold_m: float = 500.0,
    time_threshold_days: Optional[float] = None
) -> pd.DataFrame:
    """
    Analyze crossover points between bedmap and OPR tracks.

    Parameters
    ----------
    bedmap_data : geopandas.GeoDataFrame
        Bedmap data
    opr_tracks : list of xarray.Dataset
        List of OPR track datasets
    crossover_threshold_m : float
        Maximum distance for crossover detection
    time_threshold_days : float, optional
        Maximum time difference for valid crossovers

    Returns
    -------
    pandas.DataFrame
        Crossover analysis results
    """
    crossovers = []

    for i, opr_track in enumerate(opr_tracks):
        # Match bedmap to this OPR track
        matched = match_bedmap_to_opr(
            bedmap_data,
            opr_track,
            max_distance_m=crossover_threshold_m,
            time_tolerance_days=time_threshold_days
        )

        # Find valid matches
        valid_matches = matched[matched['is_matched']]

        if valid_matches.empty:
            continue

        # Get coordinate columns (handle both naming conventions)
        lon_col, lat_col = _get_lon_lat_columns(valid_matches)

        # Calculate differences at crossovers
        for _, row in valid_matches.iterrows():
            crossover = {
                'bedmap_id': row.get('source_file', ''),
                'opr_track': i,
                'longitude': row[lon_col],
                'latitude': row[lat_col],
                'distance_m': row['opr_match_distance_m'],
            }

            # Add elevation differences if available
            if 'opr_surface' in row and 'surface_altitude (m)' in row:
                crossover['surface_diff'] = (
                    row['surface_altitude (m)'] - row['opr_surface']
                )

            if 'opr_bottom' in row and 'bedrock_altitude (m)' in row:
                crossover['bed_diff'] = (
                    row['bedrock_altitude (m)'] - row['opr_bottom']
                )

            crossovers.append(crossover)

    return pd.DataFrame(crossovers)