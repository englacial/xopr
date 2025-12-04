"""
Geometry utilities for bedmap data processing.

This module provides functions for:
- Calculating haversine distances between points
- Extracting flight lines from point data
- Simplifying geometries for STAC catalog storage
- Polar projection transformations for Antarctic data
"""

import numpy as np
import pandas as pd
import shapely
from shapely.geometry import LineString, MultiLineString
from shapely.ops import transform as shapely_transform
from typing import Tuple, Optional, Union
from haversine import haversine_vector, Unit
from pyproj import Transformer, CRS

# Antarctic Polar Stereographic projection (EPSG:3031)
# This projection is centered on the South Pole and eliminates antimeridian issues
ANTARCTIC_CRS = CRS.from_epsg(3031)
WGS84_CRS = CRS.from_epsg(4326)

# Create transformers (cached for performance)
_transformer_to_polar = Transformer.from_crs(WGS84_CRS, ANTARCTIC_CRS, always_xy=True)
_transformer_from_polar = Transformer.from_crs(ANTARCTIC_CRS, WGS84_CRS, always_xy=True)


def calculate_haversine_distances(
    lat_lon_array: np.ndarray,
    unit: Unit = Unit.KILOMETERS
) -> np.ndarray:
    """
    Calculate haversine distances between consecutive points.

    Parameters
    ----------
    lat_lon_array : np.ndarray
        Array of shape (n, 2) with latitude in column 0, longitude in column 1
    unit : haversine.Unit
        Unit for distance calculation (default: KILOMETERS)

    Returns
    -------
    np.ndarray
        Array of distances between consecutive points (length n-1)
    """
    if len(lat_lon_array) < 2:
        return np.array([])

    coords = lat_lon_array
    coords_shifted = np.roll(coords, -1, axis=0)

    # Calculate distances, excluding the last wrap-around distance
    distances = haversine_vector(coords[:-1], coords_shifted[:-1], unit)

    return distances


def extract_flight_lines(
    df: pd.DataFrame,
    lon_col: str = 'longitude (degree_east)',
    lat_col: str = 'latitude (degree_north)',
    distance_threshold_km: float = 10.0,
    min_points_per_segment: int = 2
) -> Optional[MultiLineString]:
    """
    Extract flight line segments from point data.

    Breaks lines when consecutive points are more than threshold distance apart,
    indicating a gap in coverage or transition between flight lines.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing latitude and longitude columns
    lon_col : str
        Name of longitude column
    lat_col : str
        Name of latitude column
    distance_threshold_km : float
        Distance threshold in kilometers for line segmentation
    min_points_per_segment : int
        Minimum points required to create a line segment

    Returns
    -------
    MultiLineString or None
        Multiline geometry representing flight paths, or None if no valid segments
    """
    # Remove rows with missing coordinates
    valid_coords = df[[lat_col, lon_col]].dropna()

    if len(valid_coords) < min_points_per_segment:
        return None

    # Convert to numpy array (lat, lon order for haversine)
    coords_array = valid_coords[[lat_col, lon_col]].values

    # Calculate distances between consecutive points
    distances = calculate_haversine_distances(coords_array, Unit.KILOMETERS)

    # Find breakpoints where distance exceeds threshold
    breakpoints = np.where(distances > distance_threshold_km)[0] + 1

    # Add start and end indices
    breakpoints = np.concatenate([[0], breakpoints, [len(coords_array)]])

    # Create line segments
    segments = []
    for i in range(len(breakpoints) - 1):
        start_idx = breakpoints[i]
        end_idx = breakpoints[i + 1]

        segment_length = end_idx - start_idx
        if segment_length >= min_points_per_segment:
            # Extract segment coordinates (lon, lat order for shapely)
            segment_coords = coords_array[start_idx:end_idx]
            # Swap to lon, lat for shapely
            segment_coords_lonlat = segment_coords[:, [1, 0]]

            # Create LineString
            try:
                line = LineString(segment_coords_lonlat)
                if line.is_valid and not line.is_empty:
                    segments.append(line)
            except Exception as e:
                print(f"Warning: Could not create line segment: {e}")
                continue

    if not segments:
        return None

    return MultiLineString(segments)


def simplify_multiline_geometry(
    geometry: MultiLineString,
    tolerance_km: float = 10.0,
    preserve_topology: bool = False
) -> MultiLineString:
    """
    Simplify multiline geometry to reduce storage size.

    Uses vectorized pyproj transformation to Antarctic Polar Stereographic
    (EPSG:3031) for simplification, avoiding slow coordinate-by-coordinate
    iteration.

    Parameters
    ----------
    geometry : MultiLineString
        Input multiline geometry to simplify (WGS84)
    tolerance_km : float
        Simplification tolerance in kilometers
    preserve_topology : bool
        Whether to preserve topology during simplification

    Returns
    -------
    MultiLineString
        Simplified multiline geometry (WGS84)
    """
    if geometry is None or geometry.is_empty:
        return geometry

    tolerance_m = tolerance_km * 1000

    # Process each line segment with vectorized transformation
    simplified_lines = []
    for line in geometry.geoms:
        coords = np.array(line.coords)
        if len(coords) < 2:
            continue

        # Vectorized transform to polar (lon, lat -> x, y)
        lons, lats = coords[:, 0], coords[:, 1]
        x, y = _transformer_to_polar.transform(lons, lats)

        # Build polar LineString and simplify
        polar_line = LineString(zip(x, y))
        simplified_polar = polar_line.simplify(tolerance_m, preserve_topology=preserve_topology)

        if simplified_polar.is_empty:
            continue

        # Vectorized transform back to WGS84
        polar_coords = np.array(simplified_polar.coords)
        px, py = polar_coords[:, 0], polar_coords[:, 1]
        result_lons, result_lats = _transformer_from_polar.transform(px, py)

        result_line = LineString(zip(result_lons, result_lats))
        if result_line.is_valid and not result_line.is_empty:
            simplified_lines.append(result_line)

    if not simplified_lines:
        return None

    if len(simplified_lines) == 1:
        return MultiLineString([simplified_lines[0]])

    return MultiLineString(simplified_lines)


def calculate_bbox(
    df: pd.DataFrame,
    lon_col: str = 'longitude (degree_east)',
    lat_col: str = 'latitude (degree_north)'
) -> Tuple[float, float, float, float]:
    """
    Calculate bounding box from coordinate data.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing coordinate columns
    lon_col : str
        Name of longitude column
    lat_col : str
        Name of latitude column

    Returns
    -------
    tuple
        Bounding box as (min_lon, min_lat, max_lon, max_lat)
    """
    valid_coords = df[[lon_col, lat_col]].dropna()

    if valid_coords.empty:
        return None

    return (
        valid_coords[lon_col].min(),
        valid_coords[lat_col].min(),
        valid_coords[lon_col].max(),
        valid_coords[lat_col].max()
    )


# =============================================================================
# Polar Projection Functions (EPSG:3031 - Antarctic Polar Stereographic)
# =============================================================================

def transform_coords_to_polar(
    lon: Union[float, np.ndarray],
    lat: Union[float, np.ndarray]
) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
    """
    Transform WGS84 coordinates to Antarctic Polar Stereographic (EPSG:3031).

    Parameters
    ----------
    lon : float or np.ndarray
        Longitude(s) in degrees (WGS84)
    lat : float or np.ndarray
        Latitude(s) in degrees (WGS84)

    Returns
    -------
    tuple
        (x, y) coordinates in EPSG:3031 (meters)
    """
    return _transformer_to_polar.transform(lon, lat)


def transform_coords_from_polar(
    x: Union[float, np.ndarray],
    y: Union[float, np.ndarray]
) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
    """
    Transform Antarctic Polar Stereographic (EPSG:3031) to WGS84.

    Parameters
    ----------
    x : float or np.ndarray
        X coordinate(s) in EPSG:3031 (meters)
    y : float or np.ndarray
        Y coordinate(s) in EPSG:3031 (meters)

    Returns
    -------
    tuple
        (lon, lat) coordinates in WGS84 (degrees)
    """
    return _transformer_from_polar.transform(x, y)


def transform_geometry_to_polar(
    geometry: shapely.geometry.base.BaseGeometry
) -> shapely.geometry.base.BaseGeometry:
    """
    Transform a shapely geometry from WGS84 to Antarctic Polar Stereographic.

    Parameters
    ----------
    geometry : shapely.geometry.base.BaseGeometry
        Input geometry in WGS84 (EPSG:4326)

    Returns
    -------
    shapely.geometry.base.BaseGeometry
        Geometry in EPSG:3031
    """
    if geometry is None or geometry.is_empty:
        return geometry

    return shapely_transform(_transformer_to_polar.transform, geometry)


def transform_geometry_from_polar(
    geometry: shapely.geometry.base.BaseGeometry
) -> shapely.geometry.base.BaseGeometry:
    """
    Transform a shapely geometry from Antarctic Polar Stereographic to WGS84.

    Parameters
    ----------
    geometry : shapely.geometry.base.BaseGeometry
        Input geometry in EPSG:3031

    Returns
    -------
    shapely.geometry.base.BaseGeometry
        Geometry in WGS84 (EPSG:4326)
    """
    if geometry is None or geometry.is_empty:
        return geometry

    return shapely_transform(_transformer_from_polar.transform, geometry)


def get_polar_bounds(
    geometry: shapely.geometry.base.BaseGeometry
) -> Tuple[float, float, float, float]:
    """
    Get bounds of a WGS84 geometry in Antarctic Polar Stereographic projection.

    This is useful for spatial queries because rectangular bounds in
    polar projection don't have antimeridian issues.

    Parameters
    ----------
    geometry : shapely.geometry.base.BaseGeometry
        Input geometry in WGS84 (EPSG:4326)

    Returns
    -------
    tuple
        Bounds as (min_x, min_y, max_x, max_y) in EPSG:3031 (meters)
    """
    if geometry is None or geometry.is_empty:
        return None

    polar_geom = transform_geometry_to_polar(geometry)
    return polar_geom.bounds


def check_intersects_polar(
    geometry1: shapely.geometry.base.BaseGeometry,
    geometry2: shapely.geometry.base.BaseGeometry
) -> bool:
    """
    Check if two geometries intersect using polar projection.

    This avoids antimeridian crossing issues by transforming both
    geometries to EPSG:3031 before checking intersection.

    Parameters
    ----------
    geometry1 : shapely.geometry.base.BaseGeometry
        First geometry in WGS84
    geometry2 : shapely.geometry.base.BaseGeometry
        Second geometry in WGS84

    Returns
    -------
    bool
        True if geometries intersect
    """
    if geometry1 is None or geometry2 is None:
        return False
    if geometry1.is_empty or geometry2.is_empty:
        return False

    polar1 = transform_geometry_to_polar(geometry1)
    polar2 = transform_geometry_to_polar(geometry2)

    return polar1.intersects(polar2)
