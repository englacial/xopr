"""
Geometry utilities for bedmap data processing.

This module provides functions for:
- Calculating haversine distances between points
- Extracting flight lines from point data
- Simplifying geometries for STAC catalog storage
"""

import numpy as np
import pandas as pd
import shapely
from shapely.geometry import LineString, MultiLineString, Point
from typing import Tuple, List, Optional
from haversine import haversine_vector, Unit


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

    # Haversine expects (lat, lon) tuples
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

    # Return single LineString if only one segment, otherwise MultiLineString
    if len(segments) == 1:
        return MultiLineString(segments)

    return MultiLineString(segments)


def simplify_multiline_geometry(
    geometry: MultiLineString,
    tolerance_deg: float = 0.01,
    preserve_topology: bool = True
) -> MultiLineString:
    """
    Simplify multiline geometry to reduce storage size.

    Uses the Douglas-Peucker algorithm to reduce the number of vertices
    while preserving the overall shape of the flight lines.

    Parameters
    ----------
    geometry : MultiLineString
        Input multiline geometry to simplify
    tolerance_deg : float
        Simplification tolerance in degrees (larger values = more simplification)
    preserve_topology : bool
        Whether to preserve topology during simplification

    Returns
    -------
    MultiLineString
        Simplified multiline geometry
    """
    if geometry is None or geometry.is_empty:
        return geometry

    # Simplify the geometry
    simplified = geometry.simplify(tolerance_deg, preserve_topology=preserve_topology)

    # Ensure we still have a MultiLineString
    if isinstance(simplified, LineString):
        simplified = MultiLineString([simplified])

    return simplified


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

    min_lon = valid_coords[lon_col].min()
    max_lon = valid_coords[lon_col].max()
    min_lat = valid_coords[lat_col].min()
    max_lat = valid_coords[lat_col].max()

    return (min_lon, min_lat, max_lon, max_lat)


def create_bbox_polygon(
    bbox: Tuple[float, float, float, float]
) -> shapely.geometry.Polygon:
    """
    Create a polygon from a bounding box.

    Parameters
    ----------
    bbox : tuple
        Bounding box as (min_lon, min_lat, max_lon, max_lat)

    Returns
    -------
    shapely.geometry.Polygon
        Polygon representing the bounding box
    """
    if bbox is None:
        return None

    min_lon, min_lat, max_lon, max_lat = bbox

    return shapely.geometry.box(min_lon, min_lat, max_lon, max_lat)


def get_geometry_wkt(geometry: shapely.geometry.base.BaseGeometry) -> str:
    """
    Convert geometry to Well-Known Text (WKT) format.

    Parameters
    ----------
    geometry : shapely.geometry.base.BaseGeometry
        Input geometry

    Returns
    -------
    str
        WKT representation of the geometry
    """
    if geometry is None:
        return None

    return geometry.wkt


def get_geometry_bounds(geometry: shapely.geometry.base.BaseGeometry) -> Tuple[float, float, float, float]:
    """
    Get bounds of a geometry.

    Parameters
    ----------
    geometry : shapely.geometry.base.BaseGeometry
        Input geometry

    Returns
    -------
    tuple
        Bounds as (min_x, min_y, max_x, max_y)
    """
    if geometry is None or geometry.is_empty:
        return None

    return geometry.bounds