"""
Geometry operations for STAC catalog creation.

This module contains functions for handling spatial geometries including
simplification, merging, and projection transformations for polar data.
"""

from typing import List
import shapely
import pyproj
from shapely.ops import transform
import pystac


def simplify_geometry_polar_projection(
    geometry: shapely.geometry.base.BaseGeometry,
    simplify_tolerance: float = 100.0
) -> shapely.geometry.base.BaseGeometry:
    """
    Simplify geometry using appropriate polar stereographic projection.

    Parameters
    ----------
    geometry : shapely.geometry.base.BaseGeometry
        Input geometry in WGS84 coordinates
    simplify_tolerance : float, default 100.0
        Tolerance for shapely.simplify() in meters (used in polar projection)

    Returns
    -------
    shapely.geometry.base.BaseGeometry
        Simplified geometry in WGS84 coordinates
    """
    if not geometry or not geometry.is_valid:
        return geometry

    # Determine appropriate polar projection based on geometry centroid
    centroid = geometry.centroid
    lat = centroid.y

    if lat < 0:
        # Antarctic/South Polar Stereographic
        target_epsg = 3031
    else:
        # Arctic/North Polar Stereographic
        target_epsg = 3413

    # Set up coordinate transformations
    wgs84 = pyproj.CRS('EPSG:4326')
    polar_proj = pyproj.CRS(f'EPSG:{target_epsg}')

    # Transform to polar projection
    transformer_to_polar = pyproj.Transformer.from_crs(wgs84, polar_proj, always_xy=True)
    transformer_to_wgs84 = pyproj.Transformer.from_crs(polar_proj, wgs84, always_xy=True)

    # Project to polar coordinates
    projected_geom = transform(transformer_to_polar.transform, geometry)

    # Simplify in projected coordinates (tolerance in meters)
    simplified_geom = projected_geom.simplify(simplify_tolerance, preserve_topology=True)

    # Transform back to WGS84
    return transform(transformer_to_wgs84.transform, simplified_geom)


def build_collection_extent_and_geometry(
    items: List[pystac.Item]
) -> pystac.Extent:
    """
    Calculate spatial and temporal extent from a list of items.

    Parameters
    ----------
    items : list of pystac.Item
        List of STAC items to compute extent from.

    Returns
    -------
    pystac.Extent
        Combined spatial and temporal extent covering all input items.

    Raises
    ------
    ValueError
        If items list is empty.
    """
    if not items:
        raise ValueError("Cannot build extent from empty item list")

    # Build extent using bboxes
    bboxes = []
    datetimes = []

    for item in items:
        if item.bbox:
            bbox_geom = shapely.geometry.box(*item.bbox)
            bboxes.append(bbox_geom)

        if item.datetime:
            datetimes.append(item.datetime)

    if bboxes:
        union_bbox = bboxes[0]
        for bbox in bboxes[1:]:
            union_bbox = union_bbox.union(bbox)

        collection_bbox = list(union_bbox.bounds)
        spatial_extent = pystac.SpatialExtent(bboxes=[collection_bbox])
    else:
        spatial_extent = pystac.SpatialExtent(bboxes=[[-180, -90, 180, 90]])

    if datetimes:
        sorted_times = sorted(datetimes)
        temporal_extent = pystac.TemporalExtent(
            intervals=[[sorted_times[0], sorted_times[-1]]]
        )
    else:
        temporal_extent = pystac.TemporalExtent(intervals=[[None, None]])

    extent = pystac.Extent(spatial=spatial_extent, temporal=temporal_extent)
    return extent
