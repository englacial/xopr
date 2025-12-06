"""
Bedmap data integration module for xopr.

This module provides functionality to:
- Convert bedmap CSV files to cloud-optimized GeoParquet format
- Create GeoParquet STAC catalogs for bedmap data discovery
- Query and retrieve bedmap data efficiently
"""

from .converter import (
    convert_bedmap_csv,
    batch_convert_bedmap,
    parse_bedmap_metadata,
)

from .geometry import (
    extract_flight_lines,
    calculate_haversine_distances,
    simplify_multiline_geometry,
    calculate_bbox,
    get_polar_bounds,
    check_intersects_polar,
    # Expose transformers for direct use
    _transformer_to_polar,
    _transformer_from_polar,
)

from .catalog import (
    read_parquet_metadata,
    build_bedmap_geoparquet_catalog,
)

from .query import (
    query_bedmap,
    query_bedmap_catalog,
    query_bedmap_local,
    fetch_bedmap,
    get_bedmap_cache_path,
)

__all__ = [
    # Converter functions
    'convert_bedmap_csv',
    'batch_convert_bedmap',
    'parse_bedmap_metadata',
    # Geometry functions
    'extract_flight_lines',
    'calculate_haversine_distances',
    'simplify_multiline_geometry',
    'calculate_bbox',
    'get_polar_bounds',
    'check_intersects_polar',
    '_transformer_to_polar',
    '_transformer_from_polar',
    # Catalog functions
    'read_parquet_metadata',
    'build_bedmap_geoparquet_catalog',
    # Query functions
    'query_bedmap',
    'query_bedmap_catalog',
    'query_bedmap_local',
    'fetch_bedmap',
    'get_bedmap_cache_path',
]

__version__ = '0.1.0'
