"""
Bedmap data integration module for xopr.

This module provides functionality to:
- Convert bedmap CSV files to cloud-optimized GeoParquet format
- Create STAC catalogs for bedmap data discovery
- Query and retrieve bedmap data efficiently
- Compare bedmap data with OPR layer picks
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
)

from .catalog import (
    create_bedmap_stac_item,
    build_bedmap_catalog,
)

from .query import (
    query_bedmap,
    query_bedmap_stac,
    query_bedmap_local,
)

from .compare import (
    compare_with_opr,
    match_bedmap_to_opr,
    aggregate_comparisons_by_region,
    create_crossover_analysis,
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
    # Catalog functions
    'create_bedmap_stac_item',
    'build_bedmap_catalog',
    # Query functions
    'query_bedmap',
    'query_bedmap_stac',
    'query_bedmap_local',
    # Comparison functions
    'compare_with_opr',
    'match_bedmap_to_opr',
    'aggregate_comparisons_by_region',
    'create_crossover_analysis',
]

__version__ = '0.1.0'