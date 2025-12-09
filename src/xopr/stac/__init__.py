"""
STAC catalog creation utilities for Open Polar Radar data.

This module provides tools for generating STAC (SpatioTemporal Asset Catalog)
metadata for OPR datasets, enabling spatial and temporal search capabilities
across radar campaigns and data products.
"""

from .catalog import (
    create_collection, create_item,
    create_items_from_flight_data,
    export_collection_to_parquet,
    build_catalog_from_parquet_metadata
)
from .config import load_config, save_config, validate_config
from .geometry import (
    build_collection_extent_and_geometry,
    simplify_geometry_polar_projection
)
from .metadata import extract_item_metadata, discover_campaigns, discover_flight_lines, collect_uniform_metadata

__all__ = [
    # Configuration
    "load_config",
    "save_config",
    "validate_config",
    # Catalog functions
    "create_collection",
    "create_item",
    "create_items_from_flight_data",
    "export_collection_to_parquet",
    "build_catalog_from_parquet_metadata",
    # Metadata functions
    "extract_item_metadata",
    "discover_campaigns",
    "discover_flight_lines",
    "collect_uniform_metadata",
    # Geometry functions
    "build_collection_extent_and_geometry",
    "simplify_geometry_polar_projection"
]
