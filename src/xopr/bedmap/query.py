"""
Query interface for bedmap data.

This module provides functions to query and retrieve bedmap data efficiently
using STAC catalogs and DuckDB for partial reads from cloud-optimized GeoParquet files.
"""

import duckdb
import pandas as pd
import geopandas as gpd
from typing import Optional, List, Dict, Tuple, Union
from datetime import datetime
from pathlib import Path
import warnings
import json

import pystac
from pystac import Catalog, Item
import shapely
from shapely.geometry import shape, box
from rustac import DuckdbClient


def query_bedmap_stac(
    stac_catalog_path: str = 'gs://opr_stac/bedmap/catalog/',
    collections: List[str] = None,
    geometry: Optional[shapely.geometry.base.BaseGeometry] = None,
    date_range: Optional[Tuple[datetime, datetime]] = None,
    properties: Dict = {},
    max_items: Optional[int] = None
) -> List[pystac.Item]:
    """
    Query STAC catalog for bedmap items.

    Parameters
    ----------
    stac_catalog_path : str
        Path to STAC catalog (local or cloud)
    collections : list of str, optional
        Filter by bedmap version collections (e.g., ['bedmap-bm2', 'bedmap-bm3'])
    geometry : shapely geometry, optional
        Spatial filter geometry
    date_range : tuple of datetime, optional
        Temporal filter (start_date, end_date)
    properties : dict, optional
        Additional property filters
    max_items : int, optional
        Maximum number of items to return

    Returns
    -------
    list of pystac.Item
        List of matching STAC items
    """
    # Load STAC catalog
    if stac_catalog_path.startswith(('gs://', 's3://', 'http')):
        # Remote catalog
        catalog = pystac.Catalog.from_file(stac_catalog_path + 'catalog.json')
    else:
        # Local catalog
        catalog_path = Path(stac_catalog_path) / 'catalog.json'
        if catalog_path.exists():
            catalog = pystac.Catalog.from_file(str(catalog_path))
        else:
            warnings.warn(f"STAC catalog not found at {catalog_path}")
            return []

    matching_items = []

    # Iterate through collections
    for collection in catalog.get_collections():
        # Filter by collection if specified
        if collections and collection.id not in collections:
            continue

        # Iterate through items in collection
        for item in collection.get_items():
            # Check spatial filter
            if geometry and item.geometry:
                item_shape = shape(item.geometry)
                if not geometry.intersects(item_shape):
                    continue

            # Check temporal filter
            if date_range:
                start_filter, end_filter = date_range
                item_start = item.properties.get('start_datetime') or item.datetime
                item_end = item.properties.get('end_datetime') or item.datetime

                if item_start and item_end:
                    # Convert strings to datetime if needed
                    if isinstance(item_start, str):
                        item_start = datetime.fromisoformat(item_start.replace('Z', '+00:00'))
                    if isinstance(item_end, str):
                        item_end = datetime.fromisoformat(item_end.replace('Z', '+00:00'))

                    # Check overlap
                    if end_filter < item_start or start_filter > item_end:
                        continue

            # Check property filters
            if properties:
                match = True
                for key, value in properties.items():
                    if item.properties.get(key) != value:
                        match = False
                        break
                if not match:
                    continue

            matching_items.append(item)

            # Check max_items limit
            if max_items and len(matching_items) >= max_items:
                return matching_items

    return matching_items


def build_duckdb_query(
    parquet_urls: List[str],
    geometry: Optional[shapely.geometry.base.BaseGeometry] = None,
    date_range: Optional[Tuple[datetime, datetime]] = None,
    columns: Optional[List[str]] = None,
    max_items: Optional[int] = None
) -> str:
    """
    Build DuckDB SQL query for partial reads.

    Parameters
    ----------
    parquet_urls : list of str
        URLs/paths to parquet files
    geometry : shapely geometry, optional
        Spatial filter
    date_range : tuple of datetime, optional
        Temporal filter
    columns : list of str, optional
        Columns to select
    max_items : int, optional
        Limit number of rows

    Returns
    -------
    str
        DuckDB SQL query string
    """
    # Build FROM clause with union of all files
    if len(parquet_urls) == 1:
        from_clause = f"read_parquet('{parquet_urls[0]}')"
    else:
        # Union multiple files
        unions = [f"SELECT * FROM read_parquet('{url}')" for url in parquet_urls]
        from_clause = f"({' UNION ALL '.join(unions)})"

    # Build SELECT clause
    if columns:
        # Ensure we have lat/lon for geometry operations if needed
        if geometry and 'longitude (degree_east)' not in columns:
            columns = columns + ['longitude (degree_east)', 'latitude (degree_north)']
        select_clause = ', '.join([f'"{col}"' for col in columns])
    else:
        select_clause = '*'

    # Build WHERE clause
    where_conditions = []

    # Spatial filter
    if geometry:
        bounds = geometry.bounds  # (minx, miny, maxx, maxy)
        where_conditions.append(
            f'"longitude (degree_east)" >= {bounds[0]} AND '
            f'"longitude (degree_east)" <= {bounds[2]} AND '
            f'"latitude (degree_north)" >= {bounds[1]} AND '
            f'"latitude (degree_north)" <= {bounds[3]}'
        )

        # TODO: Add more precise geometry filtering if needed
        # This requires point-in-polygon checks which are more complex in SQL

    # Temporal filter
    if date_range:
        start_date, end_date = date_range
        where_conditions.append(
            f"timestamp >= '{start_date.isoformat()}' AND "
            f"timestamp <= '{end_date.isoformat()}'"
        )

    # Combine WHERE conditions
    if where_conditions:
        where_clause = ' WHERE ' + ' AND '.join(where_conditions)
    else:
        where_clause = ''

    # Build LIMIT clause
    limit_clause = f' LIMIT {max_items}' if max_items else ''

    # Combine query
    query = f"SELECT {select_clause} FROM {from_clause}{where_clause}{limit_clause}"

    return query


def query_bedmap(
    collections: List[str] = None,
    segment_paths: List[str] = None,  # Maps to institution/project for bedmap
    geometry: Optional[shapely.geometry.base.BaseGeometry] = None,
    date_range: Optional[Tuple[datetime, datetime]] = None,
    properties: Dict = {},
    max_items: Optional[int] = None,
    exclude_geometry: bool = True,
    search_kwargs: Dict = {},
    # Additional bedmap-specific parameters
    stac_catalog_path: str = 'gs://opr_stac/bedmap/catalog/',
    parquet_base_path: str = 'gs://opr_stac/bedmap/data/',
    columns: Optional[List[str]] = None
) -> gpd.GeoDataFrame:
    """
    Query bedmap data matching the query_frames interface.

    This function provides an interface similar to OPRConnection.query_frames()
    but queries bedmap data instead.

    Parameters
    ----------
    collections : list of str, optional
        Filter by bedmap version (e.g., ['bedmap-bm2', 'bedmap-bm3'])
    segment_paths : list of str, optional
        Filter by institution/project (extracted from filenames)
    geometry : shapely geometry, optional
        Spatial filter geometry
    date_range : tuple of datetime, optional
        Temporal filter (start_date, end_date)
    properties : dict, optional
        Additional property filters
    max_items : int, optional
        Maximum number of rows to return
    exclude_geometry : bool, default True
        If True, don't create geometry column (keeps lat/lon as separate columns)
    search_kwargs : dict, optional
        Additional search parameters
    stac_catalog_path : str
        Path to STAC catalog
    parquet_base_path : str
        Base path for parquet files
    columns : list of str, optional
        Specific columns to retrieve

    Returns
    -------
    geopandas.GeoDataFrame
        GeoDataFrame with bedmap data

    Examples
    --------
    >>> from shapely.geometry import box
    >>> # Query data for a specific region
    >>> bbox = box(-20, -75, -10, -70)  # lon_min, lat_min, lon_max, lat_max
    >>> df = query_bedmap(
    ...     geometry=bbox,
    ...     date_range=(datetime(1994, 1, 1), datetime(1995, 12, 31)),
    ...     collections=['bedmap-bm2']
    ... )
    """
    # Map segment_paths to institution filter if provided
    if segment_paths:
        # Extract institution from segment paths (e.g., AWI from AWI_1994_...)
        institutions = []
        for path in segment_paths:
            if '_' in path:
                institutions.append(path.split('_')[0])
        if institutions:
            properties['bedmap:institution'] = institutions

    # Query STAC catalog for matching items
    print(f"Querying STAC catalog for matching bedmap items...")
    stac_items = query_bedmap_stac(
        stac_catalog_path=stac_catalog_path,
        collections=collections,
        geometry=geometry,
        date_range=date_range,
        properties=properties,
        max_items=None  # Don't limit STAC items, limit rows instead
    )

    if not stac_items:
        warnings.warn("No matching bedmap items found in STAC catalog")
        return gpd.GeoDataFrame()

    print(f"Found {len(stac_items)} matching files in STAC catalog")

    # Get parquet file URLs
    parquet_urls = []
    for item in stac_items:
        if 'data' in item.assets:
            parquet_urls.append(item.assets['data'].href)

    if not parquet_urls:
        warnings.warn("No data assets found in STAC items")
        return gpd.GeoDataFrame()

    # Build and execute DuckDB query
    print(f"Querying {len(parquet_urls)} parquet files with DuckDB...")

    query = build_duckdb_query(
        parquet_urls=parquet_urls,
        geometry=geometry,
        date_range=date_range,
        columns=columns,
        max_items=max_items
    )

    # Execute query
    conn = duckdb.connect()

    try:
        # Enable S3 support if needed
        if any(url.startswith('s3://') for url in parquet_urls):
            conn.execute("INSTALL httpfs")
            conn.execute("LOAD httpfs")

        # Enable GCS support if needed
        if any(url.startswith('gs://') for url in parquet_urls):
            conn.execute("INSTALL httpfs")
            conn.execute("LOAD httpfs")
            # Configure GCS access if needed
            # conn.execute("SET s3_endpoint='storage.googleapis.com'")

        # Execute query
        result_df = conn.execute(query).df()

    except Exception as e:
        warnings.warn(f"Error executing DuckDB query: {e}")
        return gpd.GeoDataFrame()

    finally:
        conn.close()

    if result_df.empty:
        return gpd.GeoDataFrame()

    print(f"Retrieved {len(result_df):,} rows from parquet files")

    # Create GeoDataFrame
    if not exclude_geometry and 'longitude (degree_east)' in result_df.columns and 'latitude (degree_north)' in result_df.columns:
        # Create point geometries
        geometry_col = gpd.points_from_xy(
            result_df['longitude (degree_east)'],
            result_df['latitude (degree_north)']
        )
        gdf = gpd.GeoDataFrame(result_df, geometry=geometry_col, crs='EPSG:4326')
    else:
        # No geometry column
        gdf = gpd.GeoDataFrame(result_df)

    return gdf


def query_bedmap_local(
    parquet_dir: Union[str, Path],
    geometry: Optional[shapely.geometry.base.BaseGeometry] = None,
    date_range: Optional[Tuple[datetime, datetime]] = None,
    columns: Optional[List[str]] = None,
    max_items: Optional[int] = None,
    exclude_geometry: bool = True
) -> gpd.GeoDataFrame:
    """
    Query bedmap data from local parquet files.

    Simplified version for querying local files without STAC catalog.

    Parameters
    ----------
    parquet_dir : str or Path
        Directory containing parquet files
    geometry : shapely geometry, optional
        Spatial filter
    date_range : tuple of datetime, optional
        Temporal filter
    columns : list of str, optional
        Columns to select
    max_items : int, optional
        Maximum rows to return
    exclude_geometry : bool, default True
        Whether to exclude geometry column

    Returns
    -------
    geopandas.GeoDataFrame
        Query results
    """
    parquet_dir = Path(parquet_dir)

    # Find all parquet files
    parquet_files = sorted(parquet_dir.glob('*.parquet'))

    if not parquet_files:
        warnings.warn(f"No parquet files found in {parquet_dir}")
        return gpd.GeoDataFrame()

    # Convert to string paths
    parquet_urls = [str(f) for f in parquet_files]

    # Build and execute query
    query = build_duckdb_query(
        parquet_urls=parquet_urls,
        geometry=geometry,
        date_range=date_range,
        columns=columns,
        max_items=max_items
    )

    conn = duckdb.connect()
    try:
        result_df = conn.execute(query).df()
    finally:
        conn.close()

    if result_df.empty:
        return gpd.GeoDataFrame()

    # Create GeoDataFrame
    if not exclude_geometry and 'longitude (degree_east)' in result_df.columns:
        geometry_col = gpd.points_from_xy(
            result_df['longitude (degree_east)'],
            result_df['latitude (degree_north)']
        )
        gdf = gpd.GeoDataFrame(result_df, geometry=geometry_col, crs='EPSG:4326')
    else:
        gdf = gpd.GeoDataFrame(result_df)

    return gdf