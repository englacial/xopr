"""
Query interface for bedmap data.

This module provides functions to query and retrieve bedmap data efficiently
using GeoParquet STAC catalogs and DuckDB for partial reads from cloud-optimized files.
"""

import duckdb
import geopandas as gpd
from typing import Optional, List, Dict, Tuple, Union
from datetime import datetime
from pathlib import Path
import warnings

import shapely
from shapely.geometry import shape

from .geometry import (
    check_intersects_polar,
    get_polar_bounds,
)


def _crosses_antimeridian(geometry: shapely.geometry.base.BaseGeometry) -> bool:
    """
    Check if a geometry crosses the antimeridian (180°/-180° longitude).

    Parameters
    ----------
    geometry : shapely geometry
        Geometry to check

    Returns
    -------
    bool
        True if geometry crosses antimeridian
    """
    if geometry is None:
        return False

    bounds = geometry.bounds  # (minx, miny, maxx, maxy)
    lon_min, lon_max = bounds[0], bounds[2]

    # If min_lon > max_lon or spans > 180 degrees
    if lon_min > lon_max or (lon_max - lon_min > 180):
        return True

    return False


def _build_polar_sql_filter(polar_bounds: Tuple[float, float, float, float]) -> str:
    """
    Build SQL WHERE clause for filtering in polar coordinates.

    Uses a spherical approximation of Antarctic Polar Stereographic (EPSG:3031)
    that can be computed directly in SQL.

    Parameters
    ----------
    polar_bounds : tuple
        (x_min, y_min, x_max, y_max) in EPSG:3031 meters

    Returns
    -------
    str
        SQL WHERE clause fragment
    """
    x_min, y_min, x_max, y_max = polar_bounds

    # Spherical approximation of South Polar Stereographic (EPSG:3031)
    polar_sql = f"""
    (
        (6378137.0 * (1.0 + SIN(RADIANS(71.0))) * COS(RADIANS("latitude (degree_north)")) * SIN(RADIANS("longitude (degree_east)")))
            / (1.0 + SIN(RADIANS(-"latitude (degree_north)"))) >= {x_min}
        AND
        (6378137.0 * (1.0 + SIN(RADIANS(71.0))) * COS(RADIANS("latitude (degree_north)")) * SIN(RADIANS("longitude (degree_east)")))
            / (1.0 + SIN(RADIANS(-"latitude (degree_north)"))) <= {x_max}
        AND
        (6378137.0 * (1.0 + SIN(RADIANS(71.0))) * COS(RADIANS("latitude (degree_north)")) * COS(RADIANS("longitude (degree_east)")))
            / (1.0 + SIN(RADIANS(-"latitude (degree_north)"))) >= {y_min}
        AND
        (6378137.0 * (1.0 + SIN(RADIANS(71.0))) * COS(RADIANS("latitude (degree_north)")) * COS(RADIANS("longitude (degree_east)")))
            / (1.0 + SIN(RADIANS(-"latitude (degree_north)"))) <= {y_max}
    )
    """
    return polar_sql.strip()


def build_duckdb_query(
    parquet_urls: List[str],
    geometry: Optional[shapely.geometry.base.BaseGeometry] = None,
    date_range: Optional[Tuple[datetime, datetime]] = None,
    columns: Optional[List[str]] = None,
    max_rows: Optional[int] = None,
    use_polar_filter: bool = True
) -> str:
    """
    Build DuckDB SQL query for partial reads from parquet files.

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
    max_rows : int, optional
        Limit number of rows
    use_polar_filter : bool, default True
        Use Antarctic Polar Stereographic projection for spatial filtering.

    Returns
    -------
    str
        DuckDB SQL query string
    """
    # Build FROM clause
    if len(parquet_urls) == 1:
        from_clause = f"read_parquet('{parquet_urls[0]}')"
    else:
        unions = [f"SELECT * FROM read_parquet('{url}')" for url in parquet_urls]
        from_clause = f"({' UNION ALL '.join(unions)})"

    # Build SELECT clause
    if columns:
        if geometry and 'longitude (degree_east)' not in columns:
            columns = columns + ['longitude (degree_east)', 'latitude (degree_north)']
        select_clause = ', '.join([f'"{col}"' for col in columns])
    else:
        select_clause = '*'

    # Build WHERE clause
    where_conditions = []

    if geometry:
        crosses_am = _crosses_antimeridian(geometry)

        if use_polar_filter or crosses_am:
            polar_bounds = get_polar_bounds(geometry)
            if polar_bounds:
                where_conditions.append(_build_polar_sql_filter(polar_bounds))
        else:
            bounds = geometry.bounds
            where_conditions.append(
                f'"longitude (degree_east)" >= {bounds[0]} AND '
                f'"longitude (degree_east)" <= {bounds[2]} AND '
                f'"latitude (degree_north)" >= {bounds[1]} AND '
                f'"latitude (degree_north)" <= {bounds[3]}'
            )

    if date_range:
        start_date, end_date = date_range
        where_conditions.append(
            f"timestamp >= '{start_date.isoformat()}' AND "
            f"timestamp <= '{end_date.isoformat()}'"
        )

    where_clause = ' WHERE ' + ' AND '.join(where_conditions) if where_conditions else ''
    limit_clause = f' LIMIT {max_rows}' if max_rows else ''

    return f"SELECT {select_clause} FROM {from_clause}{where_clause}{limit_clause}"


def query_bedmap_catalog(
    catalog_path: str = 'gs://opr_stac/bedmap/',
    collections: Optional[List[str]] = None,
    geometry: Optional[shapely.geometry.base.BaseGeometry] = None,
    date_range: Optional[Tuple[datetime, datetime]] = None,
    properties: Optional[Dict] = None
) -> gpd.GeoDataFrame:
    """
    Query GeoParquet STAC catalogs for matching bedmap items.

    Parameters
    ----------
    catalog_path : str
        Base path to GeoParquet catalog files (local or cloud)
    collections : list of str, optional
        Filter by bedmap version (e.g., ['bedmap1', 'bedmap2', 'bedmap3'])
    geometry : shapely geometry, optional
        Spatial filter geometry
    date_range : tuple of datetime, optional
        Temporal filter (start_date, end_date)
    properties : dict, optional
        Additional property filters (e.g., {'institution': 'AWI'})

    Returns
    -------
    geopandas.GeoDataFrame
        Matching catalog items with asset_href for data access
    """
    # Default to all collections
    if collections is None:
        collections = ['bedmap1', 'bedmap2', 'bedmap3']

    all_items = []

    for collection in collections:
        # Build catalog file path
        if catalog_path.startswith(('gs://', 's3://', 'http')):
            catalog_file = f"{catalog_path.rstrip('/')}/{collection}.parquet"
        else:
            catalog_file = Path(catalog_path) / f"{collection}.parquet"
            if not catalog_file.exists():
                continue
            catalog_file = str(catalog_file)

        # Try to load catalog
        try:
            gdf = gpd.read_parquet(catalog_file)
        except Exception as e:
            warnings.warn(f"Could not load catalog {catalog_file}: {e}")
            continue

        if gdf.empty:
            continue

        # Spatial filter using polar projection
        if geometry is not None:
            mask = gdf.geometry.apply(
                lambda g: check_intersects_polar(geometry, g) if g is not None else False
            )
            gdf = gdf[mask]

        # Temporal filter
        if date_range and 'temporal_start' in gdf.columns:
            start_filter, end_filter = date_range

            # Parse temporal columns
            temporal_start = gpd.pd.to_datetime(gdf['temporal_start'], errors='coerce')
            temporal_end = gpd.pd.to_datetime(gdf['temporal_end'], errors='coerce')

            # Filter by overlap
            mask = ~((temporal_end < start_filter) | (temporal_start > end_filter))
            gdf = gdf[mask]

        # Property filters
        if properties:
            for key, value in properties.items():
                if key in gdf.columns:
                    if isinstance(value, list):
                        gdf = gdf[gdf[key].isin(value)]
                    else:
                        gdf = gdf[gdf[key] == value]

        if not gdf.empty:
            all_items.append(gdf)

    if not all_items:
        return gpd.GeoDataFrame()

    return gpd.pd.concat(all_items, ignore_index=True)


def query_bedmap(
    collections: Optional[List[str]] = None,
    geometry: Optional[shapely.geometry.base.BaseGeometry] = None,
    date_range: Optional[Tuple[datetime, datetime]] = None,
    properties: Optional[Dict] = None,
    max_rows: Optional[int] = None,
    columns: Optional[List[str]] = None,
    catalog_path: str = 'gs://opr_stac/bedmap/',
    exclude_geometry: bool = True
) -> gpd.GeoDataFrame:
    """
    Query bedmap data from GeoParquet catalogs and return filtered data.

    This function:
    1. Queries the GeoParquet STAC catalogs to find matching data files
    2. Uses DuckDB for efficient partial reads with spatial/temporal filtering
    3. Returns a GeoDataFrame with the requested data

    Parameters
    ----------
    collections : list of str, optional
        Filter by bedmap version (e.g., ['bedmap1', 'bedmap2', 'bedmap3'])
    geometry : shapely geometry, optional
        Spatial filter geometry
    date_range : tuple of datetime, optional
        Temporal filter (start_date, end_date)
    properties : dict, optional
        Additional property filters (e.g., {'institution': 'AWI'})
    max_rows : int, optional
        Maximum number of rows to return
    columns : list of str, optional
        Specific columns to retrieve
    catalog_path : str
        Base path to GeoParquet catalog files
    exclude_geometry : bool, default True
        If True, don't create geometry column (keeps lat/lon as separate columns)

    Returns
    -------
    geopandas.GeoDataFrame
        GeoDataFrame with bedmap data

    Examples
    --------
    >>> from shapely.geometry import box
    >>> bbox = box(-20, -75, -10, -70)
    >>> df = query_bedmap(
    ...     geometry=bbox,
    ...     date_range=(datetime(1994, 1, 1), datetime(1995, 12, 31)),
    ...     collections=['bedmap2']
    ... )
    """
    # Query catalog for matching items
    catalog_items = query_bedmap_catalog(
        catalog_path=catalog_path,
        collections=collections,
        geometry=geometry,
        date_range=date_range,
        properties=properties
    )

    if catalog_items.empty:
        warnings.warn("No matching bedmap items found in catalog")
        return gpd.GeoDataFrame()

    print(f"Found {len(catalog_items)} matching files in catalog")

    # Get data file URLs
    if 'asset_href' not in catalog_items.columns:
        warnings.warn("No asset_href column in catalog items")
        return gpd.GeoDataFrame()

    parquet_urls = catalog_items['asset_href'].tolist()

    # Build and execute DuckDB query
    print(f"Querying {len(parquet_urls)} parquet files...")

    query = build_duckdb_query(
        parquet_urls=parquet_urls,
        geometry=geometry,
        date_range=date_range,
        columns=columns,
        max_rows=max_rows
    )

    conn = duckdb.connect()
    try:
        # Enable cloud storage support
        conn.execute("INSTALL httpfs")
        conn.execute("LOAD httpfs")

        result_df = conn.execute(query).df()
    except Exception as e:
        warnings.warn(f"Error executing DuckDB query: {e}")
        return gpd.GeoDataFrame()
    finally:
        conn.close()

    if result_df.empty:
        return gpd.GeoDataFrame()

    print(f"Retrieved {len(result_df):,} rows")

    # Create GeoDataFrame
    if not exclude_geometry and 'longitude (degree_east)' in result_df.columns:
        geometry_col = gpd.points_from_xy(
            result_df['longitude (degree_east)'],
            result_df['latitude (degree_north)']
        )
        return gpd.GeoDataFrame(result_df, geometry=geometry_col, crs='EPSG:4326')

    return gpd.GeoDataFrame(result_df)


def query_bedmap_local(
    parquet_dir: Union[str, Path],
    geometry: Optional[shapely.geometry.base.BaseGeometry] = None,
    date_range: Optional[Tuple[datetime, datetime]] = None,
    columns: Optional[List[str]] = None,
    max_rows: Optional[int] = None,
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
    max_rows : int, optional
        Maximum rows to return
    exclude_geometry : bool, default True
        Whether to exclude geometry column

    Returns
    -------
    geopandas.GeoDataFrame
        Query results
    """
    parquet_dir = Path(parquet_dir)
    parquet_files = sorted(parquet_dir.glob('*.parquet'))

    if not parquet_files:
        warnings.warn(f"No parquet files found in {parquet_dir}")
        return gpd.GeoDataFrame()

    parquet_urls = [str(f) for f in parquet_files]

    query = build_duckdb_query(
        parquet_urls=parquet_urls,
        geometry=geometry,
        date_range=date_range,
        columns=columns,
        max_rows=max_rows
    )

    conn = duckdb.connect()
    try:
        result_df = conn.execute(query).df()
    finally:
        conn.close()

    if result_df.empty:
        return gpd.GeoDataFrame()

    if not exclude_geometry and 'longitude (degree_east)' in result_df.columns:
        geometry_col = gpd.points_from_xy(
            result_df['longitude (degree_east)'],
            result_df['latitude (degree_north)']
        )
        return gpd.GeoDataFrame(result_df, geometry=geometry_col, crs='EPSG:4326')

    return gpd.GeoDataFrame(result_df)
