"""
Query interface for bedmap data.

This module provides functions to query and retrieve bedmap data efficiently
using rustac for STAC catalog searches and DuckDB for partial reads from
cloud-optimized GeoParquet files.
"""

import duckdb
import geopandas as gpd
import numpy as np
from typing import Optional, List, Dict, Tuple, Union
from datetime import datetime
from pathlib import Path
import warnings
import fsspec

import antimeridian
import shapely
from shapely.geometry import shape
from rustac import DuckdbClient

from .geometry import (
    check_intersects_polar,
    get_polar_bounds,
)
from ..stac_cache import get_bedmap_catalog_path


# Cloud URLs for bedmap data files
BEDMAP_DATA_URLS = {
    'bedmap1': 'gs://opr_stac/bedmap/data/bedmap1.parquet',
    'bedmap2': 'gs://opr_stac/bedmap/data/bedmap2.parquet',
    'bedmap3': 'gs://opr_stac/bedmap/data/bedmap3.parquet',
}

# Default cache directory (matches OPRConnection convention)
DEFAULT_BEDMAP_CACHE_DIR = Path.home() / '.cache' / 'xopr' / 'bedmap'


def fetch_bedmap(
    version: str = 'all',
    cache_dir: Optional[Union[str, Path]] = None,
    force: bool = False
) -> Dict[str, Path]:
    """
    Download bedmap GeoParquet data files to local cache.

    Downloads bedmap data files from cloud storage to local disk for faster
    repeated queries. Useful when executing many queries in a loop.

    Parameters
    ----------
    version : str, default 'all'
        Which bedmap version(s) to download:
        - 'all': Download all versions (bedmap1, bedmap2, bedmap3)
        - 'bedmap1', 'bedmap2', 'bedmap3': Download specific version
    cache_dir : str or Path, optional
        Directory to store downloaded files. Defaults to ~/.cache/xopr/bedmap
    force : bool, default False
        If True, re-download even if files already exist

    Returns
    -------
    dict
        Mapping of version names to local file paths

    Examples
    --------
    >>> # Download all bedmap data
    >>> paths = fetch_bedmap()
    >>> print(paths['bedmap2'])
    /home/user/.cache/xopr/bedmap/bedmap2.parquet

    >>> # Download only bedmap3
    >>> paths = fetch_bedmap(version='bedmap3')

    >>> # Use with query_bedmap for faster repeated queries
    >>> paths = fetch_bedmap()
    >>> for bbox in many_bboxes:
    ...     df = query_bedmap(geometry=bbox, local_cache=True)
    """
    cache_dir = Path(cache_dir) if cache_dir else DEFAULT_BEDMAP_CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Determine which versions to download
    if version == 'all':
        versions = list(BEDMAP_DATA_URLS.keys())
    elif version in BEDMAP_DATA_URLS:
        versions = [version]
    else:
        raise ValueError(
            f"Unknown version '{version}'. "
            f"Valid options: 'all', {list(BEDMAP_DATA_URLS.keys())}"
        )

    downloaded = {}
    for ver in versions:
        url = BEDMAP_DATA_URLS[ver]
        local_path = cache_dir / f'{ver}.parquet'

        if local_path.exists() and not force:
            print(f"{ver}: Using cached file at {local_path}")
            downloaded[ver] = local_path
            continue

        print(f"{ver}: Downloading from {url}...")
        try:
            # Use fsspec to download from GCS
            with fsspec.open(url, 'rb') as remote_file:
                with open(local_path, 'wb') as local_file:
                    local_file.write(remote_file.read())
            print(f"{ver}: Saved to {local_path}")
            downloaded[ver] = local_path
        except Exception as e:
            warnings.warn(f"Failed to download {ver}: {e}")

    return downloaded


def get_bedmap_cache_path(
    version: Optional[str] = None,
    cache_dir: Optional[Union[str, Path]] = None
) -> Union[Path, Dict[str, Path]]:
    """
    Get paths to cached bedmap data files.

    Parameters
    ----------
    version : str, optional
        Specific version to get path for. If None, returns all cached paths.
    cache_dir : str or Path, optional
        Cache directory to check. Defaults to ~/.cache/xopr/bedmap

    Returns
    -------
    Path or dict
        Single path if version specified, otherwise dict of all cached paths
    """
    cache_dir = Path(cache_dir) if cache_dir else DEFAULT_BEDMAP_CACHE_DIR

    if version:
        return cache_dir / f'{version}.parquet'

    # Return all existing cached files
    cached = {}
    for ver in BEDMAP_DATA_URLS.keys():
        path = cache_dir / f'{ver}.parquet'
        if path.exists():
            cached[ver] = path
    return cached


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
    that can be computed directly in SQL. Works with GeoParquet files that have
    WKB-encoded Point geometry accessed via ST_X(geometry) and ST_Y(geometry).

    Parameters
    ----------
    polar_bounds : tuple
        (x_min, y_min, x_max, y_max) in EPSG:3031 meters

    Returns
    -------
    str
        SQL WHERE clause fragment
    """
    min_x, min_y, max_x, max_y = polar_bounds

    # Spherical approximation of South Polar Stereographic (EPSG:3031)
    # Uses ST_X/ST_Y to extract coordinates from WKB Point geometry
    # Note: y coordinate uses positive cos (matching EPSG:3031 convention)
    polar_sql = f"""
    (
        (6371000.0 * 2 * tan(radians(45 + ST_Y(geometry)/2)) * sin(radians(ST_X(geometry)))) >= {min_x}
        AND
        (6371000.0 * 2 * tan(radians(45 + ST_Y(geometry)/2)) * sin(radians(ST_X(geometry)))) <= {max_x}
        AND
        (6371000.0 * 2 * tan(radians(45 + ST_Y(geometry)/2)) * cos(radians(ST_X(geometry)))) >= {min_y}
        AND
        (6371000.0 * 2 * tan(radians(45 + ST_Y(geometry)/2)) * cos(radians(ST_X(geometry)))) <= {max_y}
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
    Build DuckDB SQL query for partial reads from GeoParquet files.

    Works with GeoParquet files that have WKB-encoded Point geometry.
    Uses DuckDB spatial extension for geometry operations.

    Parameters
    ----------
    parquet_urls : list of str
        URLs/paths to GeoParquet files
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

    # Build SELECT clause - GeoParquet uses geometry column
    # Extract lon/lat using ST_X/ST_Y instead of raw WKB geometry
    if columns:
        # Add lon/lat extraction if needed for spatial filtering
        if geometry and 'lon' not in columns and 'lat' not in columns:
            columns = columns + ['lon', 'lat']
        select_parts = []
        for col in columns:
            if col == 'lon':
                select_parts.append('ST_X(geometry) as lon')
            elif col == 'lat':
                select_parts.append('ST_Y(geometry) as lat')
            elif col == 'geometry':
                # Skip raw geometry, use lon/lat instead
                continue
            else:
                select_parts.append(f'"{col}"')
        select_clause = ', '.join(select_parts)
    else:
        # Select all columns plus extracted lon/lat
        select_clause = '*, ST_X(geometry) as lon, ST_Y(geometry) as lat'

    # Build WHERE clause
    where_conditions = []

    if geometry:
        crosses_am = _crosses_antimeridian(geometry)

        if use_polar_filter or crosses_am:
            polar_bounds = get_polar_bounds(geometry)
            if polar_bounds:
                where_conditions.append(_build_polar_sql_filter(polar_bounds))
        else:
            # Simple bounding box filter using ST_X/ST_Y for GeoParquet
            bounds = geometry.bounds
            where_conditions.append(
                f'ST_X(geometry) >= {bounds[0]} AND '
                f'ST_X(geometry) <= {bounds[2]} AND '
                f'ST_Y(geometry) >= {bounds[1]} AND '
                f'ST_Y(geometry) <= {bounds[3]}'
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
    collections: Optional[List[str]] = None,
    geometry: Optional[shapely.geometry.base.BaseGeometry] = None,
    date_range: Optional[Tuple[datetime, datetime]] = None,
    properties: Optional[Dict] = None,
    max_items: Optional[int] = None,
    exclude_geometry: bool = False,
    catalog_path: str = 'local',
) -> gpd.GeoDataFrame:
    """
    Query GeoParquet STAC catalogs for matching bedmap items using rustac.

    This function uses rustac's DuckdbClient to perform efficient spatial queries
    on STAC GeoParquet catalogs, following the same pattern as
    OPRConnection.query_frames().

    Parameters
    ----------
    collections : list of str, optional
        Filter by bedmap version (e.g., ['bedmap1', 'bedmap2', 'bedmap3'])
    geometry : shapely geometry, optional
        Spatial filter geometry
    date_range : tuple of datetime, optional
        Temporal filter (start_date, end_date)
    properties : dict, optional
        Additional property filters using CQL2 (e.g., {'institution': 'AWI'})
    max_items : int, optional
        Maximum number of items to return
    exclude_geometry : bool, default False
        If True, exclude geometry column from results
    catalog_path : str, default 'local'
        Catalog source: 'local' (cached, downloaded on first use),
        'cloud' (direct from GCS), or a custom path/URL pattern.

    Returns
    -------
    geopandas.GeoDataFrame
        Matching catalog items with asset_href for data access
    """
    # Resolve catalog path
    if catalog_path == 'local':
        catalog_path = get_bedmap_catalog_path()
    elif catalog_path == 'cloud':
        catalog_path = 'gs://opr_stac/bedmap/bedmap*.parquet'

    search_params = {}

    # Exclude geometry
    if exclude_geometry:
        search_params['exclude'] = ['geometry']

    # Handle collections (bedmap versions)
    if collections is not None:
        # Map collection names to their catalog patterns
        collection_list = [collections] if isinstance(collections, str) else collections
        search_params['collections'] = collection_list

    # Handle geometry filtering
    if geometry is not None:
        if hasattr(geometry, '__geo_interface__'):
            geom_dict = geometry.__geo_interface__
        else:
            geom_dict = geometry

        # Fix geometries that cross the antimeridian
        geom_dict = antimeridian.fix_geojson(geom_dict, reverse=True)

        search_params['intersects'] = geom_dict

    # Note: The bedmap STAC catalogs currently don't have datetime fields,
    # so temporal filtering at the catalog level is not supported.
    # Date filtering can be applied later in the DuckDB query step if the
    # data files have timestamp columns.
    if date_range is not None:
        warnings.warn(
            "Temporal filtering (date_range) is not supported at the catalog level "
            "for bedmap data. The date_range parameter will be applied during "
            "the data query step if timestamp columns are available.",
            UserWarning
        )

    # Handle max_items
    if max_items is not None:
        search_params['limit'] = max_items
    else:
        search_params['limit'] = 1000000

    # Handle property filters using CQL2
    filter_conditions = []

    if properties:
        for key, value in properties.items():
            if isinstance(value, list):
                # Create OR conditions for multiple values
                value_conditions = []
                for v in value:
                    value_conditions.append({
                        "op": "=",
                        "args": [{"property": key}, v]
                    })
                if len(value_conditions) == 1:
                    filter_conditions.append(value_conditions[0])
                else:
                    filter_conditions.append({
                        "op": "or",
                        "args": value_conditions
                    })
            else:
                filter_conditions.append({
                    "op": "=",
                    "args": [{"property": key}, value]
                })

    # Combine all filter conditions with AND
    if filter_conditions:
        if len(filter_conditions) == 1:
            filter_expr = filter_conditions[0]
        else:
            filter_expr = {
                "op": "and",
                "args": filter_conditions
            }
        search_params['filter'] = filter_expr

    # Perform the search using rustac
    client = DuckdbClient()
    items = client.search(catalog_path, **search_params)

    if isinstance(items, dict):
        items = items['features']

    if not items or len(items) == 0:
        warnings.warn("No items found matching the query criteria", UserWarning)
        return gpd.GeoDataFrame()

    # Convert to GeoDataFrame
    items_df = gpd.GeoDataFrame(items)

    # Set index
    if 'id' in items_df.columns:
        items_df = items_df.set_index(items_df['id'])
        items_df.index.name = 'stac_item_id'

    # Set the geometry column
    if 'geometry' in items_df.columns and not exclude_geometry:
        items_df = items_df.set_geometry(items_df['geometry'].apply(shapely.geometry.shape))
        items_df.crs = "EPSG:4326"

    return items_df


def query_bedmap(
    collections: Optional[List[str]] = None,
    geometry: Optional[shapely.geometry.base.BaseGeometry] = None,
    date_range: Optional[Tuple[datetime, datetime]] = None,
    properties: Optional[Dict] = None,
    max_rows: Optional[int] = None,
    columns: Optional[List[str]] = None,
    exclude_geometry: bool = True,
    catalog_path: str = 'local',
    local_cache: bool = False,
    cache_dir: Optional[Union[str, Path]] = None,
) -> gpd.GeoDataFrame:
    """
    Query bedmap data from GeoParquet catalogs and return filtered data.

    This function:
    1. Uses rustac to query the STAC GeoParquet catalogs for matching items
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
    exclude_geometry : bool, default True
        If True, don't create geometry column (keeps lat/lon as separate columns)
    catalog_path : str, default 'local'
        Catalog source: 'local' (cached, downloaded on first use),
        'cloud' (direct from GCS), or a custom path/URL pattern.
    local_cache : bool, default False
        If True, use locally cached bedmap data files for faster queries.
        Files will be downloaded automatically if not present.
        Recommended for repeated queries in loops.
    cache_dir : str or Path, optional
        Directory for local cache. Defaults to ~/.cache/xopr/bedmap

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

    >>> # Use local cache for faster repeated queries
    >>> for bbox in many_bboxes:
    ...     df = query_bedmap(geometry=bbox, local_cache=True)
    """
    # If using local cache, bypass STAC catalog and query local files directly
    if local_cache:
        return _query_bedmap_cached(
            collections=collections,
            geometry=geometry,
            date_range=date_range,
            columns=columns,
            max_rows=max_rows,
            exclude_geometry=exclude_geometry,
            cache_dir=cache_dir,
        )

    # Query catalog for matching items using rustac
    catalog_items = query_bedmap_catalog(
        collections=collections,
        geometry=geometry,
        date_range=date_range,
        properties=properties,
        catalog_path=catalog_path,
    )

    if catalog_items.empty:
        warnings.warn("No matching bedmap items found in catalog")
        return gpd.GeoDataFrame()

    print(f"Found {len(catalog_items)} matching files in catalog")

    # Get data file URLs from the catalog items
    # The rustac STAC GeoParquet format stores asset_href in the properties dict
    parquet_urls = []
    for idx, item in catalog_items.iterrows():
        # Primary: check properties dict for asset_href
        if 'properties' in item and isinstance(item['properties'], dict):
            props = item['properties']
            if 'asset_href' in props and props['asset_href']:
                parquet_urls.append(props['asset_href'])
                continue

        # Fallback: check for assets dict (standard STAC structure)
        if 'assets' in item and item['assets']:
            assets = item['assets']
            if isinstance(assets, dict):
                for asset_key, asset_info in assets.items():
                    if isinstance(asset_info, dict) and 'href' in asset_info:
                        parquet_urls.append(asset_info['href'])
                        break
                    elif isinstance(asset_info, str):
                        parquet_urls.append(asset_info)
                        break

        # Final fallback: check for asset_href column directly
        if 'asset_href' in catalog_items.columns:
            if item.get('asset_href'):
                parquet_urls.append(item['asset_href'])

    if not parquet_urls:
        warnings.warn("No asset URLs found in catalog items")
        return gpd.GeoDataFrame()

    # Build and execute DuckDB query
    print(f"Querying {len(parquet_urls)} GeoParquet files...")

    query = build_duckdb_query(
        parquet_urls=parquet_urls,
        geometry=geometry,
        date_range=date_range,
        columns=columns,
        max_rows=max_rows
    )

    conn = duckdb.connect()
    try:
        # Enable cloud storage and spatial extension support
        conn.execute("INSTALL httpfs; LOAD httpfs;")
        conn.execute("INSTALL spatial; LOAD spatial;")

        result_df = conn.execute(query).df()
    except Exception as e:
        warnings.warn(f"Error executing DuckDB query: {e}")
        return gpd.GeoDataFrame()
    finally:
        conn.close()

    if result_df.empty:
        return gpd.GeoDataFrame()

    print(f"Retrieved {len(result_df):,} rows")

    # Query already extracts lon/lat from geometry using ST_X/ST_Y
    if 'lon' in result_df.columns and 'lat' in result_df.columns:
        if not exclude_geometry:
            # Create geometry from lon/lat
            geometry = gpd.points_from_xy(result_df['lon'], result_df['lat'])
            gdf = gpd.GeoDataFrame(result_df, geometry=geometry, crs='EPSG:4326')
            return gdf
        else:
            # Just return with lon/lat columns (no geometry)
            return gpd.GeoDataFrame(result_df)

    return gpd.GeoDataFrame(result_df)


def _query_bedmap_cached(
    collections: Optional[List[str]] = None,
    geometry: Optional[shapely.geometry.base.BaseGeometry] = None,
    date_range: Optional[Tuple[datetime, datetime]] = None,
    columns: Optional[List[str]] = None,
    max_rows: Optional[int] = None,
    exclude_geometry: bool = True,
    cache_dir: Optional[Union[str, Path]] = None,
) -> gpd.GeoDataFrame:
    """
    Query bedmap data from locally cached files.

    Internal helper that fetches data if needed and queries local files.
    """
    cache_dir = Path(cache_dir) if cache_dir else DEFAULT_BEDMAP_CACHE_DIR

    # Determine which versions to query
    if collections:
        versions = [c for c in collections if c in BEDMAP_DATA_URLS]
    else:
        versions = list(BEDMAP_DATA_URLS.keys())

    if not versions:
        warnings.warn("No valid bedmap versions specified")
        return gpd.GeoDataFrame()

    # Check which files exist and download missing ones
    parquet_paths = []
    for ver in versions:
        local_path = cache_dir / f'{ver}.parquet'
        if not local_path.exists():
            print(f"Downloading {ver} to local cache...")
            fetch_bedmap(version=ver, cache_dir=cache_dir)

        if local_path.exists():
            parquet_paths.append(str(local_path))
        else:
            warnings.warn(f"Could not fetch {ver}")

    if not parquet_paths:
        warnings.warn("No bedmap files available")
        return gpd.GeoDataFrame()

    # Build and execute DuckDB query on local files
    query = build_duckdb_query(
        parquet_urls=parquet_paths,
        geometry=geometry,
        date_range=date_range,
        columns=columns,
        max_rows=max_rows
    )

    conn = duckdb.connect()
    try:
        conn.execute("INSTALL spatial; LOAD spatial;")
        result_df = conn.execute(query).df()
    except Exception as e:
        warnings.warn(f"Error executing DuckDB query: {e}")
        return gpd.GeoDataFrame()
    finally:
        conn.close()

    if result_df.empty:
        return gpd.GeoDataFrame()

    print(f"Retrieved {len(result_df):,} rows from local cache")

    if 'lon' in result_df.columns and 'lat' in result_df.columns:
        if not exclude_geometry:
            geom = gpd.points_from_xy(result_df['lon'], result_df['lat'])
            return gpd.GeoDataFrame(result_df, geometry=geom, crs='EPSG:4326')
        else:
            return gpd.GeoDataFrame(result_df)

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
    Query bedmap data from local GeoParquet files.

    Simplified version for querying local files without STAC catalog.

    Parameters
    ----------
    parquet_dir : str or Path
        Directory containing GeoParquet files
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
        # Load spatial extension for ST_X/ST_Y functions
        conn.execute("INSTALL spatial; LOAD spatial;")
        result_df = conn.execute(query).df()
    finally:
        conn.close()

    if result_df.empty:
        return gpd.GeoDataFrame()

    # Query already extracts lon/lat from geometry using ST_X/ST_Y
    if 'lon' in result_df.columns and 'lat' in result_df.columns:
        if not exclude_geometry:
            # Create geometry from lon/lat
            geometry = gpd.points_from_xy(result_df['lon'], result_df['lat'])
            gdf = gpd.GeoDataFrame(result_df, geometry=geometry, crs='EPSG:4326')
            return gdf
        else:
            # Just return with lon/lat columns (no geometry)
            return gpd.GeoDataFrame(result_df)

    return gpd.GeoDataFrame(result_df)
