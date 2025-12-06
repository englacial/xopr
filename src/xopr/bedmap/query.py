"""
Query interface for bedmap data.

This module provides functions to query and retrieve bedmap data efficiently
using rustac for STAC catalog searches and DuckDB for partial reads from
cloud-optimized GeoParquet files.
"""

import duckdb
import geopandas as gpd
import numpy as np
import requests
from typing import Optional, List, Dict, Tuple, Union
from datetime import datetime
from pathlib import Path
import warnings

import antimeridian
import shapely
from shapely.geometry import shape
from rustac import DuckdbClient

from .geometry import (
    check_intersects_polar,
    get_polar_bounds,
)
from ..stac_cache import get_bedmap_catalog_path


# Cloud URLs for bedmap STAC catalog files
BEDMAP_CATALOG_URLS = {
    'bedmap1': 'gs://opr_stac/bedmap/bedmap1.parquet',
    'bedmap2': 'gs://opr_stac/bedmap/bedmap2.parquet',
    'bedmap3': 'gs://opr_stac/bedmap/bedmap3.parquet',
}

# Default cache directories
# STAC catalogs go to ~/.cache/xopr/bedmap (same as stac_cache module)
DEFAULT_STAC_CACHE_DIR = Path.home() / '.cache' / 'xopr' / 'bedmap'

# Data files go to radar_cache/bedmap (matches OPRConnection convention)
# OPRConnection uses cache_dir="radar_cache" for fsspec caching;
# bedmap data goes in a "bedmap" subdirectory within that same cache
DEFAULT_RADAR_CACHE = Path('radar_cache')
DEFAULT_BEDMAP_DATA_DIR = DEFAULT_RADAR_CACHE / 'bedmap'


def _gs_to_https(gs_url: str) -> str:
    """Convert gs:// URL to public https:// URL for anonymous access."""
    if gs_url.startswith('gs://'):
        # gs://bucket/path -> https://storage.googleapis.com/bucket/path
        return gs_url.replace('gs://', 'https://storage.googleapis.com/', 1)
    return gs_url


def _download_file(url: str, local_path: Path) -> bool:
    """Download a file from URL to local path. Returns True on success."""
    https_url = _gs_to_https(url)
    try:
        response = requests.get(https_url, stream=True)
        response.raise_for_status()
        with open(local_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return True
    except Exception as e:
        warnings.warn(f"Failed to download {url}: {e}")
        return False


def fetch_bedmap(
    version: str = 'all',
    data_dir: Optional[Union[str, Path]] = None,
    force: bool = False
) -> Dict[str, List[Path]]:
    """
    Download bedmap GeoParquet data files to local cache.

    Reads the STAC catalog to get asset hrefs, then downloads all data files
    for the requested bedmap version(s).

    Parameters
    ----------
    version : str, default 'all'
        Which bedmap version(s) to download:
        - 'all': Download all versions (bedmap1, bedmap2, bedmap3)
        - 'bedmap1', 'bedmap2', 'bedmap3': Download specific version
    data_dir : str or Path, optional
        Directory to store downloaded data files. Defaults to radar_cache/bedmap/
    force : bool, default False
        If True, re-download even if files already exist

    Returns
    -------
    dict
        Mapping of version names to lists of local file paths

    Examples
    --------
    >>> # Download all bedmap data
    >>> paths = fetch_bedmap()
    >>> print(len(paths['bedmap2']))  # Number of files downloaded
    45

    >>> # Download only bedmap3
    >>> paths = fetch_bedmap(version='bedmap3')

    >>> # Use with query_bedmap for faster repeated queries
    >>> paths = fetch_bedmap()
    >>> for bbox in many_bboxes:
    ...     df = query_bedmap(geometry=bbox, local_cache=True)
    """
    # Data files go to radar_cache/bedmap/ (matches OPRConnection convention)
    data_dir = Path(data_dir) if data_dir else DEFAULT_BEDMAP_DATA_DIR
    data_dir.mkdir(parents=True, exist_ok=True)

    # Determine which versions to download
    valid_versions = list(BEDMAP_CATALOG_URLS.keys())
    if version == 'all':
        versions = valid_versions
    elif version in BEDMAP_CATALOG_URLS:
        versions = [version]
    else:
        raise ValueError(
            f"Unknown version '{version}'. "
            f"Valid options: 'all', {valid_versions}"
        )

    downloaded = {}
    for ver in versions:
        catalog_url = BEDMAP_CATALOG_URLS[ver]
        print(f"{ver}: Reading STAC catalog to get data file locations...")

        # Read catalog with DuckDB to extract asset_href column
        https_catalog_url = _gs_to_https(catalog_url)
        try:
            conn = duckdb.connect()
            conn.execute("SET enable_progress_bar = false")
            query = f"SELECT asset_href FROM read_parquet('{https_catalog_url}')"
            result = conn.execute(query).fetchall()
            conn.close()
            hrefs = [row[0] for row in result if row[0]]
        except Exception as e:
            warnings.warn(f"Failed to read catalog for {ver}: {e}")
            continue

        if not hrefs:
            warnings.warn(f"No asset hrefs found in {ver} catalog")
            continue

        print(f"{ver}: Found {len(hrefs)} data files to download")

        # Download each data file
        downloaded_files = []
        for href in hrefs:
            filename = Path(href).name
            local_path = data_dir / filename

            if local_path.exists() and not force:
                downloaded_files.append(local_path)
                continue

            if _download_file(href, local_path):
                downloaded_files.append(local_path)
                print(f"  Downloaded: {filename}")

        downloaded[ver] = downloaded_files
        print(f"{ver}: {len(downloaded_files)} files cached in {data_dir}")

    return downloaded


def get_bedmap_cache_path(
    data_dir: Optional[Union[str, Path]] = None
) -> Path:
    """
    Get path to cached bedmap data directory.

    Parameters
    ----------
    data_dir : str or Path, optional
        Data directory. Defaults to radar_cache/bedmap/

    Returns
    -------
    Path
        Path to the data directory containing cached parquet files
    """
    return Path(data_dir) if data_dir else DEFAULT_BEDMAP_DATA_DIR


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
    data_dir: Optional[Union[str, Path]] = None,
    show_progress: bool = False,
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
    data_dir : str or Path, optional
        Directory for local data cache. Defaults to radar_cache/bedmap/
    show_progress : bool, default False
        If True, show DuckDB progress bar during queries. Disabled by default
        to avoid IOPub rate limit errors in Jupyter notebooks.

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
            data_dir=data_dir,
            show_progress=show_progress,
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
        if not show_progress:
            conn.execute("SET enable_progress_bar = false")
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
    data_dir: Optional[Union[str, Path]] = None,
    show_progress: bool = False,
) -> gpd.GeoDataFrame:
    """
    Query bedmap data from locally cached files.

    Internal helper that fetches data if needed and queries local files.
    Uses STAC catalog to pre-filter files when geometry is provided.
    """
    data_dir = Path(data_dir) if data_dir else DEFAULT_BEDMAP_DATA_DIR

    # Determine which versions to query
    valid_versions = list(BEDMAP_CATALOG_URLS.keys())
    if collections:
        versions = [c for c in collections if c in valid_versions]
    else:
        versions = valid_versions

    if not versions:
        warnings.warn("No valid bedmap versions specified")
        return gpd.GeoDataFrame()

    # Check if data directory has files, if not download them
    if not data_dir.exists() or not list(data_dir.glob('*.parquet')):
        print("No cached data files found, downloading...")
        for ver in versions:
            fetch_bedmap(version=ver, data_dir=data_dir)

    # Use STAC catalog to pre-filter files when geometry is provided
    # This is much faster than querying all files
    parquet_paths = []
    if geometry is not None:
        # Query catalog to find matching files
        catalog_items = query_bedmap_catalog(
            collections=versions,
            geometry=geometry,
            date_range=date_range,
        )

        if not catalog_items.empty:
            # Map catalog items to local file paths
            for idx, item in catalog_items.iterrows():
                props = item.get('properties', {})
                if isinstance(props, dict):
                    href = props.get('asset_href', '')
                    if href:
                        fname = href.split('/')[-1]
                        local_path = data_dir / fname
                        if local_path.exists():
                            parquet_paths.append(str(local_path))

    # Fall back to all files if no geometry filter or no catalog matches
    if not parquet_paths and data_dir.exists():
        parquet_paths = [str(p) for p in sorted(data_dir.glob('*.parquet'))]

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
        if not show_progress:
            conn.execute("SET enable_progress_bar = false")
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
        # Disable progress bar to avoid IOPub rate limit in notebooks
        conn.execute("SET enable_progress_bar = false")
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
