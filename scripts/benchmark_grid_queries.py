#!/usr/bin/env python
"""
Grid-based query benchmark simulating realistic user queries.

Creates polar stereographic grids at different resolutions over Antarctica,
picks random grid cells over land, and uses them as spatial queries.
"""

import sys
import time
import random
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import box, Polygon
from shapely.ops import transform
import pyproj
import duckdb

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


def get_file_extent(parquet_path: str, format_type: str, location: str = 'local'):
    """
    Get the spatial extent of a parquet file.

    Returns (minx, miny, maxx, maxy) in lat/lon coordinates.
    """
    conn = duckdb.connect()

    if location == 'gcs':
        conn.execute("INSTALL httpfs; LOAD httpfs;")

    try:
        if format_type == 'original':
            query = f"""
                SELECT
                    MIN("longitude (degree_east)") as minx,
                    MIN("latitude (degree_north)") as miny,
                    MAX("longitude (degree_east)") as maxx,
                    MAX("latitude (degree_north)") as maxy
                FROM read_parquet('{parquet_path}')
            """
        else:
            conn.execute("INSTALL spatial; LOAD spatial;")
            query = f"""
                SELECT
                    MIN(ST_X(geometry)) as minx,
                    MIN(ST_Y(geometry)) as miny,
                    MAX(ST_X(geometry)) as maxx,
                    MAX(ST_Y(geometry)) as maxy
                FROM read_parquet('{parquet_path}')
            """

        result = conn.execute(query).fetchone()
        conn.close()
        return result  # (minx, miny, maxx, maxy)
    except Exception as e:
        conn.close()
        return None


def create_antarctic_grid(resolution_km: int, crs_epsg: int = 3031,
                          extent_latlon: tuple = None):
    """
    Create a polar stereographic grid over Antarctica.

    Parameters
    ----------
    resolution_km : int
        Grid cell size in kilometers
    crs_epsg : int
        EPSG code for the projection (default: 3031 Antarctic Polar Stereographic)
    extent_latlon : tuple, optional
        (minx, miny, maxx, maxy) in lat/lon to limit grid extent.
        If provided, only creates cells within this bounding box.

    Returns
    -------
    list of tuple
        List of (minx, miny, maxx, maxy) bounds in projected coordinates
    """
    resolution_m = resolution_km * 1000

    if extent_latlon is not None:
        # Transform lat/lon extent to polar stereographic
        transformer = pyproj.Transformer.from_crs(
            "EPSG:4326", f"EPSG:{crs_epsg}", always_xy=True
        )
        # Transform corners
        lon_min, lat_min, lon_max, lat_max = extent_latlon

        # Sample multiple points along edges to get accurate projected bounds
        lons = [lon_min, lon_max, lon_min, lon_max, (lon_min+lon_max)/2]
        lats = [lat_min, lat_max, lat_max, lat_min, (lat_min+lat_max)/2]
        xs, ys = transformer.transform(lons, lats)

        # Add buffer of 1 grid cell
        extent_x_min = min(xs) - resolution_m
        extent_x_max = max(xs) + resolution_m
        extent_y_min = min(ys) - resolution_m
        extent_y_max = max(ys) + resolution_m
    else:
        # Full Antarctic extent in EPSG:3031 (roughly -3000km to 3000km)
        extent_x_min = -3000000
        extent_x_max = 3000000
        extent_y_min = -3000000
        extent_y_max = 3000000

    cells = []
    x = extent_x_min
    while x < extent_x_max:
        y = extent_y_min
        while y < extent_y_max:
            cells.append((x, y, x + resolution_m, y + resolution_m))
            y += resolution_m
        x += resolution_m

    return cells


def is_over_land(cell_bounds, crs_from=3031, crs_to=4326):
    """
    Check if a cell is over Antarctic land mass (rough approximation).

    Uses a simple distance-from-pole check: cells within ~2000km of South Pole
    are likely over land.
    """
    minx, miny, maxx, maxy = cell_bounds
    center_x = (minx + maxx) / 2
    center_y = (miny + maxy) / 2

    # Distance from South Pole (origin in EPSG:3031)
    distance_from_pole = np.sqrt(center_x**2 + center_y**2)

    # Antarctic landmass extends roughly 2000km from pole in most directions
    # Use 2200km to include coastal areas
    return distance_from_pole < 2200000  # 2200 km


def polar_to_latlon(cell_bounds, crs_from=3031, crs_to=4326):
    """
    Convert cell bounds from polar stereographic to lat/lon polygon.

    Parameters
    ----------
    cell_bounds : tuple
        (minx, miny, maxx, maxy) in projected coordinates
    crs_from : int
        Source EPSG code
    crs_to : int
        Target EPSG code

    Returns
    -------
    shapely.geometry.Polygon
        Polygon in lat/lon coordinates
    """
    transformer = pyproj.Transformer.from_crs(
        f"EPSG:{crs_from}", f"EPSG:{crs_to}", always_xy=True
    )

    minx, miny, maxx, maxy = cell_bounds

    # Create polygon with more points along edges for better reprojection
    # (important for polar projections where lines curve)
    n_points = 10
    coords = []

    # Bottom edge
    for i in range(n_points):
        x = minx + (maxx - minx) * i / (n_points - 1)
        coords.append((x, miny))
    # Right edge
    for i in range(1, n_points):
        y = miny + (maxy - miny) * i / (n_points - 1)
        coords.append((maxx, y))
    # Top edge
    for i in range(1, n_points):
        x = maxx - (maxx - minx) * i / (n_points - 1)
        coords.append((x, maxy))
    # Left edge
    for i in range(1, n_points - 1):
        y = maxy - (maxy - miny) * i / (n_points - 1)
        coords.append((minx, y))

    # Transform all coordinates
    lons, lats = transformer.transform(
        [c[0] for c in coords],
        [c[1] for c in coords]
    )

    return Polygon(zip(lons, lats))


def select_random_land_cells(grid_cells, n_cells=5, seed=42):
    """
    Select random cells that are over Antarctic land.
    """
    random.seed(seed)

    land_cells = [c for c in grid_cells if is_over_land(c)]
    print(f"    Grid has {len(grid_cells)} total cells, {len(land_cells)} over land")

    if len(land_cells) < n_cells:
        return land_cells

    return random.sample(land_cells, n_cells)


def run_query(parquet_path: str, query_polygon: Polygon, format_type: str,
              location: str = 'local') -> dict:
    """
    Run a spatial query and return timing and row count.
    """
    bounds = query_polygon.bounds

    conn = duckdb.connect()

    if location == 'gcs':
        conn.execute("INSTALL httpfs; LOAD httpfs;")

    start_time = time.time()

    try:
        if format_type == 'original':
            query = f"""
                SELECT COUNT(*) as cnt FROM read_parquet('{parquet_path}')
                WHERE "longitude (degree_east)" >= {bounds[0]}
                  AND "longitude (degree_east)" <= {bounds[2]}
                  AND "latitude (degree_north)" >= {bounds[1]}
                  AND "latitude (degree_north)" <= {bounds[3]}
            """
        else:
            conn.execute("INSTALL spatial; LOAD spatial;")
            query = f"""
                SELECT COUNT(*) as cnt FROM read_parquet('{parquet_path}')
                WHERE ST_X(geometry) >= {bounds[0]}
                  AND ST_X(geometry) <= {bounds[2]}
                  AND ST_Y(geometry) >= {bounds[1]}
                  AND ST_Y(geometry) <= {bounds[3]}
            """

        result = conn.execute(query).fetchone()
        row_count = result[0] if result else 0
        elapsed = time.time() - start_time
        success = True
        error = None

    except Exception as e:
        elapsed = time.time() - start_time
        row_count = 0
        success = False
        error = str(e)

    conn.close()

    return {
        'time_s': elapsed,
        'rows': row_count,
        'success': success,
        'error': error
    }


def benchmark_grid_queries(output_dir: Path, location: str = 'local',
                           test_files: list = None):
    """
    Run grid-based query benchmark.

    Parameters
    ----------
    output_dir : Path
        Directory containing parquet files
    location : str
        'local' or 'gcs'
    test_files : list
        List of file stems to test. Defaults to BEDMAP1 and AWI_2016_OIR.
    """
    results = []

    resolutions = [10, 50, 200]  # km

    # Configurations to test
    configs = [
        ('original', 'zstd', False, None),
        ('geoparquet', 'zstd', False, None),
        ('geoparquet', 'zstd', True, None),
        ('geoparquet', 'zstd', False, 50000),
        ('geoparquet', 'zstd', True, 50000),
        ('geoparquet', 'zstd', False, 10000),
        ('geoparquet', 'zstd', True, 10000),
    ]

    # Test files - default to both BEDMAP1 (1.9M rows) and AWI_2016_OIR (564K rows)
    if test_files is None:
        test_files = [
            ('BEDMAP1_1966-2000_AIR_BM1', '1.9M rows'),
        ]

    for test_file_stem, file_desc in test_files:
        print(f"\n  Testing file: {test_file_stem} ({file_desc})")

        # Get file extent from the original format file
        if location == 'local':
            extent_file = output_dir / '_original_zstd' / f"{test_file_stem}.parquet"
            file_url = str(extent_file)
        else:
            file_url = f"gs://opr_stac/bedmap_benchmark/_original_zstd/{test_file_stem}.parquet"

        file_extent = get_file_extent(file_url, 'original', location)
        if file_extent:
            print(f"    File extent: lon [{file_extent[0]:.2f}, {file_extent[2]:.2f}], "
                  f"lat [{file_extent[1]:.2f}, {file_extent[3]:.2f}]")
        else:
            print(f"    Warning: Could not get file extent, using full Antarctic grid")

        for resolution_km in resolutions:
            print(f"\n    Resolution: {resolution_km} km")

            # Create grid limited to file extent
            grid_cells = create_antarctic_grid(resolution_km, extent_latlon=file_extent)
            selected_cells = select_random_land_cells(grid_cells, n_cells=5)

            # Convert to lat/lon polygons
            query_polygons = []
            for i, cell in enumerate(selected_cells):
                try:
                    poly = polar_to_latlon(cell)
                    query_polygons.append((i, cell, poly))
                except Exception as e:
                    print(f"      Warning: Could not convert cell {i}: {e}")

            print(f"      Testing {len(query_polygons)} cells")

            for format_type, compression, use_hilbert, row_group_size in configs:
                suffix = f"_{format_type}_{compression}"
                if use_hilbert:
                    suffix += "_hilbert"
                if row_group_size:
                    suffix += f"_rg{row_group_size//1000}k"

                if location == 'local':
                    parquet_path = output_dir / suffix / f"{test_file_stem}.parquet"
                    if not parquet_path.exists():
                        continue
                    file_url = str(parquet_path)
                else:
                    file_url = f"gs://opr_stac/bedmap_benchmark/{suffix}/{test_file_stem}.parquet"

                config_results = []

                for cell_idx, cell_bounds, query_poly in query_polygons:
                    result = run_query(file_url, query_poly, format_type, location)

                    config_results.append({
                        'file': test_file_stem,
                        'resolution_km': resolution_km,
                        'cell_idx': cell_idx,
                        'location': location,
                        'format': format_type,
                        'hilbert': use_hilbert,
                        'row_group_size': row_group_size or 'all',
                        'time_s': result['time_s'],
                        'rows': result['rows'],
                        'success': result['success'],
                    })

                if config_results:
                    times = [r['time_s'] for r in config_results if r['success']]
                    rows = [r['rows'] for r in config_results if r['success']]

                    if times:
                        rg_str = f"rg={row_group_size//1000}k" if row_group_size else "rg=all"
                        hilbert_str = " hilbert" if use_hilbert else ""
                        avg_rows = sum(rows) / len(rows) if rows else 0
                        print(f"        {format_type} {rg_str}{hilbert_str}: "
                              f"min={min(times):.4f}s, max={max(times):.4f}s, "
                              f"mean={sum(times)/len(times):.4f}s, avg_rows={avg_rows:.0f}")

                results.extend(config_results)

    return pd.DataFrame(results)


def main():
    print("=" * 80)
    print("Grid-Based Query Benchmark: Simulating Realistic User Queries")
    print("=" * 80)

    # Find the output directory from previous benchmark
    import glob
    temp_dirs = glob.glob('/tmp/geoparquet_cloud_benchmark_*')
    if not temp_dirs:
        print("ERROR: No benchmark output directory found. Run benchmark_geoparquet_cloud.py first.")
        return

    output_dir = Path(sorted(temp_dirs)[-1])  # Use most recent
    print(f"\nUsing output directory: {output_dir}")

    # Check if BEDMAP1 file exists
    bedmap1_path = output_dir / '_original_zstd' / 'BEDMAP1_1966-2000_AIR_BM1.parquet'
    if not bedmap1_path.exists():
        print(f"ERROR: BEDMAP1 file not found at {bedmap1_path}")
        print("Available files:")
        for p in output_dir.glob('*/*.parquet'):
            print(f"  {p}")
        return

    # Run local benchmarks
    print("\n" + "-" * 80)
    print("LOCAL QUERY BENCHMARK")
    print("-" * 80)

    local_results = benchmark_grid_queries(output_dir, location='local')

    # Run cloud benchmarks
    print("\n" + "-" * 80)
    print("CLOUD (GCS) QUERY BENCHMARK")
    print("-" * 80)

    try:
        cloud_results = benchmark_grid_queries(output_dir, location='gcs')
    except Exception as e:
        print(f"Cloud benchmark failed: {e}")
        cloud_results = pd.DataFrame()

    # Combine results
    all_results = pd.concat([local_results, cloud_results], ignore_index=True)

    # Save results
    results_dir = Path(__file__).parent / 'benchmark_results'
    results_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    all_results.to_csv(results_dir / f'grid_queries_{timestamp}.csv', index=False)
    print(f"\nSaved: grid_queries_{timestamp}.csv")

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY: GRID QUERY PERFORMANCE")
    print("=" * 80)

    for location in ['local', 'gcs']:
        loc_data = all_results[all_results['location'] == location]
        if loc_data.empty:
            continue

        print(f"\n  {location.upper()} QUERIES:")

        for resolution in [10, 50, 200]:
            res_data = loc_data[loc_data['resolution_km'] == resolution]
            if res_data.empty:
                continue

            print(f"\n    {resolution}km grid cells:")

            # Group by configuration
            for (fmt, hilbert, rg), group in res_data.groupby(['format', 'hilbert', 'row_group_size']):
                times = group['time_s'].values
                rows = group['rows'].values

                rg_str = f"rg={rg}" if rg != 'all' else "rg=all"
                hilbert_str = " hilbert" if hilbert else ""

                print(f"      {fmt} {rg_str}{hilbert_str}:")
                print(f"        Time - min: {times.min():.4f}s, max: {times.max():.4f}s, mean: {times.mean():.4f}s")
                print(f"        Rows - min: {rows.min()}, max: {rows.max()}, mean: {rows.mean():.0f}")


if __name__ == '__main__':
    main()
