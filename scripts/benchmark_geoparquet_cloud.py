#!/usr/bin/env python
"""
Comprehensive benchmark comparing parquet formats for local and cloud storage.

Tests:
1. File size comparison (snappy vs zstd, various row group sizes)
2. Local query performance
3. Cloud (GCS) query performance
4. Impact of Hilbert sorting with row groups
"""

import os
import sys
import time
import tempfile
from pathlib import Path
from datetime import datetime
import pandas as pd
import geopandas as gpd
from shapely.geometry import box
import duckdb
import warnings

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from xopr.bedmap.converter import convert_bedmap_csv, convert_bedmap_csv_geoparquet


# Test files (varying sizes) - added BEDMAP1 as 6th file
TEST_FILES = [
    '/home/espg/software/bedmap/Results/AWI_1996_DML3_AIR_BM2.csv',      # ~7.5K rows
    '/home/espg/software/bedmap/Results/AWI_1994_DML1_AIR_BM2.csv',      # ~25K rows
    '/home/espg/software/bedmap/Results/AWI_2001_DML7_AIR_BM2.csv',      # ~165K rows
    '/home/espg/software/bedmap/Results/AWI_2018_JURAS_AIR_BM3.csv',     # ~190K rows
    '/home/espg/software/bedmap/Results/AWI_2016_OIR_AIR_BM3.csv',       # ~564K rows
    '/home/espg/software/bedmap/Results/BEDMAP1_1966-2000_AIR_BM1.csv',  # ~1.9M rows
]

# GCS bucket for cloud tests
GCS_BUCKET = 'gs://opr_stac/bedmap_benchmark'


def get_file_size_kb(path):
    """Get file size in KB."""
    return os.path.getsize(path) / 1024


def count_csv_rows(csv_path):
    """Count data rows in CSV (excluding comments and header)."""
    with open(csv_path, 'r') as f:
        return sum(1 for line in f if not line.startswith('#')) - 1


def benchmark_file_sizes(output_dir: Path, test_files: list = None):
    """
    Benchmark file sizes for different formats, compressions, and row group sizes.
    """
    if test_files is None:
        test_files = TEST_FILES

    results = []

    # Test configurations: (format, compression, hilbert, row_group_size)
    configs = [
        # Original format
        ('original', 'snappy', False, None),
        ('original', 'zstd', False, None),
        # GeoParquet - single row group (default)
        ('geoparquet', 'zstd', False, None),
        ('geoparquet', 'zstd', True, None),
        # GeoParquet - 50k row groups
        ('geoparquet', 'zstd', False, 50000),
        ('geoparquet', 'zstd', True, 50000),
        # GeoParquet - 10k row groups
        ('geoparquet', 'zstd', False, 10000),
        ('geoparquet', 'zstd', True, 10000),
    ]

    for csv_path in test_files:
        csv_path = Path(csv_path)
        if not csv_path.exists():
            print(f"  Skipping {csv_path.name} - not found")
            continue

        row_count = count_csv_rows(csv_path)
        csv_size = get_file_size_kb(csv_path)

        print(f"\n  Processing {csv_path.name} ({row_count:,} rows, {csv_size:.1f} KB CSV)")

        for format_type, compression, use_hilbert, row_group_size in configs:
            try:
                suffix = f"_{format_type}_{compression}"
                if use_hilbert:
                    suffix += "_hilbert"
                if row_group_size:
                    suffix += f"_rg{row_group_size//1000}k"

                out_dir = output_dir / suffix
                out_dir.mkdir(parents=True, exist_ok=True)

                start_time = time.time()

                if format_type == 'original':
                    convert_bedmap_csv(csv_path, out_dir, compression=compression)
                else:
                    convert_bedmap_csv_geoparquet(
                        csv_path, out_dir,
                        compression=compression,
                        use_hilbert=use_hilbert,
                        row_group_size=row_group_size
                    )

                convert_time = time.time() - start_time

                output_file = out_dir / f"{csv_path.stem}.parquet"
                parquet_size = get_file_size_kb(output_file)

                # Count row groups
                import pyarrow.parquet as pq
                pq_file = pq.ParquetFile(output_file)
                num_row_groups = pq_file.metadata.num_row_groups

                results.append({
                    'file': csv_path.stem,
                    'rows': row_count,
                    'csv_size_kb': csv_size,
                    'format': format_type,
                    'compression': compression,
                    'hilbert': use_hilbert,
                    'row_group_size': row_group_size or 'all',
                    'num_row_groups': num_row_groups,
                    'parquet_size_kb': parquet_size,
                    'convert_time_s': convert_time,
                })

                rg_str = f"rg={num_row_groups}" if row_group_size else "rg=1"
                hilbert_str = " hilbert" if use_hilbert else ""
                print(f"    {format_type}/{compression} {rg_str}{hilbert_str}: {parquet_size:.1f} KB ({convert_time:.2f}s)")

            except Exception as e:
                print(f"    ERROR {format_type}/{compression}: {e}")
                import traceback
                traceback.print_exc()

    return pd.DataFrame(results)


def upload_to_gcs(local_dir: Path, gcs_path: str):
    """Upload parquet files to GCS."""
    import subprocess

    print(f"\n  Uploading to {gcs_path}...")
    for subdir in local_dir.iterdir():
        if subdir.is_dir():
            for parquet_file in subdir.glob('*.parquet'):
                gcs_dest = f"{gcs_path}/{subdir.name}/{parquet_file.name}"
                cmd = f"gsutil -q cp {parquet_file} {gcs_dest}"
                subprocess.run(cmd, shell=True, check=True)
                print(f"    Uploaded {subdir.name}/{parquet_file.name}")


def benchmark_query_performance(output_dir: Path, test_file_stem: str, location: str = 'local'):
    """
    Benchmark query performance for spatial subsets.

    Parameters
    ----------
    output_dir : Path
        Directory containing parquet files (for local) or GCS path prefix
    test_file_stem : str
        Stem of the test file to query
    location : str
        'local' or 'gcs'
    """
    results = []

    # Get spatial extent from one of the files to create appropriate query boxes
    # For BEDMAP1, extent is roughly: lon -180 to 180, lat -90 to -60
    # For AWI_2016_OIR, extent is: lon 0.4-47.7, lat -80 to -75

    if 'BEDMAP1' in test_file_stem:
        # BEDMAP1 covers most of Antarctica
        query_boxes = {
            'small': box(0, -80, 10, -75),          # ~10x5 degrees
            'medium': box(-30, -85, 30, -70),       # ~60x15 degrees
            'large': box(-90, -90, 90, -60),        # ~180x30 degrees (half continent)
        }
    else:
        # AWI_2016_OIR extent
        query_boxes = {
            'small': box(30, -80, 35, -78),
            'medium': box(20, -80, 40, -76),
            'large': box(5, -80, 45, -75),
        }

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
            file_url = f"{GCS_BUCKET}/{suffix}/{test_file_stem}.parquet"

        for box_name, query_box in query_boxes.items():
            try:
                n_iterations = 3
                times = []
                row_counts = []

                for iteration in range(n_iterations):
                    conn = duckdb.connect()

                    # For GCS, need to configure credentials
                    if location == 'gcs':
                        conn.execute("INSTALL httpfs; LOAD httpfs;")
                        conn.execute("SET s3_region='us-central1';")

                    start_time = time.time()

                    bounds = query_box.bounds

                    if format_type == 'original':
                        query = f"""
                            SELECT * FROM read_parquet('{file_url}')
                            WHERE "longitude (degree_east)" >= {bounds[0]}
                              AND "longitude (degree_east)" <= {bounds[2]}
                              AND "latitude (degree_north)" >= {bounds[1]}
                              AND "latitude (degree_north)" <= {bounds[3]}
                        """
                        result = conn.execute(query).df()
                    else:
                        conn.execute("INSTALL spatial; LOAD spatial;")
                        query = f"""
                            SELECT *, ST_X(geometry) as lon, ST_Y(geometry) as lat
                            FROM read_parquet('{file_url}')
                            WHERE ST_X(geometry) >= {bounds[0]}
                              AND ST_X(geometry) <= {bounds[2]}
                              AND ST_Y(geometry) >= {bounds[1]}
                              AND ST_Y(geometry) <= {bounds[3]}
                        """
                        result = conn.execute(query).df()

                    elapsed = time.time() - start_time
                    times.append(elapsed)
                    row_counts.append(len(result))

                    conn.close()

                avg_time = sum(times) / len(times)
                avg_rows = sum(row_counts) / len(row_counts)

                results.append({
                    'location': location,
                    'format': format_type,
                    'compression': compression,
                    'hilbert': use_hilbert,
                    'row_group_size': row_group_size or 'all',
                    'query_extent': box_name,
                    'avg_time_s': avg_time,
                    'rows_returned': int(avg_rows),
                    'file': test_file_stem,
                })

                rg_str = f"rg={row_group_size//1000}k" if row_group_size else "rg=all"
                hilbert_str = " hilbert" if use_hilbert else ""
                print(f"    [{location}] {format_type} {rg_str}{hilbert_str} / {box_name}: {avg_time:.4f}s ({int(avg_rows)} rows)")

            except Exception as e:
                print(f"    ERROR {suffix} / {box_name}: {e}")

    return pd.DataFrame(results)


def main():
    print("=" * 80)
    print("Comprehensive GeoParquet Benchmark: Local vs Cloud, Row Groups, Hilbert")
    print("=" * 80)

    # Create temp directory for outputs
    output_dir = Path(tempfile.mkdtemp(prefix='geoparquet_cloud_benchmark_'))
    print(f"\nOutput directory: {output_dir}")

    # Check DuckDB version
    conn = duckdb.connect()
    version = conn.execute("SELECT version()").fetchone()[0]
    conn.close()
    print(f"DuckDB version: {version}")

    # For initial testing, use a subset of files
    # Comment out to use all files
    test_files = [
        TEST_FILES[2],  # AWI_2001_DML7 - 165K rows (medium)
        TEST_FILES[4],  # AWI_2016_OIR - 564K rows (large)
        TEST_FILES[5],  # BEDMAP1 - 1.9M rows (very large)
    ]

    # 1. File size benchmarks
    print("\n" + "-" * 80)
    print("PHASE 1: FILE SIZE BENCHMARKS")
    print("-" * 80)

    size_results = benchmark_file_sizes(output_dir, test_files)

    if not size_results.empty:
        print("\n  FILE SIZE SUMMARY:")
        pivot = size_results.pivot_table(
            values='parquet_size_kb',
            index='file',
            columns=['format', 'hilbert', 'row_group_size'],
            aggfunc='first'
        )
        print(pivot.to_string())

    # 2. Local query benchmarks
    print("\n" + "-" * 80)
    print("PHASE 2: LOCAL QUERY PERFORMANCE")
    print("-" * 80)

    local_query_results = []

    # Test with medium and large files
    for test_file in [TEST_FILES[4], TEST_FILES[5]]:  # AWI_2016_OIR and BEDMAP1
        test_stem = Path(test_file).stem
        if not Path(test_file).exists():
            continue
        print(f"\n  Testing {test_stem}...")
        results = benchmark_query_performance(output_dir, test_stem, location='local')
        local_query_results.append(results)

    if local_query_results:
        local_df = pd.concat(local_query_results, ignore_index=True)

        print("\n  LOCAL QUERY SUMMARY:")
        pivot = local_df.pivot_table(
            values='avg_time_s',
            index=['file', 'query_extent'],
            columns=['format', 'hilbert', 'row_group_size'],
            aggfunc='first'
        )
        print(pivot.to_string())

    # 3. Upload to GCS and test cloud queries
    print("\n" + "-" * 80)
    print("PHASE 3: CLOUD (GCS) UPLOAD AND QUERY PERFORMANCE")
    print("-" * 80)

    try:
        upload_to_gcs(output_dir, GCS_BUCKET)

        cloud_query_results = []

        for test_file in [TEST_FILES[4], TEST_FILES[5]]:
            test_stem = Path(test_file).stem
            if not Path(test_file).exists():
                continue
            print(f"\n  Testing {test_stem} from GCS...")
            results = benchmark_query_performance(output_dir, test_stem, location='gcs')
            cloud_query_results.append(results)

        if cloud_query_results:
            cloud_df = pd.concat(cloud_query_results, ignore_index=True)

            print("\n  CLOUD QUERY SUMMARY:")
            pivot = cloud_df.pivot_table(
                values='avg_time_s',
                index=['file', 'query_extent'],
                columns=['format', 'hilbert', 'row_group_size'],
                aggfunc='first'
            )
            print(pivot.to_string())

    except Exception as e:
        print(f"  Cloud tests failed: {e}")
        cloud_df = pd.DataFrame()

    # 4. Save results
    print("\n" + "-" * 80)
    print("SAVING RESULTS")
    print("-" * 80)

    results_dir = Path(__file__).parent / 'benchmark_results'
    results_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    if not size_results.empty:
        size_results.to_csv(results_dir / f'cloud_file_sizes_{timestamp}.csv', index=False)
        print(f"  Saved: cloud_file_sizes_{timestamp}.csv")

    if local_query_results:
        local_df.to_csv(results_dir / f'cloud_local_queries_{timestamp}.csv', index=False)
        print(f"  Saved: cloud_local_queries_{timestamp}.csv")

    if 'cloud_df' in dir() and not cloud_df.empty:
        cloud_df.to_csv(results_dir / f'cloud_gcs_queries_{timestamp}.csv', index=False)
        print(f"  Saved: cloud_gcs_queries_{timestamp}.csv")

    # 5. Print comprehensive findings
    print("\n" + "=" * 80)
    print("COMPREHENSIVE FINDINGS")
    print("=" * 80)

    if not size_results.empty:
        print("\n  FILE SIZE IMPACT:")

        # Compare formats
        orig = size_results[size_results['format'] == 'original']['parquet_size_kb'].mean()
        geo_single = size_results[(size_results['format'] == 'geoparquet') &
                                   (size_results['row_group_size'] == 'all') &
                                   (size_results['hilbert'] == False)]['parquet_size_kb'].mean()
        geo_50k = size_results[(size_results['format'] == 'geoparquet') &
                                (size_results['row_group_size'] == 50000)]['parquet_size_kb'].mean()
        geo_10k = size_results[(size_results['format'] == 'geoparquet') &
                                (size_results['row_group_size'] == 10000)]['parquet_size_kb'].mean()

        print(f"    Original (zstd): {orig:.1f} KB (baseline)")
        print(f"    GeoParquet single rg: {geo_single:.1f} KB ({((geo_single-orig)/orig*100):+.1f}%)")
        print(f"    GeoParquet 50k rg: {geo_50k:.1f} KB ({((geo_50k-orig)/orig*100):+.1f}%)")
        print(f"    GeoParquet 10k rg: {geo_10k:.1f} KB ({((geo_10k-orig)/orig*100):+.1f}%)")

    if local_query_results:
        print("\n  LOCAL QUERY PERFORMANCE:")
        for extent in ['small', 'medium', 'large']:
            subset = local_df[local_df['query_extent'] == extent]
            if subset.empty:
                continue

            orig_time = subset[subset['format'] == 'original']['avg_time_s'].mean()
            geo_single = subset[(subset['format'] == 'geoparquet') &
                                (subset['row_group_size'] == 'all') &
                                (subset['hilbert'] == False)]['avg_time_s'].mean()
            geo_50k_hilbert = subset[(subset['format'] == 'geoparquet') &
                                      (subset['row_group_size'] == 50000) &
                                      (subset['hilbert'] == True)]['avg_time_s'].mean()

            print(f"    {extent.upper()} extent:")
            print(f"      Original: {orig_time:.4f}s")
            print(f"      GeoParquet (single rg): {geo_single:.4f}s ({((geo_single-orig_time)/orig_time*100):+.1f}%)")
            if not pd.isna(geo_50k_hilbert):
                print(f"      GeoParquet (50k rg + hilbert): {geo_50k_hilbert:.4f}s ({((geo_50k_hilbert-orig_time)/orig_time*100):+.1f}%)")

    if 'cloud_df' in dir() and not cloud_df.empty:
        print("\n  CLOUD (GCS) QUERY PERFORMANCE:")
        for extent in ['small', 'medium', 'large']:
            subset = cloud_df[cloud_df['query_extent'] == extent]
            if subset.empty:
                continue

            orig_time = subset[subset['format'] == 'original']['avg_time_s'].mean()
            geo_single = subset[(subset['format'] == 'geoparquet') &
                                (subset['row_group_size'] == 'all') &
                                (subset['hilbert'] == False)]['avg_time_s'].mean()
            geo_50k_hilbert = subset[(subset['format'] == 'geoparquet') &
                                      (subset['row_group_size'] == 50000) &
                                      (subset['hilbert'] == True)]['avg_time_s'].mean()

            print(f"    {extent.upper()} extent:")
            print(f"      Original: {orig_time:.4f}s")
            print(f"      GeoParquet (single rg): {geo_single:.4f}s ({((geo_single-orig_time)/orig_time*100):+.1f}%)")
            if not pd.isna(geo_50k_hilbert):
                print(f"      GeoParquet (50k rg + hilbert): {geo_50k_hilbert:.4f}s ({((geo_50k_hilbert-orig_time)/orig_time*100):+.1f}%)")

    print(f"\n  Temp files in: {output_dir}")
    print("  (Delete manually when done reviewing)")

    return size_results, local_df if local_query_results else pd.DataFrame()


if __name__ == '__main__':
    main()
