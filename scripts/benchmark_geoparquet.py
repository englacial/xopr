#!/usr/bin/env python
"""
Benchmark script comparing original parquet format vs GeoParquet with WKB Point geometry.

Tests:
1. File size comparison (snappy vs zstd compression)
2. Query performance for spatial subsets
3. Hilbert ordering impact
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

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from xopr.bedmap.converter import convert_bedmap_csv, convert_bedmap_csv_geoparquet
from xopr.bedmap.query import build_duckdb_query, build_duckdb_query_geoparquet


# Test files (varying sizes)
TEST_FILES = [
    '/home/espg/software/bedmap/Results/AWI_1996_DML3_AIR_BM2.csv',      # ~7.5K rows
    '/home/espg/software/bedmap/Results/AWI_1994_DML1_AIR_BM2.csv',      # ~25K rows
    '/home/espg/software/bedmap/Results/AWI_2001_DML7_AIR_BM2.csv',      # ~165K rows
    '/home/espg/software/bedmap/Results/AWI_2018_JURAS_AIR_BM3.csv',     # ~190K rows
    '/home/espg/software/bedmap/Results/AWI_2016_OIR_AIR_BM3.csv',       # ~564K rows
]


def get_file_size_kb(path):
    """Get file size in KB."""
    return os.path.getsize(path) / 1024


def benchmark_file_sizes(output_dir: Path):
    """
    Benchmark file sizes for different formats and compression options.

    Returns DataFrame with results.
    """
    results = []

    for csv_path in TEST_FILES:
        csv_path = Path(csv_path)
        if not csv_path.exists():
            print(f"  Skipping {csv_path.name} - not found")
            continue

        # Count rows
        with open(csv_path, 'r') as f:
            row_count = sum(1 for line in f if not line.startswith('#')) - 1  # -1 for header

        csv_size = get_file_size_kb(csv_path)

        print(f"\n  Processing {csv_path.name} ({row_count:,} rows, {csv_size:.1f} KB CSV)")

        # Test configurations
        configs = [
            ('original', 'snappy', False),
            ('original', 'zstd', False),
            ('geoparquet', 'snappy', False),
            ('geoparquet', 'zstd', False),
            ('geoparquet', 'snappy', True),   # With Hilbert
            ('geoparquet', 'zstd', True),     # With Hilbert
        ]

        for format_type, compression, use_hilbert in configs:
            try:
                suffix = f"_{format_type}_{compression}"
                if use_hilbert:
                    suffix += "_hilbert"

                out_dir = output_dir / suffix
                out_dir.mkdir(parents=True, exist_ok=True)

                start_time = time.time()

                if format_type == 'original':
                    convert_bedmap_csv(csv_path, out_dir, compression=compression)
                else:
                    convert_bedmap_csv_geoparquet(
                        csv_path, out_dir,
                        compression=compression,
                        use_hilbert=use_hilbert
                    )

                convert_time = time.time() - start_time

                # Get output file size
                output_file = out_dir / f"{csv_path.stem}.parquet"
                parquet_size = get_file_size_kb(output_file)

                results.append({
                    'file': csv_path.stem,
                    'rows': row_count,
                    'csv_size_kb': csv_size,
                    'format': format_type,
                    'compression': compression,
                    'hilbert': use_hilbert,
                    'parquet_size_kb': parquet_size,
                    'convert_time_s': convert_time,
                    'compression_ratio': csv_size / parquet_size,
                })

                print(f"    {format_type}/{compression}" +
                      (" (hilbert)" if use_hilbert else "") +
                      f": {parquet_size:.1f} KB ({convert_time:.2f}s)")

            except Exception as e:
                print(f"    ERROR {format_type}/{compression}: {e}")

    return pd.DataFrame(results)


def benchmark_query_performance(output_dir: Path):
    """
    Benchmark query performance for spatial subsets.

    Tests different spatial extent sizes:
    - Small: ~1 degree box
    - Medium: ~5 degree box
    - Large: ~20 degree box
    """
    results = []

    # Define spatial query boxes based on AWI_2016_OIR extent (lon: 0.4-47.7, lat: -80 to -75)
    query_boxes = {
        'small': box(30, -80, 35, -78),        # ~5x2 degrees (within data extent)
        'medium': box(20, -80, 40, -76),       # ~20x4 degrees
        'large': box(5, -80, 45, -75),         # ~40x5 degrees (most of data)
    }

    # Use the largest file for query benchmarks
    test_csv = Path(TEST_FILES[-1])  # AWI_2016_OIR
    if not test_csv.exists():
        test_csv = Path(TEST_FILES[2])  # Fallback to AWI_2001_DML7

    print(f"\n  Using {test_csv.name} for query benchmarks")

    configs = [
        ('original', 'snappy', False),
        ('original', 'zstd', False),
        ('geoparquet', 'snappy', False),
        ('geoparquet', 'zstd', False),
        ('geoparquet', 'snappy', True),
        ('geoparquet', 'zstd', True),
    ]

    for format_type, compression, use_hilbert in configs:
        suffix = f"_{format_type}_{compression}"
        if use_hilbert:
            suffix += "_hilbert"

        parquet_path = output_dir / suffix / f"{test_csv.stem}.parquet"

        if not parquet_path.exists():
            print(f"    Skipping {suffix} - file not found")
            continue

        for box_name, query_box in query_boxes.items():
            try:
                # Run multiple iterations for timing
                n_iterations = 3
                times = []
                row_counts = []

                for _ in range(n_iterations):
                    conn = duckdb.connect()

                    start_time = time.time()

                    if format_type == 'original':
                        query = build_duckdb_query(
                            [str(parquet_path)],
                            geometry=query_box,
                            use_polar_filter=False
                        )
                        result = conn.execute(query).df()
                    else:
                        conn.execute("INSTALL spatial; LOAD spatial;")
                        query = build_duckdb_query_geoparquet(
                            [str(parquet_path)],
                            geometry=query_box,
                            use_polar_filter=False
                        )
                        result = conn.execute(query).df()

                    elapsed = time.time() - start_time
                    times.append(elapsed)
                    row_counts.append(len(result))

                    conn.close()

                avg_time = sum(times) / len(times)
                avg_rows = sum(row_counts) / len(row_counts)

                results.append({
                    'format': format_type,
                    'compression': compression,
                    'hilbert': use_hilbert,
                    'query_extent': box_name,
                    'avg_time_s': avg_time,
                    'rows_returned': int(avg_rows),
                })

                print(f"    {suffix} / {box_name}: {avg_time:.4f}s ({int(avg_rows)} rows)")

            except Exception as e:
                print(f"    ERROR {suffix} / {box_name}: {e}")

    return pd.DataFrame(results)


def main():
    print("=" * 70)
    print("GeoParquet WKB Benchmark: Original vs Point Geometry")
    print("=" * 70)

    # Create temp directory for outputs
    output_dir = Path(tempfile.mkdtemp(prefix='geoparquet_benchmark_'))
    print(f"\nOutput directory: {output_dir}")

    # Check DuckDB version
    conn = duckdb.connect()
    version = conn.execute("SELECT version()").fetchone()[0]
    conn.close()
    print(f"DuckDB version: {version}")

    # 1. File size benchmarks
    print("\n" + "-" * 70)
    print("FILE SIZE BENCHMARKS")
    print("-" * 70)

    size_results = benchmark_file_sizes(output_dir)

    if not size_results.empty:
        print("\n  SUMMARY - File Sizes:")
        summary = size_results.pivot_table(
            values='parquet_size_kb',
            index='file',
            columns=['format', 'compression', 'hilbert'],
            aggfunc='first'
        )
        print(summary.to_string())

        # Calculate averages
        print("\n  AVERAGE SIZE BY FORMAT:")
        avg_by_format = size_results.groupby(['format', 'compression', 'hilbert'])['parquet_size_kb'].mean()
        print(avg_by_format.to_string())

    # 2. Query performance benchmarks
    print("\n" + "-" * 70)
    print("QUERY PERFORMANCE BENCHMARKS")
    print("-" * 70)

    query_results = benchmark_query_performance(output_dir)

    if not query_results.empty:
        print("\n  SUMMARY - Query Times:")
        summary = query_results.pivot_table(
            values='avg_time_s',
            index='query_extent',
            columns=['format', 'compression', 'hilbert'],
            aggfunc='first'
        )
        print(summary.to_string())

    # 3. Save results
    print("\n" + "-" * 70)
    print("SAVING RESULTS")
    print("-" * 70)

    results_dir = Path(__file__).parent / 'benchmark_results'
    results_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    if not size_results.empty:
        size_results.to_csv(results_dir / f'file_sizes_{timestamp}.csv', index=False)
        print(f"  Saved: file_sizes_{timestamp}.csv")

    if not query_results.empty:
        query_results.to_csv(results_dir / f'query_times_{timestamp}.csv', index=False)
        print(f"  Saved: query_times_{timestamp}.csv")

    # Print key findings
    print("\n" + "=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)

    if not size_results.empty:
        # Compare original vs geoparquet
        orig = size_results[size_results['format'] == 'original']['parquet_size_kb'].mean()
        geo = size_results[(size_results['format'] == 'geoparquet') &
                          (size_results['hilbert'] == False)]['parquet_size_kb'].mean()

        print(f"\n  File Size:")
        print(f"    Original (avg): {orig:.1f} KB")
        print(f"    GeoParquet (avg): {geo:.1f} KB")
        print(f"    Difference: {((geo - orig) / orig * 100):.1f}%")

        # Compare snappy vs zstd
        snappy = size_results[size_results['compression'] == 'snappy']['parquet_size_kb'].mean()
        zstd = size_results[size_results['compression'] == 'zstd']['parquet_size_kb'].mean()

        print(f"\n  Compression:")
        print(f"    Snappy (avg): {snappy:.1f} KB")
        print(f"    ZSTD (avg): {zstd:.1f} KB")
        print(f"    Savings with ZSTD: {((snappy - zstd) / snappy * 100):.1f}%")

    if not query_results.empty:
        # Compare query times
        orig_time = query_results[query_results['format'] == 'original']['avg_time_s'].mean()
        geo_time = query_results[(query_results['format'] == 'geoparquet') &
                                 (query_results['hilbert'] == False)]['avg_time_s'].mean()

        print(f"\n  Query Performance:")
        print(f"    Original (avg): {orig_time:.4f}s")
        print(f"    GeoParquet (avg): {geo_time:.4f}s")
        print(f"    Difference: {((geo_time - orig_time) / orig_time * 100):.1f}%")

    print(f"\n  Temp files in: {output_dir}")
    print("  (Delete manually when done reviewing)")

    return size_results, query_results


if __name__ == '__main__':
    main()
