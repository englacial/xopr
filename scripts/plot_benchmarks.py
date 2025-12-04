#!/usr/bin/env python
"""
Generate benchmark plots from CSV results.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import glob


def load_latest_results():
    """Load the most recent benchmark results."""
    results_dir = Path(__file__).parent / 'benchmark_results'

    # Load cloud benchmark results
    cloud_files = sorted(glob.glob(str(results_dir / 'cloud_*.csv')))

    cloud_sizes = None
    cloud_local = None
    cloud_gcs = None

    for f in cloud_files:
        if 'file_sizes' in f:
            cloud_sizes = pd.read_csv(f)
        elif 'local_queries' in f:
            cloud_local = pd.read_csv(f)
        elif 'gcs_queries' in f:
            cloud_gcs = pd.read_csv(f)

    # Load grid query results (both BEDMAP1 and AWI)
    grid_files = sorted(glob.glob(str(results_dir / 'grid_queries_*.csv')))
    grid_dfs = []
    for f in grid_files:
        df = pd.read_csv(f)
        grid_dfs.append(df)

    grid_results = pd.concat(grid_dfs, ignore_index=True) if grid_dfs else None

    return cloud_sizes, cloud_local, cloud_gcs, grid_results


def plot_file_sizes(df, output_dir):
    """Plot file size comparison."""
    if df is None:
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    # Prepare data - focus on BEDMAP1 which shows the biggest difference
    bedmap1 = df[df['file'] == 'BEDMAP1_1966-2000_AIR_BM1'].copy()

    # Create labels
    configs = []
    sizes = []
    colors = []

    for _, row in bedmap1.iterrows():
        label = f"{row['format']}"
        if row['hilbert']:
            label += " +hilbert"
        if row['row_group_size'] != 'all':
            label += f" rg={int(row['row_group_size'])//1000}k"
        else:
            label += " rg=all"
        configs.append(label)
        sizes.append(row['parquet_size_kb'] / 1024)  # Convert to MB

        if row['format'] == 'original':
            colors.append('#1f77b4')  # Blue for original
        elif row['hilbert']:
            colors.append('#2ca02c')  # Green for hilbert
        else:
            colors.append('#ff7f0e')  # Orange for geoparquet

    y_pos = np.arange(len(configs))
    ax.barh(y_pos, sizes, color=colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(configs)
    ax.set_xlabel('File Size (MB)')
    ax.set_title('BEDMAP1 File Size by Configuration (1.9M rows)')
    ax.axvline(x=sizes[0], color='red', linestyle='--', alpha=0.5, label='Original baseline')

    # Add size labels
    for i, (size, config) in enumerate(zip(sizes, configs)):
        pct = (size - sizes[0]) / sizes[0] * 100
        sign = '+' if pct > 0 else ''
        ax.text(size + 1, i, f'{size:.1f} MB ({sign}{pct:.0f}%)', va='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_dir / 'file_sizes_bedmap1.png', dpi=150)
    plt.close()
    print(f"  Saved: file_sizes_bedmap1.png")


def plot_local_vs_cloud(cloud_local, cloud_gcs, output_dir):
    """Plot local vs cloud query performance."""
    if cloud_local is None or cloud_gcs is None:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Filter for BEDMAP1
    local = cloud_local[cloud_local['file'] == 'BEDMAP1_1966-2000_AIR_BM1'].copy()
    gcs = cloud_gcs[cloud_gcs['file'] == 'BEDMAP1_1966-2000_AIR_BM1'].copy()

    for ax, (data, title, ylabel) in zip(axes, [
        (local, 'Local Query Performance (BEDMAP1)', 'Time (seconds)'),
        (gcs, 'Cloud (GCS) Query Performance (BEDMAP1)', 'Time (seconds)')
    ]):
        # Group by configuration and query extent
        pivot = data.pivot_table(
            values='avg_time_s',
            index='query_extent',
            columns=['format', 'hilbert', 'row_group_size'],
            aggfunc='mean'
        )

        # Reorder index
        order = ['small', 'medium', 'large']
        pivot = pivot.reindex(order)

        # Plot
        x = np.arange(len(order))
        width = 0.12

        configs = [
            ('original', False, 'all', 'Original', '#1f77b4'),
            ('geoparquet', False, 'all', 'GeoParquet', '#ff7f0e'),
            ('geoparquet', True, 'all', 'GeoParquet +hilbert', '#2ca02c'),
            ('geoparquet', False, '50000', 'GeoParquet rg=50k', '#d62728'),
            ('geoparquet', True, '50000', 'GeoParquet rg=50k +hilbert', '#9467bd'),
            ('geoparquet', False, '10000', 'GeoParquet rg=10k', '#8c564b'),
            ('geoparquet', True, '10000', 'GeoParquet rg=10k +hilbert', '#e377c2'),
        ]

        for i, (fmt, hilbert, rg, label, color) in enumerate(configs):
            try:
                # Try multiple formats for row_group_size (int or string)
                try:
                    values = pivot[(fmt, hilbert, rg)].values
                except KeyError:
                    # Try int version if string didn't work
                    rg_int = int(rg) if rg != 'all' else rg
                    values = pivot[(fmt, hilbert, rg_int)].values
                offset = (i - len(configs)/2 + 0.5) * width
                bars = ax.bar(x + offset, values, width, label=label, color=color)
            except KeyError:
                continue

        ax.set_xlabel('Query Extent')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(['Small\n(12K rows)', 'Medium\n(274K rows)', 'Large\n(1.4M rows)'])
        ax.legend(loc='upper left', fontsize=8)
        ax.set_yscale('log')

    plt.tight_layout()
    plt.savefig(output_dir / 'local_vs_cloud_bedmap1.png', dpi=150)
    plt.close()
    print(f"  Saved: local_vs_cloud_bedmap1.png")


def plot_cloud_speedup(cloud_local, cloud_gcs, output_dir):
    """Plot cloud query speedup vs original."""
    if cloud_gcs is None:
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    # Filter for BEDMAP1
    gcs = cloud_gcs[cloud_gcs['file'] == 'BEDMAP1_1966-2000_AIR_BM1'].copy()

    # Get original baseline times
    orig = gcs[(gcs['format'] == 'original') & (gcs['row_group_size'] == 'all')]
    orig_times = orig.set_index('query_extent')['avg_time_s'].to_dict()

    # Calculate speedup for each configuration
    configs = [
        ('geoparquet', False, 'all', 'GeoParquet rg=all'),
        ('geoparquet', True, 'all', 'GeoParquet rg=all +hilbert'),
        ('geoparquet', False, 50000, 'GeoParquet rg=50k'),
        ('geoparquet', True, 50000, 'GeoParquet rg=50k +hilbert'),
        ('geoparquet', False, 10000, 'GeoParquet rg=10k'),
        ('geoparquet', True, 10000, 'GeoParquet rg=10k +hilbert'),
    ]

    x = np.arange(3)  # small, medium, large
    width = 0.12

    for i, (fmt, hilbert, rg, label) in enumerate(configs):
        # Compare row_group_size as strings since CSV stores them that way
        rg_str = str(rg) if isinstance(rg, int) else rg
        subset = gcs[(gcs['format'] == fmt) &
                     (gcs['hilbert'] == hilbert) &
                     (gcs['row_group_size'].astype(str) == rg_str)]

        if subset.empty:
            continue

        speedups = []
        for extent in ['small', 'medium', 'large']:
            ext_data = subset[subset['query_extent'] == extent]
            if not ext_data.empty and extent in orig_times:
                speedup = orig_times[extent] / ext_data['avg_time_s'].values[0]
                speedups.append(speedup)
            else:
                speedups.append(0)

        offset = (i - len(configs)/2 + 0.5) * width
        bars = ax.bar(x + offset, speedups, width, label=label)

    ax.axhline(y=1, color='red', linestyle='--', label='Original baseline')
    ax.set_xlabel('Query Extent')
    ax.set_ylabel('Speedup vs Original (higher is better)')
    ax.set_title('Cloud (GCS) Query Speedup - BEDMAP1')
    ax.set_xticks(x)
    ax.set_xticklabels(['Small\n(12K rows)', 'Medium\n(274K rows)', 'Large\n(1.4M rows)'])
    ax.legend(loc='upper right', fontsize=8)

    plt.tight_layout()
    plt.savefig(output_dir / 'cloud_speedup_bedmap1.png', dpi=150)
    plt.close()
    print(f"  Saved: cloud_speedup_bedmap1.png")


def plot_grid_queries(grid_results, output_dir):
    """Plot grid-based query results."""
    if grid_results is None or grid_results.empty:
        return

    # Drop rows without file info and separate by file
    grid_results = grid_results.dropna(subset=['file'])
    files = grid_results['file'].unique()

    for file_stem in files:
        file_data = grid_results[grid_results['file'] == file_stem]

        # Short name for title
        short_name = 'BEDMAP1' if 'BEDMAP1' in file_stem else 'AWI_2016_OIR'

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        for ax, location in zip(axes, ['local', 'gcs']):
            loc_data = file_data[file_data['location'] == location]
            if loc_data.empty:
                continue

            # Group by resolution and configuration
            configs = [
                ('original', False, 'all', 'Original', '#1f77b4', '-'),
                ('geoparquet', False, 'all', 'GeoParquet', '#ff7f0e', '-'),
                ('geoparquet', True, '50000', 'GeoParquet rg=50k +hilbert', '#2ca02c', '-'),
                ('geoparquet', False, '50000', 'GeoParquet rg=50k', '#9467bd', '--'),
                ('geoparquet', False, '10000', 'GeoParquet rg=10k', '#d62728', '--'),
            ]

            resolutions = [10, 50, 200]

            for fmt, hilbert, rg, label, color, linestyle in configs:
                means = []
                stds = []

                for res in resolutions:
                    subset = loc_data[
                        (loc_data['format'] == fmt) &
                        (loc_data['hilbert'] == hilbert) &
                        (loc_data['row_group_size'].astype(str) == str(rg)) &
                        (loc_data['resolution_km'] == res)
                    ]

                    if not subset.empty:
                        means.append(subset['time_s'].mean())
                        stds.append(subset['time_s'].std())
                    else:
                        means.append(np.nan)
                        stds.append(np.nan)

                ax.errorbar(resolutions, means, yerr=stds, label=label,
                           color=color, linestyle=linestyle, marker='o', capsize=3)

            ax.set_xlabel('Grid Resolution (km)')
            ax.set_ylabel('Query Time (seconds)')
            loc_title = 'Local' if location == 'local' else 'Cloud (GCS)'
            ax.set_title(f'{loc_title} Grid Queries - {short_name}')
            ax.set_xscale('log')
            ax.set_xticks(resolutions)
            ax.set_xticklabels(['10 km', '50 km', '200 km'])
            ax.legend(loc='upper right', fontsize=8)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / f'grid_queries_{short_name.lower()}.png', dpi=150)
        plt.close()
        print(f"  Saved: grid_queries_{short_name.lower()}.png")


def plot_grid_comparison(grid_results, output_dir):
    """Plot grid query comparison between files."""
    if grid_results is None or grid_results.empty:
        return

    # Drop rows without file info
    grid_results = grid_results.dropna(subset=['file'])

    # Only plot if we have both files
    files = grid_results['file'].unique()
    if len(files) < 2:
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    # Focus on GCS queries with 50k+hilbert (best config)
    gcs_data = grid_results[
        (grid_results['location'] == 'gcs') &
        (grid_results['format'] == 'geoparquet') &
        (grid_results['hilbert'] == True) &
        (grid_results['row_group_size'].astype(str) == '50000')
    ]

    if gcs_data.empty:
        return

    x = np.arange(3)  # 10, 50, 200 km
    width = 0.35

    colors = {'BEDMAP1_1966-2000_AIR_BM1': '#1f77b4', 'AWI_2016_OIR_AIR_BM3': '#ff7f0e'}
    labels = {'BEDMAP1_1966-2000_AIR_BM1': 'BEDMAP1 (1.9M rows)',
              'AWI_2016_OIR_AIR_BM3': 'AWI_2016_OIR (564K rows)'}

    for i, file_stem in enumerate(files):
        file_data = gcs_data[gcs_data['file'] == file_stem]

        means = []
        stds = []
        for res in [10, 50, 200]:
            subset = file_data[file_data['resolution_km'] == res]
            if not subset.empty:
                means.append(subset['time_s'].mean())
                stds.append(subset['time_s'].std())
            else:
                means.append(np.nan)
                stds.append(np.nan)

        offset = (i - 0.5) * width
        ax.bar(x + offset, means, width, yerr=stds, label=labels.get(file_stem, file_stem),
               color=colors.get(file_stem, f'C{i}'), capsize=3)

    ax.set_xlabel('Grid Resolution')
    ax.set_ylabel('Query Time (seconds)')
    ax.set_title('Cloud (GCS) Grid Query Performance\n(GeoParquet with 50k row groups + Hilbert)')
    ax.set_xticks(x)
    ax.set_xticklabels(['10 km', '50 km', '200 km'])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_dir / 'grid_comparison_files.png', dpi=150)
    plt.close()
    print(f"  Saved: grid_comparison_files.png")


def main():
    print("=" * 70)
    print("Generating Benchmark Plots")
    print("=" * 70)

    output_dir = Path(__file__).parent / 'benchmark_results'

    # Load data
    print("\nLoading results...")
    cloud_sizes, cloud_local, cloud_gcs, grid_results = load_latest_results()

    print(f"  Cloud sizes: {len(cloud_sizes) if cloud_sizes is not None else 0} rows")
    print(f"  Cloud local: {len(cloud_local) if cloud_local is not None else 0} rows")
    print(f"  Cloud GCS: {len(cloud_gcs) if cloud_gcs is not None else 0} rows")
    print(f"  Grid results: {len(grid_results) if grid_results is not None else 0} rows")

    # Generate plots
    print("\nGenerating plots...")

    plot_file_sizes(cloud_sizes, output_dir)
    plot_local_vs_cloud(cloud_local, cloud_gcs, output_dir)
    plot_cloud_speedup(cloud_local, cloud_gcs, output_dir)
    plot_grid_queries(grid_results, output_dir)
    plot_grid_comparison(grid_results, output_dir)

    print("\nDone!")


if __name__ == '__main__':
    main()
