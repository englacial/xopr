#!/usr/bin/env python3
"""
Visualize GitHub traffic statistics collected by GitHub Actions.
This script generates charts from the historical traffic data.
"""

import argparse
import json
import sys
from pathlib import Path

# Check for required packages
try:
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns
except ImportError:
    print("📦 Installing required packages...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install",
                          "matplotlib", "pandas", "seaborn"])
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns

# Set style for better looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)


def load_traffic_data(data_dir='/tmp/traffic-data'):
    """Load historical traffic data from Parquet file."""
    parquet_file = Path(data_dir) / 'traffic_history.parquet'
    metadata_file = Path(data_dir) / 'metadata.json'

    if not parquet_file.exists():
        print(f"❌ No historical data found at {parquet_file}")
        print("   Run the GitHub Action first to collect data.")
        return None, None

    # Load parquet
    df = pd.read_parquet(parquet_file)

    # Load metadata if exists
    metadata = None
    if metadata_file.exists():
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)

    print(f"✅ Loaded {len(df)} days of traffic data")
    print(f"   Date range: {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}")
    if metadata:
        print(f"   File size: {metadata['file_size_bytes'] / 1024:.1f} KB")

    return df, metadata


def prepare_dataframe(df):
    """Prepare DataFrame for visualization."""
    # Ensure date is datetime and set as index
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)

    # Add derived columns
    df['clone_rate'] = df['clones_unique'] / df['clones_total'].replace(0, 1) * 100
    df['view_rate'] = df['views_unique'] / df['views_total'].replace(0, 1) * 100

    return df


def plot_traffic_overview(df, save_path=None):
    """Create overview plot with clones and views."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('GitHub Repository Traffic Analytics', fontsize=16, fontweight='bold')

    # Plot 1: Clones over time
    ax1 = axes[0, 0]
    ax1.plot(df.index, df['clones_total'], label='Total Clones', marker='o', linewidth=2)
    ax1.plot(df.index, df['clones_unique'], label='Unique Clones', marker='s', linewidth=2)
    ax1.set_title('Git Clones Over Time')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Number of Clones')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Views over time
    ax2 = axes[0, 1]
    ax2.plot(df.index, df['views_total'], label='Total Views', marker='o', linewidth=2, color='green')
    ax2.plot(df.index, df['views_unique'], label='Unique Views', marker='s', linewidth=2, color='darkgreen')
    ax2.set_title('Page Views Over Time')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Number of Views')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Combined bar chart
    ax3 = axes[1, 0]
    x_pos = range(len(df))
    width = 0.35
    ax3.bar([p - width/2 for p in x_pos], df['clones_total'], width, label='Clones', alpha=0.8)
    ax3.bar([p + width/2 for p in x_pos], df['views_total'], width, label='Views', alpha=0.8)
    ax3.set_title('Daily Traffic Comparison')
    ax3.set_xlabel('Date')
    ax3.set_ylabel('Count')
    ax3.set_xticks(x_pos[::max(1, len(df)//10)])  # Show every nth date
    ax3.set_xticklabels(df.index.strftime('%m/%d')[::max(1, len(df)//10)], rotation=45)
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Unique visitor ratios
    ax4 = axes[1, 1]
    ax4.plot(df.index, df['clone_rate'], label='Unique Clone %', marker='o', linewidth=2, color='purple')
    ax4.plot(df.index, df['view_rate'], label='Unique View %', marker='s', linewidth=2, color='orange')
    ax4.set_title('Unique Visitor Ratios')
    ax4.set_xlabel('Date')
    ax4.set_ylabel('Percentage (%)')
    ax4.set_ylim(0, 105)
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Adjust layout
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"💾 Chart saved to {save_path}")

    return fig


def plot_weekly_trends(df, save_path=None):
    """Create weekly aggregated trends."""
    # Resample to weekly data
    weekly = df.resample('W').agg({
        'clones_total': 'sum',
        'clones_unique': 'sum',
        'views_total': 'sum',
        'views_unique': 'sum'
    })

    if len(weekly) < 2:
        print("⚠️  Not enough data for weekly trends (need at least 2 weeks)")
        return None

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Weekly Traffic Trends', fontsize=14, fontweight='bold')

    # Weekly clones
    ax1 = axes[0]
    ax1.bar(range(len(weekly)), weekly['clones_total'], alpha=0.7, label='Total')
    ax1.bar(range(len(weekly)), weekly['clones_unique'], alpha=0.9, label='Unique')
    ax1.set_title('Weekly Clones')
    ax1.set_xlabel('Week')
    ax1.set_ylabel('Number of Clones')
    ax1.set_xticks(range(len(weekly)))
    ax1.set_xticklabels(weekly.index.strftime('%m/%d'), rotation=45)
    ax1.legend()

    # Weekly views
    ax2 = axes[1]
    ax2.bar(range(len(weekly)), weekly['views_total'], alpha=0.7, label='Total', color='green')
    ax2.bar(range(len(weekly)), weekly['views_unique'], alpha=0.9, label='Unique', color='darkgreen')
    ax2.set_title('Weekly Views')
    ax2.set_xlabel('Week')
    ax2.set_ylabel('Number of Views')
    ax2.set_xticks(range(len(weekly)))
    ax2.set_xticklabels(weekly.index.strftime('%m/%d'), rotation=45)
    ax2.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"💾 Weekly trends saved to {save_path}")

    return fig


def print_statistics(df):
    """Print summary statistics."""
    print("\n" + "="*60)
    print("📊 TRAFFIC STATISTICS SUMMARY")
    print("="*60)

    print(f"\n📅 Date Range: {df.index.min().strftime('%Y-%m-%d')} to {df.index.max().strftime('%Y-%m-%d')}")
    print(f"   Total Days: {len(df)}")

    print("\n📦 CLONES:")
    print(f"   Total: {df['clones_total'].sum():,}")
    print(f"   Unique: {df['clones_unique'].sum():,}")
    print(f"   Daily Average: {df['clones_total'].mean():.1f} ({df['clones_unique'].mean():.1f} unique)")
    print(f"   Peak Day: {df['clones_total'].max()} on {df['clones_total'].idxmax().strftime('%Y-%m-%d')}")

    print("\n👀 VIEWS:")
    print(f"   Total: {df['views_total'].sum():,}")
    print(f"   Unique: {df['views_unique'].sum():,}")
    print(f"   Daily Average: {df['views_total'].mean():.1f} ({df['views_unique'].mean():.1f} unique)")
    print(f"   Peak Day: {df['views_total'].max()} on {df['views_total'].idxmax().strftime('%Y-%m-%d')}")

    print("\n📈 TRENDS (Last 7 days vs Previous 7 days):")
    if len(df) >= 14:
        last_week = df.tail(7)
        prev_week = df.tail(14).head(7)

        clone_change = ((last_week['clones_total'].sum() / prev_week['clones_total'].sum()) - 1) * 100
        view_change = ((last_week['views_total'].sum() / prev_week['views_total'].sum()) - 1) * 100

        print(f"   Clone Traffic: {'+' if clone_change >= 0 else ''}{clone_change:.1f}%")
        print(f"   View Traffic: {'+' if view_change >= 0 else ''}{view_change:.1f}%")
    else:
        print("   Not enough data for trend analysis (need 14+ days)")

    print("\n" + "="*60)


def main():
    parser = argparse.ArgumentParser(description='Visualize GitHub traffic statistics')
    parser.add_argument('--data-dir', default='/tmp/traffic-data',
                       help='Directory containing traffic data (default: /tmp/traffic-data)')
    parser.add_argument('--output-dir', default='.',
                       help='Directory to save charts (default: current directory)')
    parser.add_argument('--no-show', action='store_true',
                       help='Do not display charts (only save)')

    args = parser.parse_args()

    # Load data
    df, metadata = load_traffic_data(args.data_dir)
    if df is None:
        return 1

    # Prepare DataFrame
    df = prepare_dataframe(df)

    # Print statistics
    print_statistics(df)

    # Create output directory if it doesn't exist
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate plots
    overview_path = output_dir / 'traffic_overview.png'
    plot_traffic_overview(df, save_path=overview_path)

    if len(df) >= 14:
        weekly_path = output_dir / 'traffic_weekly.png'
        plot_weekly_trends(df, save_path=weekly_path)

    # Show plots if requested
    if not args.no_show:
        plt.show()

    print("\n✅ Visualization complete!")
    return 0


if __name__ == '__main__':
    sys.exit(main())
