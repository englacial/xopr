#!/usr/bin/env python3
"""
Generate markdown report from traffic data.
Creates a human-readable report with statistics and trends.
"""

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd


def generate_report(data_dir='/tmp/traffic-data'):
    """Generate markdown report from Parquet data."""
    data_dir = Path(data_dir)
    parquet_file = data_dir / 'traffic_history.parquet'
    metadata_file = data_dir / 'metadata.json'

    if not parquet_file.exists():
        print("❌ No historical data found")
        return False

    print("📝 Generating markdown report...")

    # Load data
    df = pd.read_parquet(parquet_file)

    with open(metadata_file, 'r') as f:
        metadata = json.load(f)

    # Get all available data, sorted by date
    df = df.sort_values('date')

    # For display, show up to last 14 days (or all data if less than 14 days)
    last_date = df['date'].max()
    cutoff_date = last_date - timedelta(days=14)
    display_df = df[df['date'] > cutoff_date].copy() if len(df) > 14 else df.copy()

    # Create markdown report
    report = [
        "# GitHub Traffic Analytics Report",
        f"\n*Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}*\n",
        f"*Data range: {metadata['date_range']['start']} to {metadata['date_range']['end']}*\n",
        f"## Recent Traffic ({len(display_df)} Days Available)\n",
        "### Clones",
        "| Date | Total | Unique |",
        "|------|-------|--------|"
    ]

    for _, row in display_df.iterrows():
        report.append(f"| {row['date'].strftime('%Y-%m-%d')} | {row['clones_total']} | {row['clones_unique']} |")

    report.extend([
        "\n### Views",
        "| Date | Total | Unique |",
        "|------|-------|--------|"
    ])

    for _, row in display_df.iterrows():
        report.append(f"| {row['date'].strftime('%Y-%m-%d')} | {row['views_total']} | {row['views_unique']} |")

    # Calculate week-over-week changes if we have enough data
    if len(display_df) >= 14:
        week1 = display_df.tail(7)
        week2 = display_df.iloc[-14:-7]  # Week before last week

        clones_change = ((week1['clones_total'].sum() / max(week2['clones_total'].sum(), 1)) - 1) * 100
        views_change = ((week1['views_total'].sum() / max(week2['views_total'].sum(), 1)) - 1) * 100

        trend_section = [
            "\n## Week-over-Week Trends",
            f"- **Clone Traffic:** {'+' if clones_change >= 0 else ''}{clones_change:.1f}%",
            f"- **View Traffic:** {'+' if views_change >= 0 else ''}{views_change:.1f}%"
        ]
        report.extend(trend_section)

    # Add summary statistics from ALL collected data
    report.extend([
        "\n## Summary Statistics",
        f"- **Total Clones ({len(display_df)} days displayed):** {display_df['clones_total'].sum():,} ({display_df['clones_unique'].sum():,} unique)",
        f"- **Total Views ({len(display_df)} days displayed):** {display_df['views_total'].sum():,} ({display_df['views_unique'].sum():,} unique)",
        f"- **Average Daily Clones:** {display_df['clones_total'].mean():.1f} ({display_df['clones_unique'].mean():.1f} unique)",
        f"- **Average Daily Views:** {display_df['views_total'].mean():.1f} ({display_df['views_unique'].mean():.1f} unique)",
        "",
        "## All-Time Statistics",
        f"- **Total Records:** {metadata['record_count']} days",
        f"- **Total Clones:** {metadata['collection_stats']['total_clones']:,}",
        f"- **Total Views:** {metadata['collection_stats']['total_views']:,}",
        f"- **Collection Started:** {metadata['date_range']['start']}",
        f"- **Data Size:** {metadata['file_size_bytes'] / 1024:.1f} KB"
    ])

    # Add peak days
    if len(df) > 0:
        peak_clones_day = df.loc[df['clones_total'].idxmax()]
        peak_views_day = df.loc[df['views_total'].idxmax()]

        report.extend([
            "",
            "## Peak Days",
            f"- **Most Clones:** {peak_clones_day['clones_total']} on {peak_clones_day['date'].strftime('%Y-%m-%d')}",
            f"- **Most Views:** {peak_views_day['views_total']} on {peak_views_day['date'].strftime('%Y-%m-%d')}"
        ])

    # Add missing dates if any
    if 'missing_dates' in metadata['collection_stats'] and metadata['collection_stats']['missing_dates']:
        missing_dates = metadata['collection_stats']['missing_dates']
        report.extend([
            "",
            "## Data Quality",
            f"- **Missing Dates:** {len(missing_dates)} days",
        ])
        if len(missing_dates) <= 10:
            report.append(f"  - Dates: {', '.join(missing_dates)}")
        else:
            report.append(f"  - Recent missing: {', '.join(missing_dates[-5:])}")

    # Save report
    report_file = data_dir / 'REPORT.md'
    with open(report_file, 'w') as f:
        f.write('\n'.join(report))

    print(f"✅ Report saved to {report_file}")
    print(f"   Displayed {len(display_df)} days in recent traffic section")
    print(f"   Total historical data: {len(df)} days")

    return True


if __name__ == '__main__':
    data_dir = sys.argv[1] if len(sys.argv) > 1 else '/tmp/traffic-data'

    try:
        success = generate_report(data_dir)
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"❌ Error generating report: {e}")
        sys.exit(1)
