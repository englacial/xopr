#!/usr/bin/env python3
"""
Process GitHub traffic data with intelligent merging and gap handling.
Processes all available data from GitHub API (up to 14 days) and merges with existing history.
"""

import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd


def load_json_file(filepath):
    """Load JSON file safely."""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Could not load {filepath}: {e}")
        return None


def extract_daily_data(api_data, data_type='clones'):
    """Extract daily breakdown from API response."""
    daily_records = []

    if not api_data:
        return daily_records

    # Get the daily breakdown from the API response
    daily_data = api_data.get(data_type, [])  # 'clones' or 'views' array

    for day in daily_data:
        date = pd.to_datetime(day['timestamp']).date()
        daily_records.append({
            'date': date,
            f'{data_type}_total': day.get('count', 0),
            f'{data_type}_unique': day.get('uniques', 0)
        })

    return daily_records


def process_traffic_data(raw_dir='/tmp/traffic-data/raw', data_dir='/tmp/traffic-data'):
    """Process all available traffic data with intelligent merging."""
    raw_dir = Path(raw_dir)
    data_dir = Path(data_dir)

    today = datetime.now().strftime('%Y-%m-%d')
    timestamp = datetime.utcnow().isoformat() + 'Z'

    print(f"📅 Processing traffic data (collected on {today})")

    # Load the API responses
    clones_data = load_json_file(raw_dir / f'clones-{today}.json')
    views_data = load_json_file(raw_dir / f'views-{today}.json')
    paths_data = load_json_file(raw_dir / f'paths-{today}.json')
    referrers_data = load_json_file(raw_dir / f'referrers-{today}.json')

    # Extract ALL daily data from API responses
    clones_daily = extract_daily_data(clones_data, 'clones')
    views_daily = extract_daily_data(views_data, 'views')

    # Create DataFrame from daily data
    if clones_daily:
        df_clones = pd.DataFrame(clones_daily)
    else:
        df_clones = pd.DataFrame(columns=['date', 'clones_total', 'clones_unique'])

    if views_daily:
        df_views = pd.DataFrame(views_daily)
    else:
        df_views = pd.DataFrame(columns=['date', 'views_total', 'views_unique'])

    # Merge clones and views data on date
    if not df_clones.empty and not df_views.empty:
        df_new = pd.merge(df_clones, df_views, on='date', how='outer')
    elif not df_clones.empty:
        df_new = df_clones
        df_new['views_total'] = 0
        df_new['views_unique'] = 0
    elif not df_views.empty:
        df_new = df_views
        df_new['clones_total'] = 0
        df_new['clones_unique'] = 0
    else:
        # No data available - create empty record for today
        df_new = pd.DataFrame([{
            'date': pd.to_datetime(today).date(),
            'clones_total': 0,
            'clones_unique': 0,
            'views_total': 0,
            'views_unique': 0
        }])

    # Fill NaN values with 0
    df_new = df_new.fillna(0)

    # Add timestamp for when data was collected
    df_new['timestamp'] = pd.to_datetime(timestamp)

    # Add top paths and referrers (same for all days in this batch)
    df_new['top_paths'] = json.dumps(paths_data.get('paths', [])[:5] if paths_data else [])
    df_new['top_referrers'] = json.dumps(referrers_data.get('referrers', [])[:5] if referrers_data else [])

    # Convert date column to datetime
    df_new['date'] = pd.to_datetime(df_new['date'])

    print(f"📊 API provided {len(df_new)} days of data:")
    print(f"  Date range: {df_new['date'].min().date()} to {df_new['date'].max().date()}")
    print(f"  Total clones: {df_new['clones_total'].sum():.0f} ({df_new['clones_unique'].sum():.0f} unique)")
    print(f"  Total views: {df_new['views_total'].sum():.0f} ({df_new['views_unique'].sum():.0f} unique)")

    # Load existing parquet file if it exists
    parquet_file = data_dir / 'traffic_history.parquet'
    if parquet_file.exists():
        print("📂 Loading existing Parquet file...")
        df_existing = pd.read_parquet(parquet_file)
        df_existing['date'] = pd.to_datetime(df_existing['date'])

        # Intelligent merge: update existing dates, add new ones
        # Keep the existing data for dates not in the new data
        existing_dates = set(df_existing['date'].dt.date)
        new_dates = set(df_new['date'].dt.date)

        # Dates to keep from existing (not in new data)
        keep_dates = existing_dates - new_dates
        df_keep = df_existing[df_existing['date'].dt.date.isin(keep_dates)]

        # Combine: kept old data + all new data
        df = pd.concat([df_keep, df_new], ignore_index=True)

        print(f"  Merged data: {len(df_keep)} existing + {len(df_new)} new = {len(df)} total records")

        # Report on updates
        updated_dates = existing_dates & new_dates
        if updated_dates:
            print(f"  Updated {len(updated_dates)} existing dates with fresh data")

        new_only_dates = new_dates - existing_dates
        if new_only_dates:
            print(f"  Added {len(new_only_dates)} new dates")
    else:
        print("📄 Creating new Parquet file...")
        df = df_new

    # Sort by date
    df = df.sort_values('date').reset_index(drop=True)

    # Ensure integer types for count columns
    for col in ['clones_total', 'clones_unique', 'views_total', 'views_unique']:
        df[col] = df[col].fillna(0).astype(int)

    # Save as Parquet with compression
    df.to_parquet(parquet_file, compression='snappy', index=False)
    print(f"💾 Saved {len(df)} records to Parquet")

    # Create comprehensive metadata
    metadata = {
        'last_updated': timestamp,
        'date_range': {
            'start': df['date'].min().isoformat(),
            'end': df['date'].max().isoformat()
        },
        'record_count': len(df),
        'schema_version': '1.1',
        'columns': list(df.columns),
        'file_size_bytes': parquet_file.stat().st_size,
        'parquet_file': 'traffic_history.parquet',
        'collection_stats': {
            'total_clones': int(df['clones_total'].sum()),
            'total_views': int(df['views_total'].sum()),
            'unique_clones': int(df['clones_unique'].sum()),
            'unique_views': int(df['views_unique'].sum()),
            'days_collected': len(df),
            'avg_daily_clones': float(df['clones_total'].mean()),
            'avg_daily_views': float(df['views_total'].mean())
        }
    }

    # Check for gaps in data collection
    if len(df) > 1:
        date_range = pd.date_range(start=df['date'].min(), end=df['date'].max(), freq='D')
        collected_dates = set(df['date'].dt.date)
        expected_dates = set(date_range.date)
        missing_dates = sorted(expected_dates - collected_dates)

        if missing_dates:
            metadata['collection_stats']['missing_dates'] = [d.isoformat() for d in missing_dates]
            metadata['collection_stats']['data_completeness'] = f"{(len(collected_dates) / len(expected_dates)) * 100:.1f}%"
            print(f"⚠️  Found {len(missing_dates)} gaps in historical data")
        else:
            metadata['collection_stats']['data_completeness'] = "100%"
            print("✅ No gaps in historical data")

    # Add weekly aggregates for quick reference
    if len(df) >= 7:
        last_week = df.tail(7)
        metadata['last_week_stats'] = {
            'clones_total': int(last_week['clones_total'].sum()),
            'clones_unique': int(last_week['clones_unique'].sum()),
            'views_total': int(last_week['views_total'].sum()),
            'views_unique': int(last_week['views_unique'].sum())
        }

    # Save metadata
    metadata_file = data_dir / 'metadata.json'
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"📋 Metadata updated: {metadata['record_count']} total records")
    print(f"   File size: {metadata['file_size_bytes'] / 1024:.1f} KB")

    return df, metadata


if __name__ == '__main__':
    # Allow command line args for directories
    raw_dir = sys.argv[1] if len(sys.argv) > 1 else '/tmp/traffic-data/raw'
    data_dir = sys.argv[2] if len(sys.argv) > 2 else '/tmp/traffic-data'

    try:
        df, metadata = process_traffic_data(raw_dir, data_dir)
        sys.exit(0)
    except Exception as e:
        print(f"❌ Error processing traffic data: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
