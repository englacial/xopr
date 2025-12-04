"""
Converter module for transforming bedmap CSV files to GeoParquet format.

This module handles:
- Parsing bedmap CSV files and metadata
- Complex date/time handling with fallback strategies
- Converting data to cloud-optimized GeoParquet format with WKB Point geometry
- Hilbert curve sorting for large files (>600k rows) to optimize spatial queries
- Extracting spatial and temporal bounds
"""

import pandas as pd
import numpy as np
import geopandas as gpd
from pathlib import Path
from typing import Dict, Optional, Tuple, List, Union
from datetime import datetime, timezone
import warnings

from .geometry import (
    extract_flight_lines,
    simplify_multiline_geometry,
    calculate_bbox,
)

# Threshold for applying Hilbert sorting (rows)
HILBERT_ROW_THRESHOLD = 600_000
# Row group size for files with Hilbert sorting
HILBERT_ROW_GROUP_SIZE = 50_000


def _normalize_longitude(lon: np.ndarray) -> np.ndarray:
    """
    Normalize longitude values from 0-360 to -180 to 180 convention.

    Parameters
    ----------
    lon : numpy.ndarray
        Longitude values (may be in 0-360 or -180 to 180 range)

    Returns
    -------
    numpy.ndarray
        Longitude values in -180 to 180 range
    """
    # Convert values > 180 to negative (0-360 -> -180 to 180)
    return np.where(lon > 180, lon - 360, lon)


def parse_bedmap_metadata(csv_path: Union[str, Path]) -> Dict:
    """
    Parse metadata from bedmap CSV header lines.

    Parameters
    ----------
    csv_path : str or Path
        Path to the bedmap CSV file

    Returns
    -------
    dict
        Dictionary containing parsed metadata fields
    """
    metadata = {}

    with open(csv_path, 'r') as f:
        for line in f:
            if not line.startswith('#'):
                break

            # Remove '#' and whitespace
            line = line[1:].strip()

            # Skip empty lines
            if not line:
                continue

            # Parse key-value pairs
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip().replace(' ', '_').replace('-', '_')
                value = value.strip()

                # Parse numeric values where appropriate
                if key in ['time_coverage_start', 'time_coverage_end']:
                    try:
                        metadata[key] = int(value)
                    except ValueError:
                        metadata[key] = value
                elif key in ['electromagnetic_wave_speed_in_ice', 'firn_correction', 'centre_frequency']:
                    # Extract numeric value and unit
                    parts = value.split('(')
                    if len(parts) > 0:
                        try:
                            metadata[key] = float(parts[0].strip())
                            if len(parts) > 1:
                                metadata[f"{key}_unit"] = parts[1].rstrip(')')
                        except ValueError:
                            metadata[key] = value
                else:
                    metadata[key] = value

    return metadata


def parse_date_time_columns(
    date_series: pd.Series,
    time_series: pd.Series
) -> pd.Series:
    """
    Parse date and time columns into timestamps.

    Parameters
    ----------
    date_series : pd.Series
        Series containing date values
    time_series : pd.Series
        Series containing time values

    Returns
    -------
    pd.Series
        Series of parsed timestamps
    """
    timestamps = []

    for date_val, time_val in zip(date_series, time_series):
        if pd.isna(date_val) or pd.isna(time_val):
            timestamps.append(pd.NaT)
            continue

        try:
            # Handle various date formats
            date_str = str(date_val)
            time_str = str(time_val)

            # Common formats: YYYYMMDD, YYYY-MM-DD, YYYY/MM/DD
            if len(date_str) == 8 and date_str.isdigit():
                # YYYYMMDD format
                year = int(date_str[:4])
                month = int(date_str[4:6])
                day = int(date_str[6:8])
            elif '-' in date_str:
                parts = date_str.split('-')
                year = int(parts[0])
                month = int(parts[1]) if len(parts) > 1 else 1
                day = int(parts[2]) if len(parts) > 2 else 1
            elif '/' in date_str:
                parts = date_str.split('/')
                # Could be MM/DD/YYYY or YYYY/MM/DD
                if len(parts[0]) == 4:
                    year, month, day = int(parts[0]), int(parts[1]), int(parts[2])
                else:
                    month, day, year = int(parts[0]), int(parts[1]), int(parts[2])
            else:
                timestamps.append(pd.NaT)
                continue

            # Parse time (HH:MM:SS or HHMMSS)
            if ':' in time_str:
                time_parts = time_str.split(':')
                hour = int(time_parts[0])
                minute = int(time_parts[1]) if len(time_parts) > 1 else 0
                second = int(float(time_parts[2])) if len(time_parts) > 2 else 0
            elif len(time_str) >= 4 and time_str.replace('.', '').isdigit():
                # HHMMSS or HHMM format
                hour = int(time_str[:2])
                minute = int(time_str[2:4])
                second = int(time_str[4:6]) if len(time_str) >= 6 else 0
            else:
                hour = minute = second = 0

            # Create timestamp
            timestamp = pd.Timestamp(year=year, month=month, day=day,
                                      hour=hour, minute=minute, second=second,
                                      tz=timezone.utc)
            timestamps.append(timestamp)

        except (ValueError, TypeError) as e:
            timestamps.append(pd.NaT)

    return pd.Series(timestamps)


def create_timestamps(df: pd.DataFrame, metadata: Dict) -> pd.Series:
    """
    Create timestamps with complex fallback strategies.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing date and time columns
    metadata : dict
        Metadata dictionary with time_coverage fields

    Returns
    -------
    pd.Series
        Series of timestamps
    """
    # Check for date and time columns
    has_date = 'date' in df.columns and not df['date'].isna().all()
    has_time = 'time_UTC' in df.columns and not df['time_UTC'].isna().all()

    if has_date and has_time:
        # Primary strategy: use date and time columns
        timestamps = parse_date_time_columns(df['date'], df['time_UTC'])

        # Check if we got valid timestamps
        valid_count = timestamps.notna().sum()
        if valid_count > 0:
            return timestamps

    # Fallback strategy: use metadata time coverage
    start_year = metadata.get('time_coverage_start')
    end_year = metadata.get('time_coverage_end')

    if start_year is None:
        # No temporal information available
        warnings.warn(f"No temporal information available, using current year as placeholder")
        start_year = end_year = datetime.now().year

    # Convert to timestamps
    if start_year == end_year:
        # Single year: distribute evenly across the year
        start_time = pd.Timestamp(f"{start_year}-01-01", tz=timezone.utc)
        end_time = pd.Timestamp(f"{start_year}-12-31 23:59:59", tz=timezone.utc)
    else:
        # Multi-year: distribute across full range
        start_time = pd.Timestamp(f"{start_year}-01-01", tz=timezone.utc)
        end_time = pd.Timestamp(f"{end_year}-12-31 23:59:59", tz=timezone.utc)

    # Create evenly spaced timestamps
    n_rows = len(df)
    timestamps = pd.date_range(start_time, end_time, periods=n_rows, tz=timezone.utc)

    # Convert to microsecond precision to avoid issues with PyArrow
    timestamps = pd.Series(timestamps).dt.floor('us')

    return timestamps


def extract_temporal_extent(
    df: pd.DataFrame,
    metadata: Dict
) -> Tuple[datetime, datetime]:
    """
    Extract temporal extent from dataframe or metadata.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with timestamp column
    metadata : dict
        Metadata dictionary

    Returns
    -------
    tuple
        (start_datetime, end_datetime)
    """
    if 'timestamp' in df.columns:
        valid_timestamps = df['timestamp'].dropna()
        if not valid_timestamps.empty:
            return (valid_timestamps.min(), valid_timestamps.max())

    # Fallback to metadata
    start_year = metadata.get('time_coverage_start')
    end_year = metadata.get('time_coverage_end')

    if start_year is None:
        return (None, None)

    if start_year == end_year:
        start_time = pd.Timestamp(f"{start_year}-01-01", tz=timezone.utc)
        end_time = pd.Timestamp(f"{start_year}-12-31 23:59:59", tz=timezone.utc)
    else:
        start_time = pd.Timestamp(f"{start_year}-01-01", tz=timezone.utc)
        end_time = pd.Timestamp(f"{end_year}-12-31 23:59:59", tz=timezone.utc)

    return (start_time, end_time)


def _apply_hilbert_sorting(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Apply Hilbert curve sorting to a GeoDataFrame for spatial locality.

    Uses file-specific bounds for the Hilbert curve to maximize spatial
    locality within each file.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        GeoDataFrame with Point geometry

    Returns
    -------
    geopandas.GeoDataFrame
        Sorted GeoDataFrame
    """
    import duckdb

    # Extract coordinates
    coords_df = pd.DataFrame({
        'lon': gdf.geometry.x,
        'lat': gdf.geometry.y,
    })
    coords_df['orig_idx'] = range(len(coords_df))

    conn = duckdb.connect()
    conn.execute("INSTALL spatial; LOAD spatial;")
    conn.register('coords', coords_df)

    # Get file-specific bounds for Hilbert curve
    minx, miny, maxx, maxy = gdf.total_bounds

    # Compute Hilbert indices and sort
    sorted_order = conn.execute(f"""
        SELECT orig_idx,
               ST_Hilbert(lon, lat,
                   {{'min_x': {minx}, 'min_y': {miny}, 'max_x': {maxx}, 'max_y': {maxy}}}::BOX_2D
               ) as hilbert_idx
        FROM coords
        ORDER BY hilbert_idx
    """).fetchdf()
    conn.close()

    # Reorder GeoDataFrame
    return gdf.iloc[sorted_order['orig_idx'].values].reset_index(drop=True)


def convert_bedmap_csv(
    csv_path: Union[str, Path],
    output_dir: Union[str, Path],
    simplify_tolerance_deg: float = 0.01,
) -> Dict:
    """
    Convert a single bedmap CSV file to cloud-optimized GeoParquet format.

    Creates GeoParquet files with WKB-encoded Point geometry and zstd compression.
    For large files (>600k rows), applies Hilbert curve sorting with 50k row groups
    to optimize spatial queries.

    Parameters
    ----------
    csv_path : str or Path
        Path to the input CSV file
    output_dir : str or Path
        Directory for output GeoParquet file
    simplify_tolerance_deg : float
        Tolerance for geometry simplification in degrees

    Returns
    -------
    dict
        Dictionary containing metadata and bounds information
    """
    from shapely.geometry import Point

    csv_path = Path(csv_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Converting {csv_path.name}...")

    # Parse metadata from header
    metadata = parse_bedmap_metadata(csv_path)

    # Read CSV data
    df = pd.read_csv(csv_path, comment='#')

    # Convert trajectory_id to string (it may be numeric in some files)
    df['trajectory_id'] = df['trajectory_id'].astype(str)

    # Replace -9999 with NaN for numeric columns (but not trajectory_id)
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    for col in numeric_columns:
        df[col] = df[col].replace(-9999, np.nan)

    # Normalize longitude from 0-360 to -180 to 180 convention
    df['longitude (degree_east)'] = _normalize_longitude(
        df['longitude (degree_east)'].values
    )

    # Handle date/time conversion
    df['timestamp'] = create_timestamps(df, metadata)

    # Add source_file column (without extension)
    source_name = csv_path.stem  # Gets filename without extension
    df['source_file'] = source_name

    # Extract flight line geometry for metadata
    multiline_geom = extract_flight_lines(df)

    if multiline_geom is not None:
        simplified_geom = simplify_multiline_geometry(
            multiline_geom,
            tolerance_deg=simplify_tolerance_deg
        )
    else:
        simplified_geom = None
        warnings.warn(f"Could not extract flight lines from {csv_path.name}")

    # Calculate spatial bounds
    bbox = calculate_bbox(df)

    # Extract temporal extent
    temporal_start, temporal_end = extract_temporal_extent(df, metadata)

    # Prepare metadata dictionary
    file_metadata = {
        'source_csv': csv_path.name,
        'bedmap_version': _extract_bedmap_version(csv_path.name),
        'spatial_bounds': {
            'bbox': bbox,
            'geometry': simplified_geom.wkt if simplified_geom else None,
        },
        'temporal_bounds': {
            'start': temporal_start.isoformat() if temporal_start else None,
            'end': temporal_end.isoformat() if temporal_end else None,
        },
        'row_count': len(df),
        'original_metadata': metadata,
    }

    # Create Point geometry from lon/lat columns (already normalized above)
    geometry = [
        Point(lon, lat) for lon, lat in
        zip(df['longitude (degree_east)'], df['latitude (degree_north)'])
    ]

    # Create GeoDataFrame with WKB Point geometry
    gdf = gpd.GeoDataFrame(
        df.drop(columns=['longitude (degree_east)', 'latitude (degree_north)',
                        'date', 'time_UTC'], errors='ignore'),
        geometry=geometry,
        crs='EPSG:4326'
    )

    # Determine if Hilbert sorting is needed based on row count
    row_count = len(gdf)
    use_hilbert = row_count > HILBERT_ROW_THRESHOLD

    if use_hilbert:
        print(f"  Applying Hilbert sorting ({row_count:,} rows > {HILBERT_ROW_THRESHOLD:,} threshold)")
        gdf = _apply_hilbert_sorting(gdf)
        row_group_size = HILBERT_ROW_GROUP_SIZE
    else:
        row_group_size = None  # Use default (single row group)

    # Prepare output path
    output_path = output_dir / f"{source_name}.parquet"

    # Write as GeoParquet with zstd compression
    write_kwargs = {'compression': 'zstd'}
    if row_group_size is not None:
        write_kwargs['row_group_size'] = row_group_size

    gdf.to_parquet(output_path, **write_kwargs)

    print(f"  Written to {output_path}")
    print(f"  Rows: {row_count:,}")
    if use_hilbert:
        print(f"  Row groups: {row_group_size:,} rows each")
    if bbox:
        print(f"  Spatial extent: {bbox[0]:.2f}, {bbox[1]:.2f} to {bbox[2]:.2f}, {bbox[3]:.2f}")
    if temporal_start and temporal_end:
        print(f"  Temporal extent: {temporal_start.date()} to {temporal_end.date()}")

    return file_metadata


def _extract_bedmap_version(filename: str) -> str:
    """
    Extract bedmap version from filename.

    Parameters
    ----------
    filename : str
        Name of the CSV file

    Returns
    -------
    str
        Bedmap version (BM1, BM2, BM3, or unknown)
    """
    if '_BM1' in filename or 'BM1.' in filename:
        return 'BM1'
    elif '_BM2' in filename or 'BM2.' in filename:
        return 'BM2'
    elif '_BM3' in filename or 'BM3.' in filename:
        return 'BM3'
    else:
        return 'unknown'


def batch_convert_bedmap(
    input_dir: Union[str, Path],
    output_dir: Union[str, Path],
    pattern: str = "*.csv",
    parallel: bool = False,
    n_workers: int = 4
) -> List[Dict]:
    """
    Batch convert multiple bedmap CSV files to GeoParquet.

    Parameters
    ----------
    input_dir : str or Path
        Directory containing input CSV files
    output_dir : str or Path
        Directory for output GeoParquet files
    pattern : str
        Glob pattern for CSV files
    parallel : bool
        Whether to process files in parallel
    n_workers : int
        Number of parallel workers

    Returns
    -------
    list
        List of metadata dictionaries for all converted files
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    # Find all CSV files
    csv_files = sorted(input_dir.glob(pattern))
    print(f"Found {len(csv_files)} CSV files to convert")

    if not csv_files:
        warnings.warn(f"No CSV files found matching pattern '{pattern}' in {input_dir}")
        return []

    metadata_list = []

    if parallel and len(csv_files) > 1:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        from tqdm import tqdm

        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {
                executor.submit(convert_bedmap_csv, csv_file, output_dir): csv_file
                for csv_file in csv_files
            }

            for future in tqdm(as_completed(futures), total=len(csv_files),
                               desc="Converting files"):
                try:
                    metadata = future.result()
                    metadata_list.append(metadata)
                except Exception as e:
                    csv_file = futures[future]
                    print(f"Error processing {csv_file.name}: {e}")

    else:
        # Sequential processing
        from tqdm import tqdm

        for csv_file in tqdm(csv_files, desc="Converting files"):
            try:
                metadata = convert_bedmap_csv(csv_file, output_dir)
                metadata_list.append(metadata)
            except Exception as e:
                print(f"Error processing {csv_file.name}: {e}")

    print(f"\nSuccessfully converted {len(metadata_list)} files")
    return metadata_list