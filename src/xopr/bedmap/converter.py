"""
Converter module for transforming bedmap CSV files to GeoParquet format.

This module handles:
- Parsing bedmap CSV files and metadata
- Complex date/time handling with fallback strategies
- Converting data to cloud-optimized GeoParquet format with WKB Point geometry
- Hilbert curve sorting for large files (>600k rows) to optimize spatial queries
- Extracting spatial and temporal bounds
"""

import io
import json
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, Tuple, List, Union

import duckdb
import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.csv as pa_csv
import pyarrow.parquet as pq
from tqdm import tqdm

from .geometry import (
    extract_flight_lines,
    simplify_multiline_geometry,
    calculate_bbox,
)

# Threshold for applying Hilbert sorting (rows)
HILBERT_ROW_THRESHOLD = 600_000
# Row group size for files with Hilbert sorting
HILBERT_ROW_GROUP_SIZE = 50_000


def _write_geoparquet_with_metadata(
    gdf: gpd.GeoDataFrame,
    output_path: Path,
    bedmap_metadata: Dict,
    compression: str = 'zstd',
    row_group_size: Optional[int] = None
) -> None:
    """
    Write GeoDataFrame to GeoParquet with custom bedmap metadata in schema.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        GeoDataFrame to write
    output_path : Path
        Output file path
    bedmap_metadata : dict
        Metadata to embed in parquet schema (includes flight_line_wkb)
    compression : str
        Compression codec
    row_group_size : int, optional
        Row group size for parquet file
    """
    # First write to a temporary buffer to get the arrow table with geo metadata
    buffer = io.BytesIO()
    gdf.to_parquet(buffer, compression=compression)
    buffer.seek(0)

    # Read back as arrow table
    table = pq.read_table(buffer)

    # Get existing metadata and add bedmap_metadata
    existing_metadata = table.schema.metadata or {}
    new_metadata = {
        **existing_metadata,
        b'bedmap_metadata': json.dumps(bedmap_metadata).encode('utf-8')
    }

    # Create new schema with updated metadata
    new_schema = table.schema.with_metadata(new_metadata)
    table = table.cast(new_schema)

    # Write with custom row group size if specified
    write_kwargs = {'compression': compression}
    if row_group_size is not None:
        write_kwargs['row_group_size'] = row_group_size

    pq.write_table(table, output_path, **write_kwargs)


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


def _read_csv_fast(csv_path: Union[str, Path]) -> pd.DataFrame:
    """
    Read CSV file using pyarrow for better performance on large files.

    Uses pyarrow.csv.read_csv() which is ~2x faster than pandas for large files
    due to multi-threaded parsing. Falls back to pandas for small files or errors.

    Parameters
    ----------
    csv_path : str or Path
        Path to the CSV file

    Returns
    -------
    pandas.DataFrame
        Loaded data
    """
    csv_path = Path(csv_path)

    # Count comment lines to skip
    skip_rows = 0
    with open(csv_path, 'r') as f:
        for line in f:
            if line.startswith('#'):
                skip_rows += 1
            else:
                break

    try:
        # pyarrow is faster for medium/large files (>50k rows)
        table = pa_csv.read_csv(
            csv_path,
            read_options=pa_csv.ReadOptions(skip_rows=skip_rows),
        )
        return table.to_pandas()
    except Exception:
        # Fall back to pandas if pyarrow fails
        return pd.read_csv(csv_path, comment='#')


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
    Parse date and time columns into timestamps (vectorized).

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
    # Convert to string series for vectorized operations
    date_str = date_series.astype(str)
    time_str = time_series.astype(str)

    # Combine date and time with 'T' separator for ISO8601 format
    # Most bedmap files use YYYY-MM-DD and HH:MM:SS formats
    datetime_str = date_str + 'T' + time_str

    # Try ISO8601 format first (fastest path), fall back to mixed if needed
    # errors='coerce' converts unparseable values to NaT
    try:
        timestamps = pd.to_datetime(datetime_str, format='ISO8601', utc=True)
    except ValueError:
        # Fall back to mixed format parsing for non-ISO formats
        datetime_str_space = date_str + ' ' + time_str
        timestamps = pd.to_datetime(datetime_str_space, errors='coerce', utc=True, format='mixed')

    # Handle NaN inputs - set to NaT where either date or time was missing
    na_mask = date_series.isna() | time_series.isna()
    timestamps = timestamps.where(~na_mask, pd.NaT)

    return timestamps


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


def _apply_hilbert_sorting(gdf: gpd.GeoDataFrame, verbose: bool = True) -> gpd.GeoDataFrame:
    """
    Apply Hilbert curve sorting to a GeoDataFrame for spatial locality.

    Uses file-specific bounds for the Hilbert curve to maximize spatial
    locality within each file.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        GeoDataFrame with Point geometry
    verbose : bool
        Print progress messages

    Returns
    -------
    geopandas.GeoDataFrame
        Sorted GeoDataFrame
    """
    row_count = len(gdf)
    if verbose:
        print(f"    Hilbert sort: extracting {row_count:,} coordinates...")

    # Extract coordinates
    t0 = time.time()
    coords_df = pd.DataFrame({
        'lon': gdf.geometry.x,
        'lat': gdf.geometry.y,
    })
    coords_df['orig_idx'] = range(len(coords_df))
    if verbose:
        print(f"    Hilbert sort: coordinates extracted ({time.time() - t0:.1f}s)")

    # Get file-specific bounds for Hilbert curve
    minx, miny, maxx, maxy = gdf.total_bounds
    if verbose:
        print(f"    Hilbert sort: bounds = ({minx:.2f}, {miny:.2f}) to ({maxx:.2f}, {maxy:.2f})")

    t0 = time.time()
    conn = duckdb.connect()
    conn.execute("INSTALL spatial; LOAD spatial;")
    conn.register('coords', coords_df)
    if verbose:
        print(f"    Hilbert sort: DuckDB setup ({time.time() - t0:.1f}s)")

    # Compute Hilbert indices and sort
    if verbose:
        print(f"    Hilbert sort: computing indices for {row_count:,} rows (this may take several minutes)...")
    t0 = time.time()
    sorted_order = conn.execute(f"""
        SELECT orig_idx,
               ST_Hilbert(lon, lat,
                   {{'min_x': {minx}, 'min_y': {miny}, 'max_x': {maxx}, 'max_y': {maxy}}}::BOX_2D
               ) as hilbert_idx
        FROM coords
        ORDER BY hilbert_idx
    """).fetchdf()
    conn.close()
    if verbose:
        print(f"    Hilbert sort: indices computed ({time.time() - t0:.1f}s)")

    # Reorder GeoDataFrame
    if verbose:
        print(f"    Hilbert sort: reordering GeoDataFrame...")
    t0 = time.time()
    result = gdf.iloc[sorted_order['orig_idx'].values].reset_index(drop=True)
    if verbose:
        print(f"    Hilbert sort: complete ({time.time() - t0:.1f}s)")

    return result


def convert_bedmap_csv(
    csv_path: Union[str, Path],
    output_dir: Union[str, Path],
    simplify_tolerance_km: float = 10.0,
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
    simplify_tolerance_km : float
        Tolerance for geometry simplification in kilometers (uses polar
        stereographic projection to avoid distortion near poles)

    Returns
    -------
    dict
        Dictionary containing metadata and bounds information
    """
    csv_path = Path(csv_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    total_start = time.time()
    print(f"Converting {csv_path.name}...")

    # Parse metadata from header
    metadata = parse_bedmap_metadata(csv_path)

    # Read CSV data using pyarrow for better performance on large files
    t0 = time.time()
    df = _read_csv_fast(csv_path)
    print(f"  Read {len(df):,} rows ({time.time() - t0:.1f}s)")

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
            tolerance_km=simplify_tolerance_km
        )
    else:
        simplified_geom = None
        warnings.warn(f"Could not extract flight lines from {csv_path.name}")

    # Calculate spatial bounds
    bbox = calculate_bbox(df)

    # Extract temporal extent
    temporal_start, temporal_end = extract_temporal_extent(df, metadata)

    # Convert simplified geometry to WKB hex for storage
    flight_line_wkb = None
    if simplified_geom is not None:
        flight_line_wkb = simplified_geom.wkb_hex

    # Prepare metadata dictionary (will be stored in parquet schema)
    file_metadata = {
        'source_csv': csv_path.name,
        'bedmap_version': _extract_bedmap_version(csv_path.name),
        'spatial_bounds': {
            'bbox': list(bbox) if bbox else None,  # Convert tuple to list for JSON
        },
        'temporal_bounds': {
            'start': temporal_start.isoformat() if temporal_start else None,
            'end': temporal_end.isoformat() if temporal_end else None,
        },
        'row_count': len(df),
        'original_metadata': metadata,
        'flight_line_wkb': flight_line_wkb,  # WKB hex string for STAC catalog
    }

    # Create Point geometry from lon/lat columns (vectorized for performance)
    t0 = time.time()
    geometry = gpd.points_from_xy(
        df['longitude (degree_east)'],
        df['latitude (degree_north)']
    )

    # Create GeoDataFrame with WKB Point geometry
    gdf = gpd.GeoDataFrame(
        df.drop(columns=['longitude (degree_east)', 'latitude (degree_north)',
                        'date', 'time_UTC'], errors='ignore'),
        geometry=geometry,
        crs='EPSG:4326'
    )
    print(f"  Created GeoDataFrame ({time.time() - t0:.1f}s)")

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

    # Write as GeoParquet with zstd compression and bedmap metadata
    t0 = time.time()
    _write_geoparquet_with_metadata(
        gdf=gdf,
        output_path=output_path,
        bedmap_metadata=file_metadata,
        compression='zstd',
        row_group_size=row_group_size
    )
    print(f"  Written to {output_path} ({time.time() - t0:.1f}s)")

    print(f"  Rows: {row_count:,}")
    if use_hilbert:
        print(f"  Row groups: {row_group_size:,} rows each")
    if bbox:
        print(f"  Spatial extent: {bbox[0]:.2f}, {bbox[1]:.2f} to {bbox[2]:.2f}, {bbox[3]:.2f}")
    if temporal_start and temporal_end:
        print(f"  Temporal extent: {temporal_start.date()} to {temporal_end.date()}")

    total_time = time.time() - total_start
    print(f"  Total conversion time: {total_time:.1f}s")

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
        for csv_file in tqdm(csv_files, desc="Converting files"):
            try:
                metadata = convert_bedmap_csv(csv_file, output_dir)
                metadata_list.append(metadata)
            except Exception as e:
                print(f"Error processing {csv_file.name}: {e}")

    print(f"\nSuccessfully converted {len(metadata_list)} files")
    return metadata_list