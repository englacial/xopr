"""
STAC catalog generation for bedmap data.

This module creates GeoParquet STAC catalogs from converted bedmap data files
for use with xopr's map visualization and query infrastructure.
"""

import json
from pathlib import Path
from typing import Dict, Union
import warnings

import pyarrow.parquet as pq
import geopandas as gpd
from shapely import wkt


def read_parquet_metadata(parquet_path: Union[str, Path]) -> Dict:
    """
    Read metadata from a GeoParquet file.

    Parameters
    ----------
    parquet_path : str or Path
        Path to the GeoParquet file

    Returns
    -------
    dict
        Metadata dictionary from the parquet file
    """
    parquet_file = pq.ParquetFile(parquet_path)

    # Get metadata from parquet schema (use schema_arrow for Arrow schema with metadata)
    schema_metadata = parquet_file.schema_arrow.metadata or {}
    metadata_bytes = schema_metadata.get(b'bedmap_metadata')

    if metadata_bytes:
        return json.loads(metadata_bytes.decode())
    else:
        warnings.warn(f"No bedmap metadata found in {parquet_path}")
        return {}


def build_bedmap_geoparquet_catalog(
    parquet_dir: Union[str, Path],
    output_dir: Union[str, Path],
    base_href: str = 'gs://opr_stac/bedmap/data/',
    simplify_tolerance: float = 0.01
) -> Dict[str, gpd.GeoDataFrame]:
    """
    Build GeoParquet STAC catalogs from bedmap data files.

    Creates separate GeoParquet catalog files per bedmap version:
    - bedmap1.parquet (BM1 items)
    - bedmap2.parquet (BM2 items)
    - bedmap3.parquet (BM3 items)

    Each catalog file contains one row per data file with:
    - WKB geometry column (flight line geometry for map display)
    - Metadata columns for STAC queries
    - asset_href pointing to the data parquet file

    Parameters
    ----------
    parquet_dir : str or Path
        Directory containing bedmap parquet data files
    output_dir : str or Path
        Output directory for catalog GeoParquet files
    base_href : str
        Base URL for data assets (where data files are hosted)
    simplify_tolerance : float
        Tolerance for geometry simplification in degrees

    Returns
    -------
    dict
        Dictionary mapping version names to GeoDataFrames
    """
    parquet_dir = Path(parquet_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    parquet_files = sorted(parquet_dir.glob('*.parquet'))
    print(f"Building GeoParquet catalogs from {len(parquet_files)} files")

    # Group features by bedmap version
    version_features = {
        'BM1': [],
        'BM2': [],
        'BM3': [],
    }

    for parquet_file in parquet_files:
        print(f"  Processing {parquet_file.name}...")

        # Read metadata
        metadata = read_parquet_metadata(parquet_file)
        if not metadata:
            print(f"    Warning: No metadata found, skipping")
            continue

        # Extract geometry from metadata
        spatial_bounds = metadata.get('spatial_bounds', {})
        geometry_wkt = spatial_bounds.get('geometry')
        bbox = spatial_bounds.get('bbox')

        if geometry_wkt:
            try:
                geometry = wkt.loads(geometry_wkt)
                # Simplify for visualization
                geometry = geometry.simplify(simplify_tolerance, preserve_topology=True)
            except Exception as e:
                print(f"    Warning: Could not parse geometry: {e}")
                geometry = None
        else:
            geometry = None

        if geometry is None:
            print(f"    Warning: No geometry, skipping")
            continue

        # Extract metadata fields
        original_metadata = metadata.get('original_metadata', {})
        bedmap_version = metadata.get('bedmap_version', 'unknown')
        temporal_bounds = metadata.get('temporal_bounds', {})

        # Build asset href
        asset_href = base_href.rstrip('/') + '/' + parquet_file.name

        # Create feature dict matching what polar.html expects
        feature = {
            'geometry': geometry,
            'id': parquet_file.stem,
            'collection': f'bedmap{bedmap_version[-1]}' if bedmap_version.startswith('BM') else 'bedmap',
            'name': parquet_file.stem,
            'description': f"{original_metadata.get('project', '')} - {original_metadata.get('institution', '')}".strip(' -'),
            # Additional metadata for queries
            'asset_href': asset_href,
            'bedmap_version': bedmap_version,
            'row_count': metadata.get('row_count', 0),
            'institution': original_metadata.get('institution', ''),
            'project': original_metadata.get('project', ''),
            'bbox_minx': bbox[0] if bbox else None,
            'bbox_miny': bbox[1] if bbox else None,
            'bbox_maxx': bbox[2] if bbox else None,
            'bbox_maxy': bbox[3] if bbox else None,
            'temporal_start': temporal_bounds.get('start'),
            'temporal_end': temporal_bounds.get('end'),
        }

        # Add to appropriate version group
        if bedmap_version in version_features:
            version_features[bedmap_version].append(feature)
        else:
            print(f"    Warning: Unknown version {bedmap_version}, skipping")

    # Create and write GeoParquet catalogs per version
    catalogs = {}
    for version, features in version_features.items():
        if not features:
            continue

        # Create GeoDataFrame
        gdf = gpd.GeoDataFrame(features, crs='EPSG:4326')

        # Output filename: bedmap1.parquet, bedmap2.parquet, bedmap3.parquet
        version_num = version[-1]  # Extract '1', '2', or '3' from 'BM1', 'BM2', 'BM3'
        output_path = output_dir / f'bedmap{version_num}.parquet'

        # Write to GeoParquet
        gdf.to_parquet(output_path)

        print(f"  Created {output_path.name}: {len(gdf)} items")
        catalogs[version] = gdf

    print(f"\nGeoParquet catalogs written to {output_dir}")
    return catalogs
