"""
STAC catalog generation for bedmap data.

This module creates STAC (SpatioTemporal Asset Catalog) items and collections
from converted bedmap GeoParquet files.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Union
from datetime import datetime
import warnings

import pystac
from pystac import Catalog, Collection, Item, Asset, Extent, SpatialExtent, TemporalExtent
from pystac.extensions.projection import ProjectionExtension
import pyarrow.parquet as pq
import shapely
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

    # Get metadata from parquet schema
    metadata_bytes = parquet_file.schema.metadata.get(b'bedmap_metadata')

    if metadata_bytes:
        return json.loads(metadata_bytes.decode())
    else:
        warnings.warn(f"No bedmap metadata found in {parquet_path}")
        return {}


def create_bedmap_stac_item(
    parquet_path: Union[str, Path],
    asset_href: str,
    collection_id: str = None
) -> Item:
    """
    Create a STAC item from a bedmap GeoParquet file.

    Parameters
    ----------
    parquet_path : str or Path
        Path to the GeoParquet file
    asset_href : str
        URL/path where the parquet file will be accessible
    collection_id : str, optional
        Collection ID to associate with this item

    Returns
    -------
    pystac.Item
        STAC item representing the bedmap data file
    """
    parquet_path = Path(parquet_path)

    # Read metadata from parquet file
    metadata = read_parquet_metadata(parquet_path)

    # Extract key information
    item_id = parquet_path.stem  # e.g., AWI_1994_DML1_AIR_BM2

    # Get temporal bounds
    temporal_bounds = metadata.get('temporal_bounds', {})
    start_time = temporal_bounds.get('start')
    end_time = temporal_bounds.get('end')

    if start_time and end_time:
        # Parse ISO strings back to datetime
        start_dt = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
        end_dt = datetime.fromisoformat(end_time.replace('Z', '+00:00'))
    else:
        # Fallback to current time if no temporal info
        start_dt = end_dt = datetime.now()

    # Get spatial bounds
    spatial_bounds = metadata.get('spatial_bounds', {})
    bbox = spatial_bounds.get('bbox')
    geometry_wkt = spatial_bounds.get('geometry')

    # Parse geometry from WKT
    if geometry_wkt:
        try:
            geometry_shape = wkt.loads(geometry_wkt)
            # Convert to GeoJSON format
            geometry = shapely.geometry.mapping(geometry_shape)
        except Exception as e:
            warnings.warn(f"Could not parse geometry for {item_id}: {e}")
            # Fallback to bbox polygon
            if bbox:
                geometry = {
                    'type': 'Polygon',
                    'coordinates': [[
                        [bbox[0], bbox[1]],  # min_lon, min_lat
                        [bbox[2], bbox[1]],  # max_lon, min_lat
                        [bbox[2], bbox[3]],  # max_lon, max_lat
                        [bbox[0], bbox[3]],  # min_lon, max_lat
                        [bbox[0], bbox[1]]   # close polygon
                    ]]
                }
            else:
                # No spatial information available
                geometry = None
    else:
        # No geometry, use bbox if available
        if bbox:
            geometry = {
                'type': 'Polygon',
                'coordinates': [[
                    [bbox[0], bbox[1]],
                    [bbox[2], bbox[1]],
                    [bbox[2], bbox[3]],
                    [bbox[0], bbox[3]],
                    [bbox[0], bbox[1]]
                ]]
            }
        else:
            geometry = None

    # Extract original metadata
    original_metadata = metadata.get('original_metadata', {})

    # Create properties
    properties = {
        'bedmap:version': metadata.get('bedmap_version', 'unknown'),
        'bedmap:row_count': metadata.get('row_count', 0),
        'bedmap:source_csv': metadata.get('source_csv', ''),
    }

    # Add original metadata fields
    if original_metadata:
        properties.update({
            'bedmap:project': original_metadata.get('project', ''),
            'bedmap:institution': original_metadata.get('institution', ''),
            'bedmap:creator_name': original_metadata.get('creator_name', ''),
            'bedmap:instrument': original_metadata.get('instrument', ''),
            'bedmap:platform': original_metadata.get('platform', ''),
            'bedmap:centre_frequency': original_metadata.get('centre_frequency'),
            'bedmap:electromagnetic_wave_speed_in_ice': original_metadata.get('electromagnetic_wave_speed_in_ice'),
            'bedmap:firn_correction': original_metadata.get('firn_correction'),
            'bedmap:source_doi': original_metadata.get('source', ''),
            'bedmap:references': original_metadata.get('references', ''),
        })

    # Create STAC item
    item = Item(
        id=item_id,
        geometry=geometry,
        bbox=bbox if bbox else [-180, -90, 180, 90],  # Global bbox if not available
        datetime=None,  # We use start/end times instead
        properties=properties,
        start_datetime=start_dt,
        end_datetime=end_dt,
        collection=collection_id
    )

    # Add projection extension (Antarctic Polar Stereographic)
    proj_ext = ProjectionExtension.ext(item, add_if_missing=True)
    proj_ext.epsg = 4326  # WGS84 for now, could be 3031 for Antarctic

    # Add data asset
    item.add_asset(
        key='data',
        asset=Asset(
            href=asset_href,
            title='Bedmap GeoParquet Data',
            description=f'Bedmap radar data in GeoParquet format',
            media_type='application/x-parquet',
            roles=['data']
        )
    )

    # Add metadata asset if needed
    item.add_asset(
        key='metadata',
        asset=Asset(
            href=asset_href,  # Same file contains metadata
            title='Embedded Metadata',
            description='Metadata embedded in GeoParquet file',
            media_type='application/json',
            roles=['metadata']
        )
    )

    return item


def create_bedmap_collection(
    collection_id: str,
    title: str,
    description: str,
    items: List[Item]
) -> Collection:
    """
    Create a STAC collection for bedmap data.

    Parameters
    ----------
    collection_id : str
        Unique identifier for the collection (e.g., 'bedmap-v1')
    title : str
        Human-readable title
    description : str
        Detailed description of the collection
    items : list of Item
        List of STAC items to include in collection

    Returns
    -------
    pystac.Collection
        STAC collection object
    """
    # Calculate overall extent from items
    all_bboxes = []
    all_intervals = []

    for item in items:
        if item.bbox:
            all_bboxes.append(item.bbox)

        if item.datetime:
            all_intervals.append([item.datetime, item.datetime])
        elif item.properties.get('start_datetime') and item.properties.get('end_datetime'):
            all_intervals.append([
                item.properties['start_datetime'],
                item.properties['end_datetime']
            ])

    # Calculate overall bbox
    if all_bboxes:
        min_lon = min(bbox[0] for bbox in all_bboxes)
        min_lat = min(bbox[1] for bbox in all_bboxes)
        max_lon = max(bbox[2] for bbox in all_bboxes)
        max_lat = max(bbox[3] for bbox in all_bboxes)
        overall_bbox = [min_lon, min_lat, max_lon, max_lat]
    else:
        overall_bbox = [-180, -90, 180, -60]  # Antarctic region

    # Create extent
    extent = Extent(
        spatial=SpatialExtent(bboxes=[overall_bbox]),
        temporal=TemporalExtent(intervals=all_intervals if all_intervals else [[None, None]])
    )

    # Create collection
    collection = Collection(
        id=collection_id,
        title=title,
        description=description,
        extent=extent,
        license='CC-BY-4.0',
        keywords=['bedmap', 'antarctica', 'ice-thickness', 'radar', 'geophysics']
    )

    # Add items to collection
    for item in items:
        collection.add_item(item)

    return collection


def build_bedmap_catalog(
    parquet_dir: Union[str, Path],
    catalog_dir: Union[str, Path],
    base_href: str = 'gs://opr_stac/bedmap/data/',
    catalog_title: str = 'Bedmap Data Catalog',
    catalog_description: str = 'STAC catalog for Bedmap ice thickness data'
) -> Catalog:
    """
    Build a complete STAC catalog from bedmap GeoParquet files.

    Parameters
    ----------
    parquet_dir : str or Path
        Directory containing GeoParquet files
    catalog_dir : str or Path
        Directory to write STAC catalog
    base_href : str
        Base URL for data assets
    catalog_title : str
        Title for the root catalog
    catalog_description : str
        Description for the root catalog

    Returns
    -------
    pystac.Catalog
        Root STAC catalog
    """
    parquet_dir = Path(parquet_dir)
    catalog_dir = Path(catalog_dir)
    catalog_dir.mkdir(parents=True, exist_ok=True)

    # Create root catalog
    catalog = Catalog(
        id='bedmap-catalog',
        title=catalog_title,
        description=catalog_description
    )

    # Group parquet files by bedmap version
    version_items = {
        'BM1': [],
        'BM2': [],
        'BM3': [],
        'unknown': []
    }

    # Process all parquet files
    parquet_files = sorted(parquet_dir.glob('*.parquet'))
    print(f"Found {len(parquet_files)} parquet files to process")

    for parquet_file in parquet_files:
        # Determine asset href
        asset_href = base_href + parquet_file.name

        # Read metadata to determine version
        metadata = read_parquet_metadata(parquet_file)
        version = metadata.get('bedmap_version', 'unknown')

        # Create STAC item
        collection_id = f'bedmap-{version.lower()}'
        item = create_bedmap_stac_item(parquet_file, asset_href, collection_id)

        # Add to appropriate version group
        version_items[version].append(item)

    # Create collections for each version with items
    for version, items in version_items.items():
        if items:  # Only create collection if there are items
            collection_id = f'bedmap-{version.lower()}'

            if version == 'BM1':
                title = 'Bedmap Version 1'
                description = 'Original Bedmap compilation of Antarctic ice thickness data'
            elif version == 'BM2':
                title = 'Bedmap2'
                description = 'Bedmap2 - improved compilation of Antarctic ice thickness data'
            elif version == 'BM3':
                title = 'Bedmap3'
                description = 'Bedmap3 - latest compilation of Antarctic ice thickness data'
            else:
                title = 'Bedmap Unknown Version'
                description = 'Bedmap data with unidentified version'

            collection = create_bedmap_collection(
                collection_id=collection_id,
                title=title,
                description=description,
                items=items
            )

            catalog.add_child(collection)
            print(f"Created collection '{collection_id}' with {len(items)} items")

    # Set catalog hrefs
    catalog.normalize_and_save(
        root_href=str(catalog_dir),
        catalog_type=pystac.CatalogType.SELF_CONTAINED
    )

    print(f"\nCatalog saved to {catalog_dir}")
    print(f"  Root catalog: {catalog_dir / 'catalog.json'}")

    return catalog