"""
STAC catalog creation utilities for Open Polar Radar data.
"""

import re
import json
from pathlib import Path
from typing import List, Optional, Dict, Any, Union

import pyarrow.parquet as pq
import pystac
import stac_geoparquet
from shapely.geometry import mapping

from .metadata import extract_item_metadata
from .geometry import (
    simplify_geometry_polar_projection
)
from omegaconf import DictConfig

# STAC extension URLs
SCI_EXT = 'https://stac-extensions.github.io/scientific/v1.0.0/schema.json'
# PROJ_EXT = 'https://stac-extensions.github.io/projection/v2.0.0/schema.json'  # Not used - no collection-level geometry


def create_collection(
    collection_id: str,
    description: str,
    extent: pystac.Extent,
    license: str = "various",
    stac_extensions: Optional[List[str]] = None
) -> pystac.Collection:
    """
    Create a STAC collection for a campaign or data product grouping.

    Parameters
    ----------
    collection_id : str
        Unique identifier for the collection.
    description : str
        Human-readable description of the collection.
    extent : pystac.Extent
        Spatial and temporal extent of the collection.
    license : str, default ""
        Data license identifier.
    stac_extensions : list of str, optional
        List of STAC extension URLs to enable. If None, defaults to
        empty list.

    Returns
    -------
    pystac.Collection
        Collection object.

    Examples
    --------
    >>> from datetime import datetime
    >>> import pystac
    >>> extent = pystac.Extent(
    ...     spatial=pystac.SpatialExtent([[-180, -90, 180, 90]]),
    ...     temporal=pystac.TemporalExtent([[datetime(2016, 1, 1), datetime(2016, 12, 31)]])
    ... )
    >>> collection = create_collection("2016_campaign", "2016 Antarctic flights", extent)
    >>> item = create_item("item_001", geometry, bbox, datetime.now())
    >>> collection.add_item(item)
    """
    if stac_extensions is None:
        stac_extensions = []

    collection = pystac.Collection(
        id=collection_id,
        description=description,
        extent=extent,
        license=license,
        stac_extensions=stac_extensions
    )

    return collection


def create_item(
    item_id: str,
    geometry: Dict[str, Any],
    bbox: List[float],
    datetime: Any,
    properties: Optional[Dict[str, Any]] = None,
    assets: Optional[Dict[str, pystac.Asset]] = None,
    stac_extensions: Optional[List[str]] = None
) -> pystac.Item:
    """
    Create a STAC item for a flight line data segment.

    Parameters
    ----------
    item_id : str
        Unique identifier for the item.
    geometry : dict
        GeoJSON geometry object.
    bbox : list of float
        Bounding box coordinates [xmin, ymin, xmax, ymax].
    datetime : datetime
        Acquisition datetime.
    properties : dict, optional
        Additional metadata properties. If None, defaults to empty dict.
    assets : dict of str to pystac.Asset, optional
        Dictionary of assets (data files, thumbnails, etc.). Keys are
        asset names, values are pystac.Asset objects.
    stac_extensions : list of str, optional
        List of STAC extension URLs to enable. If None, defaults to
        file extension.

    Returns
    -------
    pystac.Item
        Item object with specified properties and assets.

    Examples
    --------
    >>> from datetime import datetime
    >>> import pystac
    >>> geometry = {"type": "Point", "coordinates": [-71.0, 42.0]}
    >>> bbox = [-71.1, 41.9, -70.9, 42.1]
    >>> props = {"instrument": "radar", "platform": "aircraft"}
    >>> assets = {
    ...     "data": pystac.Asset(href="https://example.com/data.mat", media_type="application/octet-stream")
    ... }
    >>> item = create_item("flight_001", geometry, bbox, datetime.now(), props, assets)
    """
    if properties is None:
        properties = {}
    if stac_extensions is None:
        stac_extensions = ['https://stac-extensions.github.io/file/v2.1.0/schema.json']

    item = pystac.Item(
        id=item_id,
        geometry=geometry,
        bbox=bbox,
        datetime=datetime,
        properties=properties,
        stac_extensions=stac_extensions
    )

    if assets:
        for key, asset in assets.items():
            item.add_asset(key, asset)

    return item


def create_items_from_flight_data(
    flight_data: Dict[str, Any],
    config: DictConfig,
    base_url: str = "https://data.cresis.ku.edu/data/rds/",
    campaign_name: str = "",
    primary_data_product: str = "CSARP_standard",
    verbose: bool = False,
    error_log_file: Optional[Union[str, Path]] = None
) -> List[pystac.Item]:
    """
    Create STAC items from flight line data.

    Parameters
    ----------
    flight_data : dict
        Flight metadata from discover_flight_lines(). Expected to contain
        'flight_id' and 'data_files' keys.
    config : DictConfig
        Configuration object with geometry.tolerance setting for simplification.
    base_url : str, default "https://data.cresis.ku.edu/data/rds/"
        Base URL for constructing asset hrefs.
    campaign_name : str, default ""
        Campaign name for URL construction.
    primary_data_product : str, default "CSARP_standard"
        Data product name to use as primary data source.
    verbose : bool, default False
        If True, print details for each item being processed.
    error_log_file : str or Path, optional
        Path to file where metadata extraction errors will be logged.
        If None, errors are printed to stdout (default behavior).

    Returns
    -------
    list of pystac.Item
        List of STAC Item objects, one per MAT file in the flight data.
        Each item contains geometry, temporal information, and asset links.
    """
    items = []
    flight_id = flight_data['flight_id']

    primary_data_files = flight_data['data_files'][primary_data_product].values()

    for data_file_path in primary_data_files:
        data_path = Path(data_file_path)

        try:
            # Extract metadata from MAT file only (no CSV needed)
            metadata = extract_item_metadata(data_path)
        except Exception as e:
            error_msg = f"Failed to extract metadata for {data_path}: {e}"

            if error_log_file is not None:
                # Log to file
                with open(error_log_file, 'a', encoding='utf-8') as f:
                    f.write(f"{error_msg}\n")
            else:
                # Fallback to print (current behavior)
                print(f"Warning: {error_msg}")

            continue

        item_id = f"{data_path.stem}"

        # Simplify geometry using config tolerance
        simplified_geom = simplify_geometry_polar_projection(
            metadata['geom'],
            simplify_tolerance=config.geometry.tolerance
        )
        geometry = mapping(simplified_geom)
        bbox = list(metadata['bbox'].bounds)
        datetime = metadata['date']

        rel_mat_path = f"{campaign_name}/{primary_data_product}/{flight_id}/{data_path.name}"
        data_href = base_url + rel_mat_path

        # Extract frame number from MAT filename (e.g., "Data_20161014_03_001.mat" -> "001")
        frame_match = re.search(r'_(\d+)\.mat$', data_path.name)
        frame = frame_match.group(1)

        # Extract date and segment number from flight_id (e.g., "20161014_03" -> "20161014", "03")
        # Split on underscore to avoid assuming fixed lengths
        parts = flight_id.split('_')
        date_part = parts[0]  # YYYYMMDD
        segment_num_str = parts[1]  # Segment number as string (formerly flight number)

        # Create OPR-specific properties
        properties = {
            'opr:date': date_part,
            'opr:segment': int(segment_num_str),  # Changed from opr:flight
            'opr:frame': int(frame)  # Changed from opr:segment
        }

        # Add scientific extension properties if available
        item_stac_extensions = ['https://stac-extensions.github.io/file/v2.1.0/schema.json']

        # Map metadata keys to property names
        meta_mapping = {
            'doi': 'sci:doi',
            'citation': 'sci:citation',
            'frequency': 'opr:frequency',
            'bandwidth': 'opr:bandwidth'
        }

        for key, prop in meta_mapping.items():
            if metadata.get(key) is not None:
                properties[prop] = metadata[key]

        if any(metadata.get(k) is not None for k in ['doi', 'citation']):
            item_stac_extensions.append('https://stac-extensions.github.io/scientific/v1.0.0/schema.json')

        assets = {}

        for data_product_type in flight_data['data_files'].keys():
            if data_path.name in flight_data['data_files'][data_product_type]:
                product_path = flight_data['data_files'][data_product_type][data_path.name]
                file_type = metadata.get('mimetype') # get_mat_file_type(product_path)
                if verbose:
                    print(f"[{file_type}] {product_path}")
                assets[data_product_type] = pystac.Asset(
                    href=base_url + f"{campaign_name}/{data_product_type}/{flight_id}/{data_path.name}",
                    media_type=file_type
                )
                if data_product_type == primary_data_product:
                    assets['data'] = assets[data_product_type]

        thumb_href = base_url + f"{campaign_name}/images/{flight_id}/{flight_id}_{frame}_2echo_picks.jpg"
        assets['thumbnails'] = pystac.Asset(
            href=thumb_href,
            media_type=pystac.MediaType.JPEG
        )

        flight_path_href = base_url + f"{campaign_name}/images/{flight_id}/{flight_id}_{frame}_0maps.jpg"
        assets['flight_path'] = pystac.Asset(
            href=flight_path_href,
            media_type=pystac.MediaType.JPEG
        )

        item = create_item(
            item_id=item_id,
            geometry=geometry,
            bbox=bbox,
            datetime=datetime,
            properties=properties,
            assets=assets,
            stac_extensions=item_stac_extensions
        )

        items.append(item)

    return items

def determine_hemisphere_from_geometry(items: List[pystac.Item]) -> Optional[str]:
    """
    Determine hemisphere based on item geometries by checking latitude values.

    Parameters
    ----------
    items : List[pystac.Item]
        STAC items with geometries

    Returns
    -------
    str or None
        'north' for positive latitudes, 'south' for negative, None if unable to determine
    """
    if not items:
        return None

    # Sample latitudes from items
    latitudes = []
    for item in items[:10]:  # Sample first 10 items
        if item.geometry and 'coordinates' in item.geometry:
            if item.geometry['type'] == 'Point':
                lat = item.geometry['coordinates'][1]
                latitudes.append(lat)
            elif item.geometry['type'] == 'LineString':
                for coord in item.geometry['coordinates'][:5]:  # Sample first 5 points
                    latitudes.append(coord[1])
            elif item.geometry['type'] == 'Polygon':
                for coord in item.geometry['coordinates'][0][:5]:  # Sample first 5 points
                    latitudes.append(coord[1])

    if not latitudes:
        return None

    # Check average latitude
    avg_lat = sum(latitudes) / len(latitudes)
    if avg_lat > 45:  # Clearly northern hemisphere
        return 'north'
    elif avg_lat < -45:  # Clearly southern hemisphere
        return 'south'

    return None


def determine_hemisphere_from_name(name: str) -> Optional[str]:
    """
    Determine hemisphere from collection/campaign name.

    Parameters
    ----------
    name : str
        Collection or campaign name

    Returns
    -------
    str or None
        'north' or 'south', or None if unable to determine
    """
    if 'Antarctica' in name:
        return 'south'
    elif 'Greenland' in name:
        return 'north'
    return None


def export_collection_to_parquet(
    collection: pystac.Collection,
    config: DictConfig,
    provider: str = None,
    hemisphere: str = None
) -> Optional[Path]:
    """
    Export a single STAC collection to a parquet file with collection metadata.

    This function directly converts STAC items to GeoParquet format without
    intermediate NDJSON, and includes the collection metadata in the Parquet
    file metadata as per the STAC GeoParquet specification.

    Parameters
    ----------
    collection : pystac.Collection
        STAC collection to export
    config : DictConfig
        Configuration object with output.path and logging.verbose settings
    provider : str, optional
        Data provider from config (awi, cresis, dtu, utig)
    hemisphere : str, optional
        Hemisphere ('north' or 'south'). If not provided, will attempt to detect.

    Returns
    -------
    Path or None
        Path to the created parquet file, or None if no items to export

    Examples
    --------
    >>> from omegaconf import OmegaConf
    >>> config = OmegaConf.create({'output': {'path': './output'}, 'logging': {'verbose': True}})
    >>> parquet_path = export_collection_to_parquet(collection, config, provider='cresis')
    >>> print(f"Exported to {parquet_path}")
    """
    # Extract settings from config
    output_dir = Path(config.output.path)
    verbose = config.logging.get('verbose', False)

    # Get items from collection and subcollections
    collection_items = list(collection.get_items())
    if not collection_items:
        for child_collection in collection.get_collections():
            collection_items.extend(list(child_collection.get_items()))

    if not collection_items:
        if verbose:
            print(f"  Skipping {collection.id}: no items")
        return None

    # Determine hemisphere if not provided
    if hemisphere is None:
        # Try from collection name first
        hemisphere = determine_hemisphere_from_name(collection.id)

        # Fall back to geometry-based detection
        if hemisphere is None:
            hemisphere = determine_hemisphere_from_geometry(collection_items)
            if verbose and hemisphere:
                print(f"  Detected hemisphere from geometry: {hemisphere}")

    # Get provider from config if not provided
    if provider is None:
        provider = config.data.get('provider')

    if verbose:
        if hemisphere:
            print(f"  Hemisphere: {hemisphere}")
        else:
            print(f"  WARNING: Could not determine hemisphere for {collection.id}")
        if provider:
            print(f"  Provider: {provider}")
        else:
            print(f"  WARNING: No provider specified for {collection.id}")

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Export to parquet
    parquet_file = output_dir / f"{collection.id}.parquet"

    if verbose:
        print(f"  Exporting collection: {collection.id} ({len(collection_items)} items)")

    # Build collections metadata - single collection in this case
    collection_dict = collection.to_dict()

    # Add OPR metadata to collection
    props = collection_dict.setdefault('properties', {})
    if hemisphere:
        props['opr:hemisphere'] = hemisphere
    if provider:
        props['opr:provider'] = provider

    # Clean collection links - remove item links with None hrefs
    if 'links' in collection_dict:
        collection_dict['links'] = [
            link for link in collection_dict['links']
            if not (link.get('rel') == 'item' and link.get('href') is None)
        ]
    collections_dict = {
        collection.id: collection_dict
    }

    # Clean items and add xopr metadata before export
    clean_items = []
    for item in collection_items:
        item_dict = item.to_dict()

        # Add OPR metadata to each item
        if 'properties' not in item_dict:
            item_dict['properties'] = {}
        if hemisphere:
            item_dict['properties']['opr:hemisphere'] = hemisphere
        if provider:
            item_dict['properties']['opr:provider'] = provider

        # Clean links with None hrefs
        if 'links' in item_dict:
            item_dict['links'] = [
                link for link in item_dict['links']
                if link.get('href') is not None
            ]
        clean_items.append(item_dict)

    # Convert items to Arrow format
    record_batch_reader = stac_geoparquet.arrow.parse_stac_items_to_arrow(clean_items)

    # Write to Parquet with collection metadata
    # Note: Using collection_metadata for compatibility with stac-geoparquet 0.7.0
    # In newer versions (>0.8), this should be 'collections' parameter
    stac_geoparquet.arrow.to_parquet(
        table=record_batch_reader,
        output_path=parquet_file,
        collection_metadata=collection_dict,  # Single collection metadata (cleaned)
        schema_version="1.1.0",  # Use latest schema version
        compression="snappy",  # Use snappy compression for better performance
        write_statistics=True  # Write column statistics for query optimization
    )

    if verbose:
        size_kb = parquet_file.stat().st_size / 1024
        print(f"  ✅ {collection.id}.parquet saved ({size_kb:.1f} KB)")

    return parquet_file


def build_catalog_from_parquet_metadata(
    parquet_paths: List[Path],
    output_file: Path,
    catalog_id: str = "OPR",
    catalog_description: str = "Open Polar Radar airborne data",
    verbose: bool = False
) -> None:
    """
    Build a catalog.json file from parquet files by reading their metadata.

    This function reads collection metadata from parquet files and creates a
    catalog.json file with proper links to the parquet files. Unlike
    export_collections_metadata which expects STAC collections as input,
    this function works directly with parquet files.

    Parameters
    ----------
    parquet_paths : List[Path]
        List of paths to parquet files containing STAC collections
    output_file : Path
        Output path for the catalog.json file
    catalog_id : str, optional
        Catalog ID, by default "OPR"
    catalog_description : str, optional
        Catalog description, by default "Open Polar Radar airborne data"
    base_url : str, optional
        Base URL for asset hrefs. If None, uses relative paths
    verbose : bool, optional
        If True, print progress messages

    Examples
    --------
    >>> import glob
    >>> parquet_files = [Path(p) for p in glob.glob("output/*.parquet")]
    >>> build_catalog_from_parquet_metadata(
    ...     parquet_files, Path("output/catalog.json"), verbose=True
    ... )
    """
    if verbose:
        print(f"Building catalog from {len(parquet_paths)} parquet files")

    collections_data = []

    for parquet_path in parquet_paths:
        if not parquet_path.exists():
            if verbose:
                print(f"  ⚠️  Skipping non-existent file: {parquet_path}")
            continue

        try:
            # Read parquet metadata without loading the data
            parquet_metadata = pq.read_metadata(str(parquet_path))
            file_metadata = parquet_metadata.metadata

            # Extract collection metadata from parquet file
            collection_dict = None
            if file_metadata:
                # Try new format first (stac:collections)
                if b'stac:collections' in file_metadata:
                    collections_json = file_metadata[b'stac:collections'].decode('utf-8')
                    collections_meta = json.loads(collections_json)
                    # Get the first (and should be only) collection
                    if collections_meta:
                        collection_id = list(collections_meta.keys())[0]
                        collection_dict = collections_meta[collection_id]
                # Try legacy format used by stac-geoparquet 0.7.0
                elif b'stac-geoparquet' in file_metadata:
                    geoparquet_meta = json.loads(file_metadata[b'stac-geoparquet'].decode('utf-8'))
                    if 'collection' in geoparquet_meta:
                        collection_dict = geoparquet_meta['collection']

            if not collection_dict:
                if verbose:
                    print(f"  ⚠️  No collection metadata found in {parquet_path.name}")
                continue

            # Extract relevant metadata for collections.json format
            collection_id = collection_dict.get('id', parquet_path.stem)

            # Build collection entry
            collection_entry = {
                'type': 'Collection',
                'stac_version': collection_dict.get('stac_version', '1.1.0'),
                'id': collection_id,
                'description': collection_dict.get('description', f"Collection {collection_id}"),
                'license': collection_dict.get('license', 'various'),
                'extent': collection_dict.get('extent'),
                'links': [],  # Clear links as we'll build our own
                'assets': {
                    'data': {
                        'href': f"./{parquet_path.name}",
                        'type': 'application/vnd.apache.parquet',
                        'title': 'Collection data in Apache Parquet format',
                        'roles': ['data']
                    }
                }
            }

            # Add optional fields if present
            if 'title' in collection_dict:
                collection_entry['title'] = collection_dict['title']

            # Add STAC extensions if present
            if collection_dict.get('stac_extensions'):
                collection_entry['stac_extensions'] = collection_dict['stac_extensions']

            # Add any extra fields that might be present (like sci:doi, etc)
            for key in collection_dict:
                if key.startswith('sci:') or key.startswith('sar:') or key.startswith('proj:'):
                    collection_entry[key] = collection_dict[key]

            collections_data.append(collection_entry)

            if verbose:
                num_rows = parquet_metadata.num_rows
                print(f"  ✅ Added {collection_id} ({num_rows} items)")

        except Exception as e:
            if verbose:
                print(f"  ❌ Error reading {parquet_path.name}: {e}")
            continue

    if not collections_data:
        raise ValueError("No valid collections found in parquet files")

    # Sort collections by ID for consistent output
    collections_data.sort(key=lambda x: x['id'])

    # Build the catalog structure
    catalog = {
        'type': 'Catalog',
        'id': catalog_id,
        'stac_version': '1.1.0',
        'description': catalog_description,
        'links': [
            {
                'rel': 'root',
                'href': './catalog.json',
                'type': 'application/json'
            }
        ]
    }

    # Add child links for each collection
    for collection in collections_data:
        catalog['links'].append({
            'rel': 'child',
            'href': f"./{collection['id']}.parquet",
            'type': 'application/vnd.apache.parquet',
            'title': collection.get('title', collection['description'])
        })

    # Determine common STAC extensions across all collections
    all_extensions = set()
    for collection in collections_data:
        if 'stac_extensions' in collection:
            all_extensions.update(collection['stac_extensions'])

    if all_extensions:
        catalog['stac_extensions'] = sorted(list(all_extensions))

    # Ensure output directory exists
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Write the catalog.json file
    with open(output_file, 'w') as f:
        json.dump(catalog, f, indent=2, separators=(",", ": "))

    if verbose:
        print(f"\n✅ Catalog saved to {output_file}")
        print(f"   Contains {len(collections_data)} collections")

    # Also save a collections.json file for compatibility
    collections_file = output_file.parent / "collections.json"
    with open(collections_file, 'w') as f:
        json.dump(collections_data, f, indent=2, separators=(",", ": "))

    if verbose:
        print(f"✅ Collections metadata saved to {collections_file}")
