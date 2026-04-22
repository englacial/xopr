#!/usr/bin/env python3
"""
Build STAC collections from OPR data using YAML configuration.
Produces hive-partitioned parquet files ready for upload to source.coop.
"""

import argparse
import logging
import re
import sys
import time
from pathlib import Path
from typing import Optional

from dask.distributed import Client, LocalCluster, as_completed
from omegaconf import DictConfig, OmegaConf

sys.path.append(str(Path(__file__).parent.parent / "src"))
from xopr.stac.config import load_config, save_config, validate_config
from xopr.stac.metadata import discover_campaigns, discover_flight_lines, collect_uniform_metadata
from xopr.stac.catalog import create_items_from_flight_data, create_collection, export_collection_to_parquet
from xopr.stac.geometry import build_collection_extent_and_geometry


def build_collection_parallel(campaign_path: Path, conf: DictConfig, client: Client) -> Optional[Path]:
    """
    Build a parquet collection for a single campaign using parallel processing.

    Parameters
    ----------
    campaign_path : Path
        Path to campaign directory
    conf : DictConfig
        Configuration object
    client : Client
        Dask distributed client for parallel processing

    Returns
    -------
    Path or None
        Path to created parquet file, or None if failed
    """
    campaign_name = campaign_path.name
    print(f"🔍 Discovering flight lines for campaign: {campaign_name}")
    
    # Discover flight lines
    flight_lines = discover_flight_lines(campaign_path, conf)
    if not flight_lines:
        print(f"  ❌ No flight lines found for campaign {campaign_name}")
        return None
    
    print(f"   Found {len(flight_lines)} flight lines")
    
    # Apply max_items limit
    if conf.processing.max_items:
        flight_lines = flight_lines[:conf.processing.max_items]
        print(f"   Limited to {len(flight_lines)} flights (max_items={conf.processing.max_items})")
    
    # Submit flight processing tasks
    print(f"📡 Processing {len(flight_lines)} flights in parallel...")
    futures = []
    for flight_data in flight_lines:
        future = client.submit(
            create_items_from_flight_data,
            flight_data,
            conf,
            base_url=conf.assets.base_url,
            campaign_name=campaign_name,
            primary_data_product=conf.data.primary_product,
            provider=conf.data.get("provider", "cresis"),
            verbose=False,
        )
        futures.append(future)
    
    # Collect results
    all_items = []
    completed_count = 0
    
    for future in as_completed(futures):
        try:
            items = future.result()
            all_items.extend(items)
            completed_count += 1
            print(f"   Completed flight {completed_count}/{len(flight_lines)} "
                  f"({len(items)} items)")
        except Exception as e:
            logging.warning(f"   ⚠️ Failed to process flight: {e}")
            completed_count += 1
    
    print(f"✅ Processed {len(all_items)} total items from {completed_count} flights")
    
    if not all_items:
        print(f"❌ No items created for campaign {campaign_name}")
        return None
    
    # Build collection
    print("📐 Building collection extent...")
    extent = build_collection_extent_and_geometry(all_items)
    extensions, extra_fields = collect_uniform_metadata(
        all_items, 
        ['sci:doi', 'sci:citation', 'sar:center_frequency', 'sar:bandwidth']
    )
    
    # Parse campaign name for metadata
    parts = campaign_name.split('_')
    year = parts[0] if parts else ''
    location = parts[1] if len(parts) > 1 else ''
    aircraft = parts[2] if len(parts) > 2 else ''
    
    print(f"📦 Creating STAC collection: {campaign_name}")
    collection = create_collection(
        collection_id=campaign_name,
        description=f"{year} {aircraft} flights over {location}",
        extent=extent,
        license=conf.output.get('license', 'various'),
        stac_extensions=extensions
    )
    
    for key, value in extra_fields.items():
        collection.extra_fields[key] = value
    
    for item in all_items:
        collection.add_item(item)
    
    # Export to parquet
    output_dir = Path(conf.output.path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("💾 Writing parquet file...")
    parquet_path = export_collection_to_parquet(
        collection, conf
    )
    
    print(f"✅ Successfully created: {parquet_path.name}")
    print(f"   Items: {len(all_items)}")
    print(f"   File size: {parquet_path.stat().st_size / 1024 / 1024:.1f} MB")
    return parquet_path


def process_campaign_with_fresh_cluster(campaign_path: Path, conf: DictConfig) -> Optional[Path]:
    """
    Process a single campaign with its own fresh Dask cluster.

    Parameters
    ----------
    campaign_path : Path
        Path to campaign directory
    conf : DictConfig
        Configuration object

    Returns
    -------
    Path or None
        Path to created parquet file, or None if failed
    """
    campaign_name = campaign_path.name
    print(f"\n🚀 Starting fresh Dask cluster for {campaign_name} with {conf.processing.n_workers} workers")

    # Create a fresh cluster for this campaign
    cluster = LocalCluster(
        n_workers=conf.processing.n_workers,
        threads_per_worker=1,
        memory_limit=conf.processing.get('memory_limit', '4GB'),
        silence_logs=logging.WARNING  # Reasonable log level
        # Removed processes=True as it can cause hanging issues
        # Removed death_timeout as it's not needed with default thread workers
    )

    client = None
    try:
        client = Client(cluster)
        print(f"   Dashboard: {client.dashboard_link}")

        # Process the campaign
        result = build_collection_parallel(campaign_path, conf, client)
        return result

    finally:
        # Ensure clean shutdown
        print(f"   Shutting down cluster for {campaign_name}...")

        try:
            # Close client first, then cluster
            if client:
                client.close()
            if cluster:
                cluster.close()
        except Exception as e:
            print(f"   Warning during shutdown: {e}")

        # Give OS time to reclaim resources
        time.sleep(1)


def process_catalog(conf: DictConfig):
    """
    Process catalog based on configuration.
    Each campaign gets its own fresh Dask cluster for perfect isolation.
    """
    # Discover campaigns
    campaigns = discover_campaigns(Path(conf.data.root), conf)

    # Filter by regex if specified
    if conf.data.get('campaign_filter'):
        pattern = re.compile(conf.data.campaign_filter)
        campaigns = [c for c in campaigns if pattern.match(c['name'])]

    if not campaigns:
        print("No campaigns found")
        return

    print(f"Found {len(campaigns)} campaigns to process")
    print("=" * 60)

    # Process each campaign with its own cluster
    results = []
    for i, campaign in enumerate(campaigns, 1):
        print(f"\n📊 Processing campaign {i}/{len(campaigns)}: {campaign['name']}")
        print("-" * 60)

        try:
            # Each campaign gets a fresh cluster
            path = process_campaign_with_fresh_cluster(
                Path(campaign['path']),
                conf
            )

            if path:
                results.append(path)
                print(f"✅ Successfully processed {campaign['name']}")
            else:
                print(f"⚠️ No data produced for {campaign['name']}")

        except Exception as e:
            logging.error(f"❌ Failed to process {campaign['name']}: {e}")
            print(f"❌ Failed to process {campaign['name']}: {e}")

    print("=" * 60)

    # Save config for reproducibility
    output_path = Path(conf.output.path)
    save_config(conf, output_path / "config_used.yaml")

    # Summary
    print(f"\n🎉 Successfully created {len(results)} out of {len(campaigns)} parquet files")
    if results:
        print("📋 Next step: Upload to source.coop with upload_stac_catalogs.py")


def main():
    parser = argparse.ArgumentParser(
        description="Build STAC collections from OPR data"
    )
    
    parser.add_argument("--config", "-c", required=True, help="YAML configuration file")
    parser.add_argument("--env", "-e", help="Environment (test/production)")
    parser.add_argument("overrides", nargs="*", help="Config overrides (e.g., processing.n_workers=8)")
    
    args = parser.parse_args()
    
    try:
        conf = load_config(args.config, args.overrides, args.env)
        validate_config(conf)
        
        if conf.logging.verbose:
            print(OmegaConf.to_yaml(conf))
        
        start = time.time()
        process_catalog(conf)
        print(f"Completed in {time.time() - start:.1f}s")
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
