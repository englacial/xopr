#!/usr/bin/env python
"""
Build GeoParquet STAC catalogs from bedmap data files.

Creates separate catalog files per bedmap version:
- bedmap1.parquet (BM1 items)
- bedmap2.parquet (BM2 items)
- bedmap3.parquet (BM3 items)

Usage:
    python scripts/build_bedmap_catalog.py --input scripts/output/bedmap --output scripts/output/bedmap_catalog
"""

import argparse
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from xopr.bedmap import build_bedmap_geoparquet_catalog


def main():
    parser = argparse.ArgumentParser(
        description='Build GeoParquet STAC catalogs from bedmap data files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Build catalogs from parquet files
  python scripts/build_bedmap_catalog.py

  # Build with custom paths
  python scripts/build_bedmap_catalog.py --input scripts/output/bedmap --output scripts/output/bedmap_catalog

  # Build with custom base URL
  python scripts/build_bedmap_catalog.py --base-href gs://my-bucket/bedmap/data/
        """
    )

    parser.add_argument(
        '--input', '-i',
        type=str,
        default='scripts/output/bedmap',
        help='Directory containing bedmap parquet data files'
    )

    parser.add_argument(
        '--output', '-o',
        type=str,
        default='scripts/output/bedmap_catalog',
        help='Output directory for GeoParquet catalog files'
    )

    parser.add_argument(
        '--base-href',
        type=str,
        default='gs://opr_stac/bedmap/data/',
        help='Base URL/path for data assets in catalog'
    )

    args = parser.parse_args()

    # Expand paths
    parquet_dir = Path(args.input).expanduser().resolve()
    catalog_dir = Path(args.output).expanduser().resolve()

    # Check input directory exists
    if not parquet_dir.exists():
        print(f"Error: Input directory does not exist: {parquet_dir}")
        sys.exit(1)

    # Check for parquet files
    parquet_files = list(parquet_dir.glob('*.parquet'))
    if not parquet_files:
        print(f"Error: No parquet files found in {parquet_dir}")
        print("Please run convert_bedmap.py first")
        sys.exit(1)

    print(f"Building GeoParquet STAC catalogs for bedmap data")
    print(f"  Input directory: {parquet_dir}")
    print(f"  Output directory: {catalog_dir}")
    print(f"  Base href: {args.base_href}")
    print(f"  Data files found: {len(parquet_files)}")
    print()

    # Build the GeoParquet catalogs
    try:
        catalogs = build_bedmap_geoparquet_catalog(
            parquet_dir=parquet_dir,
            output_dir=catalog_dir,
            base_href=args.base_href,
        )

        print(f"\n{'='*60}")
        print("GeoParquet Catalog Build Summary:")
        total_items = sum(len(gdf) for gdf in catalogs.values())
        print(f"  Total catalogs: {len(catalogs)}")
        print(f"  Total items: {total_items}")

        for version, gdf in catalogs.items():
            version_num = version[-1]
            print(f"    bedmap{version_num}.parquet: {len(gdf)} items")

        print(f"\nCatalog files written to: {catalog_dir}")
        print(f"{'='*60}")

    except Exception as e:
        print(f"Error building catalog: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print("\nCatalog build complete!")

if __name__ == '__main__':
    main()
