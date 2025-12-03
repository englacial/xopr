#!/usr/bin/env python
"""
Build STAC catalog from bedmap GeoParquet files.

Usage:
    python scripts/build_bedmap_catalog.py --input scripts/output/bedmap --output scripts/output/bedmap_catalog
"""

import argparse
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from xopr.bedmap import build_bedmap_catalog


def main():
    parser = argparse.ArgumentParser(
        description='Build STAC catalog from bedmap GeoParquet files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Build catalog from parquet files
  python scripts/build_bedmap_catalog.py --input scripts/output/bedmap --output scripts/output/bedmap_catalog

  # Build with custom base URL
  python scripts/build_bedmap_catalog.py --input scripts/output/bedmap --output scripts/output/bedmap_catalog --base-href s3://my-bucket/bedmap/
        """
    )

    parser.add_argument(
        '--input', '-i',
        type=str,
        default='scripts/output/bedmap',
        help='Directory containing bedmap parquet files'
    )

    parser.add_argument(
        '--output', '-o',
        type=str,
        default='scripts/output/bedmap_catalog',
        help='Output directory for STAC catalog'
    )

    parser.add_argument(
        '--base-href',
        type=str,
        default='gs://opr_stac/bedmap/data/',
        help='Base URL/path for data assets in catalog'
    )

    parser.add_argument(
        '--title',
        type=str,
        default='Bedmap Data Catalog',
        help='Title for the STAC catalog'
    )

    parser.add_argument(
        '--description',
        type=str,
        default='STAC catalog for Bedmap Antarctic ice thickness data',
        help='Description for the STAC catalog'
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

    print(f"Building STAC catalog for bedmap data")
    print(f"  Input directory: {parquet_dir}")
    print(f"  Output directory: {catalog_dir}")
    print(f"  Base href: {args.base_href}")
    print(f"  Parquet files found: {len(parquet_files)}")
    print()

    # Build the catalog
    try:
        catalog = build_bedmap_catalog(
            parquet_dir=parquet_dir,
            catalog_dir=catalog_dir,
            base_href=args.base_href,
            catalog_title=args.title,
            catalog_description=args.description
        )

        print(f"\n{'='*60}")
        print("STAC Catalog Build Summary:")
        print(f"  Catalog ID: {catalog.id}")
        print(f"  Title: {catalog.title}")

        # Count collections and items
        n_collections = len(list(catalog.get_collections()))
        n_items = sum(len(list(col.get_items())) for col in catalog.get_collections())

        print(f"  Collections: {n_collections}")
        print(f"  Total items: {n_items}")

        # List collections
        print(f"\nCollections created:")
        for collection in catalog.get_collections():
            n_col_items = len(list(collection.get_items()))
            print(f"    {collection.id}: {n_col_items} items")

        print(f"\nCatalog files written to: {catalog_dir}")
        print(f"  Root catalog: {catalog_dir / 'catalog.json'}")
        print(f"{'='*60}")

    except Exception as e:
        print(f"Error building catalog: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print("\nCatalog build complete!")
    print("\nNext steps:")
    print(f"  1. Review catalog: ls -la {catalog_dir}")
    print(f"  2. Upload to cloud: bash scripts/upload_bedmap_to_gcloud.sh")


if __name__ == '__main__':
    main()