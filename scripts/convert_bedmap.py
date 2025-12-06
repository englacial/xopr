#!/usr/bin/env python
"""
CLI script to convert bedmap CSV files to cloud-optimized GeoParquet format.

Usage:
    python scripts/convert_bedmap.py --input ~/software/bedmap/Results --output scripts/output/bedmap
"""

import argparse
import sys
from pathlib import Path
import json
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from xopr.bedmap import batch_convert_bedmap


def main():
    parser = argparse.ArgumentParser(
        description='Convert bedmap CSV files to GeoParquet format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert all CSV files in directory
  python scripts/convert_bedmap.py --input ~/software/bedmap/Results --output scripts/output/bedmap

  # Convert with parallel processing
  python scripts/convert_bedmap.py --input ~/software/bedmap/Results --output scripts/output/bedmap --parallel --workers 8

  # Convert specific pattern
  python scripts/convert_bedmap.py --input ~/software/bedmap/Results --output scripts/output/bedmap --pattern "*BM3*.csv"
        """
    )

    parser.add_argument(
        '--input', '-i',
        type=str,
        default='~/software/bedmap/Results',
        help='Input directory containing bedmap CSV files (default: ~/software/bedmap/Results)'
    )

    parser.add_argument(
        '--output', '-o',
        type=str,
        default='scripts/output/bedmap',
        help='Output directory for GeoParquet files (default: scripts/output/bedmap)'
    )

    parser.add_argument(
        '--pattern', '-p',
        type=str,
        default='*.csv',
        help='Glob pattern for CSV files (default: *.csv)'
    )

    parser.add_argument(
        '--parallel',
        action='store_true',
        help='Process files in parallel'
    )

    parser.add_argument(
        '--workers', '-w',
        type=int,
        default=4,
        help='Number of parallel workers (default: 4)'
    )

    parser.add_argument(
        '--metadata-output', '-m',
        type=str,
        help='Optional path to save conversion metadata JSON'
    )

    args = parser.parse_args()

    # Expand paths
    input_dir = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output).expanduser().resolve()

    # Check input directory exists
    if not input_dir.exists():
        print(f"Error: Input directory does not exist: {input_dir}")
        sys.exit(1)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Converting bedmap CSV files")
    print(f"  Input directory: {input_dir}")
    print(f"  Output directory: {output_dir}")
    print(f"  Pattern: {args.pattern}")

    metadata_list = batch_convert_bedmap(
        input_dir=input_dir,
        output_dir=output_dir,
        pattern=args.pattern,
        parallel=args.parallel,
        n_workers=args.workers
    )

    if not metadata_list:
        print("Warning: No files were successfully converted")
        sys.exit(1)

    # Save metadata if requested
    if args.metadata_output:
        metadata_path = Path(args.metadata_output).expanduser()
        metadata_path.parent.mkdir(parents=True, exist_ok=True)

        # Add conversion metadata
        conversion_info = {
            'conversion_date': datetime.now().isoformat(),
            'input_directory': str(input_dir),
            'output_directory': str(output_dir),
            'files_converted': len(metadata_list),
            'files': metadata_list
        }

        with open(metadata_path, 'w') as f:
            json.dump(conversion_info, f, indent=2, default=str)

        print(f"\nMetadata saved to: {metadata_path}")

    # Print summary statistics
    print(f"\n{'='*60}")
    print(f"Conversion Summary:")
    print(f"  Files converted: {len(metadata_list)}")

    # Count by bedmap version
    version_counts = {}
    total_rows = 0
    for meta in metadata_list:
        version = meta.get('bedmap_version', 'unknown')
        version_counts[version] = version_counts.get(version, 0) + 1
        total_rows += meta.get('row_count', 0)

    print(f"  Total rows: {total_rows:,}")
    print(f"\n  Files by version:")
    for version, count in sorted(version_counts.items()):
        print(f"    {version}: {count}")

    print(f"{'='*60}")
    print("\nConversion complete!")
    print(f"\nNext steps:")
    print(f"  1. Build STAC catalog: python scripts/build_bedmap_catalog.py --input {output_dir}")
    print(f"  2. Upload to cloud: bash scripts/upload_bedmap_to_gcloud.sh")


if __name__ == '__main__':
    main()