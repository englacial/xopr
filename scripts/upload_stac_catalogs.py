#!/usr/bin/env python3
"""
Upload STAC parquet collection catalogs to the correct GCS locations.

This script processes a directory of STAC parquet files and uploads them to:
  gs://opr_stac/catalog/hemisphere=<north|south>/provider=<provider>/collection=<collection>/

The hemisphere and provider are read from the opr namespace metadata in the parquet file:
  - opr:hemisphere (north or south)
  - opr:provider (awi, cresis, dtu, utig, etc.)
Collection is extracted from the filename or metadata.
"""

import os
import sys
import argparse
import subprocess
import json
from pathlib import Path
from typing import Dict, Optional, Tuple, List
import re

try:
    import pyarrow.parquet as pq
    import pandas as pd
except ImportError:
    print("Error: pyarrow and pandas are required. Install with: pip install pyarrow pandas")
    sys.exit(1)


def run_command(cmd: List[str], env=None) -> Tuple[bool, str, str]:
    """Run a command and return success status, stdout, and stderr."""
    try:
        # Use current environment if none provided
        if env is None:
            env = os.environ.copy()
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, env=env)
        return True, result.stdout, result.stderr
    except subprocess.CalledProcessError as e:
        return False, e.stdout, e.stderr


def extract_metadata_from_parquet(file_path: str) -> Dict:
    """Extract metadata from a STAC parquet file, looking for opr namespace fields."""
    try:
        # Read parquet metadata
        parquet_file = pq.ParquetFile(file_path)
        metadata = parquet_file.metadata

        extracted = {}

        # Try to get file metadata (custom metadata stored with the file)
        file_metadata = metadata.metadata

        if file_metadata:
            # Check for opr namespace metadata
            for key in [b'opr:hemisphere', b'opr:provider', b'opr:collection']:
                if key in file_metadata:
                    extracted[key.decode('utf-8')] = file_metadata[key].decode('utf-8')

        # If not found in file metadata, check the actual data
        if not extracted:
            df = pd.read_parquet(file_path)

            # Check for opr namespace columns directly
            for key in ['opr:hemisphere', 'opr:provider', 'opr:collection']:
                if key in df.columns and len(df) > 0:
                    # Get the most common value (should be same for all rows in a collection)
                    value = df[key].mode()[0] if not df[key].isna().all() else None
                    if value:
                        extracted[key] = value

            # Check in properties column if it exists
            if 'properties' in df.columns and len(df) > 0 and not extracted:
                # Properties might be a dict/JSON column
                first_props = df['properties'].iloc[0]
                if isinstance(first_props, dict):
                    for key in ['opr:hemisphere', 'opr:provider', 'opr:collection']:
                        if key in first_props:
                            extracted[key] = first_props[key]
                elif isinstance(first_props, str):
                    try:
                        props_dict = json.loads(first_props)
                        for key in ['opr:hemisphere', 'opr:provider', 'opr:collection']:
                            if key in props_dict:
                                extracted[key] = props_dict[key]
                    except json.JSONDecodeError:
                        pass

            # Also check for standard STAC collection field
            if 'collection' in df.columns and len(df) > 0:
                collection_val = df['collection'].iloc[0]
                if pd.notna(collection_val):
                    extracted['stac_collection'] = str(collection_val)

        return extracted

    except Exception as e:
        print(f"Error reading metadata from {file_path}: {e}")
        return {}


def extract_info_from_filename(filename: str) -> Dict:
    """Extract collection information from filename as fallback."""
    info = {}

    base_name = Path(filename).stem

    # Remove common suffixes
    base_name = base_name.replace('_stac', '').replace('_catalog', '')

    # Try to match year_location_platform pattern
    year_pattern = r'^(\d{4})_([A-Za-z]+)_([A-Za-z0-9]+)'
    match = re.match(year_pattern, base_name)

    if match:
        year, location, platform = match.groups()
        collection = f"{year}_{location}_{platform}"
        info['collection'] = collection
    else:
        # Use the entire base name as collection
        info['collection'] = base_name

    return info


def check_gcs_auth() -> bool:
    """Check if GCS authentication is configured."""
    # Try to list the bucket
    env = os.environ.copy()
    cmd = ["gsutil", "ls", "gs://opr_stac/"]
    success, _, stderr = run_command(cmd, env)

    if not success:
        print("ERROR: Not authenticated to Google Cloud Storage")
        print(f"Error details: {stderr}")
        print("\nTo fix this, either:")
        print('1. Set service account: export GOOGLE_APPLICATION_CREDENTIALS="$HOME/opr-stac-key.json"')
        print("2. Or use gcloud: gcloud auth application-default login")

        # Check if the environment variable is set
        if os.environ.get('GOOGLE_APPLICATION_CREDENTIALS'):
            cred_path = os.environ['GOOGLE_APPLICATION_CREDENTIALS']
            print(f"\nNote: GOOGLE_APPLICATION_CREDENTIALS is set to: {cred_path}")
            if not os.path.exists(cred_path):
                print(f"  ERROR: File does not exist: {cred_path}")
            else:
                print(f"  File exists, but authentication is still failing.")
                print(f"  Check that the service account has storage.objectAdmin role.")
        return False
    return True


def build_gcs_path(hemisphere: str, provider: str, collection: str) -> str:
    """Build the GCS path for uploading."""
    return f"gs://opr_stac/catalog/hemisphere={hemisphere}/provider={provider}/collection={collection}/stac.parquet"


def upload_file(local_path: str, gcs_path: str, dry_run: bool = True) -> bool:
    """Upload a file to GCS."""
    if dry_run:
        print(f"[DRY RUN] Would upload:")
        print(f"  FROM: {local_path}")
        print(f"    TO: {gcs_path}")
        return True

    # Check for credentials and use them explicitly
    env = os.environ.copy()
    cred_path = env.get('GOOGLE_APPLICATION_CREDENTIALS')

    if cred_path and os.path.exists(cred_path):
        # Use gsutil with explicit service account key file
        print(f"   Using credentials: {cred_path}")
        cmd = [
            "gsutil",
            "-o", f"Credentials:gs_service_key_file={cred_path}",
            "cp", local_path, gcs_path
        ]
    else:
        # Fallback to default authentication
        print("   Warning: No service account key found, using default authentication")
        cmd = ["gsutil", "cp", local_path, gcs_path]

    print(f"Uploading: {local_path} -> {gcs_path}")

    success, stdout, stderr = run_command(cmd, env)

    if not success:
        print(f"Error uploading {local_path}: {stderr}")
        # Additional debugging
        if "Anonymous caller" in stderr:
            print("\nDEBUG: Authentication issue detected!")
            print(f"  GOOGLE_APPLICATION_CREDENTIALS={env.get('GOOGLE_APPLICATION_CREDENTIALS')}")
            if cred_path:
                print(f"  File exists at {cred_path}: {os.path.exists(cred_path)}")
                if os.path.exists(cred_path):
                    print(f"  File size: {os.path.getsize(cred_path)} bytes")
                    print(f"  File readable: {os.access(cred_path, os.R_OK)}")
        return False

    return True


def process_directory(directory: str, dry_run: bool = True, verbose: bool = False):
    """Process all parquet files in a directory and upload them to GCS."""
    directory_path = Path(directory)

    if not directory_path.exists():
        print(f"Error: Directory {directory} does not exist")
        return

    # Find all parquet files
    parquet_files = list(directory_path.glob("*.parquet"))

    if not parquet_files:
        print(f"No parquet files found in {directory}")
        return

    print(f"Found {len(parquet_files)} parquet files to process")

    successful = 0
    failed = 0
    skipped = 0

    for parquet_file in parquet_files:
        print(f"\nProcessing: {parquet_file.name}")

        # Extract metadata from parquet file
        metadata = extract_metadata_from_parquet(str(parquet_file))

        # Extract info from filename as fallback
        file_info = extract_info_from_filename(parquet_file.name)

        # Get required fields from metadata, with fallbacks
        hemisphere = metadata.get('opr:hemisphere')
        provider = metadata.get('opr:provider')
        collection = (metadata.get('opr:collection') or
                     metadata.get('stac_collection') or
                     file_info.get('collection'))

        # Validate required fields
        missing_fields = []
        if not hemisphere:
            missing_fields.append('opr:hemisphere')
        if not provider:
            missing_fields.append('opr:provider')
        if not collection:
            missing_fields.append('collection')

        if missing_fields:
            print(f"  ERROR: Missing required metadata fields: {', '.join(missing_fields)}")
            print(f"  Please ensure the parquet file contains opr:hemisphere and opr:provider metadata")
            print(f"  Skipping...")
            skipped += 1
            continue

        if verbose:
            print(f"  Hemisphere: {hemisphere}")
            print(f"  Provider: {provider}")
            print(f"  Collection: {collection}")

        # Validate hemisphere value
        if hemisphere not in ['north', 'south']:
            print(f"  ERROR: Invalid hemisphere value '{hemisphere}'. Must be 'north' or 'south'")
            print(f"  Skipping...")
            skipped += 1
            continue

        # Build GCS path
        gcs_path = build_gcs_path(hemisphere, provider, collection)

        # Upload file
        if upload_file(str(parquet_file), gcs_path, dry_run):
            successful += 1
        else:
            failed += 1

    print("\n" + "="*60)
    print("PROCESSING COMPLETE")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print(f"  Skipped: {skipped}")
    if skipped > 0:
        print("\nNote: Skipped files are missing required opr metadata.")
        print("Ensure your STAC catalog creation includes:")
        print("  - opr:hemisphere (north/south)")
        print("  - opr:provider (awi/cresis/dtu/utig/etc.)")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(
        description="Upload STAC parquet catalogs to GCS with correct hive structure"
    )
    parser.add_argument("directory", help="Directory containing parquet files to upload")
    parser.add_argument("--dry-run", action="store_true", default=True,
                        help="Perform a dry run without uploading (default: True)")
    parser.add_argument("--execute", action="store_true",
                        help="Actually execute the uploads (overrides --dry-run)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print verbose output")

    args = parser.parse_args()

    # If --execute is specified, override dry_run
    dry_run = not args.execute

    # Check authentication before processing (only for actual uploads)
    if not dry_run:
        print("Checking GCS authentication...")
        if not check_gcs_auth():
            sys.exit(1)
        print("✅ Authentication successful\n")

        response = input("WARNING: This will upload files to GCS. Are you sure? (yes/no): ")
        if response.lower() != 'yes':
            print("Aborted.")
            return

    if dry_run:
        print("="*60)
        print("DRY RUN MODE - No files will be uploaded")
        print("="*60 + "\n")

    process_directory(args.directory, dry_run, args.verbose)

    if dry_run:
        print("\nTo execute the actual uploads, run with --execute flag")


if __name__ == "__main__":
    main()