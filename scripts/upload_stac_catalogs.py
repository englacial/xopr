#!/usr/bin/env python3
"""
Upload STAC parquet collection catalogs to the correct S3 locations on Source.coop.

This script processes a directory of STAC parquet files and uploads them to:
  s3://us-west-2.opendata.source.coop/englacial/xopr/catalog/hemisphere=<north|south>/provider=<provider>/collection=<collection>/

The hemisphere and provider are read from the opr namespace metadata in the parquet file:
  - opr:hemisphere (north or south)
  - opr:provider (awi, cresis, dtu, utig, etc.)
Collection is extracted from the filename or metadata.

Authentication uses a Source.coop token JSON file with aws_access_key_id,
aws_secret_access_key, aws_session_token, and region_name fields.
Pass the path via --credentials or SOURCE_COOP_CREDENTIALS env var.
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict

try:
    import pandas as pd
    import pyarrow.parquet as pq
except ImportError:
    print("Error: pyarrow and pandas are required. Install with: pip install pyarrow pandas")
    sys.exit(1)

try:
    import boto3
    from botocore.exceptions import ClientError, NoCredentialsError
except ImportError:
    print("Error: boto3 is required. Install with: pip install boto3")
    sys.exit(1)

S3_BUCKET = "us-west-2.opendata.source.coop"
S3_PREFIX = "englacial/xopr/catalog"


def load_credentials(cred_path: str) -> Dict:
    """Load Source.coop S3 credentials from a JSON file."""
    path = Path(cred_path).expanduser()
    if not path.exists():
        print(f"ERROR: Credentials file not found: {path}")
        sys.exit(1)
    with open(path) as f:
        creds = json.load(f)
    required = ['aws_access_key_id', 'aws_secret_access_key', 'aws_session_token']
    missing = [k for k in required if k not in creds]
    if missing:
        print(f"ERROR: Credentials file missing keys: {', '.join(missing)}")
        sys.exit(1)
    return creds


def create_s3_client(creds: Dict):
    """Create a boto3 S3 client from Source.coop credentials."""
    return boto3.client(
        's3',
        aws_access_key_id=creds['aws_access_key_id'],
        aws_secret_access_key=creds['aws_secret_access_key'],
        aws_session_token=creds['aws_session_token'],
        region_name=creds.get('region_name', 'us-west-2'),
    )


def check_s3_auth(s3_client) -> bool:
    """Check if S3 authentication is configured for Source.coop."""
    try:
        s3_client.list_objects_v2(Bucket=S3_BUCKET, Prefix=S3_PREFIX + "/", MaxKeys=1)
        return True
    except (ClientError, NoCredentialsError) as e:
        print("ERROR: Not authenticated to Source.coop S3")
        print(f"Error details: {e}")
        print("\nTo fix this, provide a credentials JSON file with:")
        print("  --credentials path/to/source_coop_token.json")
        print("\nThe JSON file should contain:")
        print('  {"aws_access_key_id": "...", "aws_secret_access_key": "...", "aws_session_token": "...", "region_name": "us-west-2"}')
        return False


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


def build_s3_key(hemisphere: str, provider: str, collection: str) -> str:
    """Build the S3 object key for uploading."""
    return f"{S3_PREFIX}/hemisphere={hemisphere}/provider={provider}/collection={collection}/stac.parquet"


def upload_file(s3_client, local_path: str, s3_key: str, dry_run: bool = True) -> bool:
    """Upload a file to Source.coop S3."""
    s3_uri = f"s3://{S3_BUCKET}/{s3_key}"
    if dry_run:
        print("[DRY RUN] Would upload:")
        print(f"  FROM: {local_path}")
        print(f"    TO: {s3_uri}")
        return True

    print(f"Uploading: {local_path} -> {s3_uri}")
    try:
        s3_client.upload_file(local_path, S3_BUCKET, s3_key)
        return True
    except (ClientError, NoCredentialsError) as e:
        print(f"Error uploading {local_path}: {e}")
        return False


def process_directory(s3_client, directory: str, dry_run: bool = True, verbose: bool = False):
    """Process all parquet files in a directory and upload them to Source.coop."""
    directory_path = Path(directory)

    if not directory_path.exists():
        print(f"Error: Directory {directory} does not exist")
        return

    # Find all parquet files
    parquet_files = list(directory_path.glob("**/*.parquet"))

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
            print("  Please ensure the parquet file contains opr:hemisphere and opr:provider metadata")
            print("  Skipping...")
            skipped += 1
            continue

        if verbose:
            print(f"  Hemisphere: {hemisphere}")
            print(f"  Provider: {provider}")
            print(f"  Collection: {collection}")

        # Validate hemisphere value
        if hemisphere not in ['north', 'south']:
            print(f"  ERROR: Invalid hemisphere value '{hemisphere}'. Must be 'north' or 'south'")
            print("  Skipping...")
            skipped += 1
            continue

        # Build S3 key and upload
        s3_key = build_s3_key(hemisphere, provider, collection)

        if upload_file(s3_client, str(parquet_file), s3_key, dry_run):
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
        description="Upload STAC parquet catalogs to Source.coop S3 with correct hive structure"
    )
    parser.add_argument("directory", help="Directory containing parquet files to upload")
    parser.add_argument("--credentials", default=None,
                        help="Path to Source.coop credentials JSON file "
                             "(or set SOURCE_COOP_CREDENTIALS env var)")
    parser.add_argument("--dry-run", action="store_true", default=True,
                        help="Perform a dry run without uploading (default: True)")
    parser.add_argument("--execute", action="store_true",
                        help="Actually execute the uploads (overrides --dry-run)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print verbose output")

    args = parser.parse_args()

    # Resolve credentials path
    cred_path = args.credentials or os.environ.get('SOURCE_COOP_CREDENTIALS')
    if not cred_path:
        print("ERROR: No credentials provided.")
        print("  Use --credentials path/to/token.json")
        print("  Or set SOURCE_COOP_CREDENTIALS env var")
        sys.exit(1)

    creds = load_credentials(cred_path)
    s3_client = create_s3_client(creds)

    # If --execute is specified, override dry_run
    dry_run = not args.execute

    # Check authentication before processing (only for actual uploads)
    if not dry_run:
        print("Checking Source.coop S3 authentication...")
        if not check_s3_auth(s3_client):
            sys.exit(1)
        print("Authentication successful\n")

        response = input("WARNING: This will upload files to Source.coop. Are you sure? (yes/no): ")
        if response.lower() != 'yes':
            print("Aborted.")
            return

    if dry_run:
        print("="*60)
        print("DRY RUN MODE - No files will be uploaded")
        print("="*60 + "\n")

    process_directory(s3_client, args.directory, dry_run, args.verbose)

    if dry_run:
        print("\nTo execute the actual uploads, run with --execute flag")


if __name__ == "__main__":
    main()
