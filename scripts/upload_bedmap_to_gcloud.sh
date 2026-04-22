#!/bin/bash

# Upload bedmap GeoParquet files and STAC catalog to Source Cooperative (S3)
#
# Requires AWS credentials (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY,
# and optionally AWS_SESSION_TOKEN) exported as environment variables.
#
# Usage:
#   bash scripts/upload_bedmap_to_gcloud.sh [-n|--dry-run]
#
# Options:
#   -n, --dry-run   Show what would be uploaded without uploading

DRY_RUN=""

while [[ $# -gt 0 ]]; do
  case $1 in
    -n|--dry-run)
      DRY_RUN="--dryrun"
      shift
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: $0 [-n|--dry-run]"
      exit 1
      ;;
  esac
done

if [ -n "$DRY_RUN" ]; then
  echo "Dry-run mode: no files will be uploaded"
fi

if [ -z "${AWS_ACCESS_KEY_ID:-}" ]; then
  echo "Error: AWS_ACCESS_KEY_ID not set"
  exit 1
fi

# Check the current directory is the root of the git repo
if [ ! -f scripts/upload_bedmap_to_gcloud.sh ]; then
  echo "Please run this script from the root of the git repository."
  exit 1
fi

# Set variables
PARQUET_DIR="scripts/output/bedmap"
CATALOG_DIR="scripts/output/bedmap_catalog"
S3_DATA_PATH="s3://us-west-2.opendata.source.coop/englacial/bedmap/data/"
S3_CATALOG_ROOT="s3://us-west-2.opendata.source.coop/englacial/bedmap/"

# Check if parquet files exist
if [ ! -d "$PARQUET_DIR" ]; then
  echo "Error: Parquet directory not found: $PARQUET_DIR"
  echo "Please run: python scripts/convert_bedmap.py first"
  exit 1
fi

# Check if catalog exists
if [ ! -d "$CATALOG_DIR" ]; then
  echo "Warning: STAC catalog directory not found: $CATALOG_DIR"
  echo "Please run: python scripts/build_bedmap_catalog.py first"
  echo ""
  echo "Proceeding to upload parquet files only..."
fi

# Count files
PARQUET_COUNT=$(find "$PARQUET_DIR" -name "*.parquet" 2>/dev/null | wc -l)
echo "Found $PARQUET_COUNT parquet files to upload"

# Upload parquet files
echo ""
echo "Uploading bedmap parquet files..."
echo "  Source: $PARQUET_DIR/"
echo "  Destination: $S3_DATA_PATH"

aws s3 sync "$PARQUET_DIR" "$S3_DATA_PATH" --exclude "*" --include "*.parquet" $DRY_RUN

if [ $? -eq 0 ]; then
  echo "Parquet files uploaded successfully"
else
  echo "Error uploading parquet files"
  exit 1
fi

# Upload GeoParquet STAC catalogs if they exist
if [ -d "$CATALOG_DIR" ]; then
  CATALOG_FILES=$(find "$CATALOG_DIR" -name "bedmap*.parquet" 2>/dev/null)

  if [ -n "$CATALOG_FILES" ]; then
    echo ""
    echo "Uploading bedmap GeoParquet STAC catalogs..."

    for f in "$CATALOG_DIR"/bedmap*.parquet; do
      aws s3 cp "$f" "$S3_CATALOG_ROOT$(basename "$f")" $DRY_RUN
    done

    if [ $? -eq 0 ]; then
      echo "GeoParquet catalogs uploaded successfully"
    else
      echo "Error uploading GeoParquet catalogs"
      exit 1
    fi
  else
    echo ""
    echo "Warning: No bedmap*.parquet catalog files found in $CATALOG_DIR"
    echo "Please run: python scripts/build_bedmap_catalog.py first"
  fi
fi

# Verify upload (skip in dry-run mode)
if [ -z "$DRY_RUN" ]; then
  echo ""
  echo "Verifying upload..."
  echo "Data files:"
  aws s3 ls "$S3_DATA_PATH" --no-sign-request | wc -l | xargs -I{} echo "  {} files"

  echo "Catalog files:"
  aws s3 ls "$S3_CATALOG_ROOT" --no-sign-request | grep 'bedmap[123]' || echo "  (no catalog files found)"
fi

echo ""
echo "Upload complete!"
echo "  Data: $S3_DATA_PATH"
echo "  Catalogs: ${S3_CATALOG_ROOT}bedmap*.parquet"