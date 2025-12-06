#!/bin/bash

# Upload bedmap GeoParquet files and STAC catalog to Google Cloud Storage
#
# Usage:
#   bash scripts/upload_bedmap_to_gcloud.sh [-v|--verbose] [-d|--debug] [-n|--dry-run]
#
# Options:
#   -v, --verbose   Show detailed progress
#   -d, --debug     Show debug output (very verbose)
#   -n, --dry-run   Show what would be uploaded without uploading

VERBOSE=""
DEBUG=""
DRY_RUN=""

while [[ $# -gt 0 ]]; do
  case $1 in
    -v|--verbose)
      VERBOSE="-v"
      shift
      ;;
    -d|--debug)
      DEBUG="-D"
      shift
      ;;
    -n|--dry-run)
      DRY_RUN="-n"
      shift
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: $0 [-v|--verbose] [-d|--debug] [-n|--dry-run]"
      exit 1
      ;;
  esac
done

# Build gsutil options
GSUTIL_OPTS="$VERBOSE $DEBUG $DRY_RUN"

if [ -n "$VERBOSE" ]; then
  echo "Verbose mode enabled"
fi
if [ -n "$DEBUG" ]; then
  echo "Debug mode enabled (very verbose)"
fi
if [ -n "$DRY_RUN" ]; then
  echo "Dry-run mode: no files will be uploaded"
fi

# Check the current directory is the root of the git repo
if [ ! -f scripts/upload_bedmap_to_gcloud.sh ]; then
  echo "Please run this script from the root of the git repository."
  exit 1
fi

# Set variables
PARQUET_DIR="scripts/output/bedmap"
CATALOG_DIR="scripts/output/bedmap_catalog"
GCS_DATA_PATH="gs://opr_stac/bedmap/data/"
GCS_CATALOG_ROOT="gs://opr_stac/bedmap/"

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
echo "Uploading bedmap parquet files to Google Cloud Storage..."
echo "  Source: $PARQUET_DIR/*.parquet"
echo "  Destination: $GCS_DATA_PATH"

gsutil $DEBUG -m cp $VERBOSE $DRY_RUN "$PARQUET_DIR"/*.parquet "$GCS_DATA_PATH"

if [ $? -eq 0 ]; then
  echo "✓ Parquet files uploaded successfully"
else
  echo "✗ Error uploading parquet files"
  exit 1
fi

# Upload GeoParquet STAC catalogs if they exist
if [ -d "$CATALOG_DIR" ]; then
  # Find all bedmap*.parquet catalog files
  CATALOG_FILES=$(find "$CATALOG_DIR" -name "bedmap*.parquet" 2>/dev/null)

  if [ -n "$CATALOG_FILES" ]; then
    echo ""
    echo "Uploading bedmap GeoParquet STAC catalogs..."
    echo "  Source: $CATALOG_DIR/bedmap*.parquet"
    echo "  Destination: gs://opr_stac/bedmap/"

    gsutil $DEBUG -m cp $VERBOSE $DRY_RUN "$CATALOG_DIR"/bedmap*.parquet "gs://opr_stac/bedmap/"

    if [ $? -eq 0 ]; then
      echo "✓ GeoParquet catalogs uploaded successfully"
    else
      echo "✗ Error uploading GeoParquet catalogs"
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
  gsutil $DEBUG ls "$GCS_DATA_PATH" | head -5
  echo "..."

  echo ""
  echo "Catalog files:"
  gsutil $DEBUG ls "gs://opr_stac/bedmap/bedmap*.parquet" 2>/dev/null || echo "  (no catalog files found)"
else
  echo ""
  echo "Skipping verification in dry-run mode"
fi

echo ""
echo "============================================================"
echo "Upload complete!"
echo ""
echo "Bedmap data is now available at:"
echo "  Data: $GCS_DATA_PATH"
echo "  Catalogs: ${GCS_CATALOG_ROOT}bedmap*.parquet"
echo "============================================================"