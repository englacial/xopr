#!/bin/bash

# Upload bedmap GeoParquet files and STAC catalog to Google Cloud Storage

# Check the current directory is the root of the git repo
if [ ! -f scripts/upload_bedmap_to_gcloud.sh ]; then
  echo "Please run this script from the root of the git repository."
  exit 1
fi

# Set variables
PARQUET_DIR="scripts/output/bedmap"
CATALOG_DIR="scripts/output/bedmap_catalog"
GCS_DATA_PATH="gs://opr_stac/bedmap/data/"
GCS_CATALOG_PATH="gs://opr_stac/bedmap/catalog/"

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

gsutil -m cp "$PARQUET_DIR"/*.parquet "$GCS_DATA_PATH"

if [ $? -eq 0 ]; then
  echo "✓ Parquet files uploaded successfully"
else
  echo "✗ Error uploading parquet files"
  exit 1
fi

# Upload STAC catalog if it exists
if [ -d "$CATALOG_DIR" ]; then
  echo ""
  echo "Uploading bedmap STAC catalog to Google Cloud Storage..."
  echo "  Source: $CATALOG_DIR"
  echo "  Destination: $GCS_CATALOG_PATH"

  gsutil -m cp -r "$CATALOG_DIR"/* "$GCS_CATALOG_PATH"

  if [ $? -eq 0 ]; then
    echo "✓ STAC catalog uploaded successfully"
  else
    echo "✗ Error uploading STAC catalog"
    exit 1
  fi
fi

# Verify upload
echo ""
echo "Verifying upload..."
echo "Data files:"
gsutil ls "$GCS_DATA_PATH" | head -5
echo "..."

if [ -d "$CATALOG_DIR" ]; then
  echo ""
  echo "Catalog files:"
  gsutil ls "$GCS_CATALOG_PATH" | head -5
  echo "..."
fi

echo ""
echo "============================================================"
echo "Upload complete!"
echo ""
echo "Bedmap data is now available at:"
echo "  Data: $GCS_DATA_PATH"
if [ -d "$CATALOG_DIR" ]; then
  echo "  STAC Catalog: $GCS_CATALOG_PATH"
fi
echo "============================================================"