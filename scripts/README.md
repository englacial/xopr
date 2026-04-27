# STAC Catalog Generation Workflow

## Overview

The STAC catalog generation uses a two-step YAML-based workflow:

1. **Build**: Generate parquet collections (parallel processing per campaign)
2. **Upload**: Push parquet files to source.coop with hive partitioning

## Quick Start

```bash
# 1. Generate parquet collections
python scripts/build_catalog.py --config seasons/2008_Antarctica_BaslerJKB.yml

# 2. Upload to source.coop
python scripts/upload_stac_catalogs.py catalog/2008_Antarctica_BaslerJKB/ \
  --credentials ~/.source_coop_token.json --execute
```

## Configuration

All settings are managed through YAML templates in `seasons/`. See
`seasons/README.md` for the full template schema and examples.

### Key Configuration Options

```yaml
data:
  root: "/data/opr"                    # Data location
  primary_product: "CSARP_standard"    # Main product
  campaign_filter: "2016_Antarctica_.*" # Regex filter (optional)
  campaigns:
    include: ["2016_Antarctica_DC8"]   # Explicit list (optional)
    exclude: ["test_campaign"]         # Exclude list (optional)

output:
  path: "./stac_catalog"                # Output directory
  catalog_id: "OPR"                    # Catalog ID
  catalog_description: "Open Polar Radar airborne data"
  license: "various"                   # Data license (e.g., "CC-BY-4.0")

processing:
  n_workers: 4                          # Parallel workers per campaign
  max_items: null                       # Limit items (null = all)
```

## Campaign Filtering

Three ways to filter campaigns:

1. **Regex Pattern**:
   ```bash
   python scripts/build_catalog.py --config seasons/2008_Antarctica_BaslerJKB.yml \
     data.campaign_filter="2016_Antarctica_.*"
   ```

2. **Include List** (in YAML):
   ```yaml
   data:
     campaigns:
       include: ["2016_Antarctica_DC8", "2017_Antarctica_P3"]
   ```

3. **Exclude List** (in YAML):
   ```yaml
   data:
     campaigns:
       exclude: ["test_campaign", "calibration_flight"]
   ```

## Environments

Use different settings for test/production:

```bash
# Test environment (limited data)
python scripts/build_catalog.py --config seasons/2008_Antarctica_BaslerJKB.yml --env test

# Production environment (full processing)
python scripts/build_catalog.py --config seasons/2008_Antarctica_BaslerJKB.yml --env production
```

## Command Line Overrides

Override any configuration option from command line:

```bash
python scripts/build_catalog.py --config seasons/2008_Antarctica_BaslerJKB.yml \
  processing.n_workers=16 \
  processing.max_items=10 \
  output.path=./my_output
```

## Workflow Examples

### Process Single Campaign

```bash
python scripts/build_catalog.py --config seasons/2008_Antarctica_BaslerJKB.yml \
  data.campaign_filter="2016_Antarctica_DC8"
```

### Process All 2016 Data

```bash
python scripts/build_catalog.py --config seasons/2008_Antarctica_BaslerJKB.yml \
  data.campaign_filter="2016_.*"
```

### Test Run

```bash
# Quick test with limited data
python scripts/build_catalog.py --config seasons/2008_Antarctica_BaslerJKB.yml --env test
```

### Incremental Updates

```bash
# Process new campaign
python scripts/build_catalog.py --config seasons/2024_Antarctica_NewPlatform.yml

# Upload
python scripts/upload_stac_catalogs.py catalog/2024_Antarctica_NewPlatform/ \
  --credentials ~/.source_coop_token.json --execute
```

## Output Structure

```
stac_catalog/
├── config_used.yaml                          # Configuration used for reproducibility
├── 2008_Antarctica_BaslerJKB.parquet         # Campaign collection (geoparquet)
└── 2016_Antarctica_DC8.parquet               # Campaign collection (geoparquet)
```

After upload, files are placed into the hive-partitioned layout on S3:

```
s3://us-west-2.opendata.source.coop/englacial/xopr/catalog/
  hemisphere=south/
    provider=utig/
      collection=2008_Antarctica_BaslerJKB/
        stac.parquet
```

## Benefits

- **Simpler**: Single YAML file controls everything
- **Reproducible**: Config saved with output
- **Flexible**: Easy filtering and environment switching
- **Efficient**: Parallel processing within campaigns
- **Incremental**: Add new campaigns without reprocessing
