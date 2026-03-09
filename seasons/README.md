# seasons/

YAML templates that define how to build STAC catalogs for each campaign season.
Each template captures the full provenance needed to reproducibly build and
upload the hive-partitioned parquet catalogs hosted on
[source.coop/englacial/xopr](https://source.coop/repositories/englacial/xopr/).

## Naming convention

```
{year}_{region}_{platform}.yml
```

Examples: `2008_Antarctica_BaslerJKB.yml`, `2016_Antarctica_DC8.yml`

The name matches the CReSIS campaign directory name and becomes the STAC
collection ID.

## Template schema

| Field | Type | Required | CLI-overridable | Description |
|-------|------|----------|-----------------|-------------|
| `version` | string | yes | no | Schema version (semver, e.g. `1.0.0`) |
| `data.root` | string | yes | yes | Root directory containing OPR data |
| `data.provider` | string | no | yes | Data provider identifier (`awi`, `cresis`, `dtu`, `utig`) |
| `data.primary_product` | string | yes | yes | Primary CSARP product to process |
| `data.extra_products` | list[string] | no | yes | Additional CSARP products to include |
| `data.campaigns.include` | list[string] | no | yes | Campaign names to include |
| `data.campaigns.exclude` | list[string] | no | yes | Campaign names to exclude |
| `data.campaign_filter` | string | no | yes | Regex filter on campaign names |
| `output.path` | string | yes | yes | Output directory for parquet files |
| `output.catalog_id` | string | yes | yes | STAC catalog identifier |
| `output.catalog_description` | string | no | yes | Human-readable catalog description |
| `output.license` | string | no | yes | Data license (e.g. `various`, `CC-BY-4.0`) |
| `processing.n_workers` | int | no | yes | Dask parallel workers per campaign |
| `processing.memory_limit` | string | no | yes | Per-worker memory limit (e.g. `8GB`) |
| `processing.max_items` | int/null | no | yes | Cap on items to process (`null` = all) |
| `assets.base_url` | string | yes | yes | Base URL for asset hrefs |
| `logging.verbose` | bool | no | yes | Enable verbose output |
| `logging.level` | string | no | yes | Log level (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |
| `geometry.simplify` | bool | no | yes | Simplify flight geometries |
| `geometry.tolerance` | float | no | yes | Simplification tolerance in metres |
| `sci.citation` | string | no | no | Scientific citation text |
| `sci.doi` | string | no | no | DOI for the dataset |
| `sci.override` | bool | no | no | Override per-item citations with this one |

See `config/catalog_config_schema.py` for the full Cerberus validation schema.

## Building a catalog

```bash
python scripts/build_catalog.py --config seasons/2008_Antarctica_BaslerJKB.yml
```

Override any field from the command line:

```bash
python scripts/build_catalog.py --config seasons/2008_Antarctica_BaslerJKB.yml \
  processing.n_workers=8 \
  processing.max_items=10
```

Output lands in the directory set by `output.path` (default: `../catalog/<campaign>/`).

## Uploading to source.coop

### Credential setup

1. Log into [source.coop](https://source.coop) and navigate to the repository dashboard.
2. Generate an S3 token — this downloads a JSON file containing
   `aws_access_key_id`, `aws_secret_access_key`, `aws_session_token`, and
   `region_name`.
3. Save the file somewhere safe (e.g. `~/.source_coop_token.json`).

### Upload

Dry run (default — prints what would be uploaded):

```bash
python scripts/upload_stac_catalogs.py catalog/2008_Antarctica_BaslerJKB/ \
  --credentials ~/.source_coop_token.json
```

Execute the upload:

```bash
python scripts/upload_stac_catalogs.py catalog/2008_Antarctica_BaslerJKB/ \
  --credentials ~/.source_coop_token.json --execute
```

Files are placed into the hive-partitioned layout on S3:

```
s3://us-west-2.opendata.source.coop/englacial/xopr/catalog/
  hemisphere=south/
    provider=utig/
      collection=2008_Antarctica_BaslerJKB/
        stac.parquet
```

You can also set `SOURCE_COOP_CREDENTIALS` instead of passing `--credentials`
each time.

## Adding a new season

1. Copy an existing template:
   ```bash
   cp seasons/2008_Antarctica_BaslerJKB.yml seasons/2024_Antarctica_NewPlatform.yml
   ```
2. Edit the new file — update `data.root`, `data.campaigns.include`,
   `output.path`, `assets.base_url`, `sci.*`, etc.
3. Build and verify:
   ```bash
   python scripts/build_catalog.py --config seasons/2024_Antarctica_NewPlatform.yml
   ```
4. Commit the template:
   ```bash
   git add seasons/2024_Antarctica_NewPlatform.yml
   ```
