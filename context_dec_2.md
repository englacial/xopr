# Context Notes: Bedmap Integration Implementation
**Date: December 2, 2024**
**Branch: bedmap**
**Issue: #37**

## What We Built

Implemented complete bedmap data integration for xopr to enable:
1. Converting 151 bedmap CSV files (96M rows) to cloud-optimized GeoParquet
2. Building STAC catalogs for spatial-temporal discovery
3. Querying data efficiently with DuckDB partial reads
4. Comparing bedmap ice thickness with OPR layer picks

## Architecture Decisions & Rationale

### Two-Stage Query Process
**Design**: STAC catalog filtering → DuckDB partial reads
**Why**: Minimizes data transfer from cloud storage
1. STAC identifies which files *might* contain relevant data (using multiline geometry bounds)
2. DuckDB reads only the specific rows/columns needed via SQL WHERE clause
3. Optional: Apply precise point-in-polygon filtering after retrieval

**Key Insight**: We store multiline flight geometries in STAC metadata (not data) for efficient spatial indexing without bloating the parquet files.

### Data Schema Decisions

**source_file column**: Stores filename WITHOUT extension (e.g., "AWI_1994_DML1_AIR_BM2")
- User specifically requested no extension
- Makes it easier to reference across different file formats

**No geometry column in data**: Points stored as separate lat/lon columns
- More efficient for storage and queries
- Can create point geometries on-demand when needed
- Multiline geometries only in metadata for STAC

**Temporal handling complexity**:
1. Primary: Use date/time columns if valid (not -9999)
2. Fallback: Use metadata time_coverage_start/end
3. Special case: Same year (2020-2020) → distribute across full year
4. Implementation: Floor timestamps to microseconds to avoid PyArrow precision issues

## Problems Encountered & Solutions

### 1. Trajectory ID Type Mismatch
**Problem**: Some files have numeric trajectory_ids, PyArrow expected strings
**Solution**: Force convert to string: `df['trajectory_id'] = df['trajectory_id'].astype(str)`

### 2. Timestamp Precision Error
**Problem**: "Casting from timestamp[ns] to timestamp[us, tz=UTC] would lose data"
**Solution**: Floor timestamps to microsecond precision: `timestamps.dt.floor('us')`

### 3. Flight Line Extraction Test Failure
**Problem**: Test points were >10km apart, creating single-point segments
**Initial attempt**: Used 0.1° spacing (~11.7km at -70° latitude) - TOO FAR
**Solution**: Use 0.05° spacing (~5.9km) to stay under 10km threshold

### 4. Module Import Structure
**Problem**: Circular imports when all modules imported in __init__.py
**Solution**: Initially commented out unimplemented modules, then uncommented as created

### 5. Haversine Distance Order
**Problem**: Haversine expects (lat, lon) not (lon, lat)
**Solution**: Ensure correct column order when passing to haversine functions

## Things That DIDN'T Work

1. **Using -9999 as pandas NaN directly** - Had to explicitly replace with np.nan
2. **Assuming all trajectory_ids were strings** - Some files have integers
3. **Using nanosecond timestamps** - PyArrow requires microsecond or coarser
4. **Original test coordinates** - Didn't account for actual distances at high latitudes

## Current Implementation Status

### Completed ✅
- Full bedmap module (`src/xopr/bedmap/`)
  - converter.py: CSV→GeoParquet with date handling
  - geometry.py: Haversine distance & multiline extraction
  - catalog.py: STAC generation
  - query.py: DuckDB-powered queries
  - compare.py: OPR comparison functions
- CLI scripts (convert_bedmap.py, build_bedmap_catalog.py)
- Upload script (upload_bedmap_to_gcloud.sh)
- Unit tests (all passing)
- Demo notebook with examples
- Dependencies added: duckdb>=1.0.0, haversine>=2.8.0

### Test Results
- Successfully converted 3 test files (63,343 rows)
- Multiline geometries extracted and simplified
- Temporal extent handled via metadata fallback
- All 15 bedmap tests passing

## Known Limitations & Future Improvements

### Polar Projection Issue
**Current**: Uses WGS84 (EPSG:4326) bounding boxes
**Problem**: Longitudinal convergence near poles makes rectangles inefficient
**Future Fix**:
- Add Antarctic Polar Stereographic (EPSG:3031) support
- Use actual multiline geometries for intersection (not just bbox)
- Consider H3/S2 spatial indexing

**Documented in**: `notebooks/bedmap_polar_projection_note.md`

### Other Enhancements
1. **Parallel conversion**: Already supports but not tested at scale
2. **Compression tuning**: Using snappy, could benchmark others
3. **Row group sizing**: Currently 100K rows, could optimize based on query patterns
4. **Memory usage**: Stream processing for huge files not fully tested

## File Organization

```
xopr/
├── src/xopr/bedmap/          # Main module (all implemented)
├── scripts/
│   ├── convert_bedmap.py     # CSV conversion CLI
│   ├── build_bedmap_catalog.py  # STAC builder
│   └── upload_bedmap_to_gcloud.sh  # Cloud upload
├── tests/test_bedmap.py      # Unit tests
├── notebooks/
│   ├── bedmap_demo.ipynb     # Usage examples
│   └── bedmap_polar_projection_note.md  # Future improvements
└── scripts/output/bedmap/     # Converted parquet files (3 test files)
```

## Commands to Resume Work

```bash
# Switch to branch
git checkout bedmap

# Check status
git status  # All changes staged, ready to commit

# Run tests
python -m pytest tests/test_bedmap.py -v

# Test conversion
python scripts/convert_bedmap.py --test

# Full conversion (when ready)
python scripts/convert_bedmap.py --input ~/software/bedmap/Results --output scripts/output/bedmap

# Create PR
git commit -m "feat: Add bedmap data integration..."
git push origin bedmap
```

## Query Process (Key Understanding)

The two-stage query is essential for efficiency:

```python
# Stage 1: STAC finds files that MIGHT have data
files = stac_query(geometry=bbox)  # Uses multiline bounds

# Stage 2: DuckDB reads ONLY needed rows
sql = "SELECT * WHERE lon >= ? AND lon <= ? AND lat >= ? AND lat <= ?"
data = duckdb.read_parquet(files, sql)  # Partial read!
```

This avoids downloading entire files from S3/GCS!

## Next Session TODOs

1. **Test with full dataset** (151 files, 96M rows)
2. **Benchmark query performance** at scale
3. **Implement polar projection support** if bounding box queries prove inefficient
4. **Create PR** and get review feedback
5. **Deploy to production** after PR approval

## Key Design Philosophy

"Make it work, make it right, make it fast" - we're at "make it right" stage. The implementation works correctly but could be optimized for polar regions in a future PR.

## Contact/Context
Working with @espg on the englacial/xopr repository. This implements GitHub issue #37 for integrating bedmap ice thickness data with the xopr (Open Polar Radar) toolkit. The bedmap data provides historical Antarctic ice thickness measurements that can be compared with modern OPR layer picks.