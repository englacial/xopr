# Comparing OPR bed picks against external datasets

A common xOPR workflow is "I have radar bed picks for a region — how do they compare to dataset X?". The right way to do that comparison depends on the *type* of geometry that dataset X represents. This page walks through the three matching strategies xOPR supports, the API for each, and the case-study notebook that demonstrates it.

## The shared starting point: load OPR bed picks

Every comparison workflow begins the same way: query a region for STAC items, then use [`OPRConnection.load_bed_picks`](api/xopr.opr_access.html#xopr.opr_access.OPRConnection.load_bed_picks) to load layer picks for every frame into one flat GeoDataFrame:

```python
import xopr

opr = xopr.OPRConnection(cache_dir="radar_cache")
region = xopr.geometry.get_antarctic_regions(name="Vincennes_Bay")
gdf = opr.query_frames(geometry=region)

picks = opr.load_bed_picks(gdf, target_crs="EPSG:3031")
# picks columns: geometry, wgs84, twtt, slow_time, id, collection,
#                opr:date, opr:segment, opr:frame, segment_path
```

The output schema is a strict superset of the canonical `layer_gdf` schema used by the matchers below. Once you have `picks`, the right strategy depends on what you're comparing against.

## Strategy by reference-dataset shape

| Reference dataset shape | Strategy | API | Case study |
|---|---|---|---|
| OPR flight lines (the same dataset, comparing to itself) | Spatial join on simplified line geometries → nearest-pick lookup at each crossing | [`xopr.find_intersections`](api/xopr.opr_tools.html#xopr.opr_tools.find_intersections) + [`xopr.opr_tools.compute_crossover_error`](api/xopr.opr_tools.html#xopr.opr_tools.compute_crossover_error) | [`crossovers.ipynb`](notebooks/crossovers.ipynb) |
| Continuous gridded raster (e.g. BedMachine, Bedmap-DEM) | Bilinear interpolation at each pick coordinate | `xarray.DataArray.interp` (no xopr-specific API needed) | [`bedmachine_comparison.ipynb`](notebooks/bedmachine_comparison.ipynb) |
| External point cloud (e.g. Bedmap pick database) | Morton-prefix containment + along-track disambiguation | [`xopr.bedmap.match_bedmap_to_frames`](api/xopr.bedmap.html#xopr.bedmap.match_bedmap_to_frames) + [`disambiguate_matches`](api/xopr.bedmap.html#xopr.bedmap.disambiguate_matches) | _(walkthrough notebook in development; see API docs in the meantime)_ |
| External polygon dataset (e.g. ICESat-2 ATL06 granules from CMR) | Morton-prefix containment OR shapely STRtree exact intersection | [`xopr.matching`](api/xopr.matching.html) | [`xopr_atl06_crossovers.ipynb`](notebooks/xopr_atl06_crossovers.ipynb) |

## Why morton indexing for points and polygons

For point and polygon comparisons we use the **`opr:mbox`** field that every xOPR STAC item carries: a list of 4 variable-resolution morton cells (default order 18) covering the frame's geometry, plus `opr:mpolygon` (12 cells) on every collection. These are computed at catalog-generation time by [`xopr.stac.morton.compute_mbox`](api/xopr.stac.morton.html#xopr.stac.morton.compute_mbox) and [`compute_mpolygon_from_items`](api/xopr.stac.morton.html#xopr.stac.morton.compute_mpolygon_from_items).

The key invariant: **morton indices are hierarchical strings, so containment is prefix matching**. The encoding is base-4 — every digit narrows the spatial cell to one of four quadrants — so a digit can only be `1`, `2`, `3`, or `4`. A query point at order 18 has a 19-digit morton string. If a frame's mbox cell is the 5-digit prefix `12343`, every point whose order-18 morton starts with `12343` falls inside that cell. No polygon-polygon intersection — just `point_morton.startswith(cell_morton)`. This is the cheap inverted-index lookup that makes the bedmap and ATL06 backends fast even on millions of points.

To turn morton cells back into shapely Polygons (for visualization or exact intersection), use [`mbox_to_polygons`](api/xopr.stac.morton.html#xopr.stac.morton.mbox_to_polygons):

```python
from xopr.stac.morton import mbox_to_polygons
import geopandas as gpd

polys = mbox_to_polygons(item['opr:mbox'])
gpd.GeoSeries(polys, crs='EPSG:4326').to_crs('EPSG:3031').plot()
```

## When morton is the wrong tool

Morton-prefix matching is only useful when both sides of the comparison share the same coordinate system and one side has been pre-indexed. For OPR↔raster comparisons (BedMachine), there's no point indexing the raster — the natural interface is `xarray.DataArray.interp` at the OPR pick coordinates, which gives you exact bilinear sampling in one call. Skip morton; just project everything to a common CRS and interpolate.

For OPR↔OPR self-intersection ([`crossovers.ipynb`](notebooks/crossovers.ipynb)), morton is also overkill: GeoPandas's spatial join handles the line-line intersection directly, and the per-pair work after that (compute the elevation difference at each crossing) is microseconds. Use [`find_intersections`](api/xopr.opr_tools.html#xopr.opr_tools.find_intersections) and skip the morton primitives.

## Reproducibility notes

- The bedmap and ATL06 case studies fetch external data over the network on first run; expect each notebook to take a few minutes end-to-end.
- [`bedmachine_comparison.ipynb`](notebooks/bedmachine_comparison.ipynb) requires NASA Earthdata credentials (via `earthaccess`) and is **not** re-executed by CI — see the warning at the top of that notebook for how to regenerate outputs locally.
- The morton primitives require a STAC catalog produced after [PR #77](https://github.com/englacial/xopr/pull/77). [`OPRConnection`](api/xopr.opr_access.html#xopr.opr_access.OPRConnection) syncs the catalog cache to your local machine on construction (cheap ETag check; no-op if nothing changed), so as new seasons are reprocessed under [issue #78](https://github.com/englacial/xopr/issues/78) you'll pick them up automatically. If you hit a missing-column error and suspect a stale cache, force a refresh manually:

```python
from xopr.stac_cache import sync_opr_catalogs
sync_opr_catalogs()
```
