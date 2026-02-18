"""
Find overlapping Bedmap and OPR lines.

Uses hausdorff distance metric for fuzzy matching of linestrings.
"""

import io
import os

import geopandas as gpd
import obstore as obs
import pandas as pd
import shapely.geometry


# %%
async def aio_read_parquet(store: obs.store.ObjectStore, path: str) -> gpd.GeoDataFrame:
    """
    Read parquet file from remote object storage using obstore and geopandas.

    Returns
    -------
    geopandas.GeoDataFrame

    """
    result = await store.get_async(path=path)
    bytes_ = await result.bytes_async()
    gdf: gpd.GeoDataFrame = gpd.read_parquet(path=io.BytesIO(bytes_))
    return gdf


# %%
# Load BedMAP2 catalog, filter/query by institution
store = obs.store.from_url(url="https://data.source.coop/englacial/bedmap/")
gdf: gpd.GeoDataFrame = await aio_read_parquet(store=store, path="bedmap2.parquet")
gdf.institution.unique()

gdf_cresis = gdf.query(
    expr="institution.str.startswith('Center for Remote Sensing of Ice Sheets')"
)
len(gdf_cresis)
print(gdf_cresis[["name"]])
#                            name
# 31  NASA_2002_ICEBRIDGE_AIR_BM2  # CRESIS+
# 32  NASA_2004_ICEBRIDGE_AIR_BM2
# 33  NASA_2009_ICEBRIDGE_AIR_BM2
# 34  NASA_2010_ICEBRIDGE_AIR_BM2
# 35  NASA_2011_ICEBRIDGE_AIR_BM2
# 36  NASA_2012_ICEBRIDGE_AIR_BM2  # CRESIS+

# %%
# Load OPR catalog
store = obs.store.from_url(
    url="s3://us-west-2.opendata.source.coop/englacial/",
    skip_signature=True,
    region="us-west-2",
)
stream = obs.list(
    store=store, prefix="xopr/catalog/hemisphere=south/provider=cresis", chunk_size=1
)
prefixes: list[str] = []
for batch in stream:
    collection: str = batch[0]["path"]
    print(collection)
    prefixes.append(collection)

# provider=cresis/collection=2002_Antarctica_P3chile/stac.parquet
# provider=cresis/collection=2004_Antarctica_P3chile/stac.parquet
# provider=cresis/collection=2009_Antarctica_DC8/stac.parquet
# provider=cresis/collection=2009_Antarctica_TO/stac.parquet
# provider=cresis/collection=2009_Antarctica_TO_Gambit/stac.parquet
# provider=cresis/collection=2010_Antarctica_DC8/stac.parquet
# provider=cresis/collection=2011_Antarctica_DC8/stac.parquet
# provider=cresis/collection=2011_Antarctica_TO/stac.parquet
# provider=cresis/collection=2012_Antarctica_DC8/stac.parquet
# provider=cresis/collection=2013_Antarctica_Basler/stac.parquet
# provider=cresis/collection=2013_Antarctica_P3/stac.parquet
# provider=cresis/collection=2014_Antarctica_DC8/stac.parquet
# provider=cresis/collection=2016_Antarctica_DC8/stac.parquet
# provider=cresis/collection=2017_Antarctica_Basler/stac.parquet
# provider=cresis/collection=2017_Antarctica_P3/stac.parquet
# provider=cresis/collection=2018_Antarctica_DC8/stac.parquet
# provider=cresis/collection=2018_Antarctica_Ground/stac.parquet
# provider=cresis/collection=2019_Antarctica_GV/stac.parquet
# provider=cresis/collection=2024_Antarctica_GroundGHOST2/stac.parquet

# %%
# Determine overlap data for years 2002, 2004, 2009, 2010, 2011, 2012
for year in [2002, 2004, 2009, 2010, 2011, 2012]:
    # BedMAP2 (simplified/sparse points)
    gdf_bedmap = gdf_cresis.query(expr=f"name.str.contains('{year}')").to_crs(epsg=3031)
    assert len(gdf_bedmap) == 1  # always 1

    # BedMAP2 (dense XY points)
    path = gdf_bedmap.asset_href.iloc[0].replace("gs://opr_stac/bedmap/", "bedmap/")
    gdf_bedmap_dense = (await aio_read_parquet(store=store, path=path)).to_crs(
        epsg=3031
    )
    gdf_bedmap_dense = gdf_bedmap_dense.set_index(keys="trajectory_id")
    gdf_bedmap_dense.index = gdf_bedmap_dense.index.astype(dtype=pd.UInt32Dtype())
    gdf_bedmap_dense["opr_id"] = pd.Series(dtype=pd.StringDtype())  # Set new column
    # .set_crs(crs="OGC:CRS84", allow_override=True)
    gdf_bedmap_dense.to_file(filename := f"data/{os.path.basename(path)}.gpkg")
    print(f"Saved dense BedMAP points to {filename}")

    # OPR (dense XY points)
    prefix: list = [p for p in prefixes if str(year) in p]
    assert len(prefix) == 1  # TODO could be more than 1
    gdf_opr: gpd.GeoDataFrame = (
        await aio_read_parquet(store=store, path=prefix[0])
    ).to_crs(epsg=3031)
    gdf_opr = gdf_opr.sort_values(by="datetime")
    assert len(gdf_opr) >= 1

    # Loop over dense OPR line segments, find matching dense BedMAP points
    assert gdf_opr.crs == gdf_bedmap_dense.crs
    for segment in gdf_opr.itertuples():
        gdf_unlabelled = gdf_bedmap_dense[~gdf_bedmap_dense.opr_id.notna()]
        # Get cartesian distance from all unlabelled BedMAP points to OPR line
        gdf_ = gdf_unlabelled.distance(other=segment.geometry)

        for tolerance in (0.8, 0.4, 0.2):  # match pass on <X metre to line
            seg_match = gdf_[gdf_ < tolerance].drop_duplicates()
            if len(seg_match) == 0:
                print(f"Failed to match OPR segment {segment.id}")
                break
            else:
                head = int(seg_match.head(n=1).index[0])
                tail = int(seg_match.tail(n=1).index[0])
                gdf_.loc[head:tail].plot(ylabel="distance (m)")

                # Ensure all BedMAP points in series are <200m away from OPR segment
                if not all(gdf_.loc[head:tail] < 200):
                    print(
                        f"Increasing tolerance for {segment.id}, narrowing between {head}:{tail}"
                    )
                    continue
                else:
                    print(
                        f"🙌 OPR segment {segment.id} matches BedMap points {head}:{tail}"
                    )
                    gdf_bedmap_dense.loc[head:tail, "opr_id"] = segment.id  # label
                    break

    basename = os.path.basename(path).replace(".parquet", "")
    gdf_bedmap_dense.to_file(filename := f"data/{basename}.gpkg")
    print(f"Saved dense BedMAP points labelled with OPR ids to {filename}")

    break  # TODO work on more years
