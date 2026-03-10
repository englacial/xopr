"""
Find overlapping BedMap and OPR lines.

Uses custom logic for exact matching of OPR linestrings to BedMap points.
"""

import io
import os
import re

import geopandas as gpd
import obstore as obs
import pandas as pd


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
# Load BedMap2 and BedMap3 catalog, filter/query by institution
store = obs.store.from_url(url="https://data.source.coop/englacial/bedmap/")
gdf_bm2: gpd.GeoDataFrame = await aio_read_parquet(store=store, path="bedmap2.parquet")
gdf_bm3: gpd.GeoDataFrame = await aio_read_parquet(store=store, path="bedmap3.parquet")
gdf: gpd.GeoDataFrame = pd.concat(objs=[gdf_bm2, gdf_bm3], ignore_index=True)
gdf.institution.unique()

gdf_cresis = gdf.query(
    expr="institution.str.startswith('Center for Remote Sensing of Ice Sheets')"
)
len(gdf_cresis)
print(gdf_cresis[["name"]])
#                                  name
# 31        NASA_2002_ICEBRIDGE_AIR_BM2
# 32        NASA_2004_ICEBRIDGE_AIR_BM2
# 33        NASA_2009_ICEBRIDGE_AIR_BM2
# 34        NASA_2010_ICEBRIDGE_AIR_BM2
# 35        NASA_2011_ICEBRIDGE_AIR_BM2
# 36        NASA_2012_ICEBRIDGE_AIR_BM2
# 88   CRESIS_2009_AntarcticaTO_AIR_BM3
# 89       CRESIS_2009_Thwaites_AIR_BM3
# 90    CRESIS_2013_Siple-Coast_AIR_BM3
# 98        NASA_2013_ICEBRIDGE_AIR_BM3
# 99        NASA_2014_ICEBRIDGE_AIR_BM3
# 100       NASA_2016_ICEBRIDGE_AIR_BM3
# 101       NASA_2017_ICEBRIDGE_AIR_BM3
# 102       NASA_2018_ICEBRIDGE_AIR_BM3
# 103       NASA_2019_ICEBRIDGE_AIR_BM3


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
prefixes: dict[str, str] = {}
for batch in stream:
    prefix: str = batch[0]["path"]
    shortname: str = re.findall(
        pattern=r"collection=(.*)\/stac\.parquet", string=prefix
    )[0]
    if "2024_" not in shortname:  # skip 2024_Antarctica_GroundGHOST2
        prefixes[shortname] = prefix
for prefix in prefixes.values():
    print(prefix)
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

# %%
# Determine overlap data for years 2002, 2004, 2009, 2010, 2011, 2012
# | BedMap2 | OPR |
# |---------|-----|
# | NASA_2002_ICEBRIDGE_AIR_BM2 | 2002_Antarctica_P3chile |
# | NASA_2004_ICEBRIDGE_AIR_BM2 | 2004_Antarctica_P3chile |
# | NASA_2009_ICEBRIDGE_AIR_BM2 | 2009_Antarctica_DC8 |
# | NASA_2010_ICEBRIDGE_AIR_BM2 | 2010_Antarctica_DC8 |
# | NASA_2011_ICEBRIDGE_AIR_BM2 | 2011_Antarctica_DC8 |
# | NASA_2012_ICEBRIDGE_AIR_BM2 | 2012_Antarctica_DC8 |
#
# | BedMap3 | OPR |
# |---------|-----|
# | CRESIS_2009_AntarcticaTO_AIR_BM3 | 2009_Antarctica_TO_Gambit|
# | CRESIS_2009_Thwaites_AIR_BM3     | 2009_Antarctica_TO |
# | CRESIS_2013_Siple-Coast_AIR_BM3  | 2013_Antarctica_Basler |
# | NASA_2013_ICEBRIDGE_AIR_BM3 | 2013_Antarctica_P3 |
# | NASA_2014_ICEBRIDGE_AIR_BM3 | 2014_Antarctica_DC8 |
# | NASA_2016_ICEBRIDGE_AIR_BM3 | 2016_Antarctica_DC8 |
# | NASA_2017_ICEBRIDGE_AIR_BM3 | 2017_Antarctica_Basler |
# | NASA_2018_ICEBRIDGE_AIR_BM3 | 2018_Antarctica_DC8 |
# | NASA_2019_ICEBRIDGE_AIR_BM3 | 2019_Antarctica_GV |
for shortname, prefix in reversed(prefixes.items()):  # reverse chronological loop
    year: int = int(re.findall(pattern=r"collection=(.*)_Antarctica", string=prefix)[0])
    print(f"Processing OPR campaign: {shortname}")

    # OPR (sparse XY points)
    gdf_opr: gpd.GeoDataFrame = (
        await aio_read_parquet(store=store, path=prefix)
    ).to_crs(epsg=3031)
    gdf_opr = gdf_opr.sort_values(by="datetime").reset_index(drop=True)
    assert len(gdf_opr) >= 1

    # BedMAP2 (simplified/sparse points)
    gdf_bedmap = gdf_cresis.query(expr=f"name.str.contains('{year}')").to_crs(epsg=3031)
    assert len(gdf_bedmap) == 1  # always 1

    # BedMAP2 (dense XY points)
    path = gdf_bedmap.asset_href.iloc[0].replace("gs://opr_stac/bedmap/", "bedmap/")
    gdf_bedmap_dense = (await aio_read_parquet(store=store, path=path)).to_crs(
        epsg=3031
    )
    gdf_bedmap_dense = gdf_bedmap_dense.sort_values(by="timestamp").reset_index(
        drop=True
    )
    # gdf_bedmap_dense = gdf_bedmap_dense.set_index(keys="trajectory_id")
    # gdf_bedmap_dense.index = gdf_bedmap_dense.index.astype(dtype=pd.UInt64Dtype())
    gdf_bedmap_dense["opr_id"] = pd.Series(dtype=pd.StringDtype())  # Empty new column
    # .set_crs(crs="OGC:CRS84", allow_override=True)
    # gdf_bedmap_dense.to_file(filename := f"data/{os.path.basename(path)}.gpkg")
    # print(f"Saved dense BedMAP points (unlabelled) to {filename}")

    # Loop over sparse OPR line segments, find matching dense BedMAP points
    assert gdf_opr.crs == gdf_bedmap_dense.crs
    for segment in gdf_opr.itertuples():
        gdf_unlabelled = gdf_bedmap_dense[~gdf_bedmap_dense.opr_id.notna()]
        # Get cartesian distance from all unlabelled BedMAP points to sparse OPR line
        gdf_: pd.Series = gdf_unlabelled.distance(other=segment.geometry)

        TOLERANCE: float = 0.8
        dist_match: pd.Series = gdf_[gdf_ < TOLERANCE].drop_duplicates()

        if len(dist_match) == 0:
            print(f"⛔ Failed to match OPR segment {segment.id}, reason: no matches")
            continue
        elif len(dist_match) == 1:  # only one point, cannot be a segment
            print(f"⛔ Failed to match OPR segment {segment.id}, reason: only 1 point")
            continue
        else:  # >=2, potentially have match
            head = int(dist_match.head(n=1).index[0])
            tail = int(dist_match.tail(n=1).index[0])
            # gdf_.loc[head:tail].plot(ylabel="distance (m)")

            # Fast match against sparse OPR points (if everything less than 200m away..)
            if all(gdf_.loc[head:tail] < 200):
                print(
                    f"🙌 OPR segment {segment.id} "
                    f"matches BedMap points {head}:{tail} (dist-based check)"
                )
                gdf_bedmap_dense.loc[head:tail, "opr_id"] = segment.id  # label
            else:
                print(
                    f"  Trying to match segment {segment.id} "
                    f"with timestamps for range {head}:{tail}"
                )
                df_times: pd.Series = gdf_unlabelled.loc[head:tail].timestamp
                TIMESPAN = pd.Timedelta(value=2, unit="days")
                time_match: pd.Series = df_times[
                    (df_times - segment.datetime) < TIMESPAN
                ]

                if len(time_match) >= 2:
                    head_ = int(time_match.head(n=1).index[0])
                    tail_ = int(time_match.tail(n=1).index[0])
                    print(
                        f"🙌 OPR segment {segment.id} "
                        f"matches BedMap points {head_}:{tail_} (time-based check)"
                    )
                    gdf_bedmap_dense.loc[head:tail, "opr_id"] = segment.id  # label
                    continue
                else:
                    print(
                        f"⛔ Failed to match OPR segment {segment.id}, "
                        "reason: no spatiotemporal matches"
                    )

    basename = os.path.basename(path).replace(".parquet", "")
    gdf_bedmap_dense.to_file(filename := f"data/{basename}.gpkg")
    print(f"Saved dense BedMAP points labelled with OPR ids to {filename}")

    # break  # TODO work on more years
print("Done!")
