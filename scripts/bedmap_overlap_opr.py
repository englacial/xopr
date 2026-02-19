"""
Find overlapping BedMap and OPR lines.

Uses custom logic for exact matching of OPR linestrings to BedMap points.
"""

import io
import os
import re
import tempfile
import urllib.request

import geopandas as gpd
import obstore as obs
import pandas as pd
import pyproj
import scipy.io
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
transformer = pyproj.Transformer.from_crs(
    crs_from="OGC:CRS84", crs_to="EPSG:3031", always_xy=True
)


def mat_to_linestring(url: str) -> shapely.geometry.LineString:
    """
    Read .mat file from CReSIS containing OPR data in Lon/Lat, and reproject to an
    Antarctic Polar Stereographic (EPSG:3031) linestring.

    Returns
    -------
    shapely.geometry.LineString

    """
    with tempfile.TemporaryDirectory() as tmpdir:
        mat_fpath = os.path.basename(p := url)  # e.g. Data_20101026_01_001.mat
        file_name = os.path.join(tmpdir, mat_fpath)
        urllib.request.urlretrieve(url=p, filename=file_name)  # download to tempfile
        dat = scipy.io.loadmat(file_name=file_name)

        return shapely.geometry.LineString(
            gpd.points_from_xy(
                *transformer.transform(
                    xx=dat["Longitude"].squeeze(),
                    yy=dat["Latitude"].squeeze(),
                ),
                crs="EPSG:3031",
            )
        )


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
for prefix in prefixes:
    collection_shortname: str = re.findall(
        pattern=r"collection=(.*)\/stac\.parquet", string=prefix
    )[0]
    year: int = int(re.findall(pattern=r"collection=(.*)_Antarctica", string=prefix)[0])
    if "_DC8" not in collection_shortname or "_P3" not in collection_shortname:
        continue
    print(f"Processing {collection_shortname}")

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

    # Loop over dense OPR line segments, find matching dense BedMAP points
    assert gdf_opr.crs == gdf_bedmap_dense.crs
    for segment in gdf_opr.itertuples():
        gdf_unlabelled = gdf_bedmap_dense[~gdf_bedmap_dense.opr_id.notna()]
        # Get cartesian distance from all unlabelled BedMAP points to sparse OPR line
        gdf_: pd.Series = gdf_unlabelled.distance(other=segment.geometry)

        tolerance: float = 0.4
        seg_match: pd.Series = gdf_[gdf_ < tolerance].drop_duplicates()

            if len(seg_match) == 0:
            print(f"Failed to match OPR segment {segment.id}, reason: no matches")
            continue
        elif len(seg_match) == 1:  # only one point, cannot be a segment
            print(f"Failed to match OPR segment {segment.id}, reason: only 1 point")
            continue
        else:  # >=2, potentially have match
                head = int(seg_match.head(n=1).index[0])
                tail = int(seg_match.tail(n=1).index[0])
                # gdf_.loc[head:tail].plot(ylabel="distance (m)")

            # Fast match against sparse OPR points (if everything less than 200m away..)
            if all(gdf_.loc[head:tail] < 200):
                print(
                    f"🙌 OPR segment {segment.id} matches BedMap points {head}:{tail} (fast)"
                )
            else:
                print(
                    f"Trying to match segment {segment.id} with dense points for range {head}:{tail}"
                )

                # OPR (dense XY points)
                opr_dense_geom = mat_to_linestring(url=segment.assets["data"]["href"])
                seg_match_dense: pd.Series = gdf_unlabelled.loc[head:tail].distance(
                    other=opr_dense_geom
                )
                # seg_match_dense.plot(ylabel="distance (m)")

                # Ensure all BedMAP points in series are <1m away from OPR segment
                if not all(seg_match_dense < 1.0):
                    # Try and narrow down segments once more with stricter tolerance
                    tolerance_ = 0.005
                    seg_match_new: pd.Series = seg_match_dense[
                        seg_match_dense < tolerance_
                    ].drop_duplicates()
                    seg_match_new.plot(ylabel="distance (m)")
                    head_ = int(seg_match_new.head(n=1).index[0])
                    tail_ = int(seg_match_new.tail(n=1).index[0])
                    seg_match_dense.loc[head_:tail_].plot()
                    if all(seg_match_dense.loc[head_:tail_] < 0.5):
                        print(
                            f"🙌 OPR segment {segment.id} matches BedMap points {head_}:{tail_} (slow)"
                        )
                        continue
                    else:
                        print(
                            f"Failed to match OPR segment {segment.id}, reason: too many distant points"
                        )
                else:
                    print(
                        f"🙌 OPR segment {segment.id} matches BedMap points {head}:{tail} (slow)"
                    )
                    gdf_bedmap_dense.loc[head:tail, "opr_id"] = segment.id  # label
                    continue

    basename = os.path.basename(path).replace(".parquet", "")
    gdf_bedmap_dense.to_file(filename := f"data/{basename}.gpkg")
    print(f"Saved dense BedMAP points labelled with OPR ids to {filename}")

    # break  # TODO work on more years
print("Done!")
