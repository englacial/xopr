"""
Find overlapping BedMap and OPR lines.

Uses custom logic for exact matching of OPR linestrings to BedMap points.

Run using:

    uv run scripts/bedmap_overlap_opr.py
"""

import asyncio
import io
import os
import re
import tempfile
import urllib.request

import geopandas as gpd
import h5py
import obstore as obs
import pandas as pd
import pyproj
import scipy
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

    Uses `scipy.io.loadmat` for MATLAB 5 files, or `h5py.File` for MATLAB 7.3+ files.

    Returns
    -------
    shapely.geometry.LineString

    """
    with tempfile.TemporaryDirectory() as tmpdir:
        mat_fpath = os.path.basename(p := url)  # e.g. Data_20101026_01_001.mat
        file_name = os.path.join(tmpdir, mat_fpath)
        urllib.request.urlretrieve(url=p, filename=file_name)  # download to tempfile

        try:
            dat = scipy.io.loadmat(file_name)  # MATLAB 5 files
        except NotImplementedError:
            dat = h5py.File(name=file_name)  # MATLAB 7.3+ files

        return shapely.geometry.LineString(
            gpd.points_from_xy(
                *transformer.transform(
                    xx=dat["Longitude"][:].squeeze(),
                    yy=dat["Latitude"][:].squeeze(),
                ),
                crs="EPSG:3031",
            )
        )


# %%
async def load_geodataframes() -> gpd.GeoDataFrame:
    """
    Load BedMap2 and BedMap3 GeoDataFrame rows with CReSIS institution.

    Returns
    -------
    geopandas.GeoDataFrame

    """
    # Load BedMap2 and BedMap3 catalog, filter/query by institution
    store = obs.store.from_url(url="https://data.source.coop/englacial/bedmap/")
    gdf_bm2: gpd.GeoDataFrame = await aio_read_parquet(
        store=store, path="bedmap2.parquet"
    )
    gdf_bm3: gpd.GeoDataFrame = await aio_read_parquet(
        store=store, path="bedmap3.parquet"
    )
    gdf: gpd.GeoDataFrame = pd.concat(objs=[gdf_bm2, gdf_bm3], ignore_index=True)
    gdf.institution.unique()

    gdf_cresis: gpd.GeoDataFrame = gdf.query(
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
    return gdf_cresis


# %%
def get_store_and_prefixes() -> tuple[obs.store.S3Store, dict[str, str]]:
    """
    Get s3 paths relative to the Source Coop root url, specifically for OPR campaigns
    under the CReSIS provider.

    Returns
    -------
    obstore.store.S3Store, dict[str, str]

    """
    # Load OPR catalog
    store = obs.store.from_url(
        url="s3://us-west-2.opendata.source.coop/englacial/",
        skip_signature=True,
        region="us-west-2",
    )
    stream = obs.list(
        store=store,
        prefix="xopr/catalog/hemisphere=south/provider=cresis",
        chunk_size=1,
    )
    prefixes: dict[str, str] = {}
    for batch in stream:
        prefix: str = batch[0]["path"]
        shortname: str = re.findall(
            pattern=r"collection=(.*)\/stac\.parquet", string=prefix
        )[0]
        if (
            "2024_" not in shortname and "_Gambit" not in shortname
        ):  # skip 2024_Antarctica_GroundGHOST2 and 2009_Antarctica_TO_Gambit
            prefixes[shortname] = prefix
    for prefix in prefixes.values():
        print(prefix)
    # provider=cresis/collection=2002_Antarctica_P3chile/stac.parquet
    # provider=cresis/collection=2004_Antarctica_P3chile/stac.parquet
    # provider=cresis/collection=2009_Antarctica_DC8/stac.parquet
    # provider=cresis/collection=2009_Antarctica_TO/stac.parquet
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
    return store, prefixes


# %%
async def main(  # noqa: PLR0912, PLR0914, PLR0915
    gdf_cresis: gpd.GeoDataFrame, store: obs.store.S3Store, prefixes: dict[str, str]
):
    """
    Determine overlap data for years 2002, 2004, 2009, 2010, 2011, 2012.

    | BedMap2 | OPR |
    |---------|-----|
    | NASA_2002_ICEBRIDGE_AIR_BM2 | 2002_Antarctica_P3chile |
    | NASA_2004_ICEBRIDGE_AIR_BM2 | 2004_Antarctica_P3chile |
    | NASA_2009_ICEBRIDGE_AIR_BM2 | 2009_Antarctica_DC8 |
    | NASA_2010_ICEBRIDGE_AIR_BM2 | 2010_Antarctica_DC8 |
    | NASA_2011_ICEBRIDGE_AIR_BM2 | 2011_Antarctica_DC8 |
    | NASA_2012_ICEBRIDGE_AIR_BM2 | 2012_Antarctica_DC8 |

    | BedMap3 | OPR |
    |---------|-----|
    | CRESIS_2009_AntarcticaTO_AIR_BM3 & CRESIS_2009_Thwaites_AIR_BM3 | 2009_Antarctica_TO |
    | CRESIS_2013_Siple-Coast_AIR_BM3  | 2013_Antarctica_Basler |
    | NASA_2013_ICEBRIDGE_AIR_BM3 | 2013_Antarctica_P3 |
    | NASA_2014_ICEBRIDGE_AIR_BM3 | 2014_Antarctica_DC8 |
    | NASA_2016_ICEBRIDGE_AIR_BM3 | 2016_Antarctica_DC8 |
    | NASA_2017_ICEBRIDGE_AIR_BM3 | 2017_Antarctica_Basler |
    | NASA_2018_ICEBRIDGE_AIR_BM3 | 2018_Antarctica_DC8 |
    | NASA_2019_ICEBRIDGE_AIR_BM3 | 2019_Antarctica_GV |

    """
    # reverse chronological loop from 2019 to 2002
    for shortname, prefix in reversed(prefixes.items()):  # noqa: PLR1702
        year: int = int(
            re.findall(pattern=r"collection=(.*)_Antarctica", string=prefix)[0]
        )
        print(f"Processing OPR campaign: {shortname}")

        # OPR (sparse XY points)
        gdf_opr: gpd.GeoDataFrame = (
            await aio_read_parquet(store=store, path=prefix)
        ).to_crs(epsg=3031)
        gdf_opr = gdf_opr.sort_values(by="datetime").reset_index(drop=True)
        assert len(gdf_opr) >= 1

        # BedMAP2 (simplified/sparse points)
        gdf_bedmap = gdf_cresis.query(expr=f"name.str.contains('{year}')").to_crs(
            epsg=3031
        )
        if len(gdf_bedmap) > 1:  # handle multiple campaigns in one year
            match shortname:
                case "2013_Antarctica_P3":
                    campaign = "NASA_2013_ICEBRIDGE_AIR_BM3"
                case "2013_Antarctica_Basler":
                    campaign = "CRESIS_2013_Siple-Coast_AIR_BM3"
                case "2009_Antarctica_TO":
                    campaign = (
                        "CRESIS_2009_"  # AntarcticaTO_AIR_BM3 and Thwaites_AIR_BM3
                    )
                case "2009_Antarctica_DC8":
                    campaign = "NASA_2009_ICEBRIDGE_AIR_BM2"
                case _:
                    raise ValueError(
                        f"Update code to set maching campaign for {shortname}"
                    )
            gdf_bedmap = gdf_bedmap.query(expr=f"name.str.startswith('{campaign}')")


        # BedMAP2 (dense XY points)
        paths: pd.Series = gdf_bedmap.asset_href.str.replace(
            "s3://us-west-2.opendata.source.coop/englacial/bedmap/", "bedmap/"
        )
        gdf_bedmap_dense = pd.concat(
            objs=[await aio_read_parquet(store=store, path=path) for path in paths]
        ).to_crs(epsg=3031)
        gdf_bedmap_dense = gdf_bedmap_dense.sort_values(by="timestamp").reset_index(
            drop=True
        )
        # gdf_bedmap_dense = gdf_bedmap_dense.set_index(keys="trajectory_id")
        # gdf_bedmap_dense.index = gdf_bedmap_dense.index.astype(dtype=pd.UInt64Dtype())
        gdf_bedmap_dense["opr_id"] = pd.Series(
            dtype=pd.StringDtype()
        )  # Empty new column
        # .set_crs(crs="OGC:CRS84", allow_override=True)
        # gdf_bedmap_dense.to_file(filename := f"data/{os.path.basename(path)}.gpkg")
        # print(f"Saved dense BedMAP points (unlabelled) to {filename}")

        # Loop over sparse OPR line segments, find matching dense BedMAP points
        assert gdf_opr.crs == gdf_bedmap_dense.crs
        for segment in gdf_opr.itertuples():
            gdf_unlabelled = gdf_bedmap_dense[~gdf_bedmap_dense.opr_id.notna()]
            # Get cartesian distance from all unlabelled BedMAP points to sparse OPR line
            df_dist: pd.Series = gdf_unlabelled.distance(other=segment.geometry)

            TOLERANCE: float = 0.8
            dist_match: pd.Series = df_dist[df_dist < TOLERANCE].drop_duplicates()

            if len(dist_match) == 0:
                print(
                    f"⛔ Failed to match OPR segment {segment.id}, reason: no matches"
                )
                continue
            elif len(dist_match) == 1:  # only one point, cannot be a segment
                print(
                    f"⛔ Failed to match OPR segment {segment.id}, reason: only 1 point"
                )
                continue
            else:  # >=2, potentially have match
                head = int(dist_match.head(n=1).index[0])
                tail = int(dist_match.tail(n=1).index[0])

                # Verify distance-based match, shift head and tail points if needed
                # Fast match against sparse OPR points (if everything less than 200m away..)
                attempt: int = 0
                original_head = head
                original_tail = tail
                while attempt <= 2:
                    if all(df_dist.loc[head:tail] < 200):
                        print(
                            f"🙌 OPR segment {segment.id} "
                            f"matches BedMap points {head}:{tail} (dist-based check)"
                        )
                        gdf_bedmap_dense.loc[head:tail, "opr_id"] = segment.id  # label
                        break
                    else:
                        print(
                            f"  Trying to match segment {segment.id} "
                            f"with new point bounds for range {head}:{tail}"
                        )
                        # find inflexion point based on sudden distance change
                        df_dist_delta = df_dist.loc[
                            original_head:original_tail
                        ].pct_change(periods=1)
                        if len(df_dist_delta[df_dist_delta > 200]) == 0:
                            attempt = 3
                            continue

                        if attempt == 0:
                            # Try shifting tail first
                            tail = int(df_dist_delta[df_dist_delta > 200].index[0])
                        elif attempt == 1:
                            # Try shifting head next
                            tail = original_tail
                            head = int(df_dist_delta[df_dist_delta > 200].index[0])
                        elif attempt == 2:
                            head = original_head
                            tail = original_tail

                    attempt += 1

                # Slow fallback using dense OPR xy points
                if attempt == 3:
                    # Apply temporal filter to reduce points to look at
                    df_times: pd.Series = gdf_unlabelled.loc[head:tail].timestamp
                    TIMESPAN = pd.Timedelta(value=2, unit="days")
                    time_match: pd.Series = df_times[
                        (df_times - segment.datetime) < TIMESPAN
                    ]
                    # (df_times - segment.datetime).plot(ylabel="timespan")
                    if len(time_match) >= 2:
                        head = int(time_match.head(n=1).index[0])
                        tail = int(time_match.tail(n=1).index[0])  # tail_ = int(...)
                        # df_time_delta = df_times.loc[head:tail_].diff(periods=1)
                        # tail = int(
                        #     df_time_delta[
                        #         df_time_delta > pd.Timedelta(value=1, unit="hour")
                        #     ].index[0]
                        # )
                    else:
                        print(
                            f"⛔ Failed to match OPR segment {segment.id}, reason: no close time matches"
                        )
                        continue
                    # Continue with checking using dense XY points after temporal filter
                    print(
                        f"  Trying to match segment {segment.id} "
                        f"with dense OPR points for range {head}:{tail}"
                    )
                    # OPR (dense XY points)
                    opr_dense_geom: shapely.geometry.LineString = mat_to_linestring(
                        url=segment.assets["data"]["href"]
                    )
                    df_dist_dense: pd.Series = gdf_unlabelled.loc[head:tail].distance(
                        other=opr_dense_geom
                    )
                    # df_dist_dense.plot(ylabel="distance (m)")

                    # Ensure all BedMAP points in series are <1m away from OPR segment
                    if not all(df_dist_dense < 1.0):
                        # Try and narrow down segments once more with stricter tolerance
                        tolerance_ = 0.001
                        dist_match_new: pd.Series = df_dist_dense[
                            df_dist_dense < tolerance_
                        ].drop_duplicates()
                        # TODO find a way to drop points if there is no two consecutive? or two within 100 ids?
                        # dist_match_new.plot(ylabel="distance (m)")
                        head_ = int(dist_match_new.head(n=1).index[0])
                        tail_ = int(dist_match_new.tail(n=1).index[0])
                        # df_dist_dense.loc[head_:tail_].plot()
                        if all(df_dist_dense.loc[head_:tail_] < 0.5):
                            print(
                                f"🙌 OPR segment {segment.id} matches BedMap points {head_}:{tail_} (slow)"
                            )
                            continue
                        else:
                            print(
                                f"⛔ Failed to match OPR segment {segment.id}, reason: too many distant points"
                            )
                            # raise ValueError("temp")
                    else:
                        print(
                            f"🙌 OPR segment {segment.id} matches BedMap points {head}:{tail} (slow)"
                        )
                        gdf_bedmap_dense.loc[head:tail, "opr_id"] = segment.id  # label
                        continue

        basename = os.path.basename(paths.iloc[0]).replace(".parquet", "")
        gdf_bedmap_dense.to_file(filename := f"data/{basename}.gpkg")
        print(f"Saved dense BedMAP points labelled with OPR ids to {filename}")
        print()

        # break  # TODO work on more years


# %%
if __name__ == "__main__":
    print("------------------ Loading simplified Bedmap geometries ------------------")
    gdf_cresis: gpd.GeoDataFrame = asyncio.run(main=load_geodataframes())

    print("---------------------- Getting OPR CReSIS campaigns ----------------------")
    store, prefixes = get_store_and_prefixes()

    print("---------------------------- Starting main loop ---------------------------")
    asyncio.run(main=main(gdf_cresis=gdf_cresis, store=store, prefixes=prefixes))

    print("Done!")
