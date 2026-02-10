"""
Find overlapping Bedmap and OPR lines.

Uses hausdorff distance metric for fuzzy matching of linestrings.
"""

import io

import geopandas as gpd
import obstore as obs
import pandas as pd
import shapely.geometry

# %%
# Load BedMAP2 data
store = obs.store.from_url(url="https://data.source.coop/englacial/bedmap/")
result = await store.get_async(path="bedmap2.parquet")
bytes = await result.bytes_async()

gdf: gpd.GeoDataFrame = gpd.read_parquet(path=io.BytesIO(bytes))
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
# Load OPR data
store = obs.store.from_url(
    url="s3://us-west-2.opendata.source.coop/englacial/xopr/catalog/hemisphere=south",
    skip_signature=True,
    region="us-west-2",
)
stream = obs.list(store=store, prefix="provider=cresis", chunk_size=1)
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
    # OPR
    gdf_cresis_ = gdf_cresis.query(expr=f"name.str.contains('{year}')").to_crs(
        epsg=3031
    )
    assert len(gdf_cresis_) == 1
    gdf_cresis_.to_file(filename=f"data/cresis_{year}.gpkg")
    gdf_cresis.plot()

    # Bedmap2
    prefix: list = [p for p in prefixes if str(year) in p]
    assert len(prefix) == 1
    result = await store.get_async(path=prefix[0])
    bytes = await result.bytes_async()
    gdf_bedmap_: gpd.GeoDataFrame = gpd.read_parquet(path=io.BytesIO(bytes)).to_crs(
        epsg=3031
    )
    assert len(gdf_bedmap_) >= 1

    # Fuzzy match using Hausdorff distance
    geom_references: shapely.geometry.MultiLineString = gdf_cresis_.iloc[0].geometry
    gdf_bedmap_["hausdorff_dist"] = pd.DataFrame(
        # Break multilinestring into individual linestring segments, then
        # calculate hausdorff distance for each bedmap segment against opr segments
        data=(
            gdf_bedmap_.hausdorff_distance(other=geom) for geom in geom_references.geoms
        )
    ).min()  # take minimum hausdorff distance from one-to-one bedmap/opr matches
    # gdf_bedmap_["frechet_dist"] = gdf_bedmap_.frechet_distance(other=geom_reference)
    gdf_bedmap_.to_file(filename=f"data/bedmap_{year}.gpkg", mode="w")

    # Report Bedmap IDs and their hausdorff distance to nearest OPR line segment
    print(gdf_bedmap_[["id", "hausdorff_dist"]].sort_values(by="hausdorff_dist"))

    break  # TODO work on more years
