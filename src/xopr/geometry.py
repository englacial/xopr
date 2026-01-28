"""
Geospatial operations for radar datasets

This module provides utilities for working with geospatial data in the context
of polar radar analysis. It includes functions for:

- Loading and filtering Antarctic regional boundaries from the MEaSUREs dataset
- Projecting radar dataset coordinates between different coordinate reference systems
- Reprojecting GeoJSON geometries for spatial analysis

The module handles common polar coordinate system transformations, particularly
between WGS84 (EPSG:4326) and Antarctic Polar Stereographic (EPSG:3031), and
provides tools for merging and simplifying regional boundaries.

Functions
---------
get_antarctic_regions : Load and filter Antarctic regional boundaries
get_greenland_regions : Load and filter Greenland regional boundaries
project_dataset : Project dataset coordinates to a target CRS
project_geojson : Reproject geometries between coordinate systems

See Also
--------
MEaSUREs Antarctic Boundaries: https://nsidc.org/data/nsidc-0709/versions/2
EPSG:3031 (Antarctic Polar Stereographic): https://epsg.io/3031
"""

import geopandas as gpd
import shapely
import shapely.ops
from pyproj import Transformer


def get_antarctic_regions(
    name=None,
    region=None,
    subregion=None,
    type=None,
    merge_regions=True,
    merge_in_projection="EPSG:3031",
    simplify_tolerance=None,
    regions=None, # Deprecated alias for "region"
    subregions=None # Deprecated alias for "subregion"
):
    """
    Load and filter Antarctic regional boundaries from the MEaSUREs dataset.

    The data product is derived from:

    > Maps of Antarctic ice shelves, the Antarctic coastline and Antarctic basins. The maps are assembled from 2008-2009 ice-front data from ALOS PALSAR and ENVISAT ASAR data acquired during International Polar Year, 2007-2009 (IPY), the InSAR-based grounding line data (MEaSUREs Antarctic Grounding Line from Differential Satellite Radar Interferometry), augmented with other grounding line sources, the Antarctic ice velocity map (MEaSUREs InSAR-Based Antarctica Ice Velocity Map), and the Bedmap-2 DEM.
    >
    > This data set is part of the NASA Making Earth System Data Records for Use in Research Environments (MEaSUREs) program.
    >
    > Mouginot, J., B. Scheuchl, and E. Rignot. 2017. MEaSUREs Antarctic Boundaries for IPY 2007-2009 from Satellite Radar, Version 2. [Indicate subset used]. Boulder, Colorado USA. NASA National Snow and Ice Data Center Distributed Active Archive Center. doi: http://dx.doi.org/10.5067/AXE4121732AD. [Date Accessed].
    >
    > https://nsidc.org/data/nsidc-0709/versions/2

    See these two notebooks for examples of how to use this function:
    - https://docs.englacial.org/xopr/search-and-scaling/
    - https://docs.englacial.org/xopr/crossovers/

    Parameters
    ----------
    name : str or list, optional
        NAME field value(s) to filter by
    region : str or list, optional
        REGION field value(s) to filter by
    subregion : str or list, optional
        SUBREGION field value(s) to filter by
    type : str or list, optional
        TYPE field value(s) to filter by
    merge_regions : bool, default True
        If True, return a single merged geometry; if False, return list of geometries
    measures_boundaries_url : str, default "https://storage.googleapis.com/opr_stac/reference_geometry/measures_boundaries_4326.geojson"
        URL to the GeoJSON file containing Antarctic region boundaries

    Returns
    -------
    list or dict
        If merge_regions=False: List of GeoJSON geometry dicts
        If merge_regions=True: Single GeoJSON geometry dict of merged regions

    Examples
    --------
    # Get George VI ice shelf
    >>> george_vi = get_antarctic_regions(name="George_VI", type="FL")

    # Get all ice shelves, merged into one geometry
    >>> all_shelves = get_antarctic_regions(type="FL", merge_regions=True)

    # Get multiple regions by name
    >>> regions = get_antarctic_regions(name=["George_VI", "LarsenC"])
    """

    if regions is not None:
        print("Warning: 'regions' parameter is deprecated; use 'region' instead.")
        if region is not None:
            raise ValueError("Cannot use both 'regions' and 'region' parameters.")
        region = regions
    if subregions is not None:
        print("Warning: 'subregions' parameter is deprecated; use 'subregion' instead.")
        if subregion is not None:
            raise ValueError("Cannot use both 'subregions' and 'subregion' parameters.")
        subregion = subregions

    return _get_regions(
        name=name,
        region=region,
        subregion=subregion,
        type=type,
        merge_regions=merge_regions,
        regions_geojson_url="https://storage.googleapis.com/opr_stac/reference_geometry/measures_boundaries_4326.geojson",
        merge_in_projection=merge_in_projection,
        simplify_tolerance=simplify_tolerance
    )

def get_greenland_regions(
    name=None,
    region=None,
    subregion=None,
    type=None,
    merge_regions=True,
    merge_in_projection="EPSG:3413",
    simplify_tolerance=None,
):
    """
    Load and filter Greenland regional boundaries as defined by the GrIMP project

    The data product is derived from:

    > We divide Greenland, including its peripheral glaciers and ice caps, into 260 basins grouped in seven regions:  southwest (SW), central west (CW), (iii) northwest (NW), north (NO), northeast (NE), central east (CE), and southeast (SE). These regions are selected based on ice flow regimes, climate, and the need to partition the ice sheet into zones comparable in size (200,000 km2 to 400,000 km2) and ice production (50 Gt/y to 100 Gt/y, or billion tons per year). Out of the 260 surveyed glaciers, 217 are marine-terminating, i.e., calving into icebergs and melting in contact with ocean waters, and 43 are land-terminating.The actual number of land-terminating glaciers is far larger than 43, but we lump them into larger units for simplification.
    >
    > Each glacier catchment is defined using a combination of ice flow direction and surface slope. In areas of fast flow (> 100 m), we use a composite velocity mosaic (Mouginot et al. 2017). In slowmoving areas, we use surface slope using the GIMP DEM (https://nsidc.org/data/nsidc- 0715/versions/1) (Howat et al. 2014) smoothed over 10 ice thicknesses to remove shortwavelength undulations.
    >
    > References:
    >
    > Mouginot J, Rignot E, Scheuchl B, Millan R (2017) Comprehensive annual ice sheet velocity mapping using landsat-8, sentinel-1, and radarsat-2 data. Remote Sensing 9(4).
    >
    > Howat IM, Negrete A, Smith BE (2014) The greenland ice mapping project (GIMP) land classification and surface elevation data sets. The Cryosphere 8(4):1509–1518.
    >
    > https://datadryad.org/dataset/doi:10.7280/D1WT11

    Parameters
    ----------
    name : str or list, optional
        NAME field value(s) to filter by
    region : str or list, optional
        REGION field value(s) to filter by
    subregion : str or list, optional
        SUBREGION field value(s) to filter by
    type : str or list, optional
        TYPE field value(s) to filter by
    merge_regions : bool, default True
        If True, return a single merged geometry; if False, return list of geometries
    measures_boundaries_url : str, default "https://storage.googleapis.com/opr_stac/reference_geometry/measures_boundaries_4326.geojson"
        URL to the GeoJSON file containing Antarctic region boundaries

    Returns
    -------
    list or dict
        If merge_regions=False: List of GeoJSON geometry dicts
        If merge_regions=True: Single GeoJSON geometry dict of merged regions
    """

    greenland_boundaries_url = "https://storage.googleapis.com/opr_stac/reference_geometry/greenland_boundaries_4326.geojson"

    return _get_regions(
        name=name,
        region=region,
        subregion=subregion,
        type=type,
        merge_regions=merge_regions,
        regions_geojson_url=greenland_boundaries_url,
        merge_in_projection=merge_in_projection,
        simplify_tolerance=simplify_tolerance
    )


def _get_regions(
    name=None,
    region=None,
    subregion=None,
    type=None,
    merge_regions=True,
    regions_geojson_url = None,
    merge_in_projection=None,
    simplify_tolerance=None
):
    if regions_geojson_url is None:
        raise ValueError("regions_geojson_url must be provided")

    if merge_in_projection is None:
        raise ValueError("merge_in_projection must be provided")

    # Load the boundaries GeoJSON from the reference URL
    filtered = gpd.read_file(regions_geojson_url)

    standard_columns = {
        'NAME': ['NAME', 'NAMES'],
        'REGION': ['REGION', 'Regions'],
        'SUBREGION': ['SUBREGION', 'Subregions'],
        'TYPE': ['TYPE', 'TYPES', 'GL_TYPE']
    }

    # Standardize column names
    for standard_name, possible_names in standard_columns.items():
        for col in possible_names:
            if col in filtered.columns:
                filtered = filtered.rename(columns={col: standard_name})
                break

    # Apply filters based on provided parameters
    if name is not None:
        if isinstance(name, str):
            name = [name]
        filtered = filtered[filtered['NAME'].isin(name)]

    if region is not None:
        if isinstance(region, str):
            region = [region]
        filtered = filtered[filtered['REGION'].isin(region)]

    if subregion is not None:
        if isinstance(subregion, str):
            subregion = [subregion]
        filtered = filtered[filtered['SUBREGION'].isin(subregion)]

    if type is not None:
        if isinstance(type, str):
            type = [type]
        filtered = filtered[filtered['TYPE'].isin(type)]

    if len(filtered) == 0:
        return [] if not merge_regions else None

    if merge_regions:

        if merge_in_projection:
            filtered = filtered.to_crs(merge_in_projection)

        # Check for invalid regions and attempt to fix them
        invalid_geometries = filtered[~filtered.is_valid]
        if len(invalid_geometries) > 0:
            filtered = filtered.make_valid()
            print(f"Warning: {len(invalid_geometries)} invalid geometries were fixed before merging.")
            if merge_in_projection != "EPSG:3031":
                print("Consider using merge_in_projection='EPSG:3031' to reproject before merging.")
            print(f"Invalid geometry regions were: {', '.join(invalid_geometries['NAME'])}")

        merged = shapely.ops.unary_union(filtered.geometry)  # geopandas < 1.0 compat

        if simplify_tolerance is None and (merge_in_projection is not None): # Set a reasonable default based on the size
            area_km2 = merged.area / 1e6
            if area_km2 < 10000:
                simplify_tolerance = 0
            elif area_km2 < 100000:
                print(f"Area is {area_km2:.1f} km^2, automatically applying 100m simplification tolerance")
                print(f"To disable simplification, set simplify_tolerance=0")
                simplify_tolerance = 100
            else:
                print(f"Area is {area_km2:.1f} km^2, automatically applying 1km simplification tolerance")
                print(f"To disable simplification, set simplify_tolerance=0")
                simplify_tolerance = 1000

        if simplify_tolerance and (simplify_tolerance > 0):
            merged = shapely.buffer(merged, simplify_tolerance).simplify(tolerance=simplify_tolerance)

        if merge_in_projection:
            merged = project_geojson(merged, source_crs=merge_in_projection, target_crs="EPSG:4326")

        return merged
    else:
        return filtered

def project_dataset(ds, target_crs):
    """
    Project dataset coordinates from WGS84 to a target coordinate reference system.

    Takes longitude and latitude coordinates from a dataset and projects them to
    the specified target CRS, adding 'x' and 'y' coordinate arrays to the dataset.

    Parameters
    ----------
    ds : xarray.Dataset
        Input dataset containing 'Longitude' and 'Latitude' coordinates
    target_crs : cartopy.crs.CRS or str
        Target coordinate reference system. Can be a cartopy CRS object or
        a string representation (e.g., "EPSG:3031")

    Returns
    -------
    xarray.Dataset
        Dataset with added 'x' and 'y' coordinate arrays in the target CRS

    Examples
    --------
    >>> import cartopy.crs as ccrs
    >>> projected_ds = project_dataset(ds, ccrs.SouthPolarStereo())
    >>> projected_ds = project_dataset(ds, "EPSG:3031")
    """
    if hasattr(target_crs, 'to_epsg') and target_crs.to_epsg():
        target_crs_str = f"EPSG:{target_crs.to_epsg()}"
    elif isinstance(target_crs, str):
        target_crs_str = target_crs
    else:
        target_crs_str = target_crs.to_proj4_string()

    transformer = Transformer.from_crs("EPSG:4326", target_crs_str, always_xy=True)
    projected_coords = transformer.transform(ds['Longitude'].values, ds['Latitude'].values)

    ds = ds.assign_coords({
        'x': (('slow_time'), projected_coords[0]),
        'y': (('slow_time'), projected_coords[1])
    })
    return ds

def project_geojson(geometry, source_crs="EPSG:4326", target_crs=None):
    """
    Project a geometry from one coordinate reference system to another.

    Uses pyproj.Transformer to reproject geometries between different
    coordinate reference systems. Commonly used for projecting geometries
    from WGS84 (lat/lon) to polar stereographic projections.

    Parameters
    ----------
    geometry : shapely.geometry.base.BaseGeometry
        Input geometry to be projected
    source_crs : str, default "EPSG:4326"
        Source coordinate reference system (default is WGS84)
    target_crs : str, default None
        Target coordinate reference system. If None, automatically selects
        Antarctic Polar Stereographic (EPSG:3031) for southern hemisphere
        and Arctic Polar Stereographic (EPSG:3413) for northern hemisphere based on
        the geometry's centroid latitude.

    Returns
    -------
    shapely.geometry.base.BaseGeometry
        Projected geometry in the target coordinate reference system

    Examples
    --------
    >>> from shapely.geometry import Point
    >>> point = Point(-70, -75)  # lon, lat in WGS84
    >>> projected = project_geojson(point, "EPSG:4326", "EPSG:3031")
    """

    if target_crs is None:
        if geometry.centroid.y < 0:
            target_crs = "EPSG:3031"  # Antarctic Polar Stereographic
        else:
            target_crs = "EPSG:3413"  # Arctic Polar Stereographic

    transformer = Transformer.from_crs(source_crs, target_crs, always_xy=True)
    projected_geometry = shapely.ops.transform(transformer.transform, geometry)
    return projected_geometry
