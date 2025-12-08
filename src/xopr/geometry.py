import geopandas as gpd
from cartopy import crs
from pyproj import Transformer
import shapely
import shapely.ops

def get_antarctic_regions(
    name=None,
    regions=None, 
    subregions=None,
    type=None,
    merge_regions=True,
    measures_boundaries_url = "https://storage.googleapis.com/opr_stac/reference_geometry/measures_boundaries_4326.geojson",
    merge_in_projection="EPSG:3031",
    simplify_tolerance=None
):
    """
    Load and filter Antarctic regional boundaries from the MEASURES dataset.
    
    Parameters
    ----------
    name : str or list, optional
        NAME field value(s) to filter by
    regions : str or list, optional
        REGIONS field value(s) to filter by  
    subregions : str or list, optional
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
    
    
    # Load the boundaries GeoJSON from the reference URL
    filtered = gpd.read_file(measures_boundaries_url)
    
    # Apply filters based on provided parameters
    if name is not None:
        if isinstance(name, str):
            name = [name]
        filtered = filtered[filtered['NAME'].isin(name)]
    
    if regions is not None:
        if isinstance(regions, str):
            regions = [regions] 
        filtered = filtered[filtered['Regions'].isin(regions)]
        
    if subregions is not None:
        if isinstance(subregions, str):
            subregions = [subregions]
        filtered = filtered[filtered['Subregions'].isin(subregions)]
        
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

        if simplify_tolerance is None and (merge_in_projection == "EPSG:3031"): # Set a reasonable default based on the size
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

def project_geojson(geometry, source_crs="EPSG:4326", target_crs="EPSG:3031"):
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
    target_crs : str, default "EPSG:3031"
        Target coordinate reference system (default is Antarctic Polar Stereographic)
        
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
    transformer = Transformer.from_crs(source_crs, target_crs, always_xy=True)
    projected_geometry = shapely.ops.transform(transformer.transform, geometry)
    return projected_geometry