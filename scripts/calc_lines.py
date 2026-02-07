#!/usr/bin/env python3
"""Calculate total kilometers of flight lines from OPR STAC catalog."""

import sys
import json

try:
    import geopandas as gpd
    from pyproj import Geod
    from rustac import DuckdbClient
    from shapely.geometry import box
except ImportError as e:
    # Exit with non-zero code so GitHub Actions knows it failed
    print(f"Error: Missing required module - {e}", file=sys.stderr)
    print('{"error": "Missing dependencies"}')
    sys.exit(1)

try:
    # Initialize geodesic calculator
    geod = Geod(ellps="WGS84")
    
    # Connect to parquet files
    client = DuckdbClient()
    partitioned_destination = 's3://us-west-2.opendata.source.coop/englacial/xopr/catalog/**/*parquet'
    
    # Define regions
    arctic_region = box(-180, 0, 180, 90)      # Northern hemisphere
    antarctic_region = box(-180, -90, 180, 0)   # Southern hemisphere
    
    # Calculate Arctic (Northern hemisphere)
    table_arctic = client.search_to_arrow(partitioned_destination, 
                                          intersects=arctic_region.__geo_interface__)
    if table_arctic is None or len(table_arctic) == 0:
        raise ValueError("No Arctic data found")
        
    ds_arctic = gpd.GeoDataFrame.from_arrow(table_arctic)
    arctic_meters = ds_arctic['geometry'].apply(geod.geometry_length).sum()
    arctic_km = round(arctic_meters / 1000)
    
    # Calculate Antarctic (Southern hemisphere)  
    table_antarctic = client.search_to_arrow(partitioned_destination,
                                             intersects=antarctic_region.__geo_interface__)
    if table_antarctic is None or len(table_antarctic) == 0:
        raise ValueError("No Antarctic data found")
        
    ds_antarctic = gpd.GeoDataFrame.from_arrow(table_antarctic)
    antarctic_meters = ds_antarctic['geometry'].apply(geod.geometry_length).sum()
    antarctic_km = round(antarctic_meters / 1000)
    
    # Total
    total_km = arctic_km + antarctic_km
    
    # Output as JSON for easier parsing in GitHub Actions
    result = {
        "arctic": arctic_km,
        "antarctic": antarctic_km,
        "total": total_km
    }
    
    print(json.dumps(result))
    
except ConnectionError as e:
    print(f"Error: Cannot connect to GCS - {e}", file=sys.stderr)
    print('{"error": "Connection failed"}')
    sys.exit(2)
    
except ValueError as e:
    print(f"Error: Data validation failed - {e}", file=sys.stderr) 
    print('{"error": "Invalid data"}')
    sys.exit(3)
    
except Exception as e:
    print(f"Error: Unexpected error - {e}", file=sys.stderr)
    print('{"error": "Unknown error"}')
    sys.exit(4)