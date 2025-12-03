# Note on Polar Projections and Bounding Boxes

## Current Implementation
The current bedmap implementation uses WGS84 (EPSG:4326) bounding boxes for spatial queries. This works but has limitations near the poles due to longitudinal convergence.

## Issue with Rectangular Bounding Boxes at High Latitudes
- Near the South Pole, longitude lines converge rapidly
- A "rectangular" bounding box in lat/lon covers a wedge-shaped area
- This leads to inefficient queries - we fetch much more data than needed
- Example: A 10°x10° box at -70° latitude covers much less area than at the equator

## Potential Improvements

### 1. Use Antarctic Polar Stereographic (EPSG:3031)
```python
# Convert query geometry to polar stereographic
from pyproj import Transformer
transformer = Transformer.from_crs("EPSG:4326", "EPSG:3031")
x_min, y_min = transformer.transform(lon_min, lat_min)
x_max, y_max = transformer.transform(lon_max, lat_max)
# Use x,y bounds for more efficient rectangular queries
```

### 2. Use Actual Multiline Geometry for STAC
Instead of bbox bounds, use the actual flight line geometry:
```python
# In STAC item creation
item = Item(
    geometry=simplified_multiline,  # Use actual geometry
    bbox=bbox,  # Still provide bbox for compatibility
    ...
)
```

### 3. Implement Spatial Indexing
For large datasets, consider:
- H3 hexagonal indexing (equal-area cells)
- S2 geometry (hierarchical cells)
- QuadTree indexing in polar projection

### 4. Query Strategy Modifications
```python
def query_bedmap_polar(
    geometry,  # Input in WGS84
    use_projection='EPSG:3031'  # Antarctic Polar Stereographic
):
    # 1. Transform query geometry to polar
    # 2. Find intersecting files using projected bounds
    # 3. Apply precise intersection test
    # 4. Query only truly intersecting files
```

## Implementation Priority
1. **Short term**: Keep current implementation, document limitation
2. **Medium term**: Add optional polar projection support in queries
3. **Long term**: Store both WGS84 and polar bounds in STAC for efficient querying

## Testing Near-Pole Queries
When testing, be aware that queries near the pole may return more data than expected due to the bounding box expansion. Consider using smaller query regions or post-filtering results.