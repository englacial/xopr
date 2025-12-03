# Bedmap Flight Line Coverage

Below is an Antarctic map showing Bedmap radar flight line coverage. This map loads GeoParquet STAC catalog files directly in the browser using WebAssembly.

**About Bedmap**

Bedmap is a compilation of Antarctic ice thickness data from multiple institutions and surveys spanning decades of radar measurements. The data shown here represents:
- **BM1**: Original Bedmap compilation
- **BM2**: Bedmap2 - improved compilation
- **BM3**: Bedmap3 - latest compilation

**Legend**
  - <span style="color: red; font-weight: bold;">Red</span>: Bedmap1 (BM1)
  - <span style="color: orange; font-weight: bold;">Orange</span>: Bedmap2 (BM2)
  - <span style="color: navy; font-weight: bold;">Navy</span>: Bedmap3 (BM3)

## Antarctica - Bedmap Coverage

:::{polar-map}
:width: 100%
:height: 600px
:pole: south
:dataPath: https://storage.googleapis.com/opr_stac/bedmap
:fileGroups: [{"files": ["bedmap2.parquet"], "color": "orange"}]
:defaultZoom: 3
:::

*Note: Currently showing test data from 3 AWI flights (1994-1996). Full dataset includes 151 flights from multiple institutions.*

## Data Sources

The Bedmap ice thickness compilation includes contributions from:
- **AWI** (Alfred Wegener Institute) - Germany
- **BAS** (British Antarctic Survey) - UK
- **UTIG** (University of Texas Institute for Geophysics) - USA
- **CReSIS** (Center for Remote Sensing of Ice Sheets) - USA
- And many other institutions

## Technical Details

- **Data Format**: Cloud-optimized GeoParquet
- **Coordinate System**: WGS84 (EPSG:4326)
- **Projection**: Antarctic Polar Stereographic (EPSG:3031) for visualization
- **Query Support**: Spatial queries handle antimeridian crossing correctly

## Related Resources

- [OPR Radar Data Coverage](map.md) - Modern radar data availability
- [Bedmap3 at BAS](https://www.bas.ac.uk/project/bedmap/) - Official Bedmap project
