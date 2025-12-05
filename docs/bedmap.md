# Bedmap Flight Line Coverage

Below is an Antarctic map showing Bedmap radar flight line coverage. This map loads GeoParquet STAC catalog files directly in the browser using WebAssembly.

All the data shown be queried using `query_bedmap_catalog` and retrieved as a pandas
dataframe using `query_bedmap`.

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

:::{polar-map} /_static/maps/polar.html
:width: 100%
:height: 600px
:pole: south
:dataPath: https://storage.googleapis.com/opr_stac/bedmap
:fileGroups: [{"files": ["bedmap1.parquet"], "color": "red"}, {"files": ["bedmap2.parquet"], "color": "orange"}, {"files": ["bedmap3.parquet"], "color": "navy"}]
:defaultZoom: 3
:::

## Related Resources

- [OPR Radar Data Coverage](map.md) - Modern radar data availability
- [Bedmap3 at BAS](https://www.bas.ac.uk/project/bedmap/) - Official Bedmap project
