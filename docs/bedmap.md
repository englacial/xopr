# Bedmap Flight Line Coverage

:::{polar-map} /_static/maps/polar.html
:width: 100%
:height: 600px
:pole: south
:dataPath: https://storage.googleapis.com/opr_stac/bedmap
:fileGroups: [{"files": ["bedmap1.parquet"], "color": "red"}, {"files": ["bedmap2.parquet"], "color": "orange"}, {"files": ["bedmap3.parquet"], "color": "navy"}]
:defaultZoom: 3
:::

**Legend**
  - <span style="color: red; font-weight: bold;">Red</span>: BedMap1
  - <span style="color: orange; font-weight: bold;">Orange</span>: BedMap2
  - <span style="color: navy; font-weight: bold;">Navy</span>: BedMap3

While the primary purpose of xOPR is to provide access to radar data in the OPR catalog, we know many users are interested in where other radar data exists and/or need bed or surface picks from data that is not (yet) part of the OPR catalog. To serve those use cases, xOPR has integrated a submodule to provide easy access to the BedMap dataset.

The map above shows BedMap radar flight line coverage. This map loads GeoParquet STAC catalog files directly in the browser using WebAssembly.

All the data shown can be queried using [`query_bedmap_catalog`](/xopr/api/xopr/bedmap/query.html#query_bedmap_catalog) and retrieved as a pandas dataframe using [`query_bedmap`](/xopr/api/xopr/bedmap/query.html#query_bedmap). For faster repeated queries, use [`fetch_bedmap`](/xopr/api/xopr/bedmap/query.html#fetch_bedmap) to download data locally first.

See [the BedMap demo notebook](/xopr/bedmap-demo/) for a complete example.

**About BedMap**

BedMap3 is a compilation of Antarctic ice thickness data from multiple institutions and surveys spanning decades of radar measurements. Note that we are referring to the collection of surface and bed picks, not the gridded BedMap product. For more information about this dataset, refer to the [BedMap website at BAS](https://www.bas.ac.uk/project/bedmap/) or the [ESSD publication](https://essd.copernicus.org/articles/15/2695/2023/):

> Frémand, A. C., Fretwell, P., Bodart, J. A., Pritchard, H. D., Aitken, A., Bamber, J. L., Bell, R., Bianchi, C., Bingham, R. G., Blankenship, D. D., Casassa, G., Catania, G., Christianson, K., Conway, H., Corr, H. F. J., Cui, X., Damaske, D., Damm, V., Drews, R., Eagles, G., Eisen, O., Eisermann, H., Ferraccioli, F., Field, E., Forsberg, R., Franke, S., Fujita, S., Gim, Y., Goel, V., Gogineni, S. P., Greenbaum, J., Hills, B., Hindmarsh, R. C. A., Hoffman, A. O., Holmlund, P., Holschuh, N., Holt, J. W., Horlings, A. N., Humbert, A., Jacobel, R. W., Jansen, D., Jenkins, A., Jokat, W., Jordan, T., King, E., Kohler, J., Krabill, W., Kusk Gillespie, M., Langley, K., Lee, J., Leitchenkov, G., Leuschen, C., Luyendyk, B., MacGregor, J., MacKie, E., Matsuoka, K., Morlighem, M., Mouginot, J., Nitsche, F. O., Nogi, Y., Nost, O. A., Paden, J., Pattyn, F., Popov, S. V., Rignot, E., Rippin, D. M., Rivera, A., Roberts, J., Ross, N., Ruppel, A., Schroeder, D. M., Siegert, M. J., Smith, A. M., Steinhage, D., Studinger, M., Sun, B., Tabacco, I., Tinto, K., Urbini, S., Vaughan, D., Welch, B. C., Wilson, D. S., Young, D. A., and Zirizzotti, A.: Antarctic Bedmap data: Findable, Accessible, Interoperable, and Reusable (FAIR) sharing of 60 years of ice bed, surface, and thickness data, Earth Syst. Sci. Data, 15, 2695–2710, https://doi.org/10.5194/essd-15-2695-2023, 2023. 

The dataset itself is hosted at BAS and can be found on [their data portal](https://data.bas.ac.uk/full-record.php?id=GB/NERC/BAS/PDC/01615). It should be cited as follows:

> Pritchard, H., Fretwell, P., Fremand, A., Bodart, J., Kirkham, J., Aitken, A., Bamber, J., Bell, R., Bianchi, C., Bingham, R., Blankenship, D., Casassa, G., Catania, G., Christianson, K., Conway, H., Corr, H., Cui, X., Damaske, D., Damn, V., ... Zirizzotti, A. (2024). BEDMAP3 - Ice thickness, bed and surface elevation for Antarctica - gridding products (Version 1.0) [Data set]. NERC EDS UK Polar Data Centre. https://doi.org/10.5285/2d0e4791-8e20-46a3-80e4-f5f6716025d2 

The data is arranged into three subcategories according to which iteration of BedMap first included it.
