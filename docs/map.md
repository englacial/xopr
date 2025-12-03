# Radar Data Availability

Below are Antarctic and Arctic maps showing radar data availability. These maps loads GeoParquet files (STAC catalogs of radar flight lines) directly in the browser using WebAssembly; they update in real time as any additional catalogs are added.

**Legend**
  - [CReSIS](https://cresis.ku.edu/) lines <span style="color: navy; font-weight: bold;">are Navy</span>
  - [UTIG](https://ig.utexas.edu/) lines <span style="color: orange; font-weight: bold;">are Orange</span>
  - [AWI](https://www.awi.de/en/science/geosciences/glaciology/tools/radar.html) lines <span style="color: lightblue; font-weight: bold;">are Light Blue</span>
  - [BAS](https://www.bas.ac.uk/polar-operations/sites-and-facilities/facility/phase-sensitive-radar-apres/) lines <span style="color: black; font-weight: bold;">are Black</span>
  - [DTU](https://www.space.dtu.dk/english/research-divisions/microwaves-and-remote-sensing/research-areas/radar-systems) lines <span style="color: red; font-weight: bold;">are Red</span>
  - [Columbia](https://lamont.columbia.edu/research-divisions/marine-polar-geophysics) lines <span style="color: gold; font-weight: bold;">are Yellow</span>
  - [UW](https://environment.uw.edu/news/2020/02/new-radar-technology-sheds-light-on-never-before-seen-antarctic-landscape/) lines <span style="color: purple; font-weight: bold;">are Purple</span>

## Antarctica

:::{polar-map}
:width: 100%
:height: 600px
:pole: south
:dataPath: https://storage.googleapis.com/opr_stac/catalog/hemisphere=south
:fileGroups: [{"files": ["provider=cresis/*"], "color": "navy"}, {"files": ["provider=utig/*"], "color": "orange"}]
:defaultZoom: 3
:::

## Greenland

:::{polar-map}
:width: 100%
:height: 600px
:pole: north
:dataPath: https://storage.googleapis.com/opr_stac/catalog/hemisphere=north
:fileGroups: [{"files": ["provider=cresis/*"], "color": "navy"}, {"files": ["provider=utig/*"], "color": "orange"}, {"files": ["provider=dtu/*"], "color": "red"}, {"files": ["provider=awi/*"], "color": "lightblue"}]
:defaultZoom: 3
:::

