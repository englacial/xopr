# xOPR Design

## Guiding Principles

1. xOPR is first and foremost a data access tool. It should be simple to find and retrieve data for both human-centered and automated workflows.
2. xOPR is not a radar processing tool. It is reasonable to include simple analysis tools when they are common, well-defined, and widely useful, but most interesting analysis workflows will require code outside of the core xOPR.
3. xOPR is a part of the broader OPR project and, as such, should use the OPR data product types as directly as possible. It is reasonable for xOPR to transform data as needed to smooth over data format transitions or to translate appropriately from Matlab to Python, but the core data products should not be changed.
    - See [OPR Toolbox File Guides](https://gitlab.com/openpolarradar/opr/-/wikis/OPR-Toolbox-Guide#file-guides)
4. In line with Xarray advice, xOPR does not subclass any Xarray datatypes but rather operators on native Xarray datatypes.
5. To enable reproducible workflows, xopr should pair metadata and data so that they travel together. (i.e., either as sidecar files or appends to the cached on disk format) 

## Data Terminology

Segments of raw radar sounder data are identified by a collection and a granule.

### Collections

The collection, known as a season in OPR, uniquely identifies one season, a general geographic location, and an aircraft. For example: `2022_Antarctica_BaslerMKB`.

There may be multiple collections per season if multiple aircraft were in use.

Season years are defined with respect to when the season would typically start. In Antarctica, the summer season spans the new year boundary, but the seasons are always identified by the year at the start of the summer. Even if all of the flights took place in January 2023, the season year would be 2022.

### Data granules (`YYYYMMDD_SS_FFF`)

Data granules uniquely identify a particular time period within a season. There are three parts to the granule:

- `YYYYMMDD` is the date at the start of the segment. The date does not change within a segment, even if midnight is passed.
- `SS` is the segment number. All frames within a single segment represent continguous data with no gaps.
- `FFF` is the frame number. Frames within a single segment are contiguous. Frames are used as a way to split data into manageable chunks, typically (but not always) 50 km along-track segments.

We also use the term `segment_path` to refer to the `YYYYMMDD_SS` part of the granule, which uniquely identifies a segment within a collection. In many cases, segment paths correspond with flights, however it is possible for various reasons that a flight gets split into multiple segments.
