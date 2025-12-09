import xarray as xr

from typing import Iterable, Union
import warnings
import geopandas as gpd

from xopr.util import get_ror_display_name, merge_dicts_no_conflicts

def merge_frames(frames: Iterable[xr.Dataset]) -> Union[list[xr.Dataset], xr.Dataset]:
    """
    Merge a set of radar frames into a list of merged xarray Datasets. Frames from the
    same segment (typically a flight) are concatenated along the 'slow_time' dimension.

    Parameters
    ----------
    frames : Iterable[xr.Dataset]
        An iterable of xarray Datasets representing radar frames.

    Returns
    -------
    list[xr.Dataset] or xr.Dataset
        List of merged xarray Datasets or a single merged Dataset if there is only one segment.
    """
    segments = {}

    mergable_keys = ['source_url', 'collection', 'data_product', 'granule']
    
    for frame in frames:
        # Get segment path from frame attributes
        granule = frame.attrs.get('granule')
        date, segment_id, frame_id = granule.split('_')
        segment_path = f"{date}_{segment_id}"

        if not segment_path:
            warnings.warn("Frame missing 'granule' attribute or it was not in the expected format, skipping.", UserWarning)
            continue

        if segment_path not in segments:
            segments[segment_path] = []

        segments[segment_path].append(frame)

    # Merge frames for each segment
    merged_segments = []
    for segment_path, segment_frames in segments.items():
        merged_segment = xr.concat(segment_frames, dim='slow_time', combine_attrs=merge_dicts_no_conflicts).sortby('slow_time')

        for k in mergable_keys:
            if k not in merged_segment.attrs:
                merged_segment.attrs[k] = set([v for v in {f.attrs.get(k) for f in segment_frames} if v is not None])

        merged_segments.append(merged_segment)

    if len(merged_segments) == 1:
        return merged_segments[0]
    else:
        return merged_segments

def generate_citation(ds : xr.Dataset) -> str:
        """
        Generate a citation string for the dataset based on its attributes.

        Parameters
        ----------
        ds : xr.Dataset
            The xarray Dataset containing metadata.

        Returns
        -------
        str
            A formatted citation string.
        """

        citation_string = ""
        any_citation_info = False

        citation_string += "== Data Citation ==\n"

        # Find the right key for ROR
        ror_key = None
        if 'ror' in ds.attrs:
            ror_key = 'ror'
        elif 'Organization' in ds.attrs:
            ror_key = 'Organization'

        if ror_key and ds.attrs[ror_key]:
            any_citation_info = True
            if isinstance(ds.attrs[ror_key], (set, list)):
                institution_name = ', '.join([get_ror_display_name(ror) for ror in ds.attrs[ror_key]])
            else:
                institution_name = get_ror_display_name(ds.attrs[ror_key])

            citation_string += f"This data was collected by {institution_name}.\n"

        # Find the right key for DOI
        doi_key = None
        if 'doi' in ds.attrs:
            doi_key = 'doi'
        elif 'DOI' in ds.attrs:
            doi_key = 'DOI'

        if doi_key and ds.attrs[doi_key]:
            any_citation_info = True
            citation_string += f"Please cite the dataset DOI: https://doi.org/{ds.attrs[doi_key]}\n"

        if 'funder_text' in ds.attrs and ds.attrs['funder_text']: # Old style funding acknowledgement
            any_citation_info = True
            citation_string += f"Please include the following funder acknowledgment:\n{ds.attrs['funder_text']}\n"
        
        if 'Funding' in ds.attrs and ds.attrs['Funding']: # New style funding acknowledgement
            any_citation_info = True
            citation_string += f"Data collection was funded by: {ds.attrs['Funding']}\n"

        if not any_citation_info:
            citation_string += "No specific citation information was retrieved for this dataset. By default, please cite:\n"
            citation_string += "CReSIS. 2024. REPLACE_WITH_RADAR_NAME Data, Lawrence, Kansas, USA. Digital Media. http://data.cresis.ku.edu/."

        # Add general OPR Toolbox citation
        citation_string += "\n== Processing Citation ==\n"
        citation_string += "Data was processed using the Open Polar Radar (OPR) Toolbox: https://doi.org/10.5281/zenodo.5683959\n"
        citation_string += "Please cite the OPR Toolbox as:\n"
        citation_string += "Open Polar Radar. (2024). opr (Version 3.0.1) [Computer software]. https://gitlab.com/openpolarradar/opr/. https://doi.org/10.5281/zenodo.5683959\n"
        citation_string += "And include the following acknowledgment:\n"
        citation_string += "We acknowledge the use of software from Open Polar Radar generated with support from the University of Kansas, NASA grants 80NSSC20K1242 and 80NSSC21K0753, and NSF grants OPP-2027615, OPP-2019719, OPP-1739003, IIS-1838230, RISE-2126503, RISE-2127606, and RISE-2126468.\n"

        return citation_string if citation_string else "No citation information available for this dataset."


def find_intersections(gdf : gpd.GeoDataFrame, remove_self_intersections: bool = True,
                        remove_adjacent_intersections: bool = True,
                        calculate_crossing_angles: bool = False) -> gpd.GeoDataFrame:
    """
    Find intersections between geometries in a GeoDataFrame.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        A GeoDataFrame containing geometries to check for intersections.
    remove_self_intersections : bool, optional
        If True, remove self-intersections (a geometry intersecting with itself), by default True.
    remove_adjacent_intersections : bool, optional
        If True, remove intersections between adjacent geometries (e.g., consecutive frames in a flight line), by default True.

    Returns
    -------
    gpd.GeoDataFrame
        A GeoDataFrame containing pairs of intersecting geometries.
    """

    tmp_df = gdf.reset_index()
    tmp_df['geom'] = tmp_df.geometry
    intersections = gpd.sjoin(tmp_df, tmp_df, how='inner', predicate='intersects', lsuffix='1', rsuffix='2')
    
    if remove_self_intersections:
        intersections = intersections[intersections['id_1'] != intersections['id_2']]
    
    intersections['intersection_geometry'] = intersections.apply(lambda row: row['geom_1'].intersection(row['geom_2']), axis=1)
    intersections.set_geometry('intersection_geometry', inplace=True, crs=gdf.crs)
    intersections = intersections.drop_duplicates(subset=['intersection_geometry'])
    intersections = intersections.explode(index_parts=True).reset_index(drop=True)

    intersections_tmp = intersections[['id_1', 'id_2', 'intersection_geometry', 'collection_1', 'collection_2', 'geom_1', 'geom_2']].copy()

    for k in ['opr:date', 'opr:segment', 'opr:frame']:
        intersections_tmp[f'{k}_1'] = intersections['properties_1'].apply(lambda x: x[k])
        intersections_tmp[f'{k}_2'] = intersections['properties_2'].apply(lambda x: x[k])

    intersections = intersections_tmp

    if remove_adjacent_intersections:
        intersections = intersections[
            (intersections['opr:date_1'] != intersections['opr:date_2']) |
            (intersections['opr:segment_1'] != intersections['opr:segment_2']) |
            ((intersections['opr:frame_1'] != (intersections['opr:frame_2'] + 1)) &
            (intersections['opr:frame_1'] != (intersections['opr:frame_2'] - 1)))
        ]

    if calculate_crossing_angles:
        intersections['crossing_angle'] = intersections.apply(
            lambda row: _calculate_crossing_angle(row['geom_1'], row['geom_2'], row['intersection_geometry']),
            axis=1
        )
    
    return intersections

def _calculate_crossing_angle(line1, line2, intersection_point):
    """
    Calculate the crossing angle between two lines at their intersection point.

    Parameters
    ----------
    line1 : shapely.geometry.LineString
        The first line.
    line2 : shapely.geometry.LineString
        The second line.
    intersection_point : shapely.geometry.Point
        The point of intersection between the two lines.

    Returns
    -------
    float
        The crossing angle in degrees.
    """
    from shapely.geometry import Point
    import numpy as np

    def get_line_angle(line, point):
        # Get the nearest point on the line to the intersection point
        nearest_point = line.interpolate(line.project(point))
        # Get a small segment of the line around the nearest point to calculate the angle
        buffer_distance = 1e-6  # Small distance to create a segment
        start_point = line.interpolate(max(0, line.project(nearest_point) - buffer_distance))
        end_point = line.interpolate(min(line.length, line.project(nearest_point) + buffer_distance))
        
        # Calculate the angle of the segment
        delta_x = end_point.x - start_point.x
        delta_y = end_point.y - start_point.y
        angle = np.arctan2(delta_y, delta_x)
        return angle

    angle1 = get_line_angle(line1, intersection_point)
    angle2 = get_line_angle(line2, intersection_point)

    # Calculate the absolute difference in angles and convert to degrees
    angle_diff = np.abs(angle1 - angle2)
    crossing_angle = np.degrees(angle_diff)

    # Ensure the angle is between 0 and 180 degrees
    if crossing_angle > 180:
        crossing_angle = 360 - crossing_angle
    
    if crossing_angle > 90:
        crossing_angle = 180 - crossing_angle

    return crossing_angle