"""Tests for bedmap-to-frame morton matching."""

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import LineString, Point
from mortie import geo2mort, geo_morton_polygon

from xopr.bedmap.morton_match import (
    _build_frame_groups,
    _build_mbox_lookup,
    _find_contiguous_runs,
    _morton_prefix_match,
    match_bedmap_to_frames,
    disambiguate_matches,
)


def _compute_test_mbox(geom):
    """Compute mbox from a shapely geometry for test fixtures."""
    coords = np.array(geom.coords)
    lats, lons = coords[:, 1], coords[:, 0]
    cells = geo_morton_polygon(lats, lons, n_cells=4, order=18)
    result = [int(c.characteristic) for c in cells]
    while len(result) < 4:
        result.append(result[-1])
    return result


def _make_stac_gdf():
    """Create a STAC GeoDataFrame with opr:mbox pre-computed.

    Uses dense linestrings so that mbox cells are coarse prefixes.
    """
    line1 = LineString(
        [(lon, lat) for lon, lat in
         zip(np.linspace(-70.0, -65.0, 200), np.linspace(-75.0, -77.0, 200))]
    )
    line2 = LineString(
        [(lon, lat) for lon, lat in
         zip(np.linspace(-65.0, -60.0, 200), np.linspace(-77.0, -79.0, 200))]
    )
    items = [
        {
            'id': 'Data_20091020_05_001',
            'opr:segment': 5,
            'opr:date': '20091020',
            'opr:frame': 1,
            'opr:mbox': _compute_test_mbox(line1),
            'geometry': line1,
        },
        {
            'id': 'Data_20091020_05_002',
            'opr:segment': 5,
            'opr:date': '20091020',
            'opr:frame': 2,
            'opr:mbox': _compute_test_mbox(line2),
            'geometry': line2,
        },
    ]
    return gpd.GeoDataFrame(items, geometry='geometry', crs='EPSG:4326')


def _make_bedmap_gdf():
    """Create bedmap points with trajectory_id along two frames + one outside."""
    points = [
        Point(-67.5, -76.0),   # inside item 1 area
        Point(-62.5, -78.0),   # inside item 2 area
        Point(0.0, 0.0),       # outside both
    ]
    return gpd.GeoDataFrame(
        {'value': [100, 200, 300], 'trajectory_id': ['0', '1', '2']},
        geometry=points,
        crs='EPSG:4326',
    )


def _make_layer_gdf():
    """Create OPR layer picks along the two test frame lines."""
    points = []
    data = []
    for lon, lat in zip(np.linspace(-70, -65, 50), np.linspace(-75, -77, 50)):
        points.append(Point(lon, lat))
        data.append({'segment_path': '20091020_05', 'frame': 1})
    for lon, lat in zip(np.linspace(-65, -60, 50), np.linspace(-77, -79, 50)):
        points.append(Point(lon, lat))
        data.append({'segment_path': '20091020_05', 'frame': 2})
    return gpd.GeoDataFrame(data, geometry=points, crs='EPSG:4326')


def test_build_mbox_lookup():
    stac = _make_stac_gdf()
    lookup = _build_mbox_lookup(stac)
    assert isinstance(lookup, dict)
    assert len(lookup) > 0
    for prefix, entries in lookup.items():
        assert isinstance(prefix, str)
        for seg, date, frame, item_id in entries:
            assert isinstance(seg, (int, np.integer))
            assert isinstance(frame, (int, np.integer))


def test_morton_prefix_match():
    stac = _make_stac_gdf()
    lookup = _build_mbox_lookup(stac)
    mortons = geo2mort(np.array([-76.0]), np.array([-67.5]), order=18)
    mort_strs = [str(m) for m in mortons]
    candidates, counts = _morton_prefix_match(mort_strs, lookup)
    assert len(candidates) == 1
    assert counts[0] >= 1


def test_match_bedmap_to_frames():
    stac = _make_stac_gdf()
    bedmap = _make_bedmap_gdf()
    result = match_bedmap_to_frames(stac, bedmap)
    assert 'opr_candidates' in result.columns
    assert 'n_candidates' in result.columns
    assert len(result) == 3
    # Point at (0,0) should have no matches
    assert result.iloc[2]['n_candidates'] == 0
    # Points inside flight lines should have at least one match
    assert result.iloc[0]['n_candidates'] >= 1
    assert result.iloc[1]['n_candidates'] >= 1


def test_match_returns_correct_metadata():
    stac = _make_stac_gdf()
    bedmap = _make_bedmap_gdf()
    result = match_bedmap_to_frames(stac, bedmap)
    for candidates in result['opr_candidates']:
        for seg, date, frame, item_id in candidates:
            assert date == '20091020'
            assert seg == 5


def test_find_contiguous_runs():
    # Single run
    runs = _find_contiguous_runs(np.array([3, 4, 5, 6]))
    assert len(runs) == 1
    assert len(runs[0]) == 4

    # Two runs, longest first
    runs = _find_contiguous_runs(np.array([1, 2, 3, 10, 11]))
    assert len(runs) == 2
    assert len(runs[0]) == 3
    assert len(runs[1]) == 2

    # Empty
    assert _find_contiguous_runs(np.array([], dtype=int)) == []


def test_build_frame_groups():
    stac = _make_stac_gdf()
    groups = _build_frame_groups(stac, group_size=5)
    # 2 frames with group_size=5 → 1 group containing both
    assert len(groups) == 1
    assert len(groups[0]) == 2
    frames_in_group = {f for _, _, f in groups[0]}
    assert frames_in_group == {1, 2}


def test_disambiguate_matches():
    stac = _make_stac_gdf()
    bedmap = _make_bedmap_gdf()
    result = match_bedmap_to_frames(stac, bedmap)
    layer = _make_layer_gdf()

    disambig = disambiguate_matches(result, layer, stac, group_size=5)

    assert 'assigned_segment_path' in disambig.columns
    assert 'assigned_frame' in disambig.columns
    assert 'nearest_distance_m' in disambig.columns
    assert 'run_id' in disambig.columns
    assert 'run_length' in disambig.columns
    # Point near frame 1
    assert disambig.iloc[0]['assigned_frame'] == 1
    # Point near frame 2
    assert disambig.iloc[1]['assigned_frame'] == 2
    # All distances should be finite
    assert np.all(np.isfinite(disambig['nearest_distance_m']))
