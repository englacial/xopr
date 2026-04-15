"""Tests for xopr.crossovers — frame-to-granule mapping via morton cells."""

import geopandas as gpd
import numpy as np
import pytest
from mortie import geo_morton_polygon
from pyproj import Transformer
from shapely.geometry import LineString, Polygon

from xopr.crossovers import (
    build_granule_prefix_index,
    build_granule_strtree,
    cmr_bbox_from_mpolygon,
    compute_granule_sample_mortons,
    cycle_to_dates,
    match_frames_to_granules,
    match_frames_to_granules_prefix,
    resolve_temporal_window,
    subset_frames_by_points,
)


def _compute_mbox(line):
    coords = np.array(line.coords)
    lats, lons = coords[:, 1], coords[:, 0]
    cells = geo_morton_polygon(lats, lons, n_cells=4, order=18)
    result = [int(c.characteristic) for c in cells]
    while len(result) < 4:
        result.append(result[-1])
    return result


def _make_frames_gdf():
    """Two Antarctic frames with pre-computed opr:mbox."""
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
            'id': 'Data_20191020_05_001',
            'opr:segment': 5,
            'opr:date': '20191020',
            'opr:frame': 1,
            'opr:mbox': _compute_mbox(line1),
            'geometry': line1,
        },
        {
            'id': 'Data_20191020_05_002',
            'opr:segment': 5,
            'opr:date': '20191020',
            'opr:frame': 2,
            'opr:mbox': _compute_mbox(line2),
            'geometry': line2,
        },
    ]
    return gpd.GeoDataFrame(items, geometry='geometry', crs='EPSG:4326')


def _granule_from_bbox(gid, lon_min, lat_min, lon_max, lat_max):
    """Build a fake CMR UMM granule dict covering a lon/lat rectangle."""
    # CMR returns points as (Lat, Lon) in _extract_granule output
    points = [
        (lat_min, lon_min),
        (lat_min, lon_max),
        (lat_max, lon_max),
        (lat_max, lon_min),
        (lat_min, lon_min),
    ]
    return {
        'granule_id': gid,
        's3_url': f's3://fake-bucket/{gid}.h5',
        'https_url': f'https://fake.example/{gid}.h5',
        'points': points,
    }


def test_cmr_bbox_from_mpolygon():
    """mpolygon → bbox should bound the reconstructed cell union."""
    frames = _make_frames_gdf()
    # Union both frames' mbox cells to act as a pseudo-mpolygon
    mpolygon = []
    for mbox in frames['opr:mbox']:
        mpolygon.extend(mbox)

    bbox = cmr_bbox_from_mpolygon(mpolygon, step=16)
    lon_min, lat_min, lon_max, lat_max = bbox
    # Should enclose the frame endpoints
    assert lon_min <= -70 and lon_max >= -60
    assert lat_min <= -79 and lat_max >= -75


def test_build_granule_strtree_drops_invalid():
    """Granules missing s3_url or with <3 points are dropped."""
    granules = [
        _granule_from_bbox('g_ok', -70, -78, -65, -76),
        {'granule_id': 'g_no_url', 's3_url': None, 'https_url': None,
         'points': [(0, 0), (0, 1), (1, 1)]},
        {'granule_id': 'g_few_pts', 's3_url': 's3://x', 'https_url': 'h',
         'points': [(0, 0)]},
    ]
    tree, records = build_granule_strtree(granules)
    assert len(records) == 1
    assert records[0]['granule_id'] == 'g_ok'
    # STRtree should have one geometry
    assert tree.geometries.size == 1


def test_match_frames_to_granules_basic():
    """A granule covering frame 1 only should map to frame 1 only."""
    frames = _make_frames_gdf()
    granules = [
        # Covers frame 1's corridor (-70..-65 lon, -77..-75 lat)
        _granule_from_bbox('g_near_1', -71, -77.5, -64, -74.5),
        # Far away (northern hemisphere)
        _granule_from_bbox('g_far', 10, 10, 20, 20),
    ]
    tree, recs = build_granule_strtree(granules)
    result = match_frames_to_granules(frames, tree, recs, step=16)

    assert 'atl06_granules' in result.columns
    assert 'n_granules' in result.columns
    # Frame 1 should hit g_near_1; not g_far
    f1_ids = {g['granule_id'] for g in result.iloc[0]['atl06_granules']}
    assert 'g_near_1' in f1_ids
    assert 'g_far' not in f1_ids
    assert result.iloc[0]['n_granules'] == len(f1_ids)


def test_match_frames_to_granules_both_urls_preserved():
    frames = _make_frames_gdf()
    granules = [_granule_from_bbox('g1', -71, -80, -59, -74)]
    tree, recs = build_granule_strtree(granules)
    result = match_frames_to_granules(frames, tree, recs, step=16)

    for row in result['atl06_granules']:
        for g in row:
            assert g['s3_url'].startswith('s3://')
            assert g['https_url'].startswith('https://')


def test_match_frames_to_granules_union_across_mbox_cells():
    """Multiple granules each covering one mbox cell should all appear once."""
    frames = _make_frames_gdf()
    # Granule fully covering both frames at once
    g_all = _granule_from_bbox('g_all', -72, -80, -58, -74)
    tree, recs = build_granule_strtree([g_all])
    result = match_frames_to_granules(frames, tree, recs, step=16)

    # Each frame should see g_all exactly once, even though 4 mbox cells each hit it
    for row in result['atl06_granules']:
        ids = [g['granule_id'] for g in row]
        assert ids.count('g_all') == 1


def test_match_frames_missing_mbox_raises():
    frames = _make_frames_gdf().drop(columns=['opr:mbox'])
    tree, recs = build_granule_strtree([
        _granule_from_bbox('g', -70, -77, -65, -75)
    ])
    with pytest.raises(ValueError, match='opr:mbox'):
        match_frames_to_granules(frames, tree, recs)


def test_resolve_temporal_window_exact_year():
    frames = _make_frames_gdf()
    start, end = resolve_temporal_window(frames, mode='exact_year')
    assert start == '2019-10-20'
    assert end == '2019-10-20'


def test_resolve_temporal_window_all_years():
    frames = _make_frames_gdf()
    start, end = resolve_temporal_window(frames, mode='all_years')
    assert start == '2018-10-13'
    # end is "today" — just ensure ordering
    assert end > start


def test_resolve_temporal_window_exact_year_predates_icesat2():
    """Pre-2018 catalogs should refuse exact_year mode."""
    frames = _make_frames_gdf()
    frames['opr:date'] = '20091020'
    with pytest.raises(ValueError, match='before ICESat-2'):
        resolve_temporal_window(frames, mode='exact_year')


def test_cycle_to_dates():
    """Cycle 1 begins at launch; each cycle is 91 days."""
    start, end = cycle_to_dates(1)
    assert start == '2018-10-13'
    assert end == '2019-01-12'  # 91 days later
    start22, end22 = cycle_to_dates(22)
    # Cycle 22 matches magg's convention (launch + 21 cycles)
    assert start22 == '2024-01-06'


def test_resolve_temporal_window_cycle():
    start, end = resolve_temporal_window(cycle=22)
    expected_start, expected_end = cycle_to_dates(22)
    assert (start, end) == (expected_start, expected_end)


def test_resolve_temporal_window_date_range_strings():
    start, end = resolve_temporal_window(date_range=('2024-01-01', '2024-06-30'))
    assert start == '2024-01-01'
    assert end == '2024-06-30'


def test_resolve_temporal_window_date_range_date_objects():
    from datetime import date
    start, end = resolve_temporal_window(
        date_range=(date(2024, 1, 1), date(2024, 6, 30))
    )
    assert start == '2024-01-01'
    assert end == '2024-06-30'


def test_resolve_temporal_window_exact_year_requires_frames():
    with pytest.raises(ValueError, match='requires frames_gdf'):
        resolve_temporal_window(mode='exact_year')


def test_resolve_temporal_window_invalid_mode():
    frames = _make_frames_gdf()
    with pytest.raises(ValueError, match='exact_year'):
        resolve_temporal_window(frames, mode='bogus')


def test_compute_granule_sample_mortons():
    g = _granule_from_bbox('g', -70, -77, -65, -75)
    mortons = compute_granule_sample_mortons(g['points'], n_per_segment=20)
    assert len(mortons) > 0
    assert all(isinstance(m, str) for m in mortons)


def test_build_granule_prefix_index_drops_invalid():
    granules = [
        _granule_from_bbox('g_ok', -70, -78, -65, -76),
        {'granule_id': 'g_no_url', 's3_url': None, 'https_url': None,
         'points': [(0, 0), (0, 1), (1, 1)]},
    ]
    records = build_granule_prefix_index(granules, n_per_segment=20)
    ids = [r['granule_id'] for r in records]
    assert ids == ['g_ok']
    assert 'mort_strs' in records[0]
    assert len(records[0]['mort_strs']) > 0


def test_match_frames_to_granules_prefix_basic():
    frames = _make_frames_gdf()
    granules = [
        _granule_from_bbox('g_near_1', -71, -77.5, -64, -74.5),
        _granule_from_bbox('g_far', 10, 10, 20, 20),
    ]
    records = build_granule_prefix_index(granules, n_per_segment=200)
    result = match_frames_to_granules_prefix(frames, records)

    f1_ids = {g['granule_id'] for g in result.iloc[0]['atl06_granules']}
    assert 'g_near_1' in f1_ids
    assert 'g_far' not in f1_ids


def test_prefix_and_strtree_agree_on_far_granule():
    """Both backends must reject a granule that clearly doesn't overlap."""
    frames = _make_frames_gdf()
    granules = [
        _granule_from_bbox('g_near_1', -71, -77.5, -64, -74.5),
        _granule_from_bbox('g_far', 10, 10, 20, 20),
    ]
    tree, st_recs = build_granule_strtree(granules)
    st_result = match_frames_to_granules(frames, tree, st_recs, step=32)

    pf_recs = build_granule_prefix_index(granules, n_per_segment=200)
    pf_result = match_frames_to_granules_prefix(frames, pf_recs)

    for i in range(len(frames)):
        st_ids = {g['granule_id'] for g in st_result.iloc[i]['atl06_granules']}
        pf_ids = {g['granule_id'] for g in pf_result.iloc[i]['atl06_granules']}
        assert 'g_far' not in st_ids
        assert 'g_far' not in pf_ids
        # Both should agree that g_near_1 hits frame 1
        if i == 0:
            assert 'g_near_1' in st_ids
            assert 'g_near_1' in pf_ids


def test_match_frames_prefix_missing_mbox_raises():
    frames = _make_frames_gdf().drop(columns=['opr:mbox'])
    records = build_granule_prefix_index([
        _granule_from_bbox('g', -70, -77, -65, -75)
    ])
    with pytest.raises(ValueError, match='opr:mbox'):
        match_frames_to_granules_prefix(frames, records)


def test_subset_frames_by_points_basic():
    frames = _make_frames_gdf()
    # (-76.0, -67.5) is inside frame 1; (-78.0, -62.5) inside frame 2.
    # (0, 0) falls inside neither.
    pts = [(-76.0, -67.5), (-78.0, -62.5), (0.0, 0.0)]
    subset = subset_frames_by_points(frames, pts)
    ids = list(subset['id'])
    assert 'Data_20191020_05_001' in ids
    assert 'Data_20191020_05_002' in ids
    assert len(subset) == 2


def test_subset_frames_by_points_only_one():
    frames = _make_frames_gdf()
    subset = subset_frames_by_points(frames, [(-76.0, -67.5)])
    assert len(subset) == 1
    assert subset.iloc[0]['opr:frame'] == 1


def test_subset_frames_by_points_none_match():
    frames = _make_frames_gdf()
    subset = subset_frames_by_points(frames, [(0.0, 0.0)])
    assert len(subset) == 0
    assert list(subset.columns) == list(frames.columns)


def test_subset_frames_by_points_empty_input():
    frames = _make_frames_gdf()
    subset = subset_frames_by_points(frames, np.empty((0, 2)))
    assert len(subset) == 0


def test_subset_frames_by_points_accepts_numpy_array():
    frames = _make_frames_gdf()
    arr = np.array([[-76.0, -67.5]])
    subset = subset_frames_by_points(frames, arr)
    assert len(subset) == 1


def test_subset_frames_by_points_missing_mbox_raises():
    frames = _make_frames_gdf().drop(columns=['opr:mbox'])
    with pytest.raises(ValueError, match='opr:mbox'):
        subset_frames_by_points(frames, [(-76.0, -67.5)])


def test_subset_frames_by_points_bad_shape_raises():
    frames = _make_frames_gdf()
    with pytest.raises(ValueError, match=r'\(N, 2\)'):
        subset_frames_by_points(frames, [1, 2, 3])


def test_granule_polygon_reprojection_is_3031():
    """build_granule_strtree should place polygons in the requested CRS."""
    granules = [_granule_from_bbox('g', -70, -77, -65, -75)]
    tree, recs = build_granule_strtree(granules, crs='EPSG:3031')

    # Sanity: reproject one corner of the bbox with pyproj and confirm it's
    # inside the stored polygon's bounds.
    t = Transformer.from_crs('EPSG:4326', 'EPSG:3031', always_xy=True)
    x, y = t.transform(-67.5, -76.0)  # inside the bbox
    assert recs[0]['geometry'].contains(Polygon([(x-1, y-1), (x+1, y-1), (x+1, y+1), (x-1, y+1)]).centroid)
