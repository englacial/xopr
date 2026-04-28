import pickle
import time
from unittest.mock import patch

import pytest
from rustac import DuckdbClient

import xopr
import xopr.geometry
from xopr.stac_cache import OPR_CATALOG_S3_GLOB

# Configuration flag for OPS database failure handling in tests.
# Set to 'warn' to fall back to file-based layers with a warning when db is unavailable.
# Set to 'error' to fail tests if db is unavailable (strict mode).
OPS_DB_FAILURE_MODE = 'warn'  # Options: 'warn', 'error'


def test_duckdb_client_reuse():
    """Test that OPRConnection reuses a single DuckdbClient across calls."""
    opr = xopr.OPRConnection()

    # Client is lazy-initialized on first access
    assert opr._duckdb_client is None
    client = opr.duckdb_client
    assert isinstance(client, DuckdbClient)

    # The same client instance should be used for multiple calls
    opr.get_collections()
    assert opr.duckdb_client is client

    opr.query_frames(collections=['2017_Antarctica_P3'],
                     segment_paths=['20171115_02'], max_items=1)
    assert opr.duckdb_client is client


def test_duckdb_client_pickling():
    """Test that OPRConnection can be pickled (required for Dask)."""
    opr = xopr.OPRConnection()
    opr.duckdb_client  # force initialization

    # Should pickle without error
    data = pickle.dumps(opr)
    opr2 = pickle.loads(data)

    # Client is reset to None after unpickling, recreated on next access
    assert opr2._duckdb_client is None
    assert isinstance(opr2.duckdb_client, DuckdbClient)
    assert opr2.collection_url == opr.collection_url


def test_get_collections():
    """
    Test that the get_collections function returns a non-empty list of collections.
    """
    opr = xopr.OPRConnection()
    collections = opr.get_collections()
    assert len(collections) > 0, "Expected non-empty list of collections"
    print(f"Found {len(collections)} collections: {collections}")
    for c in collections:
        assert isinstance(c['id'], str), f"Collection id should be a string, got {type(c['id'])}"

def test_get_segments(collection='2017_Antarctica_P3'):
    """
    Test that the get_segments function returns a non-empty list of segments.
    """
    opr = xopr.OPRConnection()
    segments = opr.get_segments(collection)
    assert len(segments) > 0, "Expected non-empty list of segments"
    print(f"Found {len(segments)} segments: {segments}")
    for s in segments:
        assert s['collection'] == collection, f"Segment collection mismatch: {s['collection']} != {collection}"
        assert 'segment_path' in s, "Segment dictionary should contain 'segment_path' key"

@pytest.mark.parametrize("collection,segment_path",
    [
        pytest.param('2022_Antarctica_BaslerMKB', '20230109_01', id='single_season_flight'),
        pytest.param(['2022_Antarctica_BaslerMKB'], '20230109_01', id='single_season_flight_list'),
        pytest.param(['2016_Antarctica_DC8', '2017_Antarctica_P3'], '20161117_06', id='multi_season_flight_list')
    ])
def test_load_season(collection, segment_path):
    """
    Test loading frames for a given season or list of collections.
    This checks if the frames can be loaded correctly and merged into a flight.
    It also verifies that layer information can be retrieved from either the layers db
    or from layer files.
    """
    print(f"Testing loading frames for season(s): {collection}")

    opr = xopr.OPRConnection()

    max_frames = 2
    frames = opr.query_frames(collections=collection, segment_paths=segment_path, max_items=max_frames)

    print(f"Found {len(frames)} frames for season(s) {collection} and segment path {segment_path}")

    assert len(frames) == max_frames, f"Expected {max_frames} frames, got {len(frames)}"

    flight = None
    for product_type in ['CSARP_qlook', 'CSARP_standard']:
        print(f"Loading frames for product type: {product_type}")
        loaded_frames = opr.load_frames(frames, data_product=product_type)
        assert len(loaded_frames) == max_frames, f"Loaded frames for {product_type} do not match expected count"

        merged = xopr.merge_frames(loaded_frames)
        if isinstance(merged, list):
            flight = merged[0]
        else:
            flight = merged

        collection_name = None
        if isinstance(collection, list):
            if len(collection) == 1:
                collection_name = collection[0]
        else:
            collection_name = collection

        if collection_name:
            assert flight.attrs['collection'] == collection_name, f"Flight collection attribute does not match expected: {flight.attrs['collection']} != {collection_name}"

        assert len(flight.attrs) > 0, "Merged flight should have attributes"

    # Test loading layers
    db_layers_loaded = False
    file_layers_loaded = False

    print("Loading layers from db...")
    layers = opr.get_layers(flight, source='db')
    if layers is not None and len(layers) > 0:
        db_layers_loaded = True
    else:
        if OPS_DB_FAILURE_MODE == 'error':
            raise ValueError("DB layers failed and OPS_DB_FAILURE_MODE is 'error'")
        # In 'warn' mode, continue to try file-based layers
        print("Loading layers from file...")
        layers = opr.get_layers(flight, source='files')
        if layers is not None and len(layers) > 0:
            file_layers_loaded = True

    assert db_layers_loaded or file_layers_loaded, "No layers loaded from either database or file"

def test_open_file_local_path_bypasses_cache(tmp_path):
    """Local filesystem paths should be returned as-is, no fsspec caching."""
    opr = xopr.OPRConnection(cache_dir=str(tmp_path))
    local_path = "/data/campaigns/2016_Antarctica_DC8/Data_001.mat"
    result = opr._open_file(local_path)
    assert result == local_path
    # Cache directory should remain empty — no files copied
    assert len(list(tmp_path.iterdir())) == 0


def test_open_file_remote_url_uses_cache():
    """Remote URLs should still go through fsspec caching."""
    opr = xopr.OPRConnection(cache_dir="/tmp/test_cache_unused")
    # Verify remote URLs get the filecache prefix (don't actually fetch)
    assert opr.fsspec_url_prefix == 'filecache::'
    # Local path should NOT get the prefix
    assert opr._open_file("/local/path.mat") == "/local/path.mat"


def test_cache_data(tmp_path):
    """
    Test that data is locally cached after loading.
    """

    n_frames = 2
    print(f"Testing caching of {n_frames} frames...")

    opr = xopr.OPRConnection(cache_dir=str(tmp_path))

    # List contents of the cache directory before loading
    initial_cache_contents = list(tmp_path.iterdir())
    print(f"Initial cache contents: {initial_cache_contents}")
    assert len(initial_cache_contents) == 0, "Cache directory should be empty before loading frames"

    collection, segment_path = '2016_Antarctica_DC8', '20161117_06'
    frames = opr.query_frames(collections=collection, segment_paths=segment_path, max_items=n_frames)
    assert len(frames) == n_frames, f"Expected {n_frames} frames for the given season and segment path, got {len(frames)}"

    tstart = time.time()
    loaded_frames = opr.load_frames(frames, data_product='CSARP_standard') # switching until the online catalog is rebuilt # _qlook')
    t_load_first = time.time() - tstart

    print(f"First load time: {t_load_first:.2f} seconds")
    print(f"Cache contents after first load: {list(tmp_path.iterdir())}")
    assert len(list(tmp_path.iterdir())) > 0, "Cache directory should not be empty after loading frames"

    assert len(loaded_frames) == n_frames, f"Expected {n_frames} loaded frames"

@pytest.mark.parametrize("query_params",
    [
        pytest.param({'collections': '2022_Antarctica_BaslerMKB', 'segment_paths': '20230109_01'}, id='single_season_flight'),
        pytest.param({'geometry': xopr.geometry.get_antarctic_regions(name=['LarsenD', 'LarsenE'])}, id='single_region_geometry'),
    ]
)
def test_exclude_geometry(query_params):

    max_items = 5

    opr = xopr.OPRConnection()

    items_with_geometry = opr.query_frames(**query_params, max_items=max_items)
    assert len(items_with_geometry) > 0, "Expected query to return items"

    items_without_geometry = opr.query_frames(**query_params, exclude_geometry=True, max_items=max_items)

    assert len(items_without_geometry) == len(items_with_geometry), "Expected same number of items with and without geometry"

    for item_id in items_with_geometry.index:
        w_geom = items_with_geometry.loc[item_id]
        wo_geom = items_without_geometry.loc[item_id]

        assert w_geom['geometry'] is not None, "Expected geometry to be present in items with geometry"
        assert ('geometry' not in wo_geom.keys()) or (wo_geom['geometry'] is None), "Expected geometry to be excluded in items without geometry"

        for key in wo_geom.keys():
            if key in ['geometry', 'links']:
                continue
            assert w_geom[key] == wo_geom[key], f"Expected {key} to match in both items, got {w_geom[key]} != {wo_geom[key]}"


# ---------------------------------------------------------------------------
# OPR catalog sync / caching tests
# ---------------------------------------------------------------------------


def test_sync_catalogs_false():
    """sync_catalogs=False should skip sync call."""
    with patch('xopr.opr_access.sync_opr_catalogs') as mock_sync:
        xopr.OPRConnection(sync_catalogs=False)
        mock_sync.assert_not_called()


def test_sync_catalogs_default():
    """Default construction calls sync_opr_catalogs."""
    with patch('xopr.opr_access.sync_opr_catalogs') as mock_sync:
        with patch('xopr.opr_access.get_opr_catalog_path', return_value=OPR_CATALOG_S3_GLOB):
            xopr.OPRConnection()
            mock_sync.assert_called_once()


def test_explicit_href_preserved():
    """User-provided stac_parquet_href should not trigger sync."""
    custom = "/my/custom/path/**/*.parquet"
    with patch('xopr.opr_access.sync_opr_catalogs') as mock_sync:
        opr = xopr.OPRConnection(stac_parquet_href=custom)
        assert opr.stac_parquet_href == custom
        mock_sync.assert_not_called()


def test_pickle_roundtrip():
    """Pickling OPRConnection preserves stac_parquet_href."""
    with patch('xopr.opr_access.sync_opr_catalogs'):
        with patch('xopr.opr_access.get_opr_catalog_path', return_value=OPR_CATALOG_S3_GLOB):
            opr = xopr.OPRConnection()
            data = pickle.dumps(opr)
            opr2 = pickle.loads(data)
            assert opr2._duckdb_client is None
            assert opr2.stac_parquet_href == opr.stac_parquet_href


# ---------------------------------------------------------------------------
# load_bed_picks
# ---------------------------------------------------------------------------


def _fake_layers(n=10, lat0=-75.0, lon0=-60.0):
    """Build a minimal {layer_name: xr.Dataset} dict matching get_layers' shape."""
    import numpy as np
    import xarray as xr

    slow_time = np.arange(n)
    lats = np.linspace(lat0, lat0 - 0.1, n)
    lons = np.linspace(lon0, lon0 + 0.1, n)

    surface = xr.Dataset(
        {
            'twtt': ('slow_time', np.full(n, 1e-6)),
            'elev': ('slow_time', np.full(n, 500.0)),
            'lat': ('slow_time', lats),
            'lon': ('slow_time', lons),
        },
        coords={'slow_time': slow_time},
    )
    bottom = xr.Dataset(
        {
            'twtt': ('slow_time', np.full(n, 5e-5)),
            'lat': ('slow_time', lats),
            'lon': ('slow_time', lons),
        },
        coords={'slow_time': slow_time},
    )
    return {'standard:surface': surface, 'standard:bottom': bottom}


def _fake_frames_gdf():
    """Two-frame GeoDataFrame in the query_frames shape (nested 'properties')."""
    import geopandas as gpd
    from shapely.geometry import LineString

    rows = [
        {
            'id': 'Data_20091020_05_001',
            'collection': '2009_Antarctica_DC8',
            'geometry': LineString([(-60.0, -75.0), (-59.9, -75.1)]),
            'properties': {
                'opr:date': '20091020', 'opr:segment': 5, 'opr:frame': 1,
                'opr:mbox': [10, 20, 30, 40],
            },
            'assets': {},
        },
        {
            'id': 'Data_20091020_05_002',
            'collection': '2009_Antarctica_DC8',
            'geometry': LineString([(-59.8, -75.2), (-59.7, -75.3)]),
            'properties': {
                'opr:date': '20091020', 'opr:segment': 5, 'opr:frame': 2,
                'opr:mbox': [50, 60, 70, 80],
            },
            'assets': {},
        },
    ]
    return gpd.GeoDataFrame(rows, geometry='geometry', crs='EPSG:4326').set_index('id', drop=False)


def _make_opr_with_mocked_layers():
    """OPRConnection with sync skipped and get_layers patched to a fake."""
    with patch('xopr.opr_access.sync_opr_catalogs'):
        with patch('xopr.opr_access.get_opr_catalog_path', return_value=OPR_CATALOG_S3_GLOB):
            opr = xopr.OPRConnection()
    opr.get_layers = lambda *a, **kw: _fake_layers()
    return opr


def test_load_bed_picks_geodataframe():
    """Bulk form: full GeoDataFrame of frames returns picks for every frame."""
    opr = _make_opr_with_mocked_layers()
    frames = _fake_frames_gdf()
    picks = opr.load_bed_picks(frames, target_crs='EPSG:3031', show_progress=False)

    expected_cols = {'geometry', 'wgs84', 'twtt', 'slow_time', 'id', 'collection',
                     'opr:date', 'opr:segment', 'opr:frame', 'segment_path'}
    assert expected_cols.issubset(set(picks.columns))
    assert len(picks) == 20  # 2 frames × 10 picks
    assert picks.crs.to_string() == 'EPSG:3031'
    assert set(picks['id'].unique()) == {'Data_20091020_05_001', 'Data_20091020_05_002'}
    assert (picks['segment_path'] == '20091020_05').all()


def test_load_bed_picks_series():
    """Single-row Series returns picks for that frame only."""
    opr = _make_opr_with_mocked_layers()
    frames = _fake_frames_gdf()
    one_row = frames.iloc[0]
    picks = opr.load_bed_picks(one_row, show_progress=False)

    assert len(picks) == 10
    assert (picks['id'] == 'Data_20091020_05_001').all()
    assert picks.crs.to_string() == 'EPSG:4326'  # no target_crs


def test_load_bed_picks_dict():
    """STAC-item-style dict input."""
    opr = _make_opr_with_mocked_layers()
    item_dict = {
        'id': 'Data_20091020_05_001',
        'collection': '2009_Antarctica_DC8',
        'properties': {'opr:date': '20091020', 'opr:segment': 5, 'opr:frame': 1},
        'assets': {},
    }
    picks = opr.load_bed_picks(item_dict, show_progress=False)
    assert len(picks) == 10
    assert (picks['opr:frame'] == 1).all()


def test_load_bed_picks_pystac_item_like():
    """pystac.Item-like object (anything with .to_dict() returning a STAC dict)."""
    class FakeItem:
        def to_dict(self):
            return {
                'id': 'Data_20091020_05_001',
                'collection': '2009_Antarctica_DC8',
                'properties': {'opr:date': '20091020', 'opr:segment': 5, 'opr:frame': 1},
                'assets': {},
            }

    opr = _make_opr_with_mocked_layers()
    picks = opr.load_bed_picks(FakeItem(), show_progress=False)
    assert len(picks) == 10


def test_load_bed_picks_keep_mbox():
    """keep_mbox=True attaches the source frame's opr:mbox to every pick."""
    opr = _make_opr_with_mocked_layers()
    frames = _fake_frames_gdf()
    picks = opr.load_bed_picks(frames, keep_mbox=True, show_progress=False)

    assert 'opr:mbox' in picks.columns
    f1 = picks[picks['id'] == 'Data_20091020_05_001']
    assert all(m == [10, 20, 30, 40] for m in f1['opr:mbox'])


def test_load_bed_picks_empty_when_no_layers():
    """When get_layers always returns None, the result is empty but well-formed."""
    with patch('xopr.opr_access.sync_opr_catalogs'):
        with patch('xopr.opr_access.get_opr_catalog_path', return_value=OPR_CATALOG_S3_GLOB):
            opr = xopr.OPRConnection()
    opr.get_layers = lambda *a, **kw: None

    picks = opr.load_bed_picks(_fake_frames_gdf(), show_progress=False)
    assert len(picks) == 0
    assert 'wgs84' in picks.columns
    assert 'segment_path' in picks.columns


def test_load_bed_picks_invalid_input():
    """Unsupported input types raise TypeError."""
    opr = _make_opr_with_mocked_layers()
    with pytest.raises(TypeError, match="frames must be"):
        opr.load_bed_picks(42, show_progress=False)
