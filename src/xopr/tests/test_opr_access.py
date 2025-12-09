import time

import pytest

import xopr
import xopr.geometry


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
    try:
        layers = opr.get_layers(flight, source='db')
        assert len(layers) > 1, "Expected layers to be loaded from database"
        db_layers_loaded = True
    except ValueError as e:
        pass

    if not db_layers_loaded:
        print("Loading layers from file...")
        layers = opr.get_layers(flight, source='files')
        print(layers)
        if len(layers) > 1:
            file_layers_loaded = True

    assert db_layers_loaded or file_layers_loaded, "No layers loaded from either database or file"

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
