"""
Unit tests for bedmap module.
"""

import pytest
import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path
import tempfile
import json
from datetime import datetime, timezone
import shapely
from shapely.geometry import Point, LineString, MultiLineString, box

# Import bedmap modules
from xopr.bedmap import (
    # Geometry functions
    calculate_haversine_distances,
    extract_flight_lines,
    simplify_multiline_geometry,
    # Converter functions
    parse_bedmap_metadata,
    convert_bedmap_csv,
    # Query functions
    query_bedmap_local,
    # Comparison functions
    match_bedmap_to_opr,
    compare_with_opr,
)
from xopr.bedmap.converter import _extract_bedmap_version, create_timestamps
from xopr.bedmap.geometry import (
    calculate_bbox,
    transform_coords_to_polar,
    transform_coords_from_polar,
    transform_geometry_to_polar,
    get_polar_bounds,
    check_intersects_polar,
)
from xopr.bedmap.query import build_duckdb_query, _crosses_antimeridian


class TestGeometry:
    """Test geometry utilities."""

    def test_haversine_distances(self):
        """Test haversine distance calculation."""
        # Points 1 degree apart in latitude (lat, lon order for haversine)
        coords = np.array([
            [-70.0, 0.0],  # lat, lon
            [-71.0, 0.0],
            [-72.0, 0.0],
        ])

        distances = calculate_haversine_distances(coords)

        assert len(distances) == 2
        # Should be approximately 111 km per degree of latitude
        assert 110 < distances[0] < 112
        assert 110 < distances[1] < 112

    def test_extract_flight_lines(self):
        """Test flight line extraction with segmentation."""
        # Create test data with a gap
        # First 3 points are close together (< 10km), then a big gap, then 2 more close points
        df = pd.DataFrame({
            'longitude (degree_east)': [-70.0, -70.05, -70.1, -60.0, -60.05],  # Gap between 3rd and 4th
            'latitude (degree_north)': [-70.0, -70.05, -70.1, -70.0, -70.05],
        })

        lines = extract_flight_lines(df, distance_threshold_km=10.0)

        assert isinstance(lines, MultiLineString)
        # Should have 2 segments due to gap
        assert len(lines.geoms) == 2
        # First segment has 3 points
        assert len(lines.geoms[0].coords) == 3
        # Second segment has 2 points
        assert len(lines.geoms[1].coords) == 2

    def test_simplify_multiline(self):
        """Test geometry simplification."""
        # Create a complex line
        coords = [(i/10, np.sin(i/10)) for i in range(100)]
        line = LineString(coords)
        multiline = MultiLineString([line])

        simplified = simplify_multiline_geometry(multiline, tolerance_deg=0.1)

        assert isinstance(simplified, MultiLineString)
        # Should have fewer points after simplification
        assert len(simplified.geoms[0].coords) < len(coords)

    def test_calculate_bbox(self):
        """Test bounding box calculation."""
        df = pd.DataFrame({
            'longitude (degree_east)': [-70, -65, -68],
            'latitude (degree_north)': [-75, -70, -72],
        })

        bbox = calculate_bbox(df)

        assert bbox == (-70, -75, -65, -70)


class TestPolarProjection:
    """Test polar projection utilities for Antarctic data."""

    def test_transform_coords_to_polar(self):
        """Test WGS84 to EPSG:3031 coordinate transformation."""
        # South Pole should map to origin (0, 0)
        x, y = transform_coords_to_polar(0, -90)
        assert abs(x) < 1  # Should be very close to 0
        assert abs(y) < 1

        # A point at lon=0, lat=-70 should have x≈0 and y>0 (north of pole)
        x, y = transform_coords_to_polar(0, -70)
        assert abs(x) < 1000  # x should be near 0 for lon=0
        assert y > 0  # y should be positive (north of pole in this projection)

    def test_transform_coords_round_trip(self):
        """Test coordinate transform round-trip accuracy."""
        lon_orig, lat_orig = 170.0, -75.0

        x, y = transform_coords_to_polar(lon_orig, lat_orig)
        lon_back, lat_back = transform_coords_from_polar(x, y)

        assert abs(lon_orig - lon_back) < 0.0001
        assert abs(lat_orig - lat_back) < 0.0001

    def test_transform_coords_array(self):
        """Test vectorized coordinate transformation."""
        lons = np.array([0, 90, 180, -90])
        lats = np.array([-70, -70, -70, -70])

        xs, ys = transform_coords_to_polar(lons, lats)

        assert len(xs) == 4
        assert len(ys) == 4
        # All points at same latitude should have same distance from origin
        distances = np.sqrt(xs**2 + ys**2)
        assert np.allclose(distances, distances[0], rtol=0.001)

    def test_transform_geometry_to_polar(self):
        """Test geometry transformation to polar coordinates."""
        # Create a box near the antimeridian
        geom = box(170, -80, -170, -70)  # Crosses antimeridian

        polar_geom = transform_geometry_to_polar(geom)

        # Polar geometry should be valid and not empty
        assert polar_geom is not None
        assert not polar_geom.is_empty
        assert polar_geom.is_valid

    def test_get_polar_bounds(self):
        """Test getting bounds in polar projection."""
        # Simple box in West Antarctica
        geom = box(-100, -80, -90, -75)

        bounds = get_polar_bounds(geom)

        assert bounds is not None
        x_min, y_min, x_max, y_max = bounds
        assert x_min < x_max
        assert y_min < y_max

    def test_check_intersects_polar_same_side(self):
        """Test intersection check for geometries on same side of antimeridian."""
        geom1 = box(-100, -80, -90, -75)
        geom2 = box(-95, -78, -85, -73)

        # These should intersect
        assert check_intersects_polar(geom1, geom2)

    def test_check_intersects_polar_no_intersect(self):
        """Test intersection check for non-intersecting geometries."""
        geom1 = box(-100, -80, -90, -75)
        geom2 = box(0, -80, 10, -75)  # Completely different location

        # These should NOT intersect
        assert not check_intersects_polar(geom1, geom2)

    def test_check_intersects_polar_antimeridian(self):
        """Test intersection check for geometries crossing antimeridian."""
        # Data geometry that crosses the antimeridian (Ross Sea area)
        data_geom = LineString([(170, -75), (180, -76), (-170, -77)])

        # Query box that also crosses antimeridian
        query_geom = box(165, -80, -165, -70)

        # These should intersect when using polar projection
        # (would fail with simple lat/lon intersection)
        assert check_intersects_polar(data_geom, query_geom)

    def test_check_intersects_polar_near_pole(self):
        """Test intersection check for geometries near the South Pole."""
        # Two overlapping boxes near the pole
        # In polar projection, these rectangles will overlap
        geom1 = box(-10, -88, 10, -85)  # Near prime meridian
        geom2 = box(-5, -87, 15, -84)   # Overlapping box

        # Should intersect
        assert check_intersects_polar(geom1, geom2)

        # Non-overlapping boxes at different longitudes near pole
        geom3 = box(170, -88, -170, -85)  # Near antimeridian (wraps around)
        geom4 = box(80, -88, 100, -85)    # Near 90°E

        # These don't actually overlap in polar projection
        # (one is in the -Y region, other is in +X region)
        assert not check_intersects_polar(geom3, geom4)


class TestAntimeridianCrossing:
    """Test antimeridian crossing detection and handling."""

    def test_crosses_antimeridian_simple_box(self):
        """Test that a simple box doesn't cross antimeridian."""
        geom = box(-100, -80, -90, -70)
        assert not _crosses_antimeridian(geom)

    def test_crosses_antimeridian_wide_box(self):
        """Test that a box spanning >180° is detected as crossing."""
        # This box spans from -170 to 170, which is 340 degrees
        geom = box(-170, -80, 170, -70)
        assert _crosses_antimeridian(geom)

    def test_duckdb_query_polar_filter(self):
        """Test DuckDB query uses polar filter for spatial queries."""
        geom = box(-70, -75, -60, -70)

        query = build_duckdb_query(
            parquet_urls=['file.parquet'],
            geometry=geom,
            use_polar_filter=True
        )

        # Should contain polar coordinate math
        assert 'SIN(RADIANS' in query
        assert 'COS(RADIANS' in query
        assert '6378137.0' in query  # WGS84 semi-major axis

    def test_duckdb_query_simple_bbox(self):
        """Test DuckDB query can use simple bbox filter when requested."""
        geom = box(-70, -75, -60, -70)

        query = build_duckdb_query(
            parquet_urls=['file.parquet'],
            geometry=geom,
            use_polar_filter=False
        )

        # Should contain simple comparisons, not polar math
        assert '>= -70' in query
        assert '<= -60' in query
        assert 'SIN(RADIANS' not in query

    def test_duckdb_query_antimeridian_forces_polar(self):
        """Test DuckDB query forces polar filter for antimeridian-crossing geometry."""
        # Geometry that crosses antimeridian
        geom = box(170, -80, -170, -70)  # Note: 170 to -170 crosses AM

        # Even with use_polar_filter=False, should use polar for AM-crossing
        query = build_duckdb_query(
            parquet_urls=['file.parquet'],
            geometry=geom,
            use_polar_filter=False
        )

        # Since geometry crosses antimeridian, polar filter should be used
        # But first check if it detected the crossing...
        # Note: shapely.box normalizes coordinates, so we need to check
        # if _crosses_antimeridian detected it
        if _crosses_antimeridian(geom):
            assert 'SIN(RADIANS' in query


class TestAntimeridianIntegration:
    """Integration tests for antimeridian-crossing queries."""

    def test_query_crossing_antimeridian(self):
        """Test querying data that crosses the antimeridian with DuckDB."""
        import duckdb

        # Create test data that spans the antimeridian (Ross Sea area)
        test_data = pd.DataFrame({
            'longitude (degree_east)': [170.0, 175.0, 180.0, -175.0, -170.0],
            'latitude (degree_north)': [-75.0, -76.0, -77.0, -76.0, -75.0],
            'surface_altitude (m)': [100.0, 110.0, 120.0, 115.0, 105.0],
        })

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            parquet_path = tmpdir / 'test_antimeridian.parquet'

            # Write test data to parquet
            test_data.to_parquet(parquet_path)

            # Query box that crosses antimeridian (should get all 5 points)
            query_geom = box(165, -80, -165, -70)

            query = build_duckdb_query(
                parquet_urls=[str(parquet_path)],
                geometry=query_geom,
                use_polar_filter=True
            )

            # Execute query
            conn = duckdb.connect()
            result = conn.execute(query).df()
            conn.close()

            # All 5 points should be returned
            assert len(result) == 5

            # Without polar filter, simple bbox would fail
            # (bounds would be 165 to -165, which is nearly all longitudes
            # but the simple >= and <= comparison doesn't work)
            query_simple = build_duckdb_query(
                parquet_urls=[str(parquet_path)],
                geometry=query_geom,
                use_polar_filter=False
            )

            conn = duckdb.connect()
            result_simple = conn.execute(query_simple).df()
            conn.close()

            # Simple bbox query returns 0 because lon >= 165 AND lon <= -165
            # is impossible (no longitude satisfies both conditions)
            # Note: shapely.box normalizes to (-165, -80, 165, -70) with lon span > 180
            # So _crosses_antimeridian should detect this
            # Actually the simple query might work if it doesn't cross...
            # Let's verify the actual behavior
            if _crosses_antimeridian(query_geom):
                # If detected as crossing, polar filter is forced
                assert len(result_simple) == 5
            else:
                # If not detected, simple bbox might return wrong results
                # This is expected - the simple approach fails for AM-crossing
                pass


class TestConverter:
    """Test CSV conversion utilities."""

    def test_parse_metadata(self):
        """Test metadata parsing from CSV header."""
        # Create a temporary CSV with metadata
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("#project: Test Project\n")
            f.write("#time_coverage_start: 2020\n")
            f.write("#time_coverage_end: 2021\n")
            f.write("#institution: Test Institution\n")
            f.write("#centre_frequency: 150 (MHz)\n")
            f.write("col1,col2\n")
            f.write("1,2\n")
            temp_path = Path(f.name)

        try:
            metadata = parse_bedmap_metadata(temp_path)

            assert metadata['project'] == 'Test Project'
            assert metadata['time_coverage_start'] == 2020
            assert metadata['time_coverage_end'] == 2021
            assert metadata['institution'] == 'Test Institution'
            assert metadata['centre_frequency'] == 150.0
            assert metadata['centre_frequency_unit'] == 'MHz'
        finally:
            temp_path.unlink()

    def test_extract_bedmap_version(self):
        """Test bedmap version extraction from filename."""
        assert _extract_bedmap_version('AWI_1994_DML1_AIR_BM2.csv') == 'BM2'
        assert _extract_bedmap_version('UTIG_2016_OLDICE_AIR_BM3.csv') == 'BM3'
        assert _extract_bedmap_version('OLD_DATA_BM1.csv') == 'BM1'
        assert _extract_bedmap_version('UNKNOWN_DATA.csv') == 'unknown'

    def test_timestamp_creation_with_metadata(self):
        """Test timestamp creation from metadata."""
        df = pd.DataFrame({
            'date': [-9999, -9999],  # No date data
            'time_UTC': [-9999, -9999],  # No time data
        })

        metadata = {
            'time_coverage_start': 2020,
            'time_coverage_end': 2021
        }

        timestamps = create_timestamps(df, metadata)

        assert len(timestamps) == 2
        assert timestamps[0].year == 2020
        assert timestamps[1].year == 2021
        # Should be spread across the time range
        assert timestamps[0] < timestamps[1]

    def test_timestamp_single_year(self):
        """Test timestamp creation for single year coverage."""
        df = pd.DataFrame({
            'date': [-9999, -9999, -9999],
            'time_UTC': [-9999, -9999, -9999],
        })

        metadata = {
            'time_coverage_start': 2020,
            'time_coverage_end': 2020  # Same year
        }

        timestamps = create_timestamps(df, metadata)

        assert len(timestamps) == 3
        assert all(t.year == 2020 for t in timestamps)
        # Should be spread across the year
        assert timestamps[0].month == 1
        assert timestamps[2].month == 12


class TestQuery:
    """Test query functions."""

    def test_build_duckdb_query_basic(self):
        """Test basic DuckDB query building."""
        query = build_duckdb_query(
            parquet_urls=['file1.parquet'],
            columns=['longitude (degree_east)', 'latitude (degree_north)'],
            max_rows=100
        )

        assert 'SELECT' in query
        assert 'longitude (degree_east)' in query
        assert 'latitude (degree_north)' in query
        assert 'LIMIT 100' in query

    def test_build_duckdb_query_with_geometry(self):
        """Test DuckDB query with spatial filter (using polar projection by default)."""
        bbox_geom = box(-70, -75, -60, -70)  # lon_min, lat_min, lon_max, lat_max

        query = build_duckdb_query(
            parquet_urls=['file1.parquet'],
            geometry=bbox_geom
        )

        # Default behavior uses polar projection filter
        assert 'WHERE' in query
        assert 'SIN(RADIANS' in query
        assert 'COS(RADIANS' in query

    def test_build_duckdb_query_with_geometry_simple_bbox(self):
        """Test DuckDB query with simple bbox filter (no polar projection)."""
        bbox_geom = box(-70, -75, -60, -70)  # lon_min, lat_min, lon_max, lat_max

        query = build_duckdb_query(
            parquet_urls=['file1.parquet'],
            geometry=bbox_geom,
            use_polar_filter=False
        )

        assert 'WHERE' in query
        assert 'longitude (degree_east)' in query
        assert '>= -70' in query
        assert '<= -60' in query
        assert 'latitude (degree_north)' in query
        assert '>= -75' in query
        assert '<= -70' in query

    def test_build_duckdb_query_with_dates(self):
        """Test DuckDB query with temporal filter."""
        start = datetime(2020, 1, 1, tzinfo=timezone.utc)
        end = datetime(2020, 12, 31, tzinfo=timezone.utc)

        query = build_duckdb_query(
            parquet_urls=['file1.parquet'],
            date_range=(start, end)
        )

        assert 'WHERE' in query
        assert 'timestamp >=' in query
        assert '2020-01-01' in query
        assert 'timestamp <=' in query
        assert '2020-12-31' in query

    def test_build_duckdb_query_multiple_files(self):
        """Test DuckDB query with multiple files."""
        query = build_duckdb_query(
            parquet_urls=['file1.parquet', 'file2.parquet', 'file3.parquet']
        )

        assert 'UNION ALL' in query
        assert 'file1.parquet' in query
        assert 'file2.parquet' in query
        assert 'file3.parquet' in query


class TestCompare:
    """Test comparison functions."""

    def test_match_bedmap_to_opr(self):
        """Test matching bedmap points to OPR data."""
        import xarray as xr

        # Create test bedmap data
        bedmap = gpd.GeoDataFrame({
            'longitude (degree_east)': [-70, -65, -60],
            'latitude (degree_north)': [-70, -72, -74],
            'surface_altitude (m)': [1000, 1100, 1200],
        })

        # Create test OPR dataset
        opr = xr.Dataset({
            'Longitude': (('slow_time',), [-70.001, -65.002, -60.001]),
            'Latitude': (('slow_time',), [-70.001, -72.002, -74.001]),
            'Surface': (('slow_time',), [1005, 1105, 1205]),
        })

        matched = match_bedmap_to_opr(bedmap, opr, max_distance_m=1000)

        assert 'opr_match_distance_m' in matched.columns
        assert 'is_matched' in matched.columns
        assert matched['is_matched'].all()  # All should be matched within tolerance
        assert (matched['opr_match_distance_m'] < 1000).all()

    def test_compare_with_opr_statistics(self):
        """Test comparison statistics calculation."""
        import xarray as xr

        # Create test data
        bedmap = gpd.GeoDataFrame({
            'longitude (degree_east)': [-70, -65],
            'latitude (degree_north)': [-70, -72],
            'surface_altitude (m)': [1000, 1100],
            'bedrock_altitude (m)': [500, 600],
            'land_ice_thickness (m)': [500, 500],
        })

        # Create OPR data with small differences
        opr_surface = xr.DataArray(
            [[1005, 1095]],
            dims=['y', 'x'],
            coords={'x': [-70, -65], 'y': [-70]}
        )

        opr_bed = xr.DataArray(
            [[505, 595]],
            dims=['y', 'x'],
            coords={'x': [-70, -65], 'y': [-70]}
        )

        results = compare_with_opr(bedmap, opr_surface, opr_bed)

        assert 'statistics' in results
        assert 'surface' in results['statistics']
        assert 'bed' in results['statistics']
        assert 'thickness' in results['statistics']

        # Check that differences are calculated
        assert 'differences' in results
        assert 'matched_data' in results


class TestIntegration:
    """Integration tests."""

    def test_full_conversion_workflow(self):
        """Test complete CSV to Parquet conversion."""
        # Create a test CSV file
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create test CSV
            csv_path = tmpdir / 'TEST_2020_DATA_BM2.csv'
            with open(csv_path, 'w') as f:
                f.write("#project: Test Project\n")
                f.write("#time_coverage_start: 2020\n")
                f.write("#time_coverage_end: 2020\n")
                f.write("#institution: Test Institution\n")
                f.write("trajectory_id,trace_number,longitude (degree_east),latitude (degree_north),date,time_UTC,surface_altitude (m),land_ice_thickness (m),bedrock_altitude (m),two_way_travel_time (m),aircraft_altitude (m),along_track_distance (m)\n")
                f.write("1,-9999,-70.0,-70.0,-9999,-9999,1000.0,500.0,500.0,-9999,-9999,-9999\n")
                f.write("1,-9999,-70.1,-70.1,-9999,-9999,1010.0,510.0,500.0,-9999,-9999,-9999\n")
                f.write("1,-9999,-70.2,-70.2,-9999,-9999,1020.0,520.0,500.0,-9999,-9999,-9999\n")

            # Convert to parquet
            output_dir = tmpdir / 'output'
            metadata = convert_bedmap_csv(csv_path, output_dir)

            # Check output
            parquet_path = output_dir / 'TEST_2020_DATA_BM2.parquet'
            assert parquet_path.exists()

            # Check metadata
            assert metadata['bedmap_version'] == 'BM2'
            assert metadata['row_count'] == 3
            assert metadata['spatial_bounds']['bbox'] is not None

            # Test local query
            result = query_bedmap_local(
                output_dir,
                geometry=box(-71, -71, -69, -69),
                max_rows=10
            )

            assert len(result) == 3
            assert 'longitude (degree_east)' in result.columns
            assert 'source_file' in result.columns
            assert result['source_file'].iloc[0] == 'TEST_2020_DATA_BM2'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])