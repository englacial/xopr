"""Tests for morton geometry index computation."""

import numpy as np
import pytest

from xopr.stac.morton import _extract_coords, compute_mbox, compute_mpolygon_from_items


class TestExtractCoords:
    """Test _extract_coords for different geometry types."""

    def test_point(self):
        geom = {"type": "Point", "coordinates": [-69.86, -71.35]}
        lats, lons = _extract_coords(geom)
        assert lats == pytest.approx([-71.35])
        assert lons == pytest.approx([-69.86])

    def test_linestring(self):
        geom = {
            "type": "LineString",
            "coordinates": [[-69.86, -71.35], [-69.85, -71.36], [-69.84, -71.37]]
        }
        lats, lons = _extract_coords(geom)
        np.testing.assert_allclose(lats, [-71.35, -71.36, -71.37])
        np.testing.assert_allclose(lons, [-69.86, -69.85, -69.84])

    def test_polygon(self):
        geom = {
            "type": "Polygon",
            "coordinates": [[[-69.86, -71.35], [-69.85, -71.36],
                             [-69.84, -71.37], [-69.86, -71.35]]]
        }
        lats, lons = _extract_coords(geom)
        assert len(lats) == 4
        np.testing.assert_allclose(lats[0], -71.35)

    def test_multilinestring(self):
        geom = {
            "type": "MultiLineString",
            "coordinates": [
                [[-69.86, -71.35], [-69.85, -71.36]],
                [[-69.84, -71.37], [-69.83, -71.38]]
            ]
        }
        lats, lons = _extract_coords(geom)
        assert len(lats) == 4

    def test_unsupported_type_raises(self):
        with pytest.raises(ValueError, match="Unsupported geometry type"):
            _extract_coords({"type": "GeometryCollection", "coordinates": []})


class TestComputeMbox:
    """Test compute_mbox returns correct shape and types."""

    def test_returns_4_ints(self):
        geom = {
            "type": "LineString",
            "coordinates": [[-69.86, -71.35], [-69.85, -71.36], [-69.84, -71.37]]
        }
        result = compute_mbox(geom)
        assert len(result) == 4
        assert all(isinstance(v, int) for v in result)

    def test_antarctic_coords(self):
        """Test with representative Antarctic coordinates."""
        geom = {
            "type": "LineString",
            "coordinates": [
                [-60.0, -75.0], [-61.0, -75.5], [-62.0, -76.0],
                [-63.0, -76.5], [-64.0, -77.0]
            ]
        }
        result = compute_mbox(geom)
        assert len(result) == 4
        assert all(isinstance(v, int) for v in result)

    def test_point_geometry(self):
        geom = {"type": "Point", "coordinates": [-69.86, -71.35]}
        result = compute_mbox(geom)
        assert len(result) == 4


class TestComputeMpolygon:
    """Test compute_mpolygon_from_items returns correct shape and types."""

    def test_returns_12_ints(self):
        """Test with mock items having geometry attribute."""

        class MockItem:
            def __init__(self, geom):
                self.geometry = geom

        items = [
            MockItem({
                "type": "LineString",
                "coordinates": [[-69.86, -71.35], [-69.85, -71.36], [-69.84, -71.37]]
            }),
            MockItem({
                "type": "LineString",
                "coordinates": [[-70.0, -72.0], [-70.1, -72.1], [-70.2, -72.2]]
            }),
        ]
        result = compute_mpolygon_from_items(items)
        assert len(result) == 12
        assert all(isinstance(v, int) for v in result)

    def test_dict_items(self):
        """Test with dict-based items (as from item.to_dict())."""
        items = [
            {
                "geometry": {
                    "type": "LineString",
                    "coordinates": [[-69.86, -71.35], [-69.85, -71.36]]
                }
            },
            {
                "geometry": {
                    "type": "LineString",
                    "coordinates": [[-70.0, -72.0], [-70.1, -72.1]]
                }
            },
        ]
        result = compute_mpolygon_from_items(items)
        assert len(result) == 12
        assert all(isinstance(v, int) for v in result)
