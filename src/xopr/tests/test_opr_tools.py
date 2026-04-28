"""Tests for xopr.opr_tools utilities."""

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import Point

from xopr.opr_tools import compute_crossover_error


def _picks(elev, xs, ys, crs='EPSG:3031'):
    """Build a small picks GeoDataFrame for testing."""
    geom = [Point(x, y) for x, y in zip(xs, ys)]
    return gpd.GeoDataFrame({'wgs84': elev}, geometry=geom, crs=crs)


def test_compute_crossover_error_basic():
    """Two perpendicular tracks crossing at origin; nearest picks have known elevations."""
    p1 = _picks(elev=[-1000.0, -1010.0, -1020.0], xs=[-10.0, 0.0, 10.0], ys=[0.0, 0.0, 0.0])
    p2 = _picks(elev=[-2000.0, -2050.0, -2100.0], xs=[0.0, 0.0, 0.0], ys=[-10.0, 0.0, 10.0])
    e1, e2, d = compute_crossover_error(p1, p2, Point(0.0, 0.0))
    assert e1 == -1010.0
    assert e2 == -2050.0
    assert d == pytest.approx(0.0)


def test_compute_crossover_error_offset():
    """When picks don't land exactly on the crossover, distance is non-zero."""
    p1 = _picks(elev=[-1000.0], xs=[5.0], ys=[5.0])
    p2 = _picks(elev=[-2000.0], xs=[-5.0], ys=[-5.0])
    e1, e2, d = compute_crossover_error(p1, p2, Point(0.0, 0.0))
    assert e1 == -1000.0
    assert e2 == -2000.0
    assert d == pytest.approx(np.hypot(10.0, 10.0))


def test_compute_crossover_error_empty_picks():
    """Empty input returns NaN for every output."""
    p1 = _picks(elev=[], xs=[], ys=[])
    p2 = _picks(elev=[-2000.0], xs=[0.0], ys=[0.0])
    e1, e2, d = compute_crossover_error(p1, p2, Point(0.0, 0.0))
    assert all(np.isnan(v) for v in (e1, e2, d))


def test_compute_crossover_error_different_crs_raises():
    p1 = _picks(elev=[-1000.0], xs=[0.0], ys=[0.0], crs='EPSG:3031')
    p2 = _picks(elev=[-1000.0], xs=[0.0], ys=[0.0], crs='EPSG:3413')
    with pytest.raises(ValueError, match='different CRSs'):
        compute_crossover_error(p1, p2, Point(0.0, 0.0))


def test_compute_crossover_error_geographic_crs_raises():
    p1 = _picks(elev=[-1000.0], xs=[-60.0], ys=[-75.0], crs='EPSG:4326')
    p2 = _picks(elev=[-1000.0], xs=[-60.0], ys=[-75.0], crs='EPSG:4326')
    with pytest.raises(ValueError, match='geographic CRS'):
        compute_crossover_error(p1, p2, Point(-60.0, -75.0))


def test_compute_crossover_error_missing_column_raises():
    p1 = _picks(elev=[-1000.0], xs=[0.0], ys=[0.0])
    p2 = _picks(elev=[-1000.0], xs=[0.0], ys=[0.0])
    with pytest.raises(KeyError, match='missing'):
        compute_crossover_error(p1, p2, Point(0.0, 0.0), vertical='missing')
