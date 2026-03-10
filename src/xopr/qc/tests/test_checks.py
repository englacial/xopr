"""Tests for QC check functions."""

from unittest.mock import MagicMock

import numpy as np
import pytest
import xarray as xr
from scipy.constants import c as speed_of_light

from xopr.qc.checks import (
    ensure_picks,
    heading_change,
    ice_thickness_threshold,
    minimum_agl,
    snr_bed_pick,
)


@pytest.fixture
def synthetic_ds():
    """Synthetic radar dataset with 100 traces and 200 fast-time samples."""
    n_traces = 100
    n_samples = 200
    twtt = np.linspace(0, 50e-6, n_samples)
    slow_time = np.arange(n_traces, dtype=float)

    # Build a Data array where the bed-pick sample is bright
    data = np.random.rand(n_traces, n_samples) * 0.01
    bottom_twtt_val = 40e-6
    bed_idx = np.searchsorted(twtt, bottom_twtt_val)
    data[:, bed_idx] = 10.0  # strong bed return

    ds = xr.Dataset(
        {
            "Data": (["slow_time", "twtt"], data),
            "standard:surface": ("slow_time", np.full(n_traces, 10e-6)),
            "standard:bottom": ("slow_time", np.full(n_traces, bottom_twtt_val)),
            "Latitude": ("slow_time", np.linspace(-75, -74, n_traces)),
            "Longitude": ("slow_time", np.linspace(100, 101, n_traces)),
            "Heading": ("slow_time", np.zeros(n_traces)),
        },
        coords={"slow_time": slow_time, "twtt": twtt},
    )
    return ds


# ---- ice_thickness_threshold -----------------------------------------


def _synthetic_thickness(surface_twtt=10e-6, bottom_twtt=40e-6, epsilon=3.15):
    v_ice = speed_of_light / np.sqrt(epsilon)
    return (bottom_twtt - surface_twtt) * v_ice / 2.0


def test_ice_thickness_all_pass(synthetic_ds):
    thickness = _synthetic_thickness()
    result = ice_thickness_threshold(synthetic_ds, min_thickness_m=thickness - 1)
    assert result["qc_ice_thickness_threshold"].all()
    assert result["qc"].all()


def test_ice_thickness_all_fail(synthetic_ds):
    thickness = _synthetic_thickness()
    result = ice_thickness_threshold(synthetic_ds, min_thickness_m=thickness + 1)
    assert not result["qc_ice_thickness_threshold"].any()


def test_ice_thickness_nan_bottom(synthetic_ds):
    synthetic_ds["standard:bottom"].values[50:] = np.nan
    thickness = _synthetic_thickness()
    result = ice_thickness_threshold(synthetic_ds, min_thickness_m=thickness - 1)
    assert result["qc_ice_thickness_threshold"][:50].all()
    assert not result["qc_ice_thickness_threshold"][50:].any()


def test_qc_variable_created(synthetic_ds):
    result = ice_thickness_threshold(synthetic_ds, min_thickness_m=0)
    assert "qc" in result
    np.testing.assert_array_equal(
        result["qc"].values, result["qc_ice_thickness_threshold"].values
    )


def test_qc_variable_accumulation(synthetic_ds):
    """Running a check twice with different data should AND the masks."""
    result = ice_thickness_threshold(synthetic_ds, min_thickness_m=0)
    result["standard:bottom"].values[50:] = result["standard:surface"].values[50:]
    thickness = _synthetic_thickness()
    result = ice_thickness_threshold(result, min_thickness_m=thickness - 1)
    assert result["qc"][:50].all()
    assert not result["qc"][50:].any()


def test_copy_semantics(synthetic_ds):
    result = ice_thickness_threshold(synthetic_ds, min_thickness_m=0)
    assert "qc" not in synthetic_ds
    assert "qc_ice_thickness_threshold" not in synthetic_ds
    assert "qc" in result


def test_missing_variable():
    ds = xr.Dataset({"standard:surface": ("slow_time", [1.0])})
    with pytest.raises(ValueError, match="standard:bottom"):
        ice_thickness_threshold(ds)


# ---- snr_bed_pick ----------------------------------------------------


def test_snr_bed_pick_all_pass(synthetic_ds):
    """Strong bed return should pass even a moderate SNR threshold."""
    result = snr_bed_pick(synthetic_ds, min_snr_db=5.0)
    assert result["qc_snr_bed_pick"].all()


def test_snr_bed_pick_all_fail(synthetic_ds):
    """Extremely high threshold should fail everything."""
    result = snr_bed_pick(synthetic_ds, min_snr_db=100.0)
    assert not result["qc_snr_bed_pick"].any()


def test_snr_bed_pick_nan_bottom(synthetic_ds):
    """NaN bottom picks should be flagged as failing."""
    synthetic_ds["standard:bottom"].values[50:] = np.nan
    result = snr_bed_pick(synthetic_ds, min_snr_db=5.0)
    assert not result["qc_snr_bed_pick"][50:].any()


def test_snr_bed_pick_missing_data():
    ds = xr.Dataset({"standard:bottom": ("slow_time", [1.0])})
    with pytest.raises(ValueError, match="Data"):
        snr_bed_pick(ds)


# ---- heading_change --------------------------------------------------


def test_heading_change_straight_pass(synthetic_ds):
    """Constant heading should always pass."""
    result = heading_change(synthetic_ds, max_deg_per_km=2.0)
    assert result["qc_heading_change"].all()


def test_heading_change_sharp_turn(synthetic_ds):
    """A sudden 90-degree turn should fail."""
    synthetic_ds["Heading"].values[50] = np.pi / 2
    result = heading_change(synthetic_ds, max_deg_per_km=0.01)
    # Trace 50 (and 51 due to backward diff) should fail
    assert not result["qc_heading_change"].values[50]


def test_heading_change_wraparound(synthetic_ds):
    """Heading wrapping from +π to −π should not produce a large change."""
    # Set all headings to just below +π then jump to just above -π
    synthetic_ds["Heading"].values[:] = np.pi - 0.001
    synthetic_ds["Heading"].values[50] = -np.pi + 0.001
    result = heading_change(synthetic_ds, max_deg_per_km=2.0)
    # The actual change is ~0.002 rad ≈ 0.1°, should pass
    assert result["qc_heading_change"].values[50]


def test_heading_change_missing_var():
    ds = xr.Dataset({"Latitude": ("slow_time", [1.0])})
    with pytest.raises(ValueError, match="Heading"):
        heading_change(ds)


# ---- minimum_agl -----------------------------------------------------


def test_minimum_agl_all_pass(synthetic_ds):
    """Surface TWTT of 10μs gives AGL ~1500m, should pass 100m threshold."""
    result = minimum_agl(synthetic_ds, min_agl_m=100.0)
    assert result["qc_minimum_agl"].all()


def test_minimum_agl_all_fail(synthetic_ds):
    """Setting threshold above the AGL should fail everything."""
    agl = 10e-6 * speed_of_light / 2.0  # ~1500m
    result = minimum_agl(synthetic_ds, min_agl_m=agl + 1)
    assert not result["qc_minimum_agl"].any()


def test_minimum_agl_partial(synthetic_ds):
    """Low surface TWTT traces should fail."""
    low_twtt = 2 * 50.0 / speed_of_light  # 50m AGL
    synthetic_ds["standard:surface"].values[50:] = low_twtt
    result = minimum_agl(synthetic_ds, min_agl_m=100.0)
    assert result["qc_minimum_agl"][:50].all()
    assert not result["qc_minimum_agl"][50:].any()


def test_minimum_agl_nan_surface(synthetic_ds):
    """NaN surface picks should fail."""
    synthetic_ds["standard:surface"].values[50:] = np.nan
    result = minimum_agl(synthetic_ds, min_agl_m=100.0)
    assert result["qc_minimum_agl"][:50].all()
    assert not result["qc_minimum_agl"][50:].any()


def test_minimum_agl_missing_var():
    ds = xr.Dataset({"Data": ("slow_time", [1.0])})
    with pytest.raises(ValueError, match="standard:surface"):
        minimum_agl(ds)


# ---- ensure_picks ----------------------------------------------------


def test_ensure_picks_already_present(synthetic_ds):
    result = ensure_picks(synthetic_ds)
    assert "standard:surface" in result
    assert "standard:bottom" in result


def test_ensure_picks_no_opr_raises():
    ds = xr.Dataset({"Data": ("slow_time", [1.0])},
                    coords={"slow_time": [0.0]})
    with pytest.raises(ValueError, match="OPRConnection"):
        ensure_picks(ds)


def test_ensure_picks_loads_bottom():
    n = 100
    slow_time = np.arange(n, dtype=float)
    ds = xr.Dataset(
        {"standard:surface": ("slow_time", np.full(n, 10e-6))},
        coords={"slow_time": slow_time},
    )
    bottom_vals = np.full(n, 40e-6)
    mock_opr = MagicMock()
    mock_opr.get_layers.return_value = {
        "standard:surface": xr.Dataset(
            {"twtt": ("slow_time", np.full(n, 10e-6))},
            coords={"slow_time": slow_time},
        ),
        "standard:bottom": xr.Dataset(
            {"twtt": ("slow_time", bottom_vals)},
            coords={"slow_time": slow_time},
        ),
    }
    result = ensure_picks(ds, opr=mock_opr)
    assert "standard:bottom" in result
    np.testing.assert_array_equal(result["standard:bottom"].values, bottom_vals)
    mock_opr.get_layers.assert_called_once()


def test_ensure_picks_loads_both():
    n = 50
    slow_time = np.arange(n, dtype=float)
    ds = xr.Dataset(
        {"Data": ("slow_time", np.ones(n))},
        coords={"slow_time": slow_time},
    )
    mock_opr = MagicMock()
    mock_opr.get_layers.return_value = {
        "standard:surface": xr.Dataset(
            {"twtt": ("slow_time", np.full(n, 10e-6))},
            coords={"slow_time": slow_time},
        ),
        "standard:bottom": xr.Dataset(
            {"twtt": ("slow_time", np.full(n, 40e-6))},
            coords={"slow_time": slow_time},
        ),
    }
    result = ensure_picks(ds, opr=mock_opr)
    assert "standard:surface" in result
    assert "standard:bottom" in result


def test_ensure_picks_no_layers_raises():
    ds = xr.Dataset({"Data": ("slow_time", [1.0])},
                    coords={"slow_time": [0.0]})
    mock_opr = MagicMock()
    mock_opr.get_layers.return_value = None
    with pytest.raises(ValueError, match="No layer data"):
        ensure_picks(ds, opr=mock_opr)


def test_ensure_picks_copy_semantics():
    n = 100
    slow_time = np.arange(n, dtype=float)
    ds = xr.Dataset(
        {"standard:surface": ("slow_time", np.full(n, 10e-6))},
        coords={"slow_time": slow_time},
    )
    mock_opr = MagicMock()
    mock_opr.get_layers.return_value = {
        "standard:surface": xr.Dataset(
            {"twtt": ("slow_time", np.full(n, 10e-6))},
            coords={"slow_time": slow_time},
        ),
        "standard:bottom": xr.Dataset(
            {"twtt": ("slow_time", np.full(n, 40e-6))},
            coords={"slow_time": slow_time},
        ),
    }
    ensure_picks(ds, opr=mock_opr)
    assert "standard:bottom" not in ds


def test_ensure_picks_alias_bottom():
    """Layers with ':bottom' alias are stored as 'standard:bottom'."""
    n = 50
    slow_time = np.arange(n, dtype=float)
    ds = xr.Dataset(
        {"standard:surface": ("slow_time", np.full(n, 10e-6))},
        coords={"slow_time": slow_time},
    )
    bottom_vals = np.full(n, 40e-6)
    mock_opr = MagicMock()
    mock_opr.get_layers.return_value = {
        "standard:surface": xr.Dataset(
            {"twtt": ("slow_time", np.full(n, 10e-6))},
            coords={"slow_time": slow_time},
        ),
        ":bottom": xr.Dataset(
            {"twtt": ("slow_time", bottom_vals)},
            coords={"slow_time": slow_time},
        ),
    }
    result = ensure_picks(ds, opr=mock_opr)
    assert "standard:bottom" in result
    assert ":bottom" not in result
    np.testing.assert_array_equal(result["standard:bottom"].values, bottom_vals)


def test_ensure_picks_alias_surface():
    """Layers with ':surface' alias are stored as 'standard:surface'."""
    n = 50
    slow_time = np.arange(n, dtype=float)
    ds = xr.Dataset(
        {"standard:bottom": ("slow_time", np.full(n, 40e-6))},
        coords={"slow_time": slow_time},
    )
    surface_vals = np.full(n, 10e-6)
    mock_opr = MagicMock()
    mock_opr.get_layers.return_value = {
        ":surface": xr.Dataset(
            {"twtt": ("slow_time", surface_vals)},
            coords={"slow_time": slow_time},
        ),
        "standard:bottom": xr.Dataset(
            {"twtt": ("slow_time", np.full(n, 40e-6))},
            coords={"slow_time": slow_time},
        ),
    }
    result = ensure_picks(ds, opr=mock_opr)
    assert "standard:surface" in result
    assert ":surface" not in result
    np.testing.assert_array_equal(result["standard:surface"].values, surface_vals)


def test_ensure_picks_alias_both():
    """Both layers via aliases are stored under canonical names."""
    n = 30
    slow_time = np.arange(n, dtype=float)
    ds = xr.Dataset(
        {"Data": ("slow_time", np.ones(n))},
        coords={"slow_time": slow_time},
    )
    mock_opr = MagicMock()
    mock_opr.get_layers.return_value = {
        ":surface": xr.Dataset(
            {"twtt": ("slow_time", np.full(n, 10e-6))},
            coords={"slow_time": slow_time},
        ),
        ":bottom": xr.Dataset(
            {"twtt": ("slow_time", np.full(n, 40e-6))},
            coords={"slow_time": slow_time},
        ),
    }
    result = ensure_picks(ds, opr=mock_opr)
    assert "standard:surface" in result
    assert "standard:bottom" in result


def test_ensure_picks_prefers_canonical():
    """Canonical name is preferred over alias when both exist."""
    n = 30
    slow_time = np.arange(n, dtype=float)
    ds = xr.Dataset(
        {"Data": ("slow_time", np.ones(n))},
        coords={"slow_time": slow_time},
    )
    canonical_vals = np.full(n, 10e-6)
    alias_vals = np.full(n, 99e-6)
    mock_opr = MagicMock()
    mock_opr.get_layers.return_value = {
        "standard:surface": xr.Dataset(
            {"twtt": ("slow_time", canonical_vals)},
            coords={"slow_time": slow_time},
        ),
        ":surface": xr.Dataset(
            {"twtt": ("slow_time", alias_vals)},
            coords={"slow_time": slow_time},
        ),
        "standard:bottom": xr.Dataset(
            {"twtt": ("slow_time", np.full(n, 40e-6))},
            coords={"slow_time": slow_time},
        ),
    }
    result = ensure_picks(ds, opr=mock_opr)
    np.testing.assert_array_equal(result["standard:surface"].values, canonical_vals)
