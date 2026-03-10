"""Tests for QC runner."""

from unittest.mock import MagicMock

import numpy as np
import pytest
import xarray as xr

from xopr.qc.checks import _apply_qc_mask
from xopr.qc.runner import run_qc


@pytest.fixture
def synthetic_ds():
    n_traces = 100
    n_samples = 200
    twtt = np.linspace(0, 50e-6, n_samples)
    slow_time = np.arange(n_traces, dtype=float)

    return xr.Dataset(
        {
            "Data": (["slow_time", "twtt"], np.random.rand(n_traces, n_samples)),
            "standard:surface": ("slow_time", np.full(n_traces, 10e-6)),
            "standard:bottom": ("slow_time", np.full(n_traces, 40e-6)),
            "Latitude": ("slow_time", np.linspace(-75, -74, n_traces)),
            "Longitude": ("slow_time", np.linspace(100, 101, n_traces)),
            "Heading": ("slow_time", np.zeros(n_traces)),
        },
        coords={"slow_time": slow_time, "twtt": twtt},
    )


def test_run_qc_default(synthetic_ds):
    result = run_qc(synthetic_ds)
    assert "qc" in result
    assert "qc_ice_thickness_threshold" in result
    assert "qc_snr_bed_pick" in result
    assert "qc_heading_change" in result
    assert "qc_minimum_agl" in result


def test_run_qc_with_params(synthetic_ds):
    result = run_qc(
        synthetic_ds,
        checks={"ice_thickness_threshold": {"min_thickness_m": 0}},
    )
    assert result["qc"].all()


def test_run_qc_invalid_check(synthetic_ds):
    with pytest.raises(ValueError, match="Unknown QC check"):
        run_qc(synthetic_ds, checks={"nonexistent_check": {}})


def test_run_qc_callable_key(synthetic_ds):
    """A callable can be used as a check key."""
    def always_pass(ds):
        mask = xr.DataArray(np.ones(ds.sizes["slow_time"], dtype=bool),
                            dims="slow_time")
        return _apply_qc_mask(ds, mask, "always_pass")

    result = run_qc(synthetic_ds, checks={always_pass: {}})
    assert "qc_always_pass" in result
    assert result["qc"].all()


def test_run_qc_mixed_keys(synthetic_ds):
    """String and callable keys can be mixed."""
    def flag_none(ds):
        mask = xr.DataArray(np.zeros(ds.sizes["slow_time"], dtype=bool),
                            dims="slow_time")
        return _apply_qc_mask(ds, mask, "flag_none")

    result = run_qc(
        synthetic_ds,
        checks={
            "ice_thickness_threshold": {"min_thickness_m": 0},
            flag_none: {},
        },
    )
    # ice_thickness passes all, flag_none fails all → AND is all False
    assert not result["qc"].any()


def test_run_qc_bad_key_type(synthetic_ds):
    with pytest.raises(TypeError, match="strings or callables"):
        run_qc(synthetic_ds, checks={42: {}})


def test_run_qc_auto_loads_picks():
    """run_qc loads picks via opr when they are missing."""
    n_traces = 50
    n_samples = 100
    slow_time = np.arange(n_traces, dtype=float)
    ds = xr.Dataset(
        {
            "Data": (["slow_time", "twtt"], np.random.rand(n_traces, n_samples)),
            "Latitude": ("slow_time", np.linspace(-75, -74, n_traces)),
            "Longitude": ("slow_time", np.linspace(100, 101, n_traces)),
            "Heading": ("slow_time", np.zeros(n_traces)),
        },
        coords={
            "slow_time": slow_time,
            "twtt": np.linspace(0, 50e-6, n_samples),
        },
    )

    mock_opr = MagicMock()
    mock_opr.get_layers.return_value = {
        "standard:surface": xr.Dataset(
            {"twtt": ("slow_time", np.full(n_traces, 10e-6))},
            coords={"slow_time": slow_time},
        ),
        "standard:bottom": xr.Dataset(
            {"twtt": ("slow_time", np.full(n_traces, 40e-6))},
            coords={"slow_time": slow_time},
        ),
    }

    result = run_qc(ds, opr=mock_opr)
    assert "qc" in result
    assert "standard:bottom" in result
    assert "standard:surface" in result
    mock_opr.get_layers.assert_called_once()
