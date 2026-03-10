"""
QC runner — orchestrates multiple quality control checks on a dataset.
"""

import numpy as np
import xarray as xr

from .checks import (
    ensure_picks,
    heading_change,
    ice_thickness_threshold,
    minimum_agl,
    snr_bed_pick,
)

_CHECKS = {
    "ice_thickness_threshold": ice_thickness_threshold,
    "snr_bed_pick": snr_bed_pick,
    "heading_change": heading_change,
    "minimum_agl": minimum_agl,
}


def run_qc(ds, checks=None, opr=None):
    """
    Run one or more QC checks on a dataset.

    If ``standard:surface`` or ``standard:bottom`` variables are missing
    from the dataset, they are automatically loaded from layer picks via
    *opr*.

    Parameters
    ----------
    ds : xarray.Dataset
        Input radar dataset.
    checks : dict, optional
        Mapping of checks to run. Keys are either a registered check
        name (string matching a key in ``_CHECKS``) or a callable. Values
        are dicts of keyword arguments passed to the check function
        (use ``{}`` for defaults).

        Examples::

            # Registered check with default params
            {"ice_thickness_threshold": {}}

            # Registered check with custom params
            {"ice_thickness_threshold": {"min_thickness_m": 300}}

            # Custom function
            {my_custom_check: {"threshold": 0.5}}

            # Mix of both
            {"ice_thickness_threshold": {}, my_custom_check: {}}

        ``None`` (default) runs all registered checks with default
        parameters.
    opr : xopr.OPRConnection, optional
        Connection used to load layer picks when ``standard:surface`` or
        ``standard:bottom`` is not already in *ds*.

    Returns
    -------
    xarray.Dataset
        Dataset with QC mask variables added.

    Raises
    ------
    ValueError
        If a string key does not match any registered check, or if picks
        are missing and *opr* is ``None``.
    """
    if checks is None:
        checks = {name: {} for name in _CHECKS}

    # Resolve string keys to callables; validate up front
    resolved = []
    for key, kwargs in checks.items():
        if callable(key):
            resolved.append((key, kwargs))
        elif isinstance(key, str):
            if key not in _CHECKS:
                raise ValueError(
                    f"Unknown QC check '{key}'. "
                    f"Registered checks: {list(_CHECKS.keys())}"
                )
            resolved.append((_CHECKS[key], kwargs))
        else:
            raise TypeError(
                f"Check keys must be strings or callables, got {type(key)}"
            )

    ds = ensure_picks(ds, opr=opr)

    for fn, kwargs in resolved:
        ds = fn(ds, **kwargs)

    return ds
