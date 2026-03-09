"""
QC runner — orchestrates multiple quality control checks on a dataset.
"""

from .checks import ensure_picks, ice_thickness_threshold

_CHECKS = {
    "ice_thickness_threshold": ice_thickness_threshold,
}


def run_qc(ds, checks=None, check_params=None, opr=None):
    """
    Run one or more QC checks on a dataset.

    If ``Surface`` or ``Bottom`` variables are missing from the dataset,
    they are automatically loaded from layer picks via *opr*.

    Parameters
    ----------
    ds : xarray.Dataset
        Input radar dataset.
    checks : list of str, optional
        Check names to run. ``None`` (default) runs all registered checks.
    check_params : dict[str, dict], optional
        Keyword arguments for individual checks, keyed by check name.
        For example: ``{"ice_thickness_threshold": {"min_thickness_m": 300}}``.
    opr : xopr.OPRConnection, optional
        Connection used to load layer picks when ``Surface`` or ``Bottom``
        is not already in *ds*.

    Returns
    -------
    xarray.Dataset
        Dataset with QC mask variables added.

    Raises
    ------
    ValueError
        If an unknown check name is provided, or if picks are missing
        and *opr* is ``None``.
    """
    if checks is None:
        checks = list(_CHECKS.keys())

    unknown = set(checks) - set(_CHECKS.keys())
    if unknown:
        raise ValueError(f"Unknown QC checks: {unknown}")

    if check_params is None:
        check_params = {}

    ds = ensure_picks(ds, opr=opr)

    for name in checks:
        kwargs = check_params.get(name, {})
        ds = _CHECKS[name](ds, **kwargs)

    return ds
