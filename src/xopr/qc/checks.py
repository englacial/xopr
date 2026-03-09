"""
Quality control checks for polar radar datasets.

Each check function takes an xarray Dataset and returns a modified copy
with a per-trace boolean mask added as a new variable.
"""

import numpy as np
import xarray as xr
from scipy.constants import c as speed_of_light


_REQUIRED_LAYERS = ("standard:surface", "standard:bottom")


def ensure_picks(ds, opr=None):
    """
    Ensure ``standard:surface`` and ``standard:bottom`` variables exist.

    If either variable is missing, layer picks are loaded via
    ``opr.get_layers()`` and their twtt values are assigned to the
    dataset.

    Parameters
    ----------
    ds : xarray.Dataset
        Radar dataset, potentially missing required layers.
    opr : xopr.OPRConnection, optional
        An OPR connection used to fetch layers. Required only when
        required layers are missing from *ds*.

    Returns
    -------
    xarray.Dataset
        Copy of *ds* with ``standard:surface`` and ``standard:bottom``
        variables present.

    Raises
    ------
    ValueError
        If picks are missing and *opr* is ``None``, or if the required
        layers cannot be loaded.
    """
    if all(v in ds for v in _REQUIRED_LAYERS):
        return ds

    if opr is None:
        missing = [v for v in _REQUIRED_LAYERS if v not in ds]
        raise ValueError(
            f"Dataset is missing {missing} and no OPRConnection was "
            "provided to load layer picks. Pass an opr= argument or "
            "add the variables to the dataset manually."
        )

    ds = ds.copy()
    layers = opr.get_layers(ds)
    if layers is None:
        raise ValueError("No layer data found for this dataset.")

    for layer_key in _REQUIRED_LAYERS:
        if layer_key not in ds:
            if layer_key not in layers:
                raise ValueError(
                    f"Layer '{layer_key}' not found. "
                    f"Available layers: {list(layers.keys())}"
                )
            twtt = layers[layer_key]["twtt"]
            twtt = twtt.reindex(
                slow_time=ds.slow_time,
                method="nearest",
                tolerance=np.timedelta64(5, "s"),
                fill_value=np.nan,
            )
            ds[layer_key] = twtt

    return ds


def _apply_qc_mask(ds, mask, name):
    """
    Add a QC mask to a dataset and update the combined ``qc`` variable.

    Parameters
    ----------
    ds : xarray.Dataset
        Input dataset.
    mask : xarray.DataArray
        Boolean DataArray with dimension ``slow_time``. True means the
        trace passed the check.
    name : str
        Check name; stored as ``qc_{name}`` in the returned dataset.

    Returns
    -------
    xarray.Dataset
        Copy of *ds* with ``qc_{name}`` added. The combined ``qc``
        variable is the element-wise AND of all individual masks.
    """
    ds = ds.copy()
    var_name = f"qc_{name}"
    ds[var_name] = mask
    if "qc" in ds:
        ds["qc"] = ds["qc"] & mask
    else:
        ds["qc"] = mask.copy()
    return ds


def ice_thickness_threshold(ds, min_thickness_m=500.0, epsilon_ice=3.15):
    """
    Flag traces where ice thickness is below a minimum threshold.

    Ice thickness is computed from the ``standard:surface`` and
    ``standard:bottom`` two-way travel time picks using the radar wave
    speed in ice.

    Parameters
    ----------
    ds : xarray.Dataset
        Must contain ``standard:surface`` and ``standard:bottom``
        variables (units: seconds). Use :func:`ensure_picks` or pass
        ``opr`` to :func:`run_qc` to populate these automatically.
    min_thickness_m : float, optional
        Minimum ice thickness in metres. Default 500.
    epsilon_ice : float, optional
        Relative permittivity of ice. Default 3.15.

    Returns
    -------
    xarray.Dataset
        Copy with ``qc_ice_thickness_threshold`` and ``qc`` variables.

    Raises
    ------
    ValueError
        If ``standard:surface`` or ``standard:bottom`` is missing.
    """
    for var in _REQUIRED_LAYERS:
        if var not in ds:
            raise ValueError(f"Dataset is missing required variable '{var}'")

    v_ice = speed_of_light / np.sqrt(epsilon_ice)
    thickness = (ds["standard:bottom"] - ds["standard:surface"]) * v_ice / 2.0
    mask = xr.DataArray(
        thickness.values >= min_thickness_m,
        dims="slow_time",
    )
    # NaN picks produce NaN thickness → comparison is False
    return _apply_qc_mask(ds, mask, "ice_thickness_threshold")
