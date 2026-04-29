"""Morton geometry index computation for STAC items and collections.

Defines the two index primitives stored on xopr STAC objects:

- ``opr:mbox`` — 4 variable-resolution morton cells per item (see
  :func:`compute_mbox`).
- ``opr:mpolygon`` — 12 variable-resolution morton cells per collection
  (see :func:`compute_mpolygon_from_items`).

Plus :func:`mbox_to_polygons` for materializing the cells back into
shapely Polygons (e.g. for visualization or exact intersection tests).
"""

import numpy as np
from mortie import geo2mort, geo_morton_polygon, morton_polygon_from_array
from mortie.tools import mort2polygon
from shapely import make_valid
from shapely.geometry import Polygon


def _extract_coords(geometry):
    """Extract (lats, lons) from a GeoJSON geometry dict.

    Parameters
    ----------
    geometry : dict
        GeoJSON geometry with 'type' and 'coordinates' keys.
        Supports Point, LineString, Polygon, and MultiLineString.

    Returns
    -------
    lats : ndarray
    lons : ndarray
    """
    geom_type = geometry['type']
    coords = geometry['coordinates']

    if geom_type == 'Point':
        return np.array([coords[1]]), np.array([coords[0]])
    elif geom_type == 'LineString':
        arr = np.asarray(coords)
        return arr[:, 1], arr[:, 0]
    elif geom_type == 'Polygon':
        arr = np.asarray(coords[0])  # exterior ring
        return arr[:, 1], arr[:, 0]
    elif geom_type == 'MultiLineString':
        arrays = [np.asarray(line) for line in coords]
        arr = np.concatenate(arrays, axis=0)
        return arr[:, 1], arr[:, 0]
    else:
        raise ValueError(f"Unsupported geometry type: {geom_type}")


def _pad_cells(cells, target):
    """Pad cell list to exact target length by repeating the last cell."""
    result = [int(c.characteristic) for c in cells]
    while len(result) < target:
        result.append(result[-1])
    return result


def compute_mbox(geometry, order=18):
    """Compute a morton bounding box (4 cells) from a GeoJSON geometry.

    Parameters
    ----------
    geometry : dict
        GeoJSON geometry dict.
    order : int, optional
        Morton tessellation order, by default 18.

    Returns
    -------
    list of int
        List of exactly 4 characteristic integers. If the geometry is too
        compact to produce 4 distinct cells, the last cell is repeated.
    """
    lats, lons = _extract_coords(geometry)
    cells = geo_morton_polygon(lats, lons, n_cells=4, order=order)
    return _pad_cells(cells, 4)


def compute_mpolygon_from_items(items, order=18):
    """Compute a morton polygon (12 cells) from a list of STAC items.

    Parameters
    ----------
    items : list of pystac.Item or list of dict
        STAC items with geometry.
    order : int, optional
        Morton tessellation order, by default 18.

    Returns
    -------
    list of int
        List of exactly 12 characteristic integers. If the geometry is too
        compact to produce 12 distinct cells, the last cell is repeated.
    """
    all_morton = []
    for item in items:
        geom = item.geometry if hasattr(item, 'geometry') else item['geometry']
        lats, lons = _extract_coords(geom)
        all_morton.append(geo2mort(lats, lons, order=order))

    merged = np.concatenate(all_morton)
    cells = morton_polygon_from_array(merged, n_cells=12)
    return _pad_cells(cells, 12)


def mbox_to_polygons(mbox, step=32):
    """Materialize an ``opr:mbox`` (or ``opr:mpolygon``) as shapely Polygons in EPSG:4326.

    Each morton cell is reconstructed via :func:`mortie.tools.mort2polygon`,
    which returns a list of ``(lat, lon)`` vertices traced along the cell
    boundary. Polygons are returned in WGS84 ``(lon, lat)`` order. To
    project them, wrap in a ``GeoSeries`` and call ``.to_crs(...)``.

    Parameters
    ----------
    mbox : iterable of int
        Morton cell characteristics (typically 4 cells for ``opr:mbox``,
        12 for ``opr:mpolygon``). Duplicates are preserved — the result
        has one polygon per input integer.
    step : int, default 32
        Per-side sampling for :func:`mortie.tools.mort2polygon`. Higher
        values give smoother boundaries at higher cost.

    Returns
    -------
    list of shapely.geometry.Polygon
        One polygon per input cell, in input order, in EPSG:4326.

    Examples
    --------
    >>> polys = mbox_to_polygons(item['opr:mbox'])
    >>> import geopandas as gpd
    >>> gpd.GeoSeries(polys, crs='EPSG:4326').to_crs('EPSG:3031')
    """
    polys = []
    for cell in mbox:
        verts = mort2polygon(int(cell), step=step)  # [[lat, lon], ...]
        arr = np.asarray(verts)
        poly = Polygon(zip(arr[:, 1], arr[:, 0]))  # (lon, lat)
        if not poly.is_valid:
            poly = make_valid(poly)
        polys.append(poly)
    return polys
