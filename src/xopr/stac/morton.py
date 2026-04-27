"""Morton geometry index computation for STAC items and collections."""

import numpy as np
from mortie import geo2mort, geo_morton_polygon, morton_polygon_from_array


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
