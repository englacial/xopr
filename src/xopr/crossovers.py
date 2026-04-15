"""Map xopr STAC frames to ICESat-2 (or other CMR-indexed) granules.

One CMR query per invocation builds a local granule index; per-frame
matching then intersects each frame's ``opr:mbox`` cells (4 morton cells
per frame, variable resolution) against that index. Two backends are
provided and produce the same hits:

- :func:`match_frames_to_granules` — shapely STRtree over projected
  granule polygons; exact polygon-polygon intersection per mbox cell.
- :func:`match_frames_to_granules_prefix` — bidirectional morton prefix
  matching against each granule's own variable-resolution mbox
  (generalizes :mod:`xopr.bedmap.morton_match`).

The CMR query helper is a minimal vendored subset of
``magg.catalog.query_cmr`` (englacial/magg) so xopr avoids a runtime
dependency on magg.
"""

import time
from typing import Optional

import numpy as np
import requests
from mortie import geo2mort
from mortie.tools import mort2polygon
from pyproj import Transformer
from shapely import STRtree, make_valid
from shapely.geometry import Polygon

CMR_GRANULES_URL = "https://cmr.earthdata.nasa.gov/search/granules.umm_json"


def _cell_polygon_4326(cell, step=32):
    """Reconstruct a shapely Polygon in EPSG:4326 from a morton cell id."""
    verts = mort2polygon(int(cell), step=step)  # [[lat, lon], ...]
    arr = np.asarray(verts)
    return Polygon(zip(arr[:, 1], arr[:, 0]))  # (lon, lat)


def _reproject_polygon(poly, transformer):
    x, y = transformer.transform(*poly.exterior.coords.xy)
    out = Polygon(zip(x, y))
    if not out.is_valid:
        out = make_valid(out)
    return out


def cmr_bbox_from_mpolygon(mpolygon, step=32):
    """Derive a (lon_min, lat_min, lon_max, lat_max) bbox from an ``opr:mpolygon``.

    Parameters
    ----------
    mpolygon : iterable of int
        Morton cell characteristics (collection-level coverage).
    step : int
        Per-side sampling for :func:`mortie.tools.mort2polygon`.

    Returns
    -------
    tuple of float
        Bounding box in lon/lat.
    """
    polys = [_cell_polygon_4326(c, step=step) for c in mpolygon]
    union = polys[0]
    for p in polys[1:]:
        union = union.union(p)
    minx, miny, maxx, maxy = union.bounds
    return (float(minx), float(miny), float(maxx), float(maxy))


def _extract_granule(umm_item):
    """Pull ``granule_id``, s3/https URLs, and boundary points from a UMM item."""
    umm = umm_item.get("umm", {})
    granule_id = umm.get("GranuleUR", "")

    s3_url = None
    https_url = None
    for url_obj in umm.get("RelatedUrls", []):
        url = url_obj.get("URL", "")
        if url.startswith("s3://") and url.endswith(".h5"):
            s3_url = url
        elif (
            url.startswith("https://")
            and url.endswith(".h5")
            and url_obj.get("Type") == "GET DATA"
        ):
            https_url = url

    points = []
    gpolygons = (
        umm.get("SpatialExtent", {})
        .get("HorizontalSpatialDomain", {})
        .get("Geometry", {})
        .get("GPolygons", [])
    )
    if gpolygons:
        for p in gpolygons[0].get("Boundary", {}).get("Points", []):
            if "Latitude" in p and "Longitude" in p:
                points.append((p["Latitude"], p["Longitude"]))

    return {
        "granule_id": granule_id,
        "s3_url": s3_url,
        "https_url": https_url,
        "points": points,
    }


def query_cmr_granules(
    start_date: str,
    end_date: str,
    short_name: str = "ATL06",
    version: str = "007",
    provider: str = "NSIDC_CPRD",
    bbox: Optional[tuple] = None,
    polygon: Optional[list] = None,
    page_size: int = 2000,
    sleep_s: float = 0.1,
):
    """Query NASA CMR for granules matching temporal + spatial filters.

    Follows the pattern in ``magg.catalog.query_cmr`` (one call, paginated).
    Pass either ``bbox`` or ``polygon`` — polygon is used if both given.

    Parameters
    ----------
    start_date, end_date : str
        ``YYYY-MM-DD`` inclusive.
    short_name, version, provider : str
        CMR product identifiers. Defaults target ICESat-2 ATL06 v007 at NSIDC.
    bbox : tuple of float, optional
        ``(lon_min, lat_min, lon_max, lat_max)``.
    polygon : list of (lon, lat), optional
        Closed CCW ring in geodetic coordinates. Sent as a CMR ``polygon=``
        query parameter for true polygon intersection (no bbox reduction).
    page_size : int
    sleep_s : float
        Delay between pages to be polite to CMR.

    Returns
    -------
    list of dict
        Each dict has ``granule_id``, ``s3_url``, ``https_url``, ``points``
        (list of ``(lat, lon)`` tuples).
    """
    temporal = f"{start_date}T00:00:00Z,{end_date}T23:59:59Z"
    params = {
        "provider": provider,
        "short_name": short_name,
        "version": version,
        "page_size": page_size,
        "sort_key": "start_date",
        "temporal": temporal,
        "offset": 0,
    }
    if polygon is not None:
        params["polygon"] = ",".join(f"{lon},{lat}" for lon, lat in polygon)
    elif bbox is not None:
        params["bounding_box"] = f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}"

    headers = {"Accept": "application/vnd.nasa.cmr.umm_json+json"}

    all_items = []
    total_hits = None
    while True:
        resp = requests.get(CMR_GRANULES_URL, params=params, headers=headers, timeout=60)
        resp.raise_for_status()
        if total_hits is None:
            total_hits = int(resp.headers.get("CMR-Hits", 0))
        items = resp.json().get("items", [])
        if not items:
            break
        all_items.extend(items)
        if len(items) < page_size or len(all_items) >= total_hits:
            break
        params["offset"] = int(params["offset"]) + len(items)
        time.sleep(sleep_s)

    return [_extract_granule(it) for it in all_items]


def build_granule_strtree(granules, crs="EPSG:3031"):
    """Build a shapely STRtree over granule polygons in the target CRS.

    Granules missing a valid ``s3_url`` or with fewer than 3 boundary points
    are dropped.

    Parameters
    ----------
    granules : list of dict
        Output of :func:`query_cmr_granules`.
    crs : str
        Projection used for the index. Default EPSG:3031 (Antarctic polar
        stereographic). Use EPSG:3413 for Greenland.

    Returns
    -------
    tree : shapely.STRtree
    records : list of dict
        Same order as the tree's inputs; each dict is the input granule
        dict with an added ``geometry`` key (the projected Polygon).
    """
    transformer = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    polys = []
    records = []
    for g in granules:
        pts = g.get("points") or []
        if not g.get("s3_url") or len(pts) < 3:
            continue
        lats = np.array([p[0] for p in pts])
        lons = np.array([p[1] for p in pts])
        x, y = transformer.transform(lons, lats)
        try:
            poly = Polygon(zip(x, y))
            if not poly.is_valid:
                poly = make_valid(poly)
            if poly.is_empty:
                continue
        except Exception:
            continue
        polys.append(poly)
        rec = dict(g)
        rec["geometry"] = poly
        records.append(rec)

    return STRtree(polys), records


def match_frames_to_granules(
    frames_gdf,
    granule_tree,
    granule_records,
    crs="EPSG:3031",
    step=32,
):
    """Attach a list of intersecting granules to each xopr STAC frame.

    For each frame, reconstructs its 4 ``opr:mbox`` cell polygons via
    :func:`mortie.tools.mort2polygon`, reprojects to ``crs``, queries the
    STRtree per-cell, and unions the hits by ``granule_id``.

    Parameters
    ----------
    frames_gdf : GeoDataFrame
        Must have an ``opr:mbox`` column (list of 4 morton cell ints).
    granule_tree : shapely.STRtree
    granule_records : list of dict
        As returned by :func:`build_granule_strtree`.
    crs : str
        Must match the CRS used to build ``granule_tree``.
    step : int
        Per-side sampling for :func:`mortie.tools.mort2polygon`.

    Returns
    -------
    GeoDataFrame
        Copy of ``frames_gdf`` with added columns:

        - ``atl06_granules`` : list of dicts with ``granule_id``, ``s3_url``,
          ``https_url``.
        - ``n_granules`` : int.
    """
    if "opr:mbox" not in frames_gdf.columns:
        raise ValueError(
            "frames_gdf is missing 'opr:mbox'; load a catalog produced after "
            "xopr PR #77 (see issue #78 for deployment status)."
        )

    transformer = Transformer.from_crs("EPSG:4326", crs, always_xy=True)

    granule_lists = []
    counts = []
    for mbox in frames_gdf["opr:mbox"].values:
        seen = {}
        for cell in mbox:
            cell_poly = _reproject_polygon(_cell_polygon_4326(cell, step=step), transformer)
            hits = granule_tree.query(cell_poly, predicate="intersects")
            for idx in hits:
                rec = granule_records[int(idx)]
                gid = rec["granule_id"]
                if gid not in seen:
                    seen[gid] = {
                        "granule_id": gid,
                        "s3_url": rec.get("s3_url"),
                        "https_url": rec.get("https_url"),
                    }
        granule_lists.append(list(seen.values()))
        counts.append(len(seen))

    out = frames_gdf.copy()
    out["atl06_granules"] = granule_lists
    out["n_granules"] = counts
    return out


ICESAT2_LAUNCH = "2018-10-13"
_ICESAT2_CYCLE_DAYS = 91


def cycle_to_dates(cycle):
    """Convert an ICESat-2 repeat cycle number to a ``(start, end)`` date pair.

    Mirrors ``magg.catalog.cycle_to_dates``.

    Parameters
    ----------
    cycle : int
        ICESat-2 cycle number (1-based).

    Returns
    -------
    (start_date, end_date) : tuple of str (YYYY-MM-DD)
    """
    from datetime import date, timedelta

    launch = date.fromisoformat(ICESAT2_LAUNCH)
    start = launch + timedelta(days=(cycle - 1) * _ICESAT2_CYCLE_DAYS)
    end = start + timedelta(days=_ICESAT2_CYCLE_DAYS)
    return start.isoformat(), end.isoformat()


def resolve_temporal_window(
    frames_gdf=None, mode="exact_year", cycle=None, date_range=None
):
    """Derive ``(start_date, end_date)`` strings for the CMR query.

    Four ways to specify the window, checked in this order:

    1. ``date_range=(start, end)`` — explicit override, each a ``YYYY-MM-DD``
       string or :class:`datetime.date`.
    2. ``cycle=N`` — an ICESat-2 repeat cycle (91 days starting from launch).
    3. ``mode='exact_year'`` — min/max of ``frames_gdf['opr:date']``.
    4. ``mode='all_years'`` — ICESat-2 launch (2018-10-13) to today.

    Parameters
    ----------
    frames_gdf : GeoDataFrame, optional
        Only required when ``mode='exact_year'`` and neither ``date_range``
        nor ``cycle`` is given. Must then have an ``opr:date`` column
        (``YYYYMMDD`` strings).
    mode : {'exact_year', 'all_years'}
        Used only when ``date_range`` and ``cycle`` are both None.
    cycle : int, optional
    date_range : tuple of (str or date), optional

    Returns
    -------
    (start_date, end_date) : tuple of str (YYYY-MM-DD)
    """
    from datetime import date, datetime

    def _to_iso(d):
        return d.isoformat() if hasattr(d, "isoformat") else str(d)

    if date_range is not None:
        start, end = date_range
        return _to_iso(start), _to_iso(end)

    if cycle is not None:
        return cycle_to_dates(cycle)

    launch = date.fromisoformat(ICESAT2_LAUNCH)

    if mode == "exact_year":
        if frames_gdf is None:
            raise ValueError("mode='exact_year' requires frames_gdf")
        dates = [datetime.strptime(d, "%Y%m%d").date() for d in frames_gdf["opr:date"]]
        dmin, dmax = min(dates), max(dates)
        if dmax < launch:
            raise ValueError(
                f"Catalog ends {dmax} (before ICESat-2 launch {launch}); "
                "use mode='all_years', cycle=N, or date_range=... instead."
            )
        return dmin.isoformat(), dmax.isoformat()
    elif mode == "all_years":
        return launch.isoformat(), date.today().isoformat()
    else:
        raise ValueError(f"mode must be 'exact_year' or 'all_years', got {mode!r}")


# ---------------------------------------------------------------------------
# Prefix-string matching backend (alternative to STRtree)
# ---------------------------------------------------------------------------


def _densify_boundary(points, n_per_segment=50):
    """Return (lats, lons) arrays that linearly interpolate each edge.

    Parameters
    ----------
    points : list of (lat, lon)
        Polygon boundary (closed or not).
    n_per_segment : int
        Samples between each consecutive pair of vertices.

    Returns
    -------
    lats, lons : ndarray
    """
    if not points:
        return np.array([]), np.array([])
    pts = list(points)
    if pts[0] != pts[-1]:
        pts.append(pts[0])
    lats, lons = [], []
    for (lat1, lon1), (lat2, lon2) in zip(pts[:-1], pts[1:]):
        lats.append(np.linspace(lat1, lat2, n_per_segment, endpoint=False))
        lons.append(np.linspace(lon1, lon2, n_per_segment, endpoint=False))
    return np.concatenate(lats), np.concatenate(lons)


def compute_granule_sample_mortons(points, order=18, n_per_segment=50):
    """Compute full-order morton indices for densified boundary samples of a
    granule polygon.

    The returned values are fixed-order (``order``) morton strings suitable
    for the same prefix-match pattern used by
    :mod:`xopr.bedmap.morton_match`: a frame mbox cell (prefix) contains a
    sample point iff ``sample_str.startswith(mbox_str)``.

    Parameters
    ----------
    points : list of (lat, lon)
        Granule polygon boundary.
    order : int
        Morton tessellation order. Default 18 (matches ``opr:mbox``).
    n_per_segment : int
        Samples per boundary edge.

    Returns
    -------
    list of str
        Morton strings, one per sample.
    """
    lats, lons = _densify_boundary(points, n_per_segment=n_per_segment)
    if lats.size == 0:
        return []
    mortons = geo2mort(lats, lons, order=order)
    return [str(m) for m in mortons]


def build_granule_prefix_index(granules, order=18, n_per_segment=50):
    """Precompute sampled morton indices for each granule.

    Parameters
    ----------
    granules : list of dict
        Output of :func:`query_cmr_granules`.
    order : int
    n_per_segment : int
        Forwarded to :func:`compute_granule_sample_mortons`.

    Returns
    -------
    list of dict
        Same order as input (skipping granules without enough boundary
        points). Each dict has ``granule_id``, ``s3_url``, ``https_url``,
        and ``mort_strs`` (list of order-``order`` morton strings).
    """
    records = []
    for g in granules:
        pts = g.get("points") or []
        if not g.get("s3_url") or len(pts) < 3:
            continue
        try:
            mort_strs = compute_granule_sample_mortons(
                pts, order=order, n_per_segment=n_per_segment
            )
        except Exception:
            continue
        if not mort_strs:
            continue
        records.append({
            "granule_id": g["granule_id"],
            "s3_url": g.get("s3_url"),
            "https_url": g.get("https_url"),
            "mort_strs": mort_strs,
        })
    return records


def match_frames_to_granules_prefix(frames_gdf, granule_records):
    """Attach intersecting granules via morton prefix matching.

    For each of a frame's 4 ``opr:mbox`` cells (variable-resolution
    prefixes), any granule with a sample point whose full-order morton
    starts with that prefix is treated as overlapping. This is the same
    prefix pattern used for bedmap↔frame matching.

    Caveat: correctness depends on the granule being densely sampled
    enough that at least one sample lands inside any overlapping frame
    mbox cell. The default ``n_per_segment=50`` is ample for thin ATL06
    strips; polygons with small interior features relative to boundary
    spacing may need more samples.

    Parameters
    ----------
    frames_gdf : GeoDataFrame
        Must have an ``opr:mbox`` column (list of 4 morton cell ints).
    granule_records : list of dict
        Output of :func:`build_granule_prefix_index`.

    Returns
    -------
    GeoDataFrame
        Copy of ``frames_gdf`` with added columns:

        - ``atl06_granules`` : list of dicts with ``granule_id``,
          ``s3_url``, ``https_url``.
        - ``n_granules`` : int.
    """
    if "opr:mbox" not in frames_gdf.columns:
        raise ValueError(
            "frames_gdf is missing 'opr:mbox'; load a catalog produced after "
            "xopr PR #77 (see issue #78 for deployment status)."
        )

    from collections import defaultdict

    # Invert the match: for each distinct frame-prefix string, remember which
    # frames own it. Then for each granule sample we only probe one dict key
    # per distinct prefix length — O(samples * distinct_lengths) instead of
    # O(frames * granules * samples * 4).
    prefix_to_frames = defaultdict(list)
    for f_idx, mbox in enumerate(frames_gdf["opr:mbox"].values):
        for cell in mbox:
            prefix_to_frames[str(int(cell))].append(f_idx)

    distinct_lengths = sorted({len(p) for p in prefix_to_frames})
    n_frames = len(frames_gdf)
    frame_hits = [set() for _ in range(n_frames)]

    for g_idx, rec in enumerate(granule_records):
        for ms in rec["mort_strs"]:
            ms_len = len(ms)
            for L in distinct_lengths:
                if L > ms_len:
                    break  # distinct_lengths is sorted ascending
                hit_frames = prefix_to_frames.get(ms[:L])
                if hit_frames is not None:
                    for f_idx in hit_frames:
                        frame_hits[f_idx].add(g_idx)

    granule_lists = []
    counts = []
    for f_idx in range(n_frames):
        gs = [
            {
                "granule_id": granule_records[g]["granule_id"],
                "s3_url": granule_records[g].get("s3_url"),
                "https_url": granule_records[g].get("https_url"),
            }
            for g in sorted(frame_hits[f_idx])
        ]
        granule_lists.append(gs)
        counts.append(len(gs))

    out = frames_gdf.copy()
    out["atl06_granules"] = granule_lists
    out["n_granules"] = counts
    return out


__all__ = [
    "cmr_bbox_from_mpolygon",
    "query_cmr_granules",
    "build_granule_strtree",
    "match_frames_to_granules",
    "compute_granule_sample_mortons",
    "build_granule_prefix_index",
    "match_frames_to_granules_prefix",
    "resolve_temporal_window",
    "cycle_to_dates",
    "ICESAT2_LAUNCH",
]
