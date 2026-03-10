"""Match bedmap picks to OPR frames via morton index prefix containment."""

import time
from collections import defaultdict

import numpy as np
import pandas as pd
import geopandas as gpd
from mortie import geo2mort
from scipy.spatial import cKDTree


def _build_mbox_lookup(stac_gdf):
    """Build a lookup from mbox prefix strings to frame metadata.

    Parameters
    ----------
    stac_gdf : GeoDataFrame
        STAC catalog loaded as a GeoDataFrame. Must have columns
        ``opr:mbox``, ``opr:segment``, ``opr:frame``, ``opr:date``.

    Returns
    -------
    dict
        Mapping ``{prefix_str: [(segment, date, frame, item_id), ...]}``.
    """
    lookup = {}

    for idx, row in stac_gdf.iterrows():
        mbox = row['opr:mbox']
        meta = (
            row['opr:segment'],
            row['opr:date'],
            row['opr:frame'],
            row.get('id', idx),
        )

        for cell in mbox:
            prefix = str(cell)
            lookup.setdefault(prefix, []).append(meta)

    return lookup


def _morton_prefix_match(bedmap_mort_strs, mbox_lookup):
    """Find candidate frames for each bedmap point via prefix matching.

    Parameters
    ----------
    bedmap_mort_strs : array-like of str
        String representations of bedmap morton indices (order 18).
    mbox_lookup : dict
        Output of :func:`_build_mbox_lookup`.

    Returns
    -------
    candidates : list of list of tuple
        For each bedmap point, a list of ``(segment, date, frame, item_id)``
        tuples that are candidate matches.
    n_candidates : ndarray of int
        Count of candidates per point.
    """
    prefixes = list(mbox_lookup.keys())
    n = len(bedmap_mort_strs)
    candidates = [[] for _ in range(n)]
    counts = np.zeros(n, dtype=np.int32)

    for i, mort_str in enumerate(bedmap_mort_strs):
        for prefix in prefixes:
            if mort_str.startswith(prefix):
                metas = mbox_lookup[prefix]
                candidates[i].extend(metas)
                counts[i] += len(metas)

    return candidates, counts


def match_bedmap_to_frames(stac_gdf, bedmap_gdf, order=18):
    """Match bedmap picks to candidate OPR frames via morton containment.

    For each bedmap point, computes its morton index and checks which
    STAC item mbox cells contain it (via string prefix matching).
    The STAC catalog must already contain ``opr:mbox`` per item.

    Parameters
    ----------
    stac_gdf : GeoDataFrame
        OPR STAC catalog with columns ``opr:mbox``, ``opr:segment``,
        ``opr:frame``, ``opr:date``.
    bedmap_gdf : GeoDataFrame
        Bedmap data with Point geometry.
    order : int, optional
        Morton tessellation order, by default 18.

    Returns
    -------
    GeoDataFrame
        Copy of ``bedmap_gdf`` with added columns:

        - ``opr_candidates`` : list of (segment, date, frame, item_id)
        - ``n_candidates`` : number of candidate matches
    """
    lats = bedmap_gdf.geometry.y.values
    lons = bedmap_gdf.geometry.x.values

    bedmap_mortons = geo2mort(lats, lons, order=order)
    bedmap_mort_strs = [str(m) for m in bedmap_mortons]

    mbox_lookup = _build_mbox_lookup(stac_gdf)

    candidates, counts = _morton_prefix_match(bedmap_mort_strs, mbox_lookup)

    result = bedmap_gdf.copy()
    result['opr_candidates'] = candidates
    result['n_candidates'] = counts

    return result


def _build_frame_groups(stac_gdf, group_size):
    """Group adjacent frames within each (date, segment).

    Parameters
    ----------
    stac_gdf : GeoDataFrame
        STAC catalog with ``opr:date``, ``opr:segment``, ``opr:frame``.
    group_size : int
        Number of adjacent frames per group.

    Returns
    -------
    list of list of tuple
        Each inner list contains ``(segment, date, frame)`` tuples.
    """
    groups = []
    for (date, seg), sub in stac_gdf.groupby(['opr:date', 'opr:segment']):
        frames = sorted(sub['opr:frame'].values)
        for i in range(0, len(frames), group_size):
            chunk = [(int(seg), str(date), int(f)) for f in frames[i:i + group_size]]
            groups.append(chunk)
    return groups


def _find_contiguous_runs(positions):
    """Find contiguous runs in a sorted integer array.

    Parameters
    ----------
    positions : ndarray of int
        Sorted array of along-track positions.

    Returns
    -------
    list of ndarray
        Contiguous runs, sorted by length descending.
    """
    if len(positions) == 0:
        return []
    diffs = np.diff(positions)
    breaks = np.where(diffs > 1)[0] + 1
    return sorted(np.split(positions, breaks), key=len, reverse=True)


def disambiguate_matches(result_gdf, layer_gdf, stac_gdf, group_size=5):
    """Resolve ambiguous morton matches using along-track runs.

    Groups adjacent frames, finds contiguous runs of bedmap points
    through each group in ``trajectory_id`` order, and assigns the
    longest runs first while masking out claimed points. Within each
    assigned group, nearest-neighbor to individual frame layer picks.

    Parameters
    ----------
    result_gdf : GeoDataFrame
        Output of :func:`match_bedmap_to_frames`.
    layer_gdf : GeoDataFrame
        OPR layer picks with ``segment_path``, ``frame``, and Point geometry.
    stac_gdf : GeoDataFrame
        STAC catalog for building frame groups.
    group_size : int
        Number of adjacent frames per group. Default 5.

    Returns
    -------
    GeoDataFrame
        Copy with added columns:

        - ``assigned_segment_path`` : str
        - ``assigned_frame`` : int
        - ``nearest_distance_m`` : float
        - ``run_id`` : int (-1 if global fallback)
        - ``run_length`` : int (0 if global fallback)
    """
    result = result_gdf.copy().reset_index(drop=True)
    n = len(result)

    # --- Step 1: Along-track ordering ---
    tid = result['trajectory_id'].astype(int).values
    track_order = tid.argsort()
    inv_track = track_order.argsort()

    # --- Step 2: Build frame groups ---
    groups = _build_frame_groups(stac_gdf, group_size)
    n_groups = len(groups)
    print(f"  {n_groups} frame groups (group_size={group_size})")

    frame_to_group = {}
    for g_idx, group in enumerate(groups):
        for seg, date, frame in group:
            frame_to_group[(seg, date, frame)] = g_idx

    # --- Step 3: Map candidates → along-track positions per group ---
    t0 = time.time()
    group_pos_lists = [[] for _ in range(n_groups)]
    candidates_col = result['opr_candidates'].values

    for orig_idx in range(n):
        pos = inv_track[orig_idx]
        seen = set()
        for seg, date, frame, _ in candidates_col[orig_idx]:
            g_idx = frame_to_group.get((seg, date, frame))
            if g_idx is not None and g_idx not in seen:
                group_pos_lists[g_idx].append(pos)
                seen.add(g_idx)

    group_positions = [np.unique(p) for p in group_pos_lists]
    print(f"  Candidate mapping: {time.time()-t0:.1f}s")

    # --- Step 4: Pre-compute all runs, sort by length desc ---
    t0 = time.time()
    all_runs = []
    for g_idx in range(n_groups):
        for run in _find_contiguous_runs(group_positions[g_idx]):
            all_runs.append((len(run), g_idx, run))
    all_runs.sort(key=lambda x: x[0], reverse=True)

    top_len = all_runs[0][0] if all_runs else 0
    print(f"  {len(all_runs)} initial runs, longest={top_len}")

    # --- Step 5: Greedy assignment — longest first, re-split on conflict ---
    available = np.ones(n, dtype=bool)
    assigned_group_arr = np.full(n, -1, dtype=int)
    run_id_arr = np.full(n, -1, dtype=int)
    run_len_arr = np.zeros(n, dtype=int)
    next_run_id = 0
    n_assigned = 0

    for _, g_idx, positions in all_runs:
        avail_mask = available[positions]
        if not avail_mask.any():
            continue
        for sub_run in _find_contiguous_runs(positions[avail_mask]):
            assigned_group_arr[sub_run] = g_idx
            available[sub_run] = False
            run_id_arr[sub_run] = next_run_id
            run_len_arr[sub_run] = len(sub_run)
            next_run_id += 1
            n_assigned += len(sub_run)

    print(f"  Assigned {n_assigned}/{n} ({n_assigned/n:.1%}) in "
          f"{next_run_id} runs, {time.time()-t0:.1f}s")

    # Map back to original index space
    orig_group = assigned_group_arr[inv_track]
    result['run_id'] = run_id_arr[inv_track]
    result['run_length'] = run_len_arr[inv_track]

    # --- Step 6: Per-group nearest-neighbor → individual frames ---
    t0 = time.time()
    assigned_seg = np.full(n, '', dtype=object)
    assigned_frm = np.full(n, -1, dtype=int)
    nearest_dist = np.full(n, np.nan, dtype=float)

    layer_proj = layer_gdf.to_crs('EPSG:3031')
    result_proj = result_gdf.to_crs('EPSG:3031')
    result_xy = np.column_stack([result_proj.geometry.x.values,
                                 result_proj.geometry.y.values])
    layer_xy = np.column_stack([layer_proj.geometry.x.values,
                                layer_proj.geometry.y.values])
    layer_seg_arr = layer_gdf['segment_path'].values
    layer_frm_arr = layer_gdf['frame'].values

    # Pre-build layer lookup: (segment_path, frame) → indices
    layer_lookup = defaultdict(list)
    for i in range(len(layer_gdf)):
        layer_lookup[(layer_seg_arr[i], layer_frm_arr[i])].append(i)

    for g_idx, group_frames in enumerate(groups):
        mask_indices = np.where(orig_group == g_idx)[0]
        if len(mask_indices) == 0:
            continue

        # Collect layer pick indices for this group's frames
        group_layer_idx = []
        for seg, date, frame in group_frames:
            key = (f"{date}_{seg:02d}", frame)
            group_layer_idx.extend(layer_lookup.get(key, []))

        if not group_layer_idx:
            continue

        group_layer_idx = np.array(group_layer_idx)
        tree = cKDTree(layer_xy[group_layer_idx])
        dists, nn_idx = tree.query(result_xy[mask_indices])

        actual_idx = group_layer_idx[nn_idx]
        assigned_seg[mask_indices] = layer_seg_arr[actual_idx]
        assigned_frm[mask_indices] = layer_frm_arr[actual_idx]
        nearest_dist[mask_indices] = dists

    # Fallback: unassigned points get global nearest-neighbor
    unassigned = np.where(assigned_frm == -1)[0]
    if len(unassigned) > 0:
        tree = cKDTree(layer_xy)
        dists, idxs = tree.query(result_xy[unassigned])
        assigned_seg[unassigned] = layer_seg_arr[idxs]
        assigned_frm[unassigned] = layer_frm_arr[idxs]
        nearest_dist[unassigned] = dists

    result['assigned_segment_path'] = assigned_seg
    result['assigned_frame'] = assigned_frm
    result['nearest_distance_m'] = nearest_dist

    print(f"  Per-group NN: {time.time()-t0:.1f}s "
          f"({len(unassigned)} used global fallback)")

    return result
