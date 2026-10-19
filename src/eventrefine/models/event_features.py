"""Deterministic point addressing and seven-channel event feature construction.

Event features preserve the frozen checkpoint's per-pixel intervals, pixel
counts, occupied-cell counts and clip-global normalized time coordinates.
"""
from __future__ import annotations

import numpy as np
import torch

BUCKET = 128            # attention scope = SCOPE*BUCKET = 1024 tokens (a 512-bucket config from the
                        # scene-scale upstream example makes the attention span the whole cloud)
NB = 48                      # minimum; bucketize() raises it until no bucket exceeds BUCKET
                             # 48 * 512 = 24,576 == swinable_alignment(512, 8) -> one pool block exactly
ROWS = NB * BUCKET
S = 4                        # spatial/temporal quantisation for the cell (used as bucketize default)


def _splitmix64(x: np.ndarray) -> np.ndarray:
    """Deterministic 64-bit mixer (numpy, overflow-wrapped) — pure function of the integer input."""
    x = (x + np.uint64(0x9E3779B97F4A7C15)).astype(np.uint64)
    z = x.copy()
    z = ((z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)).astype(np.uint64)
    z = ((z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)).astype(np.uint64)
    return (z ^ (z >> np.uint64(31))).astype(np.uint64)


def bucketize(
    locs: np.ndarray,
    S: int = 4,
    *,
    time_origin: float | None = None,
    time_scale: float = 64.0,
) -> dict:
    """Deterministic, DENSE spatial layout for one clip.

    Design (why not per-cell buckets): the model's attention scopes come from the row count alone
    (`swin_plan.gen_scopes_from_plan(N)`), so the buckets the model sees are contiguous BUCKET-row
    chunks. A per-cell bucket table therefore either leaves empty buckets (device-side assert,
    measured) or explodes the row count. Instead:

      1. give every point a spatial key  = ((x//S) * K + (y//S)) * K + (tau//S)
      2. stable-sort by that key -> spatially coherent order, deterministic in any schedule
      3. the model's bucket b is simply rows [b*BUCKET, (b+1)*BUCKET) of that order, so every bucket
         is populated and holds spatially nearby events (which is what PSH approximates)
      4. rows = ceil(N / ALIGN) * ALIGN with ALIGN = BUCKET*SCOPE*6, matching the pool alignment

    locs: (N,4) int64 [clip, x, y, t]; tau uses (t - t0)/64 with t0 of THIS clip (past-only).
    """
    n = locs.shape[0]
    t = locs[:, 3].astype(np.float64)
    t0 = float(t.min()) if time_origin is None else float(time_origin)
    if float(time_scale) <= 0:
        raise ValueError("time_scale must be positive")
    tau = (t - t0) / float(time_scale)
    K = 4096
    key = (((locs[:, 1] // S).astype(np.int64) * K + (locs[:, 2] // S).astype(np.int64)) * K
           + np.floor(tau / S).astype(np.int64))
    # coarsen the key until no 128-row chunk would mix wildly different cells (kept simple: the sort
    # itself guarantees contiguity, so no coarsening pass is needed).
    order = np.lexsort((np.arange(n), key))          # stable: spatial key, then original index
    row_all = np.empty(n, dtype=np.int64)
    row_all[order] = np.arange(n, dtype=np.int64)    # dense: no padding inside the point range
    ALIGN = BUCKET * 8 * 6
    rows = int(np.ceil(max(1, n) / ALIGN) * ALIGN)
    nb = rows // BUCKET
    counts = np.bincount(np.arange(n, dtype=np.int64) // BUCKET, minlength=nb)
    return {'row': row_all, 'bucket': key, 'counts': counts, 'n': n, 'nb': nb, 'rows': rows,
            'tau': tau, 't0': t0, 'S': S, 'align': ALIGN,
            'max_bucket': int(counts.max(initial=0)), 'key_unique': int(np.unique(key).size)}



def stem_features(
    locs: np.ndarray,
    feats: np.ndarray,
    bu: dict,
    *,
    feature_time_origin: float | None = None,
    history_locs: np.ndarray | None = None,
    history_feats: np.ndarray | None = None,
    history_raw_timestamps: np.ndarray | None = None,
    current_raw_timestamps: np.ndarray | None = None,
) -> np.ndarray:
    """Build model features for current-window rows using only causal history.

    The first four training features are point polarity, clip-global normalized
    time, log same-polarity interval, and log opposite-polarity interval. Event
    history before the current window preserves the interval at window cuts.
    The 512 ms occupied-cell count uses only the supplied causal prefix and current closed window.
    """
    if locs.ndim != 2 or locs.shape[1] != 4 or feats.ndim != 2 or feats.shape[0] != locs.shape[0]:
        raise ValueError("locs/features must align with shapes (N,4) and (N,C)")
    if feats.shape[1] < 4 or locs.shape[0] == 0:
        raise ValueError("features need four channels and the current window must be non-empty")
    n = int(locs.shape[0])
    t = locs[:, 3].astype(np.float64)
    pol = (feats[:, 3] > 0.5).astype(np.int64)
    out = np.zeros((n, 7), dtype=np.float32)

    if history_locs is None:
        history_locs = np.empty((0, 4), dtype=locs.dtype)
    if history_feats is None:
        history_feats = np.empty((0, feats.shape[1]), dtype=feats.dtype)
    history_locs = np.asarray(history_locs)
    history_feats = np.asarray(history_feats)
    if history_locs.ndim != 2 or history_locs.shape[1] != 4 or history_feats.ndim != 2 or history_feats.shape[0] != history_locs.shape[0]:
        raise ValueError("history locations/features must align with shapes (H,4) and (H,C)")
    if history_feats.shape[0] and history_feats.shape[1] < 4:
        raise ValueError("history features need at least four channels")
    h_raw = (history_locs[:, 3].astype(np.float64) if history_raw_timestamps is None
             else np.asarray(history_raw_timestamps, dtype=np.float64))
    c_raw = (t if current_raw_timestamps is None
             else np.asarray(current_raw_timestamps, dtype=np.float64))
    if h_raw.shape != (len(history_locs),) or c_raw.shape != (n,):
        raise ValueError("raw causal timestamps must align with history/current event rows")
    if not np.isfinite(h_raw).all() or not np.isfinite(c_raw).all():
        raise ValueError("raw causal timestamps must be finite")
    if h_raw.size and c_raw.size and float(h_raw.max()) >= float(c_raw.min()):
        raise ValueError("feature history must be strictly before the current window")

    combined_locs = np.concatenate([history_locs, locs], axis=0)
    combined_feats = np.concatenate([history_feats[:, :4], feats[:, :4]], axis=0)
    combined_b = combined_locs[:, 0].astype(np.int64)
    combined_x = combined_locs[:, 1].astype(np.int64)
    combined_y = combined_locs[:, 2].astype(np.int64)
    combined_t = combined_locs[:, 3].astype(np.float64)
    combined_pol = (combined_feats[:, 3] > 0.5).astype(np.int64)
    offset = len(history_locs)
    combined_n = len(combined_locs)
    combined_dsp = np.zeros(combined_n, dtype=np.float64)
    combined_dop = np.zeros(combined_n, dtype=np.float64)
    combined_npix = np.zeros(combined_n, dtype=np.float64)
    pixel_key = (combined_b * 1000 + combined_x) * 1000 + combined_y
    pixel_order = np.lexsort((combined_t, pixel_key))
    sorted_keys = pixel_key[pixel_order]
    pixel_bounds = np.flatnonzero(np.r_[True, sorted_keys[1:] != sorted_keys[:-1]])
    for group_index, start in enumerate(pixel_bounds):
        end = pixel_bounds[group_index + 1] if group_index + 1 < len(pixel_bounds) else combined_n
        ids = pixel_order[start:end]
        last_seen: dict[int, float] = {}
        counts = {0: 0, 1: 0}
        for row in ids:
            p = int(combined_pol[row])
            tr = combined_t[row]
            same_time = last_seen.get(p)
            other_time = last_seen.get(1 - p)
            combined_dsp[row] = tr - same_time if same_time is not None else 1.0e4
            combined_dop[row] = tr - other_time if other_time is not None else 1.0e4
            counts[p] += 1
            combined_npix[row] = sum(counts.values())
            last_seen[p] = tr

    current_dsp = combined_dsp[offset:]
    current_dop = combined_dop[offset:]
    current_npix = combined_npix[offset:]

    origin = float(feature_time_origin if feature_time_origin is not None else bu.get("t0", t.min()))
    tau = (t - origin) / 64.0
    out[:, 0] = pol
    out[:, 1] = tau
    out[:, 2] = np.log1p(current_dsp)
    out[:, 3] = np.log1p(current_dop)
    out[:, 4] = np.log1p(current_npix)

    combined_cell = combined_b * 100000 + (combined_x // 8) * 1000 + (combined_y // 8)
    cell_order = np.lexsort((combined_t, combined_cell))
    sorted_cells = combined_cell[cell_order]
    cell_bounds = np.flatnonzero(np.r_[True, sorted_cells[1:] != sorted_cells[:-1]])
    combined_ncell = np.zeros(combined_n, dtype=np.float64)
    for group_index, start in enumerate(cell_bounds):
        end = cell_bounds[group_index + 1] if group_index + 1 < len(cell_bounds) else combined_n
        ids = cell_order[start:end]
        times = combined_t[ids]
        left = 0
        for right in range(len(ids)):
            while times[right] - times[left] > 512.0:
                left += 1
            combined_ncell[ids[right]] = right - left + 1
    out[:, 5] = np.log1p(combined_ncell[offset:])
    out[:, 6] = tau / 125.0
    return out


def scatter_to_rows(values: np.ndarray, bu: dict, mu=None, sd=None, dtype=torch.float32,
                    device='cuda'):
    """Place per-point values into the deterministic row layout (padding rows stay zero)."""
    v = values.astype(np.float32)
    if mu is not None:
        mu = np.asarray(mu, dtype=np.float32); sd = np.asarray(sd, dtype=np.float32)
        v = (v - mu) / sd
    rows = torch.zeros((bu['rows'], v.shape[1]), dtype=dtype, device=device)
    idx = torch.from_numpy(bu['row']).to(device)
    rows[idx] = torch.from_numpy(v).to(device)
    return rows


def coord_rows(locs: np.ndarray, bu: dict, device='cuda'):
    """Padded coordinate rows (only used to keep the module signature; XFMR documents them as unused)."""
    n = locs.shape[0]
    t = locs[:, 3].astype(np.float64)
    co = np.zeros((n, 3), dtype=np.float32)
    co[:, 0] = locs[:, 1]
    co[:, 1] = locs[:, 2]
    co[:, 2] = (t - bu['t0']) / 64.0
    rows = torch.zeros((bu['rows'], 3), dtype=torch.float32, device=device)
    rows[torch.from_numpy(bu['row']).to(device)] = torch.from_numpy(co).to(device)
    return rows


def label_rows(seg: np.ndarray, bu: dict, device='cuda'):
    lab = torch.full((bu['rows'],), -1, dtype=torch.long, device=device)
    lab[torch.from_numpy(bu['row']).to(device)] = torch.from_numpy(seg).to(device)
    return lab
