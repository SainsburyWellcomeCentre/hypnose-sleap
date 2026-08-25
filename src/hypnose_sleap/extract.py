""".slp -> per-video parquet, with one session-consistent centroid.

Gates each point on ``score_thresh``, selects the node set present in at least
``presence_frac`` of occupied frames, interpolates internal gaps up to ``gap_limit``
frames, then averages the selected nodes into ``centroid_x`` / ``centroid_y``. Writes
one ``movement_analysis/sleap_tracking_video<N>_<behav>__<video>.parquet`` per video
plus the session's quality ``.yml``.

    from hypnose_sleap.extract import extract_session
    extract_session(57, 20260717, model="eeg_surgery")
    extract_session(57, 20260717, model=None, derivatives=tmp)   # unknown provenance
"""
from __future__ import annotations

import gc
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import sleap_io

from hypnose_sleap import parameters, quality
from hypnose_sleap.io import layout

# The missing-value the point helpers below return, at module scope so they share one.
nan = float("nan")


def node_names_for_instance(inst, default_nodes):
    if getattr(inst, "skeleton", None):
        return [node.name for node in inst.skeleton.nodes]

    pts = getattr(inst, "points", None)
    if isinstance(pts, list) and pts and isinstance(pts[0], dict) and "name" in pts[0]:
        return [pt["name"] for pt in pts]

    if default_nodes:
        return default_nodes

    if pts is not None:
        return [f"node_{idx}" for idx in range(len(pts))]

    return []


def extract_points_and_scores(inst, n_nodes):
    """Extract x,y coordinates and confidence scores from instance."""
    pts_raw = None

    # Try to get numpy array first (most reliable for PredictedPointsArray)
    if hasattr(inst, "numpy") and callable(inst.numpy):
        try:
            pts_raw = inst.numpy()  # Returns (n_nodes, 2) array of [x, y]
        except Exception:
            pass

    # Fallback to points_array
    if pts_raw is None and hasattr(inst, "points_array") and inst.points_array is not None:
        pts_raw = inst.points_array

    # Fallback to points
    if pts_raw is None and hasattr(inst, "points") and inst.points is not None:
        pts_raw = inst.points

    if pts_raw is None:
        pts_seq = []
    elif isinstance(pts_raw, list):
        pts_seq = pts_raw
    elif hasattr(pts_raw, "tolist"):
        pts_seq = pts_raw.tolist()
    else:
        pts_seq = list(pts_raw)

    def as_xy(point):
        if point is None:
            return (nan, nan)
        # Handle numpy arrays directly (from numpy() method)
        if hasattr(point, "__len__") and len(point) == 2 and isinstance(point[0], (int, float, np.integer, np.floating)):
            return (float(point[0]), float(point[1]))
        if hasattr(point, "x") and hasattr(point, "y"):
            return (point.x, point.y)
        if isinstance(point, dict):
            if "xy" in point and point["xy"] is not None:
                return (point["xy"][0], point["xy"][1])
            return (point.get("x", nan), point.get("y", nan))
        return (nan, nan)

    xy = []
    for idx in range(n_nodes):
        xy.append(as_xy(pts_seq[idx] if idx < len(pts_seq) else None))

    # Extract confidence scores from PredictedPointsArray
    scores = None
    if hasattr(inst, "points") and hasattr(inst.points, "__len__"):
        # For PredictedPointsArray, confidence is in element [1]
        try:
            scores = [inst.points[idx][1] if idx < len(inst.points) else nan for idx in range(n_nodes)]
        except Exception:
            pass

    # Fallback to point_confidences attribute
    if scores is None and hasattr(inst, "point_confidences") and inst.point_confidences is not None:
        scores_raw = inst.point_confidences
        scores = [scores_raw[idx] if idx < len(scores_raw) else nan for idx in range(n_nodes)]

    return xy, scores


def infer_video_file_from_slp(slp_path: Path) -> str:
    """Infer the source AVI name (with behav prefix) from a .slp filename.

    Example: "2025-12-11T14-31-20__VideoData_1904-01-16T03-00-00.predictions.slp"
    -> "2025-12-11T14-31-20__VideoData_1904-01-16T03-00-00.avi"
    """
    stem = slp_path.name
    if stem.endswith(".slp"):
        stem = stem[:-4]
    if stem.endswith(".predictions"):
        stem = stem[:-12]
    # Keep prefix if present to disambiguate identical basenames from different behav folders
    if "__" in stem:
        left, right = stem.split("__", 1)
        stem = f"{left}__{right}"
    if not stem.endswith(".avi"):
        stem = f"{stem}.avi"
    return stem


def to_number(val):
    if val is None:
        return nan
    if isinstance(val, (float, int)):
        return val
    if hasattr(val, "item"):
        try:
            return val.item()
        except Exception:
            pass
    if hasattr(val, "__len__") and len(val) > 0:
        first = val[0]
        if isinstance(first, (float, int)):
            return first
        if hasattr(first, "item"):
            try:
                return first.item()
            except Exception:
                return nan
        return nan
    try:
        return float(val)
    except Exception:
        return nan


def _compute_session_centroid(df: pd.DataFrame, selected_nodes: List[str],
                              score_thresh: float, gap_limit: int) -> Tuple[np.ndarray, np.ndarray]:
    """Confidence-gate, gap-limited interpolate, then average the selected nodes to a centroid.

    Steps (per instance/track, in frame order):
      1. Mask each node's (x, y) to NaN where the point is absent or its score < score_thresh.
      2. Interpolate each node across internal gaps up to gap_limit *frames* (no extrapolation).
      3. Centroid = mean over selected nodes; NaN only when all selected nodes are absent.

    Returns (centroid_x, centroid_y) aligned to df rows. df must have a 0..n-1 RangeIndex.
    """
    n = len(df)
    cx = np.full(n, np.nan)
    cy = np.full(n, np.nan)
    xcols = [f"{node}_x" for node in selected_nodes]
    ycols = [f"{node}_y" for node in selected_nodes]

    for _, g in df.groupby("instance", sort=False):
        pos = g.index.to_numpy()               # positions into df (RangeIndex)
        frames = g["frame"].to_numpy(dtype=float)
        if frames.size == 0 or np.isnan(frames).all():
            continue

        data: Dict[str, np.ndarray] = {}
        for node in selected_nodes:
            x = g[f"{node}_x"].to_numpy(dtype=float).copy()
            y = g[f"{node}_y"].to_numpy(dtype=float).copy()
            scol = f"{node}_score"
            if scol in g.columns:
                s = g[scol].to_numpy(dtype=float)
                mask = np.isnan(x) | np.isnan(y) | ~(s >= score_thresh)
            else:
                mask = np.isnan(x) | np.isnan(y)
            x[mask] = np.nan
            y[mask] = np.nan
            data[f"{node}_x"] = x
            data[f"{node}_y"] = y

        fint = frames.astype(np.int64)
        sub = pd.DataFrame(data, index=fint)
        # Reindex onto a contiguous frame grid so the gap limit counts real frames,
        # interpolate short internal gaps only, then restrict back to observed frames.
        full = np.arange(int(fint.min()), int(fint.max()) + 1)
        filled = (sub.reindex(full)
                     .interpolate(method="index", limit=gap_limit, limit_area="inside")
                     .loc[fint])

        with np.errstate(invalid="ignore"):
            gx = np.nanmean(filled[xcols].to_numpy(dtype=float), axis=1)
            gy = np.nanmean(filled[ycols].to_numpy(dtype=float), axis=1)
        cx[pos] = gx
        cy[pos] = gy

    return cx, cy


def extract_session(
    subjid,
    date,
    *,
    model,
    derivatives=None,
    node_pool=None,
    skip_empty: bool = False,
    score_thresh: float = None,
    presence_frac: float = None,
    gap_limit: int = None,
) -> dict:
    """Extract one session's ``.slp`` files to per-video parquet plus a quality report.

    - ``model`` is required and may be None: nothing in a ``.slp`` records which model
      wrote it, so unknown provenance is recorded as null rather than guessed;
    - ``derivatives`` overrides the configured root, which is how the gate re-derives
      into a temp tree instead of over the baselines;
    - returns ``{results, outputs, quality, centroid_nodes}``.
    """
    params = parameters.extract_parameters(
        score_thresh=score_thresh, presence_frac=presence_frac, gap_limit=gap_limit
    )
    score_thresh = params["score_thresh"]
    presence_frac = params["presence_frac"]
    gap_limit = params["gap_limit"]

    node_pool_set = set(node_pool) if node_pool else None

    session = layout.layout_for(derivatives).find_session(subjid, date=str(date))
    results = layout.results_dir(session)
    if not results.exists():
        raise FileNotFoundError(f"Results directory not found: {results}")

    slp_files = layout.find_outputs(results, "*.slp")
    if not slp_files:
        raise FileNotFoundError(f"No .slp files found in {results}")

    print(f"Found {len(slp_files)} video(s) to process:")
    for i, f in enumerate(slp_files, 1):
        print(f"  {i}. {f.name}")

    default_nodes: List[str] = []
    session_nodes: List[str] = []

    # ----- Pass 1: load every video, flatten to a table, accumulate presence stats -----
    per_video: List[Tuple[int, str, str, pd.DataFrame]] = []
    present_counts: Dict[str, int] = {}
    occupied_total = 0

    for video_number, slp_path in enumerate(slp_files, 1):
        video_file_basename = infer_video_file_from_slp(slp_path)
        safe_tag = video_file_basename.replace(".avi", "")
        output_name = f"sleap_tracking_video{video_number}_{safe_tag}.parquet"
        print(f"\n[{video_number}/{len(slp_files)}] Reading: {slp_path.name}")

        labels = None
        try:
            labels = sleap_io.load_slp(str(slp_path))

            if getattr(labels, "skeletons", None):
                default_nodes = [node.name for node in labels.skeletons[0].nodes]

            rows = []
            for lf in getattr(labels, "labeled_frames", []):
                frame_idx = getattr(lf, "frame_idx", getattr(lf, "frame", pd.NA))
                for inst_idx, inst in enumerate(getattr(lf, "instances", [])):
                    node_names = node_names_for_instance(inst, default_nodes)
                    if not node_names:
                        continue
                    xy, scores = extract_points_and_scores(inst, len(node_names))

                    row = {"frame": frame_idx, "instance": inst_idx, "video_file": video_file_basename}
                    track = getattr(inst, "track", None)
                    if track is not None:
                        row["track"] = getattr(track, "name", None) or getattr(track, "id", None) or str(track)

                    for node_name, (x, y) in zip(node_names, xy):
                        row[f"{node_name}_x"] = to_number(x)
                        row[f"{node_name}_y"] = to_number(y)
                    if scores is not None:
                        for node_name, score in zip(node_names, scores):
                            row[f"{node_name}_score"] = to_number(score)

                    rows.append(row)

        except Exception as exc:
            if skip_empty:
                print(f"  ⚠️ Failed to read {slp_path.name}: {exc}; skipping")
                continue
            raise
        finally:
            del labels
            gc.collect()

        df = pd.DataFrame(rows)
        if df.empty:
            if skip_empty:
                print(f"  ⚠️ No pose data found in {slp_path}, skipping this file")
                continue
            raise ValueError(f"No pose data found in {slp_path}")

        df = df[pd.notna(df["frame"])]
        df.sort_values(["frame", "instance"], inplace=True)
        df.reset_index(drop=True, inplace=True)

        if not session_nodes:
            session_nodes = ([n for n in default_nodes if f"{n}_x" in df.columns]
                             or [c[:-2] for c in df.columns if c.endswith("_x")])

        # Accumulate confidence-gated presence for session-level node selection.
        # "occupied" = frames where >=1 node is present (i.e. the animal is on screen).
        node_masks = []
        for node in session_nodes:
            xcol, ycol, scol = f"{node}_x", f"{node}_y", f"{node}_score"
            if xcol not in df.columns or ycol not in df.columns:
                continue
            m = ~np.isnan(df[xcol].to_numpy(dtype=float)) & ~np.isnan(df[ycol].to_numpy(dtype=float))
            if scol in df.columns:
                m &= df[scol].to_numpy(dtype=float) >= score_thresh
            present_counts[node] = present_counts.get(node, 0) + int(m.sum())
            node_masks.append(m)
        if node_masks:
            occupied_total += int(np.any(np.vstack(node_masks), axis=0).sum())

        per_video.append((video_number, output_name, video_file_basename, df))
        print(f"    {len(df)} rows, frames {int(df['frame'].min())}-{int(df['frame'].max())}")

    if not per_video:
        raise ValueError("No pose data found in any .slp files for this session")

    # ----- Session-level node selection -----
    pool = [n for n in session_nodes if node_pool_set is None or n in node_pool_set]
    selected_nodes = [
        n for n in pool
        if occupied_total > 0 and present_counts.get(n, 0) / occupied_total >= presence_frac
    ]
    if not selected_nodes:
        selected_nodes = [n for n in pool if present_counts.get(n, 0) > 0]
        print("  ⚠️ No nodes met presence_frac; falling back to all nodes with any detections")
    if not selected_nodes:
        raise ValueError("No nodes with valid detections available for centroid computation")

    print(f"\nSession node selection (score>={score_thresh}, presence>={presence_frac:.0%} of "
          f"{occupied_total:,} occupied frames):")
    for n in pool:
        frac = present_counts.get(n, 0) / occupied_total if occupied_total else 0.0
        print(f"    {'[x]' if n in selected_nodes else '[ ]'} {n:<14} {frac:6.1%}")
    print(f"  Centroid nodes: {', '.join(selected_nodes)}")

    # ----- Pass 2: confidence gate + gap-limited interpolation + centroid, then write -----
    centroid_nodes_str = ";".join(selected_nodes)
    outputs = []
    for video_number, output_name, video_file_basename, df in per_video:
        cx, cy = _compute_session_centroid(df, selected_nodes, score_thresh, gap_limit)
        df["centroid_x"] = cx
        df["centroid_y"] = cy
        df["centroid_nodes"] = centroid_nodes_str

        output_path = layout.write_path(results, output_name)
        df.to_parquet(output_path, index=False)
        outputs.append(output_path)
        n_centroid = int(np.count_nonzero(~np.isnan(cx)))
        print(f"  ✓ Saved {output_path.name}: {len(df)} rows, {n_centroid} frames with centroid")

    report_path = quality.write_report(
        results,
        subjid=session.subjid,
        date=session.date,
        slp_files=slp_files,
        selection={
            "centroid_nodes": selected_nodes,
            "occupied_frames": occupied_total,
            "selection_pct": {
                n: 100 * present_counts.get(n, 0) / occupied_total if occupied_total else 0.0
                for n in pool
            },
        },
        params=params,
        model=model,
    )

    print("\n✅ All videos processed!")
    return {
        "results": results,
        "outputs": outputs,
        "quality": report_path,
        "centroid_nodes": selected_nodes,
    }


__all__ = [
    "extract_session",
    "node_names_for_instance", "extract_points_and_scores",
    "infer_video_file_from_slp", "to_number",
]
