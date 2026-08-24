import sleap_io
import pandas as pd
import numpy as np
import re
from pathlib import Path
import json
import gc
from typing import Dict, Iterable, List, Optional, Tuple, Union
from hypnose_behavior.io.paths import get_derivatives_root, get_data_root
from hypnose_behavior.io.loaders import load_all_streams, load_odor_mapping
from hypnose_behavior.io.load_results import load_session_results
from hypnose_behavior.utils.helpers import _get_from_cache, _update_cache


def _read_table(path: Union[str, Path]) -> pd.DataFrame:
    """Read a tracking table from .parquet or .csv.

    Parquet preserves dtypes (tz-aware datetimes, nullable Int64) natively, so no
    re-parsing is needed. CSV keeps the historical utf-8/latin1 fallback.
    """
    path = Path(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    try:
        return pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin1")


def _peek_video_file(path: Union[str, Path]) -> Optional[str]:
    """Return the first 'video_file' value from a tracking table, or None."""
    path = Path(path)
    try:
        if path.suffix == ".parquet":
            df = pd.read_parquet(path, columns=["video_file"])
        else:
            df = pd.read_csv(path, nrows=1)
        if "video_file" in df.columns and len(df) and pd.notna(df.iloc[0]["video_file"]):
            return str(df.iloc[0]["video_file"])
    except Exception:
        pass
    return None


def _find_tracking_files(results_dir: Path) -> List[Path]:
    """Find per-video sleap tracking files, preferring .parquet over .csv per stem."""
    by_stem: Dict[str, Path] = {}
    for ext in ("parquet", "csv"):  # parquet first so it wins for a given stem
        for f in sorted(results_dir.glob(f"sleap_tracking_video*.{ext}")):
            if f.name.startswith("._"):
                continue
            by_stem.setdefault(f.stem, f)
    return sorted(by_stem.values(), key=lambda p: p.name)


def _find_combined_file(results_dir: Path) -> Optional[Path]:
    """Find the combined timestamps file, preferring .parquet over .csv."""
    for ext in ("parquet", "csv"):
        matches = [m for m in sorted(results_dir.glob(f"*_combined_sleap_tracking_timestamps.{ext}"))
                   if not m.name.startswith("._")]
        if matches:
            return matches[0]
    return None


def _resolve_deriv_root(base_dir: Optional[Union[str, Path]]) -> Path:
    """Resolve the derivatives root from an optional base directory."""
    if base_dir:
        candidate = Path(base_dir).expanduser().resolve()
        if candidate.name != "derivatives" and (candidate / "derivatives").exists():
            return (candidate / "derivatives").resolve()
        return candidate
    return get_derivatives_root()


def _available_sessions(deriv_root: Path, subjid: int) -> Dict[str, Path]:
    """Return {date_str: session_dir} for a subject, or {} if the subject is missing."""
    subj_dirs = sorted(deriv_root.glob(f"sub-{int(subjid):03d}_id-*"))
    if not subj_dirs:
        return {}
    sessions: Dict[str, Path] = {}
    for ses_dir in subj_dirs[0].glob("ses-*_date-*"):
        m = re.search(r"date-(\d+)$", ses_dir.name)
        if m:
            sessions[m.group(1)] = ses_dir
    return sessions


def _normalize_date_arg(date_input, available: List[str]) -> List[str]:
    """Expand a date argument (None | single | list | (start, end) range) to a sorted
    list of available date strings. Mirrors the convention used elsewhere in the repo."""
    if date_input is None:
        return sorted(available)

    if isinstance(date_input, tuple) and len(date_input) == 2:
        start_dt = pd.to_datetime(str(date_input[0]), format="%Y%m%d", errors="coerce")
        end_dt = pd.to_datetime(str(date_input[1]), format="%Y%m%d", errors="coerce")
        if pd.isna(start_dt) or pd.isna(end_dt) or end_dt < start_dt:
            return []
        wanted = [d.strftime("%Y%m%d") for d in pd.date_range(start_dt, end_dt, freq="D")]
    elif isinstance(date_input, (list, set, tuple)):
        wanted = [str(d) for d in date_input]
    else:
        wanted = [str(date_input)]

    return sorted([d for d in wanted if d in available])


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


def sleap_labels_and_centroid(
    subjid,
    date,
    base_dir=None,
    node_pool=None,
    skip_empty: bool = False,
    score_thresh: float = 0.4,
    presence_frac: float = 0.7,
    gap_limit: int = 120,
):
    """
    Load all .slp files for a subject/date, flatten every frame/instance to a parquet
    table, and append a per-frame centroid computed with a robust, session-consistent
    pipeline:

      1. Confidence gate  - points with score < `score_thresh` are treated as missing.
      2. Node selection   - a single node set is chosen for the whole session: nodes
                            present (after gating) in >= `presence_frac` of *occupied*
                            frames (frames with >=1 node). Selection is drawn from
                            `node_pool` (default: all skeleton nodes), so it adapts to
                            whatever nodes a given model tracks.
      3. Interpolation    - each selected node's internal gaps up to `gap_limit` frames
                            are linearly interpolated (no extrapolation); longer gaps
                            (e.g. the animal off-screen) stay NaN.
      4. Centroid         - mean of the selected nodes per frame. Because the node set
                            is fixed and short gaps are filled, the centroid does not
                            jump when an individual node flickers in and out.

    The selected node set is recorded per output in a `centroid_nodes` column for QC.
    Returns a list of saved parquet paths. If skip_empty is True, files with no pose
    data (or unreadable .slp) are skipped instead of raising.
    """

    nan = float("nan")

    def resolve_deriv_root():
        if base_dir:
            candidate = Path(base_dir).expanduser().resolve()
            if candidate.name != "derivatives" and (candidate / "derivatives").exists():
                return (candidate / "derivatives").resolve()
            return candidate
        return get_derivatives_root()

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

    node_pool_set = set(node_pool) if node_pool else None

    deriv_dir = resolve_deriv_root()
    if not deriv_dir.exists():
        raise FileNotFoundError(f"Derivatives directory not found: {deriv_dir}")

    sub_str = f"sub-{str(subjid).zfill(3)}"
    date_str = str(date)

    subject_dirs = sorted(deriv_dir.glob(f"{sub_str}_id-*"))
    if not subject_dirs:
        raise FileNotFoundError(f"No subject directory found for {sub_str} under {deriv_dir}")
    subject_dir = subject_dirs[0]

    session_dirs = sorted(subject_dir.glob(f"ses-*_date-{date_str}"))
    if not session_dirs:
        raise FileNotFoundError(f"No session found for date {date_str} in {subject_dir}")
    session_dir = session_dirs[0]

    results_dir = session_dir / "saved_analysis_results"
    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    slp_files = sorted(results_dir.glob("*.slp"))
    if not slp_files:
        raise FileNotFoundError(f"No .slp files found in {results_dir}")

    print(f"Found {len(slp_files)} video(s) to process:")
    for i, f in enumerate(slp_files, 1):
        print(f"  {i}. {f.name}")

    default_nodes: List[str] = []
    session_nodes: List[str] = []

    # ----- Pass 1: load every video, flatten to a table, accumulate presence stats -----
    per_video: List[Tuple[int, Path, str, pd.DataFrame]] = []
    present_counts: Dict[str, int] = {}
    occupied_total = 0

    for video_number, slp_path in enumerate(slp_files, 1):
        video_file_basename = infer_video_file_from_slp(slp_path)
        safe_tag = video_file_basename.replace(".avi", "")
        output_path = results_dir / f"sleap_tracking_video{video_number}_{safe_tag}.parquet"
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

        per_video.append((video_number, output_path, video_file_basename, df))
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
    for video_number, output_path, video_file_basename, df in per_video:
        cx, cy = _compute_session_centroid(df, selected_nodes, score_thresh, gap_limit)
        df["centroid_x"] = cx
        df["centroid_y"] = cy
        df["centroid_nodes"] = centroid_nodes_str

        df.to_parquet(output_path, index=False)
        outputs.append(output_path)
        n_centroid = int(np.count_nonzero(~np.isnan(cx)))
        print(f"  ✓ Saved {output_path.name}: {len(df)} rows, {n_centroid} frames with centroid")

    print("\n✅ All videos processed!")
    return outputs


def get_video_frame_times(root, verbose=True):
    """
    Get synchronized timestamps for all video frames across all video files.
    
    Returns:
    --------
    pd.DataFrame with columns:
        - frame: global frame index across all videos
        - local_frame: frame index within the video file
        - time: synchronized timestamp (tz-aware Europe/London)
        - video_path: path to the .avi file
        - video_file: basename of the video file
    """
    
    # Load video metadata (already time-synchronized by load_all_streams)
    data = load_all_streams(root, verbose=verbose)
    video_data = data.get('video_data', pd.DataFrame())
    
    if video_data.empty:
        if verbose:
            print("No video data found")
        return pd.DataFrame()
    
    # video_data.index is already the synchronized time
    # _frame column is the frame index within each video file
    # _path column is the path to the video file
    
    result = pd.DataFrame({
        'time': video_data.index,
        'local_frame': video_data['_frame'],
        'video_path': video_data['_path'],
        'hw_counter': video_data.get('hw_counter'),
        'hw_timestamp': video_data.get('hw_timestamp'),
    })
    
    # Add global frame index (continuous across all videos)
    result['frame'] = range(len(result))

    # Normalize dtypes for hardware columns
    if 'hw_counter' in result:
        result['hw_counter'] = pd.to_numeric(result['hw_counter'], errors='coerce').astype('Int64')
    if 'hw_timestamp' in result:
        result['hw_timestamp'] = pd.to_numeric(result['hw_timestamp'], errors='coerce')
    
    # Add disambiguated video file name: behav folder + basename
    def _disambig_name(p):
        if pd.isna(p):
            return None
        try:
            path = Path(p)
            behav = path.parent.parent.name  # .../behav/<behav>/VideoData/file.avi
            base = path.name
            return f"{behav}__{base}"
        except Exception:
            return Path(p).name if pd.notna(p) else None

    result['video_file'] = result['video_path'].apply(_disambig_name)
    
    # Reorder columns
    result = result[['frame', 'local_frame', 'hw_counter', 'hw_timestamp', 'time', 'video_path', 'video_file']]
    
    if verbose:
        print(f"Found {len(result)} video frames across {result['video_file'].nunique()} video files")
        print(f"Time range: {result['time'].min()} to {result['time'].max()}")
    
    return result


def add_timestamps_to_sleap_tracking(subjid, date, save_output=True):
    """
    Add synchronized timestamps to all SLEAP tracking CSVs for a session.
    Matches sleap_tracking_videox.csv files to video files by index (1-indexed filename to 0-indexed video order).
    Handles gaps in frame coverage and partial tracking.
    
    Parameters:
    -----------
    subjid : int
        Subject ID
    date : int or str
        Session date (e.g., 20251027)
    save_output : bool, optional
        If True, saves the combined timestamped CSV (default: True)
    
    Returns:
    --------
    pd.DataFrame : Combined SLEAP tracking data with added columns:
        - 'time': synchronized timestamp (tz-aware Europe/London)
        - 'global_frame': frame index across all videos in session
        - 'video_file': video filename
    """
    import re
    
    # Build path to derivatives directory
    base_path = get_data_root() / "rawdata"
    derivatives_dir = base_path.resolve().parent / "derivatives"
    
    # Find subject and session directories
    sub_str = f"sub-{str(subjid).zfill(3)}"
    subject_dirs = list(derivatives_dir.glob(f"{sub_str}_id-*"))
    if not subject_dirs:
        raise FileNotFoundError(f"No subject directory found for {sub_str}")
    subject_dir = subject_dirs[0]
    
    date_str = str(date)
    session_dirs = list(subject_dir.glob(f"ses-*_date-{date_str}"))
    if not session_dirs:
        raise FileNotFoundError(f"No session found for date {date_str}")
    session_dir = session_dirs[0]
    
    # Find all sleap_tracking_videox.csv files
    results_dir = session_dir / "saved_analysis_results"
    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")
    
    tracking_csvs = _find_tracking_files(results_dir)
    if not tracking_csvs:
        raise FileNotFoundError(f"No sleap_tracking_video* files (.parquet/.csv) found in {results_dir}")
    
    print(f"Found {len(tracking_csvs)} SLEAP tracking file(s)")
    
    # Step 1: Find behavior directory and experiment folders
    behav_dirs = list(base_path.glob(f"{sub_str}_id-*/{Path(session_dirs[0]).name}/behav"))
    if not behav_dirs:
        raise FileNotFoundError(f"Behavior directory not found")
    behav_dir = behav_dirs[0]
    
    # Find all experiment folders (numbered time-stamped directories)
    exp_folders = sorted([d for d in behav_dir.iterdir() 
                         if d.is_dir() and re.match(r'\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}', d.name)])
    
    if not exp_folders:
        raise FileNotFoundError(f"No experiment folders found in {behav_dir}")
    
    print(f"Found {len(exp_folders)} experiment folder(s)")
    
    # Step 2: Load video frame times from each experiment
    all_frame_times = []
    for exp_idx, exp_folder in enumerate(exp_folders):
        try:
            frame_times = get_video_frame_times(exp_folder, verbose=False)
            if not frame_times.empty:
                all_frame_times.append(frame_times)
                print(f"  Loaded {len(frame_times)} frames from experiment {exp_idx}")
        except Exception as e:
            print(f"  Warning: Could not load experiment {exp_idx}: {e}")
            continue
    
    if not all_frame_times:
        raise ValueError("No video metadata found for this session")
    
    # Step 3: Combine and sort all video frame times
    combined_frame_times = pd.concat(all_frame_times, ignore_index=True)
    combined_frame_times['time'] = pd.to_datetime(combined_frame_times['time'], errors='coerce')
    combined_frame_times['local_frame'] = pd.to_numeric(combined_frame_times['local_frame'], errors='coerce').astype('Int64')
    combined_frame_times = combined_frame_times.sort_values('time').reset_index(drop=True)
    combined_frame_times['global_frame'] = pd.Series(range(len(combined_frame_times)), dtype='Int64')
    
    print(f"Total: {len(combined_frame_times):,} frames from {combined_frame_times['video_file'].nunique()} video file(s)")
    
    # Get unique video files in order
    video_files_ordered = combined_frame_times['video_file'].unique()
    frames_by_video = {vf: df.copy() for vf, df in combined_frame_times.groupby('video_file')}
    print(f"\nVideo files in order:")
    for i, vf in enumerate(video_files_ordered):
        print(f"  {i+1}. {vf}")
    
    # Step 4: Match SLEAP files to videos by index
    # Extract video number from filename (sleap_tracking_video2.csv -> 2)
    sleap_video_mapping = {}  # maps video_file to SLEAP CSV path
    
    for csv_path in tracking_csvs:
        # Try to extract mapping hints
        video_file_hint = _peek_video_file(csv_path)

        name_hint = None
        m_name = re.search(r"sleap_tracking_video\d+_(.+)\.(?:csv|parquet)", csv_path.name)
        if m_name:
            name_hint = m_name.group(1)
            if not name_hint.endswith(".avi"):
                name_hint = f"{name_hint}.avi"

        m_num = re.search(r'sleap_tracking_video(\d+)', csv_path.name)
        video_num = int(m_num.group(1)) if m_num else None
        video_idx = video_num - 1 if video_num is not None else None

        # Resolve target video_file in priority: column hint, name hint, numeric index
        target_video_file = None
        source = None
        if video_file_hint and video_file_hint in frames_by_video:
            target_video_file = video_file_hint
            source = "video_file column"
        elif name_hint and name_hint in frames_by_video:
            target_video_file = name_hint
            source = "filename suffix"
        elif video_idx is not None and video_idx < len(video_files_ordered):
            target_video_file = video_files_ordered[video_idx]
            source = "numeric index"

        if not target_video_file:
            print(f"Warning: Could not map {csv_path.name} (hints: {video_file_hint}, {name_hint}); skipping")
            continue

        sleap_video_mapping[target_video_file] = csv_path
        extra = f" via {source}" if source else ""
        num_txt = f" (video {video_num})" if video_num is not None else ""
        print(f"Matched {csv_path.name}{num_txt} to {target_video_file}{extra}")
    
    if not sleap_video_mapping:
        raise ValueError("No SLEAP files could be matched to videos")
    
    # Step 5: Add timestamps to each SLEAP tracking CSV
    all_tracking = []
    
    for video_file, csv_path in sleap_video_mapping.items():
        tracking_df = _read_table(csv_path)
        
        # Get frame times for this specific video
        video_frames = frames_by_video.get(video_file, pd.DataFrame()).copy()
        if video_frames.empty:
            print(f"Warning: Video '{video_file}' not found in frame times, skipping {csv_path.name}")
            continue
        
        print(f"\nProcessing {csv_path.name}:")
        print(f"  SLEAP frames: {tracking_df['frame'].min():.0f} - {tracking_df['frame'].max():.0f} ({len(tracking_df)} rows)")
        print(f"  Video local frames: {video_frames['local_frame'].min():.0f} - {video_frames['local_frame'].max():.0f}")
        
        # Merge to add timestamps - match on 'frame' column (local frame index within video)
        # Use left join to keep all SLEAP frames
        # Prefer hardware counter/timestamp for downstream alignment; merge keeps both
        merge_cols = ['local_frame', 'time', 'global_frame']
        if 'hw_counter' in video_frames.columns:
            merge_cols.append('hw_counter')
        if 'hw_timestamp' in video_frames.columns:
            merge_cols.append('hw_timestamp')

        result = tracking_df.merge(
            video_frames[merge_cols],
            left_on='frame',
            right_on='local_frame',
            how='left'
        ).drop(columns=['local_frame'], errors='ignore')
        
        result['video_file'] = video_file
        
        # Check how many frames got timestamps
        matched_frames = result['time'].notna().sum()
        print(f"  Matched {matched_frames}/{len(result)} frames to timestamps")
        
        all_tracking.append(result)
    
    if not all_tracking:
        raise ValueError("No SLEAP tracking data could be matched to video metadata")
    
    # Step 6: Combine and sort all tracking data by time
    combined = pd.concat(all_tracking, ignore_index=True)
    combined = combined.sort_values('time', na_position='last').reset_index(drop=True)
    
    # Reorder columns - put frame/time/video info first
    priority_cols = ['frame', 'time', 'global_frame', 'hw_counter', 'hw_timestamp', 'video_file', 'instance']
    other_cols = [c for c in combined.columns if c not in priority_cols]
    priority_cols = [c for c in priority_cols if c in combined.columns]
    combined = combined[priority_cols + other_cols]
    
    # Save output
    if save_output:
        output_filename = f"sub-{str(subjid).zfill(3)}_ses-{date_str}_combined_sleap_tracking_timestamps.parquet"
        output_path = results_dir / output_filename
        combined.to_parquet(output_path, index=False)
        print(f"\nSaved: {output_path}")
    
    if 'time' in combined.columns and combined['time'].notna().any():
        duration = (combined['time'].max() - combined['time'].min()).total_seconds()
        print(f"Combined: {len(combined):,} frames, {duration/60:.1f} minutes")
        print(f"  Frames with timestamps: {combined['time'].notna().sum()}")
        print(f"  Frames without timestamps (gaps): {combined['time'].isna().sum()}")
    else:
        print(f"Combined: {len(combined):,} frames (no timestamps matched)")
    
    return combined



def annotate_videos_with_sleap_and_trials(subjid, date, base_dir=None, output_suffix="sleap_visualization",
                                          centroid_radius=8, centroid_color='red', rotate_deg=0,
                                          time_window=None, reward_display_s: float = 1.0,
                                          video_indices=None, mark_timepoint: str | None = None):
    """
    Annotate behavior videos with SLEAP centroid tracking and odor overlays.

    Creates MP4 videos with:
    - Red dot at animal centroid position (from SLEAP tracking) on frames that have centroid data
    - Odor label box near the poke port (top-center in unrotated view), rotated with the video
    - Optional trimming to a time window
    - Optional rotation by 90/180/270 degrees (clockwise)
    
    Parameters:
    -----------
    subjid : int
        Subject ID
    date : int or str
        Session date (e.g., 20251029)
    base_dir : str or Path, optional
        Base data directory (default: get_data_root())
    output_suffix : str, optional
        Suffix for output files (default: "sleap_visualization")
    centroid_radius : int, optional
        Radius of centroid marker in pixels (default: 8)
    centroid_color : str, optional
        Color of centroid marker (default: 'red')
    rotate_deg : int, optional
        Rotate output video by 0, 90, 180, or 270 degrees clockwise (default: 0)
    time_window : tuple[str | pd.Timedelta, str | pd.Timedelta] | list[tuple] | None, optional
        Provide a single (start, end) pair to trim once, or a list of pairs to render
        multiple clips per video. Windows are relative to the first frame timestamp of
        that video (e.g., ("0:00:00", "0:10:00") for first 10 minutes). Each window
        outputs a separate file suffixed with _window1, _window2, etc. Passing None keeps
        the full video (_full suffix).
    reward_display_s : float, optional
        Duration to display reward labels after supply port onset (default: 1.0 seconds)
    video_indices : int | Iterable[int] | None, optional
        1-based video numbers to process (matching *_videoN.csv ordering). If None, process all.
    mark_timepoint : str | None, optional
        Absolute clock time ("HH:MM:SS" or "HH:MM:SS.mmm") within the session to mark with a
        downward green triangle at the center of the video for 2 seconds. Ignored if None or not parsable.
    
    Returns:
    --------
    list : Paths to generated output MP4 files
    """
    from pathlib import Path
    import pandas as pd
    import numpy as np
    import cv2
    from PIL import Image, ImageDraw, ImageFont
    from tqdm import tqdm

    def rotate_point(x, y, w, h, deg):
        if deg == 0:
            return x, y
        if deg == 90:
            return h - 1 - y, x
        if deg == 180:
            return w - 1 - x, h - 1 - y
        if deg == 270:
            return y, w - 1 - x
        raise ValueError("rotate_deg must be one of {0, 90, 180, 270}")

    def rotate_frame(frame_bgr, deg):
        if deg == 0:
            return frame_bgr
        if deg == 90:
            return cv2.rotate(frame_bgr, cv2.ROTATE_90_CLOCKWISE)
        if deg == 180:
            return cv2.rotate(frame_bgr, cv2.ROTATE_180)
        if deg == 270:
            return cv2.rotate(frame_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)
        raise ValueError("rotate_deg must be one of {0, 90, 180, 270}")

    def parse_single_time_window(window):
        if window is None:
            return None
        if len(window) != 2:
            raise ValueError("Each time window must be a (start, end) pair")
        start_td = pd.to_timedelta(window[0])
        end_td = pd.to_timedelta(window[1])
        if pd.isna(start_td) or pd.isna(end_td):
            raise ValueError("Could not parse time window values")
        if end_td < start_td:
            raise ValueError("time_window end must be >= start")
        return start_td, end_td

    def normalize_time_windows(window_spec):
        if window_spec is None:
            return [None]

        if isinstance(window_spec, (list, tuple)):
            if not window_spec:
                return [None]

            first = window_spec[0]
            is_sequence = isinstance(first, (list, tuple)) and not isinstance(first, str)

            # If the user passed a single (start, end) tuple/list, treat it as one window
            if not is_sequence and len(window_spec) == 2 and not isinstance(window_spec[0], (list, tuple)):
                return [parse_single_time_window(window_spec)]

            # Otherwise treat the outer iterable as a collection of windows
            windows = []
            for idx, win in enumerate(window_spec, 1):
                if win is None:
                    windows.append(None)
                    continue
                if not isinstance(win, (list, tuple)):
                    raise ValueError(f"Time window #{idx} must be a (start, end) pair")
                windows.append(parse_single_time_window(win))
            return windows

        raise ValueError("time_window must be None, a single (start, end) pair, or an iterable of such pairs")

    def parse_mark_timepoint(ts_str):
        if not ts_str:
            return None
        ts_str = str(ts_str).strip()
        # Try as clock time with date
        ts_abs = pd.to_datetime(f"{date_str} {ts_str}", errors="coerce")
        if pd.isna(ts_abs):
            # Try as timedelta from midnight
            td = pd.to_timedelta(ts_str, errors="coerce")
            if pd.isna(td):
                return None
            # Anchor to date midnight
            ts_abs = pd.to_datetime(date_str, errors="coerce") + td
        return ts_abs

    def disambig_video_name(path: Path) -> str:
        """Return behav-prefixed video name used in combined timestamps."""
        try:
            behav = path.parent.parent.name  # .../behav/<behav>/VideoData/file.avi
            return f"{behav}__{path.name}"
        except Exception:
            return path.name
    
    date_str = str(date)

    # Default base directory (use configured data root if none supplied)
    if base_dir is None:
        base_dir = get_data_root()
    else:
        base_dir = Path(base_dir)
    
    # Load behavior data (used for valve/odor timings if available)
    behavior = load_session_results(subjid, date)
    
    # Find directories
    deriv_dir = base_dir / "derivatives"
    subj_pattern = f"sub-{subjid:03d}_*"
    subj_dirs = list(deriv_dir.glob(subj_pattern))
    
    if not subj_dirs:
        raise FileNotFoundError(f"No subject directory found for pattern {subj_pattern}")
    
    subj_dir = subj_dirs[0]
    session_pattern = f"ses-*_date-{date_str}"
    session_dirs = list(subj_dir.glob(session_pattern))
    
    if not session_dirs:
        raise FileNotFoundError(f"No session found for date {date}")
    
    session_dir = session_dirs[0]
    results_dir = session_dir / "saved_analysis_results"
    
    # Find video files
    rawdata_dir = base_dir / "rawdata"
    rawdata_subj_dirs = list(rawdata_dir.glob(f"sub-{subjid:03d}_*"))
    
    if not rawdata_subj_dirs:
        raise FileNotFoundError(f"No rawdata subject directory found")
    
    rawdata_subj_dir = rawdata_subj_dirs[0]
    rawdata_session_dir = next(rawdata_subj_dir.glob(f"ses-*_date-{date_str}"), None)
    
    if not rawdata_session_dir:
        raise FileNotFoundError(f"No rawdata session directory found for date {date}")
    
    video_files_all = sorted(
        vf for vf in rawdata_session_dir.glob("behav/*/VideoData/*.avi")
        if not vf.name.startswith("._")
    )

    # Filter by requested indices (1-based)
    if video_indices is None:
        video_files = [(idx, vf) for idx, vf in enumerate(video_files_all, 1)]
    else:
        if isinstance(video_indices, (int, np.integer)):
            selected = {int(video_indices)}
        else:
            try:
                selected = {int(v) for v in video_indices}
            except Exception:
                selected = set()
        video_files = [(idx, vf) for idx, vf in enumerate(video_files_all, 1) if idx in selected]

    if not video_files:
        raise FileNotFoundError(f"No video files found for requested indices: {video_indices}")
    
    if not video_files:
        raise FileNotFoundError(f"No video files found in {rawdata_session_dir}")
    
    cache_kind = "sleap_annotate"
    cached = _get_from_cache(subjid, date, kind=cache_kind)
    combined_df = None
    valve_events = None
    supply_events = None

    if isinstance(cached, dict):
        combined_df = cached.get("combined_df")
        valve_events = cached.get("valve_events")
        supply_events = cached.get("supply_events")

    if combined_df is None:
        combined_ts_file = _find_combined_file(results_dir)
        if combined_ts_file is None:
            raise FileNotFoundError(f"No combined timestamps file found in {results_dir}")

        combined_df = _read_table(combined_ts_file)
        combined_df["video_file"] = combined_df["video_file"].astype(str)
        print(f"Loaded combined timestamps: {len(combined_df)} frames")
    else:
        combined_df = combined_df.copy()
        combined_df["video_file"] = combined_df["video_file"].astype(str)
        print(f"Loaded combined timestamps from cache: {len(combined_df)} frames")
    
    # Get odor/valve timings directly from raw streams with real timestamps
    if valve_events is None or supply_events is None:
        valve_events = []
        supply_events = []
        try:
            # Pick the first experiment root from the first video path (behav/<exp>/VideoData)
            sample_video = next(
                iter(
                    sorted(
                        vf for vf in (base_dir / "rawdata").glob(
                            f"sub-{subjid:03d}_*/ses-*_date-{date}/behav/*/VideoData/*.avi"
                        )
                        if not vf.name.startswith("._")
                    )
                )
            )
            exp_root = sample_video.parent.parent  # .../behav/<exp>

            streams = load_all_streams(exp_root, apply_corrections=True, verbose=False)
            odor_map = load_odor_mapping(exp_root, data=streams, verbose=False)

            olfactometer_valves = odor_map.get('olfactometer_valves', {}) if odor_map else {}
            valve_to_odor = odor_map.get('valve_to_odor', {}) if odor_map else {}

            for olf_id, valve_df in (olfactometer_valves or {}).items():
                if valve_df is None or getattr(valve_df, 'empty', True):
                    continue
                for valve_idx, valve_col in enumerate(valve_df.columns):
                    valve_key = f"{olf_id}{valve_idx}"
                    odor_name = valve_to_odor.get(valve_key)
                    if not odor_name or str(odor_name).lower() == 'purge':
                        continue
                    series = valve_df[valve_col].astype(bool)
                    on_edges = series & ~series.shift(1, fill_value=False)
                    off_edges = ~series & series.shift(1, fill_value=False)
                    on_times = list(series.index[on_edges])
                    off_times = list(series.index[off_edges])
                    j = 0
                    for on_time in on_times:
                        while j < len(off_times) and off_times[j] <= on_time:
                            j += 1
                        if j >= len(off_times):
                            break
                        valve_events.append({
                            'start_time': on_time,
                            'end_time': off_times[j],
                            'odor_name': str(odor_name)
                        })
            valve_events.sort(key=lambda ev: ev['start_time'])

            # Supply port pulses (reward delivery), show for reward_display_s after onset
            for port_num, key in ((1, 'pulse_supply_1'), (2, 'pulse_supply_2')):
                series = streams.get(key)
                if series is None or getattr(series, 'empty', True):
                    continue
                for ts in getattr(series, 'index', []):
                    supply_events.append({
                        'start_time': ts,
                        'end_time': ts + pd.Timedelta(seconds=float(reward_display_s)),
                        'port': port_num
                    })
            supply_events.sort(key=lambda ev: ev['start_time'])
        except Exception:
            valve_events = []
            supply_events = []

        # Persist to cache for reuse
        _update_cache(subjid, [date], {date: {
            "combined_df": combined_df,
            "valve_events": valve_events,
            "supply_events": supply_events,
        }}, kind=cache_kind)

    def lookup_odor(ts):
        if ts is None or not valve_events:
            return None
        for ev in valve_events:
            if ev['start_time'] <= ts <= ev['end_time']:
                return ev['odor_name']
            if ev['start_time'] > ts:
                break
        return None
    
    # Load fonts
    # Try common Windows fonts first; fall back to macOS, then default.
    font_large = None
    font_small = None
    for font_path in [
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/ARIAL.TTF",
        "/System/Library/Fonts/Helvetica.ttc",
    ]:
        try:
            font_large = ImageFont.truetype(font_path, 100)
            font_small = ImageFont.truetype(font_path, 60)
            break
        except Exception:
            continue
    if font_large is None or font_small is None:
        font_large = ImageFont.load_default()
        font_small = ImageFont.load_default()
    
    # Parse optional marker timepoint (absolute clock within session)
    mark_ts_abs = parse_mark_timepoint(mark_timepoint) if mark_timepoint else None
    if mark_timepoint and mark_ts_abs is None:
        print(f"Warning: could not parse mark_timepoint='{mark_timepoint}', skipping marker")

    time_windows_to_process = normalize_time_windows(time_window)

    output_paths = []
    
    # Process each video
    for video_idx, (video_num, video_path) in enumerate(video_files, 1):
        video_key = disambig_video_name(video_path)
        print(f"\nProcessing video {video_idx}/{len(video_files)} (original #{video_num}): {video_key}")
        
        # Filter combined_df for this video
        df_video = combined_df[combined_df['video_file'] == video_key].copy()
        
        if df_video.empty:
            print(f"  ⚠️ No timestamps found for video {video_key}, skipping")
            continue
        
        # Convert time column to datetime
        df_video['time'] = pd.to_datetime(df_video['time'])
        df_video = df_video.reset_index(drop=True)

        # Aggregate per frame (average across instances) to ensure a single centroid per frame
        df_video = (df_video
                .groupby('frame', as_index=False)
                .agg({'centroid_x': 'mean', 'centroid_y': 'mean', 'time': 'first'}))

        print(f"  Found {len(df_video)} frames with timestamps")

        # Probe video metadata once; reopen per window for decoding
        cap_probe = cv2.VideoCapture(str(video_path))
        fps = cap_probe.get(cv2.CAP_PROP_FPS)
        width = int(cap_probe.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap_probe.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap_probe.get(cv2.CAP_PROP_FRAME_COUNT))
        cap_probe.release()

        print(f"  Video: {width}x{height} @ {fps} fps, {total_frames} frames")

        df_video_base = df_video.copy()

        for window_idx, window in enumerate(time_windows_to_process, 1):
            df_window = df_video_base.copy()
            window_label = f"window{window_idx}" if window is not None else "full"

            if window is not None:
                base_time = df_window['time'].min()
                if pd.isna(base_time):
                    print(f"  ⚠️ Window {window_label}: missing base timestamp, skipping")
                    continue
                start_abs = base_time + window[0]
                end_abs = base_time + window[1]
                df_window = df_window[(df_window['time'] >= start_abs) & (df_window['time'] <= end_abs)].copy()
                if df_window.empty:
                    print(f"  ⚠️ Window {window_label}: no frames in requested time bounds, skipping")
                    continue
                print(f"  Window {window_label}: {window[0]} - {window[1]} ({len(df_window)} frames)")
            else:
                print(f"  Window {window_label}: full duration ({len(df_window)} frames)")

            df_window['frame'] = pd.to_numeric(df_window['frame'], errors='coerce').astype('Int64')
            df_window = df_window.dropna(subset=['frame'])
            row_map = df_window.set_index('frame').to_dict('index')
            valid_frames = sorted(row_map.keys())
            if not valid_frames:
                print(f"  ⚠️ Window {window_label}: no valid frames with centroid, skipping")
                continue

            start_frame = int(valid_frames[0])
            end_frame = int(valid_frames[-1])

            # Load video with OpenCV for frame-by-frame processing
            cap = cv2.VideoCapture(str(video_path))
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

            # Rotation-aware output size
            if rotate_deg in (90, 270):
                out_size = (height, width)
            else:
                out_size = (width, height)

            suffix_full = "_full" if window is None else f"_window{window_idx}"
            output_path = results_dir / f"{output_suffix}_rotdeg_{rotate_deg}_video{video_num}{suffix_full}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(output_path), fourcc, fps, out_size)
            output_paths.append(output_path)

            print(f"  Saving to: {output_path}")

            frames_to_process = end_frame - start_frame + 1

            mark_window = None
            if mark_ts_abs is not None:
                mark_window = (mark_ts_abs, mark_ts_abs + pd.Timedelta(seconds=2.0))

            with tqdm(total=frames_to_process, desc="  Encoding", unit="frames") as pbar:
                current_frame = start_frame
                while current_frame <= end_frame:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    frame = rotate_frame(frame, rotate_deg)
                    frame_h, frame_w = frame.shape[:2]

                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_pil = Image.fromarray(frame_rgb)
                    draw = ImageDraw.Draw(frame_pil)

                    row = row_map.get(current_frame)

                    if row is not None and not pd.isna(row.get('centroid_x')) and not pd.isna(row.get('centroid_y')):
                        cx_raw = row['centroid_x']
                        cy_raw = row['centroid_y']
                        cx_rot, cy_rot = rotate_point(cx_raw, cy_raw, width, height, rotate_deg)
                        cx, cy = int(cx_rot), int(cy_rot)
                        draw.ellipse([cx - centroid_radius, cy - centroid_radius,
                                     cx + centroid_radius, cy + centroid_radius],
                                    fill=centroid_color, outline=centroid_color)

                    frame_time = pd.to_datetime(row.get('time')) if (row is not None and 'time' in row) else None
                    odor_label = None
                    if frame_time is not None and valve_events:
                        odor_label = lookup_odor(frame_time.to_datetime64())

                    if odor_label:
                        display_odor = re.sub(r"(?i)^odor[_\-\s]*", "", str(odor_label)) or str(odor_label)
                        odor_text = f"Odor: {display_odor}"
                        anchor_x_raw = width // 2
                        anchor_y_raw = 30
                        anchor_x, anchor_y = rotate_point(anchor_x_raw, anchor_y_raw, width, height, rotate_deg)

                        if rotate_deg == 90:
                            anchor_x -= 185
                            anchor_y -= 30

                        draw.text((anchor_x, anchor_y), odor_text, fill=(255, 40, 90), font=font_small)

                    active_ports = set()
                    if frame_time is not None and supply_events:
                        active_ports = {
                            ev['port'] for ev in supply_events
                            if ev['start_time'] <= frame_time <= ev['end_time']
                        }

                    if active_ports:
                        reward_text = "Reward"
                        reward_color = (0, 220, 0)
                        x_offset = 140
                        y_offset = 5

                        if rotate_deg == 0:
                            y_offset = 40

                        bbox_reward = draw.textbbox((0, 0), reward_text, font=font_small)
                        text_h = bbox_reward[3] - bbox_reward[1]

                        if 2 in active_ports:
                            bl_x_raw = x_offset
                            bl_y_raw = height - y_offset - text_h
                            bl_x, bl_y = rotate_point(bl_x_raw, bl_y_raw, width, height, rotate_deg)
                            draw.text((bl_x, bl_y), reward_text, fill=reward_color, font=font_small)

                        if 1 in active_ports:
                            br_x_raw = width - (x_offset + 230)
                            br_y_raw = height - y_offset - text_h 
                            br_x, br_y = rotate_point(br_x_raw, br_y_raw, width, height, rotate_deg)
                            draw.text((br_x, br_y), reward_text, fill=reward_color, font=font_small)

                            if rotate_deg == 90:
                                br_y -= 20

                    if mark_window and frame_time is not None:
                        if mark_window[0] <= frame_time <= mark_window[1]:
                            cx = frame_w // 2
                            cy = frame_h // 2
                            tri = [(cx - 60, cy - 40), (cx + 60, cy - 40), (cx, cy + 40)]
                            draw.polygon(tri, fill=(0, 200, 0))

                    frame_annotated = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)
                    out.write(frame_annotated)

                    current_frame += 1
                    pbar.update(1)

            cap.release()
            out.release()

            print(f"  ✓ Completed {window_label}!")
    
    print(f"\n✅ All videos processed and saved!")
    return output_paths


def process_sleap_sessions(subjid: Union[int, Iterable[int]],
                           date: Optional[Union[int, Iterable[int], Tuple[int, int]]] = None,
                           base_dir: Optional[Union[str, Path]] = None,
                           node_pool: Optional[List[str]] = None,
                           save_output: bool = True,
                           score_thresh: float = 0.4,
                           presence_frac: float = 0.7,
                           gap_limit: int = 120,
                           recompute: bool = False) -> Dict[int, Dict[str, List[Tuple[str, str]]]]:
    """
    Wrapper to run SLEAP centroid extraction and timestamp merging across subjects/dates.

    Parameters
    ----------
    subjid : int | Iterable[int]
        Single subject id or collection of ids.
    date : int | Iterable[int] | tuple[int, int] | None
        Single date, list of dates, inclusive range (start, end), or None to process all
        available dates for each subject.
    base_dir : str | Path | None
        Optional base directory. If provided, will try <base_dir>/derivatives for SLEAP files.
    node_pool : list[str] | None
        Candidate nodes to select the centroid from (default: all skeleton nodes).
        Forwarded to sleap_labels_and_centroid.
    save_output : bool
        Forwarded to add_timestamps_to_sleap_tracking.
    score_thresh, presence_frac, gap_limit :
        Centroid pipeline parameters, forwarded to sleap_labels_and_centroid.
    recompute : bool
        If False (default), skip sessions where the combined timestamps CSV already exists.
        If True, always recompute even when outputs are present.

    Returns
    -------
    dict
        Nested summary keyed by subject id with lists of (date, reason) tuples for
        "success", "failed", and "skipped" entries.
    """

    def resolve_deriv_root() -> Path:
        if base_dir:
            candidate = Path(base_dir).expanduser().resolve()
            if candidate.name != "derivatives" and (candidate / "derivatives").exists():
                return (candidate / "derivatives").resolve()
            return candidate
        return get_derivatives_root()

    def to_subject_list(val: Union[int, Iterable[int]]) -> List[int]:
        if isinstance(val, (list, tuple, set)):
            return [int(v) for v in val]
        return [int(val)]

    def available_sessions_for_subject(root: Path, subject: int) -> Dict[str, Path]:
        subj_dirs = sorted(root.glob(f"sub-{subject:03d}_id-*"))
        if not subj_dirs:
            return {}
        subj_dir = subj_dirs[0]
        sessions = {}
        for ses_dir in subj_dir.glob("ses-*_date-*"):
            match = re.search(r"date-(\d+)$", ses_dir.name)
            if match:
                sessions[match.group(1)] = ses_dir
        return sessions

    def normalize_dates(date_input, available: List[str]) -> Tuple[List[str], List[Tuple[str, str]]]:
        """Return (dates_to_process, skipped_reasons)."""
        skipped_local: List[Tuple[str, str]] = []

        if date_input is None:
            return sorted(available), skipped_local

        if isinstance(date_input, tuple) and len(date_input) == 2:
            start_raw, end_raw = date_input
            start_dt = pd.to_datetime(str(start_raw), format="%Y%m%d", errors="coerce")
            end_dt = pd.to_datetime(str(end_raw), format="%Y%m%d", errors="coerce")
            if pd.isna(start_dt) or pd.isna(end_dt) or end_dt < start_dt:
                skipped_local.append((f"{start_raw}-{end_raw}", "Invalid date range"))
                return [], skipped_local
            dates = [d.strftime("%Y%m%d") for d in pd.date_range(start_dt, end_dt, freq="D")]
        else:
            dates = [date_input] if not isinstance(date_input, (list, set, tuple)) else list(date_input)
            dates = [str(d) for d in dates]

        normalized: List[str] = []
        for d in dates:
            if d in available:
                normalized.append(d)
            else:
                skipped_local.append((d, "Session directory not found"))
        return sorted(normalized), skipped_local

    def count_tracking_files(results_dir: Path) -> int:
        return len(_find_tracking_files(results_dir))

    def process_single(subj: int, date_str: str, session_dir: Path, results_dir: Path, centroid_found: int):
        messages = [f"\nSubject {subj:02d} Date {date_str} - Processing SLEAP Output:"]
        centroid_done = 0
        timestamp_found = 0
        matched_videos = 0
        saved_flag = False

        try:
            outputs = sleap_labels_and_centroid(subj, int(date_str), base_dir=base_dir, node_pool=node_pool,
                                                skip_empty=True, score_thresh=score_thresh,
                                                presence_frac=presence_frac, gap_limit=gap_limit)
            centroid_done = len(outputs)
            timestamp_found = count_tracking_files(results_dir)

            combined = add_timestamps_to_sleap_tracking(subj, int(date_str), save_output=save_output)
            if combined is not None and not combined.empty:
                matched_videos = combined["video_file"].nunique()
            saved_flag = _find_combined_file(results_dir) is not None

            messages.append(f"Centroid Processing: found {centroid_found} video(s) to process.")
            messages.append(f"        Successfully processed {centroid_done} videos.")
            messages.append(f"Timestamp Processing: found {timestamp_found} video(s) to process.")
            messages.append(f"         Successfully matched {matched_videos} video(s) to sleap files, combined sleap file saved = {saved_flag}")

            return {"status": "success", "reason": "", "messages": messages, "subj": subj, "date": date_str}
        except FileNotFoundError as exc:
            return {"status": "skipped", "reason": str(exc), "messages": messages + [f"  Skipped: {exc}"], "subj": subj, "date": date_str}
        except Exception as exc:
            return {"status": "failed", "reason": str(exc), "messages": messages + [f"  Failed: {exc}"], "subj": subj, "date": date_str}

    summary: Dict[int, Dict[str, List[Tuple[str, str]]]] = {}
    deriv_root = resolve_deriv_root()
    subjects = to_subject_list(subjid)

    tasks: List[Tuple[int, str, Path, Path, int]] = []

    for subj in subjects:
        summary[subj] = {"success": [], "failed": [], "skipped": []}

        sessions = available_sessions_for_subject(deriv_root, subj)
        if not sessions:
            summary[subj]["skipped"].append(("ALL", "Subject directory not found"))
            print(f"Subject {subj:02d}: no subject directory found under {deriv_root}")
            continue

        dates_to_process, skipped_dates = normalize_dates(date, list(sessions.keys()))
        summary[subj]["skipped"].extend(skipped_dates)

        if not dates_to_process:
            print(f"Subject {subj:02d}: no matching dates to process")
            continue

        for date_str in dates_to_process:
            session_dir = sessions.get(date_str)
            results_dir = session_dir / "saved_analysis_results"

            if not results_dir.exists():
                summary[subj]["skipped"].append((date_str, "Results directory not found"))
                print(f"Subject {subj:02d} Date {date_str}: results directory missing, skipping")
                continue

            if _find_combined_file(results_dir) is not None and not recompute:
                summary[subj]["skipped"].append((date_str, "Existing combined tracking file, skipping directory"))
                print(f"Subject {subj:02d} Date {date_str} - Existing combined tracking file, skipping directory (recompute=False)")
                continue

            slp_files = sorted(results_dir.glob("*.slp"))
            if not slp_files:
                summary[subj]["skipped"].append((date_str, "No .slp files found"))
                print(f"Subject {subj:02d} Date {date_str}: no .slp files, skipping")
                continue

            tasks.append((subj, date_str, session_dir, results_dir, len(slp_files)))

    if tasks:
        for task in tasks:
            result = process_single(*task)
            status = result["status"]
            if status == "success":
                summary[result["subj"]]["success"].append((result["date"], result["reason"]))
            elif status == "failed":
                summary[result["subj"]]["failed"].append((result["date"], result["reason"]))
            else:
                summary[result["subj"]]["skipped"].append((result["date"], result["reason"]))
            for line in result["messages"]:
                print(line)

    print("\nSummary:")
    for subj, stats in summary.items():
        total = len(stats["success"]) + len(stats["failed"]) + len(stats["skipped"])
        print(f"  {subj}:")
        print(f"    Successful: {len(stats['success'])}/{total}")
        print(f"    Failed: {len(stats['failed'])}/{total}")
        print(f"    Skipped: {len(stats['skipped'])}/{total}")

    return summary


def sleap_node_quality_report(subjid: int,
                              date: Optional[Union[int, Iterable[int], Tuple[int, int]]] = None,
                              base_dir: Optional[Union[str, Path]] = None,
                              score_thresh: float = 0.4,
                              presence_frac: float = 0.7,
                              verbose: bool = True) -> pd.DataFrame:
    """
    Per-node tracking-quality report for one subject across one or more sessions.

    For every skeleton node it reports how often the node is present, its confidence-score
    distribution, and whether it would be selected for the centroid at the given thresholds.
    Handy for judging whether a new SLEAP model tracks better than an old one.

    Parameters
    ----------
    subjid : int
        Subject id.
    date : int | list[int] | (start, end) tuple | None
        Session date(s). None => all available sessions for the subject. A 2-tuple is an
        inclusive (start, end) range; a list/set is treated as explicit dates.
    base_dir : str | Path | None
        Optional base directory (tries <base_dir>/derivatives).
    score_thresh : float
        A point counts as confident when its score >= this (drives `pres_pct_occ_gated`
        and `selected`).
    presence_frac : float
        Gated presence (of occupied frames) required for a node to be marked `selected`.
    verbose : bool
        Print a formatted table per session.

    Returns
    -------
    pd.DataFrame with one row per (date, node):
        subject, date, node, n_frames, n_occupied, pres_pct_all, pres_pct_occ,
        pres_pct_occ_gated, avg_score, score_p10, score_p25, score_p50, score_p75,
        score_p90, selected
    """
    deriv_root = _resolve_deriv_root(base_dir)
    sessions = _available_sessions(deriv_root, int(subjid))
    if not sessions:
        raise FileNotFoundError(f"No subject directory found for sub-{int(subjid):03d} under {deriv_root}")

    dates = _normalize_date_arg(date, list(sessions.keys()))
    if not dates:
        raise FileNotFoundError("No matching sessions found for the requested date(s)")

    records: List[dict] = []

    for date_str in dates:
        results_dir = sessions[date_str] / "saved_analysis_results"
        slp_files = sorted(p for p in results_dir.glob("*.slp") if not p.name.startswith("._"))
        if not slp_files:
            if verbose:
                print(f"{date_str}: no .slp files, skipping")
            continue

        nodes: Optional[List[str]] = None
        present_cnt: Optional[np.ndarray] = None      # coords present per node
        gated_cnt: Optional[np.ndarray] = None        # coords present AND score >= thresh
        score_sum: Optional[np.ndarray] = None
        scores_by_node: Optional[List[List[np.ndarray]]] = None
        n_frames = 0
        n_occupied = 0

        for slp_path in slp_files:
            labels = None
            try:
                labels = sleap_io.load_slp(str(slp_path))
                if nodes is None:
                    nodes = [n.name for n in labels.skeletons[0].nodes]
                    k = len(nodes)
                    present_cnt = np.zeros(k, dtype=np.int64)
                    gated_cnt = np.zeros(k, dtype=np.int64)
                    score_sum = np.zeros(k, dtype=float)
                    scores_by_node = [[] for _ in range(k)]

                arr = np.asarray(labels.numpy(return_confidence=True), dtype=float)
                if arr.ndim == 4:  # (frames, instances, nodes, 3) -> rows of (nodes, 3)
                    arr = arr.reshape(arr.shape[0] * arr.shape[1], arr.shape[2], arr.shape[3])
            except Exception as exc:
                if verbose:
                    print(f"  ⚠️ Failed to read {slp_path.name}: {exc}; skipping")
                continue
            finally:
                del labels
                gc.collect()

            x, y, s = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
            present = ~np.isnan(x) & ~np.isnan(y)
            gated = present & (s >= score_thresh)
            n_frames += arr.shape[0]
            n_occupied += int(present.any(axis=1).sum())
            present_cnt += present.sum(axis=0)
            gated_cnt += gated.sum(axis=0)
            for j in range(len(nodes)):
                sj = s[present[:, j], j]
                if sj.size:
                    score_sum[j] += float(np.nansum(sj))
                    scores_by_node[j].append(sj)

        if not nodes or n_frames == 0:
            continue

        for j, name in enumerate(nodes):
            all_scores = np.concatenate(scores_by_node[j]) if scores_by_node[j] else np.array([])
            pcnt = int(present_cnt[j])
            gcnt = int(gated_cnt[j])
            if all_scores.size:
                p10, p25, p50, p75, p90 = (float(v) for v in np.percentile(all_scores, [10, 25, 50, 75, 90]))
                avg = float(all_scores.mean())
            else:
                p10 = p25 = p50 = p75 = p90 = avg = float("nan")
            pres_occ_gated = 100 * gcnt / n_occupied if n_occupied else 0.0
            records.append({
                "subject": int(subjid),
                "date": date_str,
                "node": name,
                "n_frames": n_frames,
                "n_occupied": n_occupied,
                "pres_pct_all": 100 * pcnt / n_frames if n_frames else 0.0,
                "pres_pct_occ": 100 * pcnt / n_occupied if n_occupied else 0.0,
                "pres_pct_occ_gated": pres_occ_gated,
                "avg_score": avg,
                "score_p10": p10,
                "score_p25": p25,
                "score_p50": p50,
                "score_p75": p75,
                "score_p90": p90,
                "selected": pres_occ_gated >= 100 * presence_frac,
            })

        if verbose:
            print(f"\n=== sub-{int(subjid):03d} date-{date_str}  "
                  f"({n_frames:,} frames, {n_occupied:,} occupied) ===")
            print(f"{'node':<14}{'pres%all':>9}{'pres%occ':>9}{'gated%occ':>10}"
                  f"{'avg':>7}{'p10':>7}{'p50':>7}{'p90':>7}  sel")
            for r in records[-len(nodes):]:
                print(f"{r['node']:<14}{r['pres_pct_all']:9.1f}{r['pres_pct_occ']:9.1f}"
                      f"{r['pres_pct_occ_gated']:10.1f}{r['avg_score']:7.2f}"
                      f"{r['score_p10']:7.2f}{r['score_p50']:7.2f}{r['score_p90']:7.2f}"
                      f"   {'[x]' if r['selected'] else '[ ]'}")

    return pd.DataFrame(records)