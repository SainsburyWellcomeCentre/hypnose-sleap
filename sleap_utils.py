import sleap_io
import pandas as pd
import numpy as np
import re
from pathlib import Path
import json
from typing import Dict, Iterable, List, Optional, Tuple, Union
from hypnose_analysis.paths import get_derivatives_root, get_data_root
from hypnose_analysis.utils.classification_utils import load_all_streams, load_odor_mapping
from hypnose_analysis.utils.metrics_utils import load_session_results
from hypnose_analysis.utils.visualization_utils import _get_from_cache, _update_cache

def sleap_labels_and_centroid(
    subjid,
    date,
    base_dir=None,
    core_nodes=None,
    skip_empty: bool = False,
    anchor_threshold: Optional[float] = 150.0,
):
    """
    Load all .slp files for a subject/date, flatten every frame/instance to CSV,
    and append per-frame centroids from available core nodes.
    Centroid uses an anchor-based filter: nodes farther than `anchor_threshold`
    pixels from the previous centroid are ignored until they re-enter the window
    (set anchor_threshold=None to disable filtering).
    Returns a list of saved CSV paths.
    If skip_empty is True, files with no pose data (or unreadable .slp) are skipped instead of raising.
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

    core_nodes = core_nodes or [
        "right_ear",
        "left_ear",
        "center_head",
        "neck",
        "center",
        "center_back",
        "tail_base",
    ]

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

    default_nodes = []

    outputs = []
    for video_number, slp_path in enumerate(slp_files, 1):
        video_file_basename = infer_video_file_from_slp(slp_path)
        safe_tag = video_file_basename.replace(".avi", "")
        output_path = results_dir / f"sleap_tracking_video{video_number}_{safe_tag}.csv"
        print(f"\n[{video_number}/{len(slp_files)}] Processing: {slp_path.name}")

        try:
            labels = sleap_io.load_slp(str(slp_path))
        except Exception as exc:
            if skip_empty:
                print(f"  ⚠️ Failed to read {slp_path.name}: {exc}; skipping")
                continue
            raise
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

        df = pd.DataFrame(rows)
        if df.empty:
            if skip_empty:
                print(f"  ⚠️ No pose data found in {slp_path}, skipping this file")
                continue
            raise ValueError(f"No pose data found in {slp_path}")

        df = df[pd.notna(df["frame"])]
        df.sort_values(["frame", "instance"], inplace=True)
        df.reset_index(drop=True, inplace=True)

        centroid_x_cols = [f"{node}_x" for node in core_nodes if f"{node}_x" in df.columns]
        centroid_y_cols = [f"{node}_y" for node in core_nodes if f"{node}_y" in df.columns]
        available_nodes = [node for node in core_nodes if f"{node}_x" in df.columns]

        if centroid_x_cols and centroid_y_cols:
            xs = df[centroid_x_cols].to_numpy(dtype=float)
            ys = df[centroid_y_cols].to_numpy(dtype=float)

            n_frames = len(df)
            centroid_x_out = np.full(n_frames, np.nan)
            centroid_y_out = np.full(n_frames, np.nan)
            nodes_used: List[str] = [""] * n_frames

            anchor_x = np.nan
            anchor_y = np.nan

            for i in range(n_frames):
                row_x = xs[i]
                row_y = ys[i]

                valid_mask = ~np.isnan(row_x) & ~np.isnan(row_y)

                if anchor_threshold is not None and not np.isnan(anchor_x) and valid_mask.any():
                    dist = np.sqrt((row_x - anchor_x) ** 2 + (row_y - anchor_y) ** 2)
                    within_mask = valid_mask & (dist <= float(anchor_threshold))
                else:
                    within_mask = valid_mask

                if within_mask.any():
                    cx = float(np.nanmean(row_x[within_mask]))
                    cy = float(np.nanmean(row_y[within_mask]))
                    centroid_x_out[i] = cx
                    centroid_y_out[i] = cy
                    anchor_x = cx
                    anchor_y = cy
                    nodes_used[i] = ";".join(np.array(available_nodes)[within_mask])
                else:
                    nodes_used[i] = ""

            df["centroid_x"] = centroid_x_out
            df["centroid_y"] = centroid_y_out
            df["nodes_for_centroid"] = nodes_used
        else:
            df["centroid_x"] = pd.NA
            df["centroid_y"] = pd.NA
            df["nodes_for_centroid"] = ""

        df.to_csv(output_path, index=False)

        outputs.append(output_path)
        print(f"  ✓ Saved to: {output_path.name}")
        print(f"    Total rows: {len(df)}")
        print(f"    Frame range: {int(df['frame'].min())} to {int(df['frame'].max())}")

    if not outputs:
        raise ValueError("No pose data found in any .slp files for this session")

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
    
    tracking_csvs = sorted([f for f in results_dir.glob("sleap_tracking_video*.csv") 
                           if not f.name.startswith('._')])
    if not tracking_csvs:
        raise FileNotFoundError(f"No sleap_tracking_videox.csv files found in {results_dir}")
    
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
        video_file_hint = None
        try:
            df_head = pd.read_csv(csv_path, nrows=1)
            if "video_file" in df_head.columns and pd.notna(df_head.loc[0, "video_file"]):
                video_file_hint = str(df_head.loc[0, "video_file"])
        except Exception:
            pass

        name_hint = None
        m_name = re.search(r"sleap_tracking_video\d+_(.+)\.csv", csv_path.name)
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
        try:
            tracking_df = pd.read_csv(csv_path, encoding='utf-8')
        except UnicodeDecodeError:
            tracking_df = pd.read_csv(csv_path, encoding='latin1')
        
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
        output_filename = f"sub-{str(subjid).zfill(3)}_ses-{date_str}_combined_sleap_tracking_timestamps.csv"
        output_path = results_dir / output_filename
        combined.to_csv(output_path, index=False)
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
    time_window : tuple[str | pd.Timedelta, str | pd.Timedelta] | None, optional
        If provided, trims the video to [start, end] relative to the first frame timestamp
        of that video (e.g., ("0:00:00", "0:10:00") for first 10 minutes).
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

    def parse_time_window(window):
        if window is None:
            return None
        if len(window) != 2:
            raise ValueError("time_window must be a tuple of (start, end)")
        start_td = pd.to_timedelta(window[0])
        end_td = pd.to_timedelta(window[1])
        if end_td < start_td:
            raise ValueError("time_window end must be >= start")
        return start_td, end_td

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
        combined_ts_file = list(results_dir.glob("*_combined_sleap_tracking_timestamps.csv"))
        if not combined_ts_file:
            raise FileNotFoundError(f"No combined timestamps file found in {results_dir}")

        combined_df = pd.read_csv(combined_ts_file[0])
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

        # Apply optional time window relative to first frame time
        window = parse_time_window(time_window)
        if window:
            start_abs = df_video['time'].min() + window[0]
            end_abs = df_video['time'].min() + window[1]
            df_video = df_video[(df_video['time'] >= start_abs) & (df_video['time'] <= end_abs)].copy()
            if df_video.empty:
                print("  ⚠️ No frames in requested time window, skipping video")
                continue
            print(f"  Trimmed to window: {window[0]} - {window[1]} ({len(df_video)} frames)")
        else:
            start_abs = None
            end_abs = None
        
        print(f"  Found {len(df_video)} frames with timestamps")
        
        # Load video with OpenCV for frame-by-frame processing
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"  Video: {width}x{height} @ {fps} fps, {total_frames} frames")
        
        # Frame mapping by local frame index for fast lookup
        df_video['frame'] = pd.to_numeric(df_video['frame'], errors='coerce').astype('Int64')
        df_video = df_video.dropna(subset=['frame'])
        row_map = df_video.set_index('frame').to_dict('index')
        valid_frames = sorted(row_map.keys())
        if not valid_frames:
            print("  ⚠️ No valid frames with centroid, skipping video")
            cap.release()
            continue

        start_frame = int(valid_frames[0])
        end_frame = int(valid_frames[-1])

        # Seek to start frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        # Rotation-aware output size
        if rotate_deg in (90, 270):
            out_size = (height, width)
        else:
            out_size = (width, height)

        # Output video writer (include rotation to avoid overwriting). Mark _full when no trimming.
        suffix_full = "_full" if window is None else ""
        output_path = results_dir / f"{output_suffix}_rotdeg_{rotate_deg}_video{video_num}{suffix_full}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, out_size)
        output_paths.append(output_path)
        
        print(f"  Saving to: {output_path}")
        
        # Use tqdm for progress bar
        frames_to_process = end_frame - start_frame + 1

        # Precompute marker window for this video if applicable
        mark_window = None
        if mark_ts_abs is not None:
            mark_window = (mark_ts_abs, mark_ts_abs + pd.Timedelta(seconds=2.0))

        with tqdm(total=frames_to_process, desc="  Encoding", unit="frames") as pbar:
            current_frame = start_frame
            while current_frame <= end_frame:
                ret, frame = cap.read()
                if not ret:
                    break

                # Rotate raw frame first so overlays align post-rotation
                frame = rotate_frame(frame, rotate_deg)

                # Determine dimensions after rotation
                frame_h, frame_w = frame.shape[:2]

                # Convert BGR to RGB for PIL
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_pil = Image.fromarray(frame_rgb)
                draw = ImageDraw.Draw(frame_pil)

                row = row_map.get(current_frame)

                # Plot centroid if available for this frame
                if row is not None and not pd.isna(row.get('centroid_x')) and not pd.isna(row.get('centroid_y')):
                    cx_raw = row['centroid_x']
                    cy_raw = row['centroid_y']
                    cx_rot, cy_rot = rotate_point(cx_raw, cy_raw, width, height, rotate_deg)
                    cx, cy = int(cx_rot), int(cy_rot)
                    draw.ellipse([cx - centroid_radius, cy - centroid_radius,
                                 cx + centroid_radius, cy + centroid_radius],
                                fill=centroid_color, outline=centroid_color)

                # Odor overlay (text only, neon red); hidden when no odor
                frame_time = pd.to_datetime(row.get('time')) if (row is not None and 'time' in row) else None
                odor_label = None
                if frame_time is not None and valve_events:
                    odor_label = lookup_odor(frame_time.to_datetime64())

                if odor_label:
                    display_odor = re.sub(r"(?i)^odor[_\-\s]*", "", str(odor_label)) or str(odor_label)
                    odor_text = f"Odor: {display_odor}"
                    # Anchor in unrotated coords then rotate anchor point
                    anchor_x_raw = width // 2
                    anchor_y_raw = 30
                    anchor_x, anchor_y = rotate_point(anchor_x_raw, anchor_y_raw, width, height, rotate_deg)

                    # If rotated 90°, nudge left/up to keep on-screen
                    if rotate_deg == 90:
                        anchor_x -= 185
                        anchor_y -= 30

                    draw.text((anchor_x, anchor_y), odor_text, fill=(255, 40, 90), font=font_small)

                # Reward overlays (text only, green) shown when supply ports pulse (1s duration)
                active_ports = set()
                if frame_time is not None and supply_events:
                    active_ports = {
                        ev['port'] for ev in supply_events
                        if ev['start_time'] <= frame_time <= ev['end_time']
                    }

                if active_ports:
                    reward_text = "Reward"
                    reward_color = (0, 220, 0)
                    x_offset = 140  # pull toward center
                    y_offset = 5    # bind very close to bottom edge

                    # Add extra bottom padding when unrotated
                    if rotate_deg == 0:
                        y_offset = 40

                    bbox_reward = draw.textbbox((0, 0), reward_text, font=font_small)
                    text_h = bbox_reward[3] - bbox_reward[1]

                    # SupplyPort2 -> left (0°) / top (90°)
                    if 2 in active_ports:
                        bl_x_raw = x_offset
                        bl_y_raw = height - y_offset - text_h
                        bl_x, bl_y = rotate_point(bl_x_raw, bl_y_raw, width, height, rotate_deg)
                        draw.text((bl_x, bl_y), reward_text, fill=reward_color, font=font_small)

                    # SupplyPort1 -> right (0°) / bottom (90°)
                    if 1 in active_ports:
                        br_x_raw = width - (x_offset + 230)
                        br_y_raw = height - y_offset - text_h 
                        br_x, br_y = rotate_point(br_x_raw, br_y_raw, width, height, rotate_deg)
                        draw.text((br_x, br_y), reward_text, fill=reward_color, font=font_small)

                        if rotate_deg == 90:
                            br_y -= 20

                # Mark specific timepoint with a downward green triangle for 2 seconds
                if mark_window and frame_time is not None:
                    if mark_window[0] <= frame_time <= mark_window[1]:
                        cx = frame_w // 2
                        cy = frame_h // 2
                        tri = [(cx - 60, cy - 40), (cx + 60, cy - 40), (cx, cy + 40)]
                        draw.polygon(tri, fill=(0, 200, 0))

                # Convert back to BGR for OpenCV
                frame_annotated = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)
                out.write(frame_annotated)

                current_frame += 1
                pbar.update(1)
        
        cap.release()
        out.release()
        
        print(f"  ✓ Completed!")
    
    print(f"\n✅ All videos processed and saved!")
    return output_paths


def process_sleap_sessions(subjid: Union[int, Iterable[int]],
                           date: Optional[Union[int, Iterable[int], Tuple[int, int]]] = None,
                           base_dir: Optional[Union[str, Path]] = None,
                           core_nodes: Optional[List[str]] = None,
                           save_output: bool = True,
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
    core_nodes : list[str] | None
        Forwarded to sleap_labels_and_centroid.
    save_output : bool
        Forwarded to add_timestamps_to_sleap_tracking.
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
        return len([f for f in results_dir.glob("sleap_tracking_video*.csv") if not f.name.startswith("._")])

    def process_single(subj: int, date_str: str, session_dir: Path, results_dir: Path, centroid_found: int):
        messages = [f"\nSubject {subj:02d} Date {date_str} - Processing SLEAP Output:"]
        centroid_done = 0
        timestamp_found = 0
        matched_videos = 0
        saved_flag = False

        try:
            outputs = sleap_labels_and_centroid(subj, int(date_str), base_dir=base_dir, core_nodes=core_nodes, skip_empty=True)
            centroid_done = len(outputs)
            timestamp_found = count_tracking_files(results_dir)

            combined = add_timestamps_to_sleap_tracking(subj, int(date_str), save_output=save_output)
            if combined is not None and not combined.empty:
                matched_videos = combined["video_file"].nunique()
            output_filename = f"sub-{subj:03d}_ses-{date_str}_combined_sleap_tracking_timestamps.csv"
            saved_flag = (results_dir / output_filename).exists()

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

            combined_path = results_dir / f"sub-{subj:03d}_ses-{date_str}_combined_sleap_tracking_timestamps.csv"
            if combined_path.exists() and not recompute:
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