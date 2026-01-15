import sleap_io
import pandas as pd
import numpy as np
from pathlib import Path
import json
from hypnose_analysis.paths import get_derivatives_root, get_data_root
from hypnose_analysis.utils.classification_utils import load_all_streams
from hypnose_analysis.utils.metrics_utils import load_session_results

def sleap_labels_and_centroid(subjid, date, base_dir=None, core_nodes=None):
    """
    Load all .slp files for a subject/date, flatten every frame/instance to CSV,
    and append per-frame centroids from available core nodes.
    Returns a list of saved CSV paths.
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
        pts_raw = None
        if hasattr(inst, "points_array") and inst.points_array is not None:
            pts_raw = inst.points_array
        elif hasattr(inst, "points") and inst.points is not None:
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
            if hasattr(point, "x") and hasattr(point, "y"):
                return (point.x, point.y)
            if isinstance(point, dict):
                if "xy" in point and point["xy"] is not None:
                    return (point["xy"][0], point["xy"][1])
                return (point.get("x", nan), point.get("y", nan))
            if hasattr(point, "__len__") and len(point) >= 2:
                return (point[0], point[1])
            return (nan, nan)

        xy = []
        for idx in range(n_nodes):
            xy.append(as_xy(pts_seq[idx] if idx < len(pts_seq) else None))

        scores = None
        if hasattr(inst, "point_confidences") and inst.point_confidences is not None:
            scores_raw = inst.point_confidences
            scores = [scores_raw[idx] if idx < len(scores_raw) else nan for idx in range(n_nodes)]
        elif pts_seq and isinstance(pts_seq[0], dict) and "score" in pts_seq[0]:
            scores = [pts_seq[idx].get("score", nan) if idx < len(pts_seq) else nan for idx in range(n_nodes)]

        return xy, scores

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
        output_path = results_dir / f"sleap_tracking_video{video_number}.csv"
        print(f"\n[{video_number}/{len(slp_files)}] Processing: {slp_path.name}")

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

                row = {"frame": frame_idx, "instance": inst_idx}
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
            raise ValueError(f"No pose data found in {slp_path}")

        df = df[pd.notna(df["frame"])]
        df.sort_values(["frame", "instance"], inplace=True)
        df.reset_index(drop=True, inplace=True)

        centroid_x_cols = [f"{node}_x" for node in core_nodes if f"{node}_x" in df.columns]
        centroid_y_cols = [f"{node}_y" for node in core_nodes if f"{node}_y" in df.columns]

        df["centroid_x"] = df[centroid_x_cols].mean(axis=1, skipna=True) if centroid_x_cols else pd.NA
        df["centroid_y"] = df[centroid_y_cols].mean(axis=1, skipna=True) if centroid_y_cols else pd.NA

        df.to_csv(output_path, index=False)

        outputs.append(output_path)
        print(f"  ✓ Saved to: {output_path.name}")
        print(f"    Total rows: {len(df)}")
        print(f"    Frame range: {int(df['frame'].min())} to {int(df['frame'].max())}")

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
    })
    
    # Add global frame index (continuous across all videos)
    result['frame'] = range(len(result))
    
    # Add video file basename for convenience
    result['video_file'] = result['video_path'].apply(lambda x: Path(x).name if pd.notna(x) else None)
    
    # Reorder columns
    result = result[['frame', 'local_frame', 'time', 'video_path', 'video_file']]
    
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
        match = re.search(r'sleap_tracking_video(\d+)', csv_path.name)
        if not match:
            print(f"Warning: Could not extract video number from {csv_path.name}, skipping")
            continue
        
        video_num = int(match.group(1))  # 1-indexed from filename
        video_idx = video_num - 1  # Convert to 0-indexed
        
        if video_idx >= len(video_files_ordered):
            print(f"Warning: {csv_path.name} references video {video_num}, but only {len(video_files_ordered)} video(s) exist")
            continue
        
        video_file = video_files_ordered[video_idx]
        sleap_video_mapping[video_file] = csv_path
        print(f"Matched {csv_path.name} (video {video_num}) to {video_file}")
    
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
        result = tracking_df.merge(
            video_frames[['local_frame', 'time', 'global_frame']],
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
    priority_cols = ['frame', 'time', 'global_frame', 'video_file', 'instance']
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
                                          centroid_radius=8, centroid_color='red', show_fps_counter=True,
                                          show_timestamp=True, show_trial_state=True):
    """
    Annotate behavior videos with SLEAP centroid tracking and trial state overlays.
    
    Creates MP4 videos with:
    - Red dot at animal centroid position (from SLEAP tracking)
    - Frame counter (bottom right)
    - Synchronized timestamp from video metadata (bottom left)
    - Trial state indicator: "WITHIN TRIAL" (white) or "OUTSIDE TRIAL" (blue)
    
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
    show_fps_counter : bool, optional
        Show frame counter (default: True)
    show_timestamp : bool, optional
        Show synchronized timestamp (default: True)
    show_trial_state : bool, optional
        Show trial state overlay (default: True)
    
    Returns:
    --------
    list : Paths to generated output MP4 files
    """
    from pathlib import Path
    import pandas as pd
    import cv2
    from PIL import Image, ImageDraw, ImageFont
    from tqdm import tqdm
    
    # Default base directory (use configured data root if none supplied)
    if base_dir is None:
        base_dir = get_data_root()
    else:
        base_dir = Path(base_dir)
    
    # Load behavior data for trial state
    behavior = load_session_results(subjid, date)
    
    # Find directories
    deriv_dir = base_dir / "derivatives"
    subj_pattern = f"sub-{subjid:03d}_*"
    subj_dirs = list(deriv_dir.glob(subj_pattern))
    
    if not subj_dirs:
        raise FileNotFoundError(f"No subject directory found for pattern {subj_pattern}")
    
    subj_dir = subj_dirs[0]
    session_pattern = f"ses-*_date-{date}"
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
    rawdata_session_dir = next(rawdata_subj_dir.glob(f"ses-*_date-{date}"), None)
    
    if not rawdata_session_dir:
        raise FileNotFoundError(f"No rawdata session directory found for date {date}")
    
    video_files = sorted(rawdata_session_dir.glob("behav/*/VideoData/*.avi"))
    
    if not video_files:
        raise FileNotFoundError(f"No video files found in {rawdata_session_dir}")
    
    # Load combined timestamps file
    combined_ts_file = list(results_dir.glob("*_combined_sleap_tracking_timestamps.csv"))
    if not combined_ts_file:
        raise FileNotFoundError(f"No combined timestamps file found in {results_dir}")
    
    combined_df = pd.read_csv(combined_ts_file[0])
    print(f"Loaded combined timestamps: {len(combined_df)} frames")
    
    # Get trial times
    trials = behavior.get('initiated_sequences', pd.DataFrame())
    if not trials.empty:
        trials = trials.copy()
        trials['sequence_start'] = pd.to_datetime(trials['sequence_start'])
        trials['sequence_end'] = pd.to_datetime(trials['sequence_end'])
        print(f"Loaded {len(trials)} trials")
    else:
        print("⚠️ No trials found - will show OUTSIDE TRIAL for all frames")
    
    # Load fonts
    try:
        font_large = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 40)
        font_small = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 24)
    except:
        font_large = ImageFont.load_default()
        font_small = ImageFont.load_default()
    
    output_paths = []
    
    # Process each video
    for video_idx, video_path in enumerate(video_files, 1):
        print(f"\nProcessing video {video_idx}/{len(video_files)}: {video_path.name}")
        
        # Filter combined_df for this video
        video_name = video_path.stem  # e.g., VideoData_1904-01-02T03-00-00
        df_video = combined_df[combined_df['video_file'].str.contains(video_name, na=False)].copy()
        
        if df_video.empty:
            print(f"  ⚠️ No timestamps found for video {video_name}, skipping")
            continue
        
        # Convert time column to datetime
        df_video['time'] = pd.to_datetime(df_video['time'])
        df_video = df_video.reset_index(drop=True)
        
        print(f"  Found {len(df_video)} frames with timestamps")
        
        # Load video with OpenCV for frame-by-frame processing
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"  Video: {width}x{height} @ {fps} fps, {total_frames} frames")
        
        # Output video writer
        output_path = results_dir / f"{output_suffix}_video{video_idx}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        output_paths.append(output_path)
        
        print(f"  Saving to: {output_path.name}")
        
        # Use tqdm for progress bar
        with tqdm(total=total_frames, desc="  Encoding", unit="frames") as pbar:
            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Convert BGR to RGB for PIL
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_pil = Image.fromarray(frame_rgb)
                draw = ImageDraw.Draw(frame_pil)
                
                # Plot centroid if available
                if show_fps_counter and frame_idx < len(df_video):
                    row = df_video.iloc[frame_idx]
                    if not pd.isna(row['centroid_x']) and not pd.isna(row['centroid_y']):
                        cx = int(row['centroid_x'])
                        cy = int(row['centroid_y'])
                        # Draw simple red dot
                        draw.ellipse([cx - centroid_radius, cy - centroid_radius, 
                                     cx + centroid_radius, cy + centroid_radius], 
                                    fill=centroid_color, outline=centroid_color)
                
                # Add frame counter in bottom right
                if show_fps_counter:
                    counter_text = f"{frame_idx}/{total_frames}"
                    counter_bbox = draw.textbbox((0, 0), counter_text, font=font_small)
                    counter_width = counter_bbox[2] - counter_bbox[0]
                    counter_x = width - counter_width - 20
                    counter_y = height - 50
                    draw.rectangle([counter_x - 10, counter_y - 10, counter_x + counter_width + 10, counter_y + 40],
                                  fill='black', outline='white', width=2)
                    draw.text((counter_x, counter_y), counter_text, fill='white', font=font_small)
                
                # Add timestamp in bottom left from CSV
                if show_timestamp and frame_idx < len(df_video):
                    row = df_video.iloc[frame_idx]
                    frame_time = row['time']
                    timestamp_text = frame_time.strftime("%H:%M:%S")
                    
                    timestamp_bbox = draw.textbbox((0, 0), timestamp_text, font=font_small)
                    timestamp_width = timestamp_bbox[2] - timestamp_bbox[0]
                    timestamp_x = 20
                    timestamp_y = height - 50
                    draw.rectangle([timestamp_x - 10, timestamp_y - 10, timestamp_x + timestamp_width + 10, timestamp_y + 40],
                                  fill='black', outline='white', width=2)
                    draw.text((timestamp_x, timestamp_y), timestamp_text, fill='white', font=font_small)
                
                # Determine trial state from timestamps
                if show_trial_state:
                    trial_state = "OUTSIDE TRIAL"
                    trial_color = (100, 100, 255)  # Blue RGB
                    
                    if not trials.empty and frame_idx < len(df_video):
                        row = df_video.iloc[frame_idx]
                        frame_time = row['time']
                        
                        for _, trial in trials.iterrows():
                            if trial['sequence_start'] <= frame_time <= trial['sequence_end']:
                                trial_state = "WITHIN TRIAL"
                                trial_color = (255, 255, 255)  # White RGB
                                break
                    
                    # Add trial state in top center
                    state_bbox = draw.textbbox((0, 0), trial_state, font=font_large)
                    state_width = state_bbox[2] - state_bbox[0]
                    state_x = (width - state_width) // 2
                    state_y = 20
                    draw.rectangle([state_x - 15, state_y - 10, state_x + state_width + 15, state_y + 50],
                                  fill='black', outline=trial_color, width=3)
                    draw.text((state_x, state_y), trial_state, fill=trial_color, font=font_large)
                
                # Convert back to BGR for OpenCV
                frame_annotated = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)
                out.write(frame_annotated)
                
                frame_idx += 1
                pbar.update(1)
        
        cap.release()
        out.release()
        
        print(f"  ✓ Completed!")
    
    print(f"\n✅ All videos processed and saved!")
    return output_paths