"""Per-video parquet + harp streams -> one combined parquet per session.

- Reads synchronised video frame times via `hypnose_behavior.io.loaders.load_all_streams`,
  one call per ``behav/`` experiment folder.
- Matches each per-video table to its video by ``video_file`` column, then filename
  suffix, then numeric index.
- Left-joins timestamps onto SLEAP frames, so every tracked frame survives even when
  no timestamp matches.
- Writes ``sub-XXX_ses-YYYYMMDD_combined_sleap_tracking_timestamps.parquet``.

The `hypnose_behavior` import is lazy: `extract` must run without it.

    from hypnose_sleap.timestamps import combine_session
    combine_session(57, 20260717)
    combine_session(57, 20260717, derivatives=tmp, rawdata=raw)   # gate: temp tree
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional, Union

import pandas as pd

from hypnose_sleap.io import layout

COMBINED_FILENAME = "sub-{subject}_ses-{date}_combined_sleap_tracking_timestamps.parquet"


def load_all_streams(root, *args, **kwargs):
    """`hypnose_behavior.io.loaders.load_all_streams`, imported on first call.

    Named rather than inlined so importing this module does not require
    `hypnose_behavior` -- only combining a session does. It is also the single import
    surface: when that function moves, this line is the only one to update.
    """
    from hypnose_behavior.io.loaders import load_all_streams as _impl
    return _impl(root, *args, **kwargs)


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


def combine_session(subjid, date, *, derivatives=None, rawdata=None,
                    save_output: bool = True) -> pd.DataFrame:
    """Add synchronised timestamps to one session's per-video tables and combine them.

    - each table is matched to its video by the ``video_file`` column, then by the
      filename suffix, then by the 1-based number in ``sleap_tracking_video<N>``;
    - the join is a left join on the local frame index, so a SLEAP frame with no
      timestamp survives with ``time`` unset rather than being dropped;
    - ``derivatives`` and ``rawdata`` override the two roots independently, which is how
      the gate reads the real harp streams while writing only into a temp tree.

    Returns the combined table; `save_output` also writes it beside the per-video ones.
    """
    session = layout.layout_for(derivatives).find_session(subjid, date=str(date))
    results = layout.results_dir(session)
    if not results.exists():
        raise FileNotFoundError(f"Results directory not found: {results}")

    tracking_files = layout.find_tracking_tables(results)
    if not tracking_files:
        raise FileNotFoundError(
            f"No sleap_tracking_video* files (.parquet/.csv) found in {results}")

    print(f"Found {len(tracking_files)} SLEAP tracking file(s)")

    # Step 1: the raw session's experiment folders, from the one video walk
    raw_session = layout.layout_for(rawdata, name="rawdata").find_session(subjid, date=str(date))
    exp_folders = layout.session_experiments(raw_session)
    if not exp_folders:
        raise FileNotFoundError(f"No experiment folders with video in {raw_session.path}")

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
    # Extract video number from filename (sleap_tracking_video2.parquet -> 2)
    sleap_video_mapping = {}  # maps video_file to SLEAP table path

    for table_path in tracking_files:
        # Try to extract mapping hints
        video_file_hint = _peek_video_file(table_path)

        name_hint = None
        m_name = re.search(r"sleap_tracking_video\d+_(.+)\.(?:csv|parquet)", table_path.name)
        if m_name:
            name_hint = m_name.group(1)
            if not name_hint.endswith(".avi"):
                name_hint = f"{name_hint}.avi"

        m_num = re.search(r'sleap_tracking_video(\d+)', table_path.name)
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
            print(f"Warning: Could not map {table_path.name} (hints: {video_file_hint}, {name_hint}); skipping")
            continue

        sleap_video_mapping[target_video_file] = table_path
        extra = f" via {source}" if source else ""
        num_txt = f" (video {video_num})" if video_num is not None else ""
        print(f"Matched {table_path.name}{num_txt} to {target_video_file}{extra}")

    if not sleap_video_mapping:
        raise ValueError("No SLEAP files could be matched to videos")

    # Step 5: Add timestamps to each per-video table
    all_tracking = []

    for video_file, table_path in sleap_video_mapping.items():
        tracking_df = _read_table(table_path)

        # Get frame times for this specific video
        video_frames = frames_by_video.get(video_file, pd.DataFrame()).copy()
        if video_frames.empty:
            print(f"Warning: Video '{video_file}' not found in frame times, skipping {table_path.name}")
            continue

        print(f"\nProcessing {table_path.name}:")
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
        output_filename = COMBINED_FILENAME.format(
            subject=f"{int(session.subjid):03d}", date=session.date)
        output_path = layout.write_path(results, output_filename)
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


__all__ = [
    "COMBINED_FILENAME", "load_all_streams",
    "get_video_frame_times", "combine_session",
]
