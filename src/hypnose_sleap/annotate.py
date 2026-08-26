"""Annotated overlay video: centroid, odour label and reward markers.

- Reads the session's combined parquet through `layout.find_combined_table`, so a table
  written flat or grouped, as ``.parquet`` or ``.csv``, all resolve.
- Draws the centroid per frame, the active odour near the poke port, and reward markers
  for a window after each supply-port pulse.
- Optional rotation (0/90/180/270), time windows, and a marked timepoint.
- Writes one ``.mp4`` per video per window, beside the session's other results.

The drawing is byte-for-byte the old `annotate_videos_with_sleap_and_trials`: the same
coordinate rotation, the same font sizes, the same anchor offsets and the same colours.
What changed is how the session is found (the shared layout rather than a private glob)
and two things that were dead:

- ``sleap_utils.py:918`` called `load_session_results` into a local named ``behavior``
  and never read it -- a whole session's results loaded per render, for nothing.
- ``:1092`` loaded a 100 px ``font_large`` that nothing ever drew with; only the 60 px
  ``font_small`` reaches a `draw.text`.

The font is resolved rather than hard-coded to ``C:/Windows/Fonts/arial.ttf``, so a
render works on Linux and macOS with no configuration -- see `find_font`. On a Windows
machine Arial still wins, so the pixels do not move.
- ``:1273-1274`` adjusted ``br_y`` *after* the `draw.text` that used it, so the value
  was discarded at the next loop iteration. Removing it cannot move a pixel.

Output goes flat into ``saved_analysis_results/``, not into ``movement_analysis/``: that
subfolder is where the tracking tables `hypnose_behavior` reads live, and a rendering is
not one of them.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from hypnose_sleap.io import layout

# The filename stem, and the size the overlay text is drawn at.
DEFAULT_SUFFIX = "sleap_visualization"
FONT_SIZE = 60

# The overlay font is resolved, not hard-coded, so a render works on a machine that has
# never seen a Windows font directory. In order: `HYPNOSE_OVERLAY_FONT`, `overlay_font`
# in `parameters.yml`, a search of the platform font directories, matplotlib's bundled
# DejaVuSans, then Pillow's default. Arial leads the candidate list because every clip
# rendered so far used it, so a Windows machine keeps drawing the same pixels.
FONT_ENV = "HYPNOSE_OVERLAY_FONT"

FONT_CANDIDATES = (
    "arial.ttf", "Arial.ttf", "ARIAL.TTF",
    "DejaVuSans.ttf", "LiberationSans-Regular.ttf", "FreeSans.ttf",
    "NotoSans-Regular.ttf", "Helvetica.ttc",
)

FONT_DIRS = (
    "C:/Windows/Fonts",
    "~/AppData/Local/Microsoft/Windows/Fonts",
    "/usr/share/fonts",
    "/usr/local/share/fonts",
    "~/.fonts",
    "~/.local/share/fonts",
    "/System/Library/Fonts",
    "/Library/Fonts",
    "~/Library/Fonts",
)

ODOR_COLOR = (255, 40, 90)
REWARD_COLOR = (0, 220, 0)
MARK_COLOR = (0, 200, 0)


def rotate_point(x, y, w, h, deg):
    """One point through the same rotation the frame gets."""
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
    """The frame, rotated clockwise by ``deg``."""
    import cv2

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
    """One ``(start, end)`` pair as timedeltas, or None."""
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
    """A list of windows from one pair, a list of pairs, or None.

    ``None`` yields ``[None]`` -- one clip, the full video. A bare ``(start, end)`` is
    one window; anything deeper is a collection of them.
    """
    if window_spec is None:
        return [None]

    if isinstance(window_spec, (list, tuple)):
        if not window_spec:
            return [None]

        first = window_spec[0]
        is_sequence = isinstance(first, (list, tuple)) and not isinstance(first, str)

        if (not is_sequence and len(window_spec) == 2
                and not isinstance(window_spec[0], (list, tuple))):
            return [parse_single_time_window(window_spec)]

        windows = []
        for idx, win in enumerate(window_spec, 1):
            if win is None:
                windows.append(None)
                continue
            if not isinstance(win, (list, tuple)):
                raise ValueError(f"Time window #{idx} must be a (start, end) pair")
            windows.append(parse_single_time_window(win))
        return windows

    raise ValueError(
        "time_window must be None, a single (start, end) pair, or an iterable of such pairs"
    )


def parse_mark_timepoint(ts_str, date_str):
    """An absolute clock time within the session, or None when unparsable."""
    if not ts_str:
        return None
    ts_str = str(ts_str).strip()
    ts_abs = pd.to_datetime(f"{date_str} {ts_str}", errors="coerce")
    if pd.isna(ts_abs):
        td = pd.to_timedelta(ts_str, errors="coerce")
        if pd.isna(td):
            return None
        ts_abs = pd.to_datetime(date_str, errors="coerce") + td
    return ts_abs


def _configured_font() -> Optional[Path]:
    """An explicitly chosen font: the env var, else ``overlay_font`` in the config."""
    from hypnose_sleap.io.paths import _env_path

    override = _env_path(FONT_ENV)
    if override:
        if not Path(override).is_file():
            raise SystemExit(f"{FONT_ENV} points at a missing file: {override}")
        return Path(override)

    from hypnose_sleap import parameters
    configured = parameters.parameters().get("overlay_font")
    if configured:
        path = Path(str(configured)).expanduser()
        if not path.is_file():
            raise SystemExit(
                f"`overlay_font` in parameters.yml points at a missing file: {path}"
            )
        return path
    return None


def find_font() -> Optional[Path]:
    """A TrueType font for the overlay, or None to fall back to Pillow's default.

    Searched rather than hard-coded so a render works on Linux and macOS untouched.
    `FONT_CANDIDATES` is tried in order across `FONT_DIRS`, name by name, so the
    preference is by *font*, not by whichever directory happens to be listed first.
    matplotlib's bundled DejaVuSans is the last resort before Pillow's own default: it
    ships with the package, so it is present wherever this repo's stack is.
    """
    configured = _configured_font()
    if configured is not None:
        return configured

    dirs = [Path(d).expanduser() for d in FONT_DIRS]
    dirs = [d for d in dirs if d.is_dir()]
    for name in FONT_CANDIDATES:
        for directory in dirs:
            direct = directory / name
            if direct.is_file():
                return direct
            # Linux nests fonts a few levels down (dejavu/, truetype/liberation/, ...).
            found = next((p for p in directory.rglob(name) if p.is_file()), None)
            if found is not None:
                return found

    try:
        import matplotlib
        bundled = Path(matplotlib.__file__).parent / "mpl-data" / "fonts" / "ttf" / "DejaVuSans.ttf"
        if bundled.is_file():
            return bundled
    except Exception:
        pass
    return None


def _load_font():
    """The overlay font at `FONT_SIZE`, scalable if one was found.

    Pillow's bitmap default is the last resort and does not match the size the overlay
    was designed at, so it says so rather than silently drawing tiny text.
    """
    from PIL import ImageFont

    path = find_font()
    if path is not None:
        try:
            return ImageFont.truetype(str(path), FONT_SIZE)
        except Exception as exc:
            print(f"Warning: could not load {path} ({exc}), falling back to the default font")

    try:
        return ImageFont.load_default(size=FONT_SIZE)  # Pillow >= 10.1
    except TypeError:
        print("Warning: no scalable font found; overlay text will be drawn at bitmap size.")
        return ImageFont.load_default()


def session_events(exp_root, *, reward_display_s: float = 1.0) -> dict:
    """Odour valve windows and supply-port pulses, from one experiment's harp streams.

    Returns ``{'valve': [...], 'supply': [...]}``, both sorted by start time. A stream
    that cannot be read yields empty lists: the overlay degrades to centroid-only rather
    than failing the render, which is what the old code did.
    """
    valve_events, supply_events = [], []
    try:
        from hypnose_behavior.io.loaders import load_all_streams, load_odor_mapping

        streams = load_all_streams(exp_root, apply_corrections=True, verbose=False)
        odor_map = load_odor_mapping(exp_root, data=streams, verbose=False)

        olfactometer_valves = odor_map.get("olfactometer_valves", {}) if odor_map else {}
        valve_to_odor = odor_map.get("valve_to_odor", {}) if odor_map else {}

        for olf_id, valve_df in (olfactometer_valves or {}).items():
            if valve_df is None or getattr(valve_df, "empty", True):
                continue
            for valve_idx, valve_col in enumerate(valve_df.columns):
                valve_key = f"{olf_id}{valve_idx}"
                odor_name = valve_to_odor.get(valve_key)
                if not odor_name or str(odor_name).lower() == "purge":
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
                        "start_time": on_time,
                        "end_time": off_times[j],
                        "odor_name": str(odor_name),
                    })
        valve_events.sort(key=lambda ev: ev["start_time"])

        for port_num, key in ((1, "pulse_supply_1"), (2, "pulse_supply_2")):
            series = streams.get(key)
            if series is None or getattr(series, "empty", True):
                continue
            for ts in getattr(series, "index", []):
                supply_events.append({
                    "start_time": ts,
                    "end_time": ts + pd.Timedelta(seconds=float(reward_display_s)),
                    "port": port_num,
                })
        supply_events.sort(key=lambda ev: ev["start_time"])
    except Exception:
        return {"valve": [], "supply": []}

    return {"valve": valve_events, "supply": supply_events}


def lookup_odor(ts, valve_events):
    """The odour active at ``ts``, or None. Events are sorted, so it stops early."""
    if ts is None or not valve_events:
        return None
    for ev in valve_events:
        if ev["start_time"] <= ts <= ev["end_time"]:
            return ev["odor_name"]
        if ev["start_time"] > ts:
            break
    return None


def _draw_overlay(draw, row, *, width, height, rotate_deg, frame_w, frame_h,
                  centroid_radius, centroid_color, font, valve_events, supply_events,
                  mark_window):
    """Everything drawn onto one frame. Coordinates rotate with the frame."""
    frame_time = None

    if row is not None and not pd.isna(row.get("centroid_x")) and not pd.isna(row.get("centroid_y")):
        cx_rot, cy_rot = rotate_point(row["centroid_x"], row["centroid_y"],
                                      width, height, rotate_deg)
        cx, cy = int(cx_rot), int(cy_rot)
        draw.ellipse([cx - centroid_radius, cy - centroid_radius,
                      cx + centroid_radius, cy + centroid_radius],
                     fill=centroid_color, outline=centroid_color)

    if row is not None and "time" in row:
        frame_time = pd.to_datetime(row.get("time"))

    odor_label = None
    if frame_time is not None and valve_events:
        odor_label = lookup_odor(frame_time.to_datetime64(), valve_events)

    if odor_label:
        display_odor = re.sub(r"(?i)^odor[_\-\s]*", "", str(odor_label)) or str(odor_label)
        odor_text = f"Odor: {display_odor}"
        anchor_x, anchor_y = rotate_point(width // 2, 30, width, height, rotate_deg)
        if rotate_deg == 90:
            anchor_x -= 185
            anchor_y -= 30
        draw.text((anchor_x, anchor_y), odor_text, fill=ODOR_COLOR, font=font)

    active_ports = set()
    if frame_time is not None and supply_events:
        active_ports = {
            ev["port"] for ev in supply_events
            if ev["start_time"] <= frame_time <= ev["end_time"]
        }

    if active_ports:
        reward_text = "Reward"
        x_offset = 140
        y_offset = 40 if rotate_deg == 0 else 5
        bbox_reward = draw.textbbox((0, 0), reward_text, font=font)
        text_h = bbox_reward[3] - bbox_reward[1]

        if 2 in active_ports:
            bl_x, bl_y = rotate_point(x_offset, height - y_offset - text_h,
                                      width, height, rotate_deg)
            draw.text((bl_x, bl_y), reward_text, fill=REWARD_COLOR, font=font)

        if 1 in active_ports:
            br_x, br_y = rotate_point(width - (x_offset + 230), height - y_offset - text_h,
                                      width, height, rotate_deg)
            draw.text((br_x, br_y), reward_text, fill=REWARD_COLOR, font=font)

    if mark_window and frame_time is not None:
        if mark_window[0] <= frame_time <= mark_window[1]:
            cx, cy = frame_w // 2, frame_h // 2
            draw.polygon([(cx - 60, cy - 40), (cx + 60, cy - 40), (cx, cy + 40)],
                         fill=MARK_COLOR)


def annotate_session(subjid, date, *, derivatives=None, rawdata=None,
                     output_suffix: str = DEFAULT_SUFFIX, centroid_radius: int = 8,
                     centroid_color: str = "red", rotate_deg: int = 0,
                     time_window=None, reward_display_s: float = 1.0,
                     video_indices=None, mark_timepoint: Optional[str] = None) -> list:
    """Render one session's annotated videos. Returns the paths written.

    ``video_indices`` are 1-based and number the videos in `layout.session_videos`
    order, which is the order that numbers the ``sleap_tracking_video<N>`` tables, so
    ``--video 2`` means the same video to both.
    """
    import cv2
    from PIL import Image, ImageDraw
    from tqdm import tqdm

    date_str = str(date)
    session = layout.layout_for(derivatives).find_session(subjid, date=date_str)
    results = layout.results_dir(session)
    raw_session = layout.layout_for(rawdata, name="rawdata").find_session(subjid, date=date_str)

    combined_file = layout.find_combined_table(results)
    if combined_file is None:
        raise FileNotFoundError(f"No combined timestamps file found in {results}")

    from hypnose_sleap.timestamps import _read_table
    combined_df = _read_table(combined_file)
    combined_df["video_file"] = combined_df["video_file"].astype(str)
    print(f"Loaded combined timestamps: {len(combined_df)} frames")

    videos_all = layout.session_videos(raw_session)
    if video_indices is None:
        videos = list(enumerate(videos_all, 1))
    else:
        if isinstance(video_indices, (int, np.integer)):
            selected = {int(video_indices)}
        else:
            try:
                selected = {int(v) for v in video_indices}
            except Exception:
                selected = set()
        videos = [(idx, vf) for idx, vf in enumerate(videos_all, 1) if idx in selected]

    if not videos:
        raise FileNotFoundError(f"No video files found for requested indices: {video_indices}")

    experiments = layout.session_experiments(raw_session)
    if not experiments:
        raise FileNotFoundError(f"No experiment folders with video in {raw_session.path}")
    events = session_events(experiments[0], reward_display_s=reward_display_s)
    valve_events, supply_events = events["valve"], events["supply"]

    font = _load_font()
    mark_ts_abs = parse_mark_timepoint(mark_timepoint, date_str) if mark_timepoint else None
    if mark_timepoint and mark_ts_abs is None:
        print(f"Warning: could not parse mark_timepoint='{mark_timepoint}', skipping marker")
    mark_window = ((mark_ts_abs, mark_ts_abs + pd.Timedelta(seconds=2.0))
                   if mark_ts_abs is not None else None)

    windows = normalize_time_windows(time_window)
    output_paths = []

    for video_idx, (video_num, video_path) in enumerate(videos, 1):
        video_name = layout.video_key(video_path)
        print(f"\nProcessing video {video_idx}/{len(videos)} (original #{video_num}): {video_name}")

        df_video = combined_df[combined_df["video_file"] == video_name].copy()
        if df_video.empty:
            print(f"  No timestamps found for video {video_name}, skipping")
            continue

        df_video["time"] = pd.to_datetime(df_video["time"])
        df_video = df_video.reset_index(drop=True)
        df_video = (df_video
                    .groupby("frame", as_index=False)
                    .agg({"centroid_x": "mean", "centroid_y": "mean", "time": "first"}))
        print(f"  Found {len(df_video)} frames with timestamps")

        cap_probe = cv2.VideoCapture(str(video_path))
        fps = cap_probe.get(cv2.CAP_PROP_FPS)
        width = int(cap_probe.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap_probe.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap_probe.get(cv2.CAP_PROP_FRAME_COUNT))
        cap_probe.release()
        print(f"  Video: {width}x{height} @ {fps} fps, {total_frames} frames")

        for window_idx, window in enumerate(windows, 1):
            df_window = df_video.copy()
            window_label = f"window{window_idx}" if window is not None else "full"

            if window is not None:
                base_time = df_window["time"].min()
                if pd.isna(base_time):
                    print(f"  Window {window_label}: missing base timestamp, skipping")
                    continue
                start_abs, end_abs = base_time + window[0], base_time + window[1]
                df_window = df_window[(df_window["time"] >= start_abs)
                                      & (df_window["time"] <= end_abs)].copy()
                if df_window.empty:
                    print(f"  Window {window_label}: no frames in requested time bounds, skipping")
                    continue
                print(f"  Window {window_label}: {window[0]} - {window[1]} ({len(df_window)} frames)")
            else:
                print(f"  Window {window_label}: full duration ({len(df_window)} frames)")

            df_window["frame"] = pd.to_numeric(df_window["frame"], errors="coerce").astype("Int64")
            df_window = df_window.dropna(subset=["frame"])
            row_map = df_window.set_index("frame").to_dict("index")
            valid_frames = sorted(row_map.keys())
            if not valid_frames:
                print(f"  Window {window_label}: no valid frames with centroid, skipping")
                continue

            start_frame, end_frame = int(valid_frames[0]), int(valid_frames[-1])

            cap = cv2.VideoCapture(str(video_path))
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

            out_size = (height, width) if rotate_deg in (90, 270) else (width, height)
            suffix_full = "_full" if window is None else f"_window{window_idx}"
            output_path = results / (
                f"{output_suffix}_rotdeg_{rotate_deg}_video{video_num}{suffix_full}.mp4"
            )
            output_path.parent.mkdir(parents=True, exist_ok=True)
            out = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"),
                                  fps, out_size)
            output_paths.append(output_path)
            print(f"  Saving to: {output_path}")

            frames_to_process = end_frame - start_frame + 1
            with tqdm(total=frames_to_process, desc="  Encoding", unit="frames") as pbar:
                current_frame = start_frame
                while current_frame <= end_frame:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    frame = rotate_frame(frame, rotate_deg)
                    frame_h, frame_w = frame.shape[:2]
                    frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    draw = ImageDraw.Draw(frame_pil)

                    _draw_overlay(
                        draw, row_map.get(current_frame),
                        width=width, height=height, rotate_deg=rotate_deg,
                        frame_w=frame_w, frame_h=frame_h,
                        centroid_radius=centroid_radius, centroid_color=centroid_color,
                        font=font, valve_events=valve_events, supply_events=supply_events,
                        mark_window=mark_window,
                    )

                    out.write(cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR))
                    current_frame += 1
                    pbar.update(1)

            cap.release()
            out.release()
            print(f"  Completed {window_label}")

    print("\nAll videos processed and saved.")
    return output_paths


__all__ = [
    "DEFAULT_SUFFIX", "FONT_SIZE", "FONT_ENV", "FONT_CANDIDATES", "FONT_DIRS",
    "find_font", "rotate_point", "rotate_frame",
    "parse_single_time_window", "normalize_time_windows",
    "parse_mark_timepoint", "session_events", "lookup_odor", "annotate_session",
]
