"""Session discovery, and where SLEAP output goes inside a session.

- `rawdata` and `derivatives`: two `SessionLayout` objects over
  `hypnose_helpers.io.layout`, both with ``subject_pattern="{subject}_id-*"``.
- Outputs go to ``saved_analysis_results/movement_analysis/``, which is what
  `hypnose_behavior.io.layout.find_tracking_file` reads.
- Reads use ``rglob``, so flat and grouped sessions both resolve; `write_path` creates
  the parent.
- `session_videos` and `video_key` are the one implementation of the
  ``behav/*/VideoData/*.avi`` walk and the ``<behav>__<video>`` naming.

No `hypnose_behavior` import: `extract` runs without it. `MOVEMENT_SUBFOLDER` agreeing
with that package is asserted by `qc/check_layout.py`, not by importing it here.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from hypnose_helpers.io.layout import (  # noqa: F401  (re-exported: one import for callers)
    DuplicateSessionError,
    SessionLayout,
    SessionRef,
    filter_sessions,
    list_sessions,
    normalize_subjid,
    parse_session_dirname,
    parse_subject,
    parse_subject_dirname,
)

from hypnose_sleap.io.paths import get_derivatives_root, get_rawdata_root

# Every subject directory in this dataset is `sub-NNN_id-XXX`.
SUBJECT_PATTERN = "{subject}_id-*"

# Where a session keeps its analysis outputs, and the subfolder SLEAP writes into.
RESULTS_DIRNAME = "saved_analysis_results"
MOVEMENT_SUBFOLDER = "movement_analysis"

# The raw video tree inside one session, and the extensions worth walking.
VIDEO_GLOB = "behav/*/VideoData/*.avi"

rawdata = SessionLayout(get_rawdata_root, name="rawdata", subject_pattern=SUBJECT_PATTERN)
derivatives = SessionLayout(
    get_derivatives_root, name="derivatives", subject_pattern=SUBJECT_PATTERN
)


def layout_for(root=None, *, name: str = "derivatives") -> SessionLayout:
    """The configured derivatives layout, or an ad-hoc one rooted at ``root``."""
    if root is None:
        return derivatives
    return SessionLayout(root, name=name, subject_pattern=SUBJECT_PATTERN)


def results_dir(session) -> Path:
    """The analysis-output directory of a session directory or a `SessionRef`.

    Nothing is checked: a session never analysed yields a path that does not exist, and
    whether that is an error or a session to skip is the caller's decision.
    """
    return Path(getattr(session, "path", session)) / RESULTS_DIRNAME


def movement_dir(results) -> Path:
    """The ``movement_analysis/`` subfolder of a results directory."""
    return Path(results) / MOVEMENT_SUBFOLDER


def write_path(results, name: str) -> Path:
    """Where to write one SLEAP output, with the parent created.

    Always the grouped path -- new output is never written flat.
    """
    path = movement_dir(results) / name
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def find_outputs(results, pattern: str) -> list:
    """Every file under a results directory matching ``pattern``, flat or grouped.

    - ``rglob``, so a session written before the grouping still resolves;
    - skips ``._`` resource forks;
    - ordered by file name.
    """
    root = Path(results)
    if not root.exists():
        return []
    return sorted(
        (p for p in root.rglob(pattern) if not p.name.startswith("._")),
        key=lambda p: p.name,
    )


def find_output(results, pattern: str) -> Optional[Path]:
    """The single file matching ``pattern``, or None. Raises when several match."""
    matches = find_outputs(results, pattern)
    if not matches:
        return None
    if len(matches) > 1:
        raise DuplicateSessionError(
            f"{len(matches)} files match {pattern!r} in {results}:\n  "
            + "\n  ".join(str(p) for p in matches)
        )
    return matches[0]


def video_key(video: Path) -> str:
    """The ``<behav>__<video>.avi`` name a video is known by downstream.

    The prefix disambiguates identical basenames in different ``behav/`` folders, which
    happens whenever a session was restarted within the same hour.
    """
    video = Path(video)
    return f"{video.parents[1].name}__{video.name}"


def session_videos(session) -> list:
    """Every ``.avi`` in a raw session, ordered by behav folder then file name.

    Ordering is what numbers the ``sleap_tracking_video<N>`` outputs, so it is the walk
    order, not an accident of the filesystem.
    """
    root = Path(getattr(session, "path", session))
    return sorted(
        (v for v in root.glob(VIDEO_GLOB) if not v.name.startswith("._")),
        key=lambda v: (v.parents[1].name, v.name),
    )


__all__ = [
    "rawdata", "derivatives", "layout_for", "SUBJECT_PATTERN",
    "RESULTS_DIRNAME", "MOVEMENT_SUBFOLDER", "VIDEO_GLOB",
    "results_dir", "movement_dir", "write_path", "find_outputs", "find_output",
    "video_key", "session_videos",
    "SessionRef", "SessionLayout", "DuplicateSessionError",
    "list_sessions", "filter_sessions", "normalize_subjid",
    "parse_subject", "parse_subject_dirname", "parse_session_dirname",
]
