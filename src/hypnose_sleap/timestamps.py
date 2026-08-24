"""Per-video parquet + harp streams -> one combined parquet per session.

- Reads synchronised video frame times via `hypnose_behavior.io.loaders.load_all_streams`,
  one call per ``behav/`` experiment folder.
- Matches each per-video table to its video by ``video_file`` column, then filename
  suffix, then numeric index.
- Left-joins timestamps onto SLEAP frames, so every tracked frame survives even when
  no timestamp matches.
- Writes ``sub-XXX_ses-YYYYMMDD_combined_sleap_tracking_timestamps.parquet``.

The `hypnose_behavior` import is lazy: `extract` must run without it.

Phase 4 fills this in.
"""
from __future__ import annotations
