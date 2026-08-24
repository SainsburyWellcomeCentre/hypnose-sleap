"""Annotated overlay video: centroid, odour label and reward markers.

- Reads the combined parquet via `hypnose_behavior.io.layout.find_tracking_file`.
- Draws the centroid per frame, the active odour near the poke port, and reward
  markers for a window after each supply-port pulse.
- Optional rotation (0/90/180/270), time windows, and a marked timepoint.
- Writes one ``.mp4`` per video per window.

Phase 6 fills this in.
"""
from __future__ import annotations
