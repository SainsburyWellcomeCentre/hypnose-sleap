"""``.slp`` -> per-video parquet, with a session-consistent centroid.

Per session, in two passes over the ``.slp`` files:

- gate each point to NaN where it is absent or its score is below ``score_thresh``;
- select one node set for the whole session -- nodes present in at least
  ``presence_frac`` of occupied frames -- falling back to any node with detections;
- interpolate each selected node across internal gaps up to ``gap_limit`` frames;
- centroid = mean of the selected nodes, recorded in a ``centroid_nodes`` column.

Writes ``movement_analysis/sleap_tracking_video<N>_<behav>__<video>.parquet`` plus the
quality report from `quality.py`.

Phase 3 fills this in; `qc/_common.py` imports `extract_session` from here to switch
the regression gate from `disk` to `rederive`.
"""
from __future__ import annotations
