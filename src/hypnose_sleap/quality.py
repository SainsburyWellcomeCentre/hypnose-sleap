"""Per-node tracking quality, reported and written beside the parquet.

Writes ``movement_analysis/sleap_quality_sub-XXX_ses-YYYYMMDD.yml`` containing:

- ``model``: ``{role, path, training_config_md5}``;
- ``parameters``: ``{score_thresh, presence_frac, gap_limit}``;
- ``frames``: ``{total, occupied}``;
- per node ``{pres_pct_occ, pres_pct_occ_gated, score_p50, selected}``;
- ``centroid_nodes``, which must equal the ``centroid_nodes`` column in the parquet.

Emitted by `extract`, so the report is produced with the parquet rather than asked for
afterwards.

Phase 3 fills this in.
"""
from __future__ import annotations
