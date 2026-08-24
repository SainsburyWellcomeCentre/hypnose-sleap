"""Session discovery, and where SLEAP output goes inside a session.

- Two `SessionLayout` objects over `hypnose_helpers.io.layout`, one per tree, with
  ``subject_pattern="{subject}_id-*"``.
- Outputs land in ``saved_analysis_results/movement_analysis/``, matching
  `hypnose_behavior.io.layout.MOVEMENT_SUBFOLDER`.
- Reads use ``rglob``, so flat and grouped sessions both resolve; writes create the parent.

Phase 2 fills this in.
"""
from __future__ import annotations
