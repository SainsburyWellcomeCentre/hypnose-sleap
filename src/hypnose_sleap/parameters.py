"""Pipeline parameters and model-role resolution.

- ``configs/parameters.yml``: ``score_thresh``, ``presence_frac``, ``gap_limit``, ``batch_size``.
- ``configs/models.yml``: role -> model path. Roles are ``naive``, ``eeg_surgery``,
  ``eeg_headstage``.
- ``configs/models.local.yml`` (git-ignored) holds ``model_root``. An absolute path in
  ``models.yml`` wins; a relative one resolves under ``model_root``.

Phase 2 fills this in.
"""
from __future__ import annotations
