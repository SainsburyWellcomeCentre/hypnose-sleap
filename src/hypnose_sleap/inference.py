"""Video enumeration and ``sleap-track``.

- Walks ``rawdata/sub-*/ses-*/behav/*/VideoData/*.avi`` through the shared selectors.
- Resolves the model from a role name (`parameters.py`) or an explicit path.
- Writes ``movement_analysis/<behav>__<video>.predictions.slp``, the ``<behav>__``
  prefix keeping identical basenames from different behav folders apart.
- ``--dry-run`` lists the videos it would process and exits.

Phase 5 fills this in.
"""
from __future__ import annotations
