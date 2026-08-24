"""Copying data between the local machine and the server.

- ``fetch``: remote rawdata ``.avi`` -> local rawdata, same tree.
- ``push``: local derivatives ``.parquet`` / ``.yml`` -> remote derivatives.
- Endpoints come from ``transfer: {remote, local}`` in ``configs/data_locations.yml``;
  ``--from`` / ``--to`` override either end.
- ``push`` prints its plan and refuses to overwrite an existing destination without ``--force``.

Phase 5 fills this in.
"""
from __future__ import annotations
