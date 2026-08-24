"""Resolved rawdata and derivatives roots for this repo.

Wraps `hypnose_helpers.io.paths.DataLocations` over ``configs/``, so the active profile
is the same one `hypnose-behavior` reads on this machine.

- Resolution order: ``HYPNOSE_*`` env vars > active profile > ``data/`` fallback.
- Roots are exposed as functions, never resolved Paths, so the qc harness can redirect
  them at runtime.

Phase 2 fills this in.
"""
from __future__ import annotations

from pathlib import Path


def get_repo_root() -> Path:
    """The hypnose-sleap repo root (``paths.py`` -> ``io`` -> package -> ``src`` -> root)."""
    return Path(__file__).resolve().parents[3]
