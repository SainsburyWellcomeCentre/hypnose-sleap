"""Resolved rawdata and derivatives roots for this repo.

Wraps `hypnose_helpers.io.paths.DataLocations` over ``configs/``, with the same
``HYPNOSE`` env prefix and profile names as `hypnose-behavior`, so both repos resolve
to the same tree on a machine.

- Resolution order: ``HYPNOSE_*`` env vars > active profile > ``data/`` fallback.
- Roots are exposed as functions, never resolved Paths, so a caller can redirect them
  at runtime via env var plus ``cache_clear()``.
- `resolve_profile` and `transfer_endpoints` resolve a *named* profile, which is what
  ``fetch`` and ``push`` need alongside the active one.
"""
from __future__ import annotations

from pathlib import Path

from typing import Optional

from hypnose_helpers.io.paths import (
    DataLocations, DERIV_SUBDIR, PROFILES_FILENAME, RAW_SUBDIR, env_path, read_yaml,
)

ENV_PREFIX = "HYPNOSE"


def get_repo_root() -> Path:
    """The hypnose-sleap repo root (``paths.py`` -> ``io`` -> package -> ``src`` -> root)."""
    return Path(__file__).resolve().parents[3]


def get_config_dir() -> Path:
    return get_repo_root() / "configs"


_locations = DataLocations(
    config_dir=get_config_dir(),
    data_root=get_repo_root() / "data",
    env_prefix=ENV_PREFIX,
)

# Bound methods, so `get_derivatives_root()` keeps working at call sites and still
# exposes `.cache_clear()`.
load_profiles = _locations.load_profiles
get_active = _locations.get_active
set_active = _locations.set_active
reload = _locations.reload
get_data_root = _locations.get_data_root
get_rawdata_root = _locations.get_rawdata_root
get_server_root = _locations.get_server_root
get_derivatives_root = _locations.get_derivatives_root

_local_path = _locations._local_path
_profiles_path = _locations._profiles_path
_active_profile = _locations._active_profile
_env_path = env_path


def resolve_profile(name: str) -> dict:
    """``{'name', 'rawdata', 'derivatives'}`` for one named profile.

    - ``derivatives`` defaults to the sibling of ``rawdata`` when the profile omits it;
    - raises `KeyError` naming the available profiles when ``name`` is not one.
    """
    profiles = load_profiles()
    profile = profiles.get(name)
    if not isinstance(profile, dict) or not profile.get("rawdata"):
        raise KeyError(
            f"no data-location profile named {name!r} in {_profiles_path()}. "
            f"Available: {', '.join(profiles) or '(none)'}"
        )
    raw = str(profile["rawdata"])
    deriv = profile.get("derivatives") or str(Path(raw).parent / DERIV_SUBDIR)
    return {"name": name, "rawdata": Path(raw), "derivatives": Path(deriv)}


def is_remote(name: Optional[str] = None) -> bool:
    """Whether a profile is the shared server. Defaults to the active profile."""
    name = name or get_active()
    if not name:
        return False
    profile = load_profiles().get(name)
    return bool(isinstance(profile, dict) and profile.get("remote"))


def require_local(verb: str, *, allow_remote: bool = False,
                  profile: Optional[str] = None) -> None:
    """Refuse to run a writing verb against a remote profile.

    `.slp` files run to tens of MB per video and belong on local disk until `push`
    moves the parquet across. ``profile`` names the profile the verb will actually
    write to, defaulting to the active one; ``allow_remote=True`` is the deliberate
    override.
    """
    if allow_remote:
        return
    name = profile or get_active()
    if not is_remote(name):
        return
    root = resolve_profile(name)["derivatives"] if profile else get_derivatives_root()
    raise SystemExit(
        f"refusing to run `{verb}` against the remote profile {name!r} "
        f"({root}).\n"
        f"  Write locally, then `hypnose-sleap push`:\n"
        f"      hypnose-set-data-location local_1\n"
        f"  Or override deliberately with --allow-remote."
    )


def transfer_endpoints() -> dict:
    """The ``transfer: {remote, local}`` profile names from ``data_locations.yml``.

    Returns ``{'remote': <name>, 'local': <name>}``. Missing keys come back as None, so
    a caller can require ``--from`` / ``--to`` instead of guessing an endpoint.
    """
    transfer = read_yaml(_profiles_path()).get("transfer") or {}
    return {"remote": transfer.get("remote"), "local": transfer.get("local")}


__all__ = [
    "ENV_PREFIX", "get_repo_root", "get_config_dir",
    "get_data_root", "get_rawdata_root", "get_server_root", "get_derivatives_root",
    "load_profiles", "get_active", "set_active", "reload",
    "resolve_profile", "transfer_endpoints", "is_remote", "require_local",
    "RAW_SUBDIR", "DERIV_SUBDIR", "PROFILES_FILENAME",
]
