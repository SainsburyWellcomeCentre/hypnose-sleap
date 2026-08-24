"""Pipeline parameters and model-role resolution, from ``configs/``.

- `parameters()` reads ``parameters.yml``: ``score_thresh``, ``presence_frac``,
  ``gap_limit``, ``batch_size``. `resolve` layers a CLI override over it.
- `resolve_model()` turns a role name (``naive``, ``eeg_surgery``, ``eeg_headstage``) or
  an explicit path into a model directory.
- `model_provenance()` returns what the quality report records: role, resolved path and
  the md5 of ``training_config.json``.

Relative paths in ``models.yml`` resolve under ``model_root`` from the git-ignored
``models.local.yml``; an absolute path wins.
"""
from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path
from typing import Optional

from hypnose_helpers.io.paths import read_yaml

from hypnose_sleap.io.paths import get_config_dir

PARAMETERS_FILENAME = "parameters.yml"
MODELS_FILENAME = "models.yml"
MODELS_LOCAL_FILENAME = "models.local.yml"

# Every parameter a call site may ask for, with the value used if the config omits it.
DEFAULTS = {
    "score_thresh": 0.4,
    "presence_frac": 0.7,
    "gap_limit": 120,
    "batch_size": 64,
}


@lru_cache
def parameters() -> dict:
    """``parameters.yml`` layered over `DEFAULTS`.

    An unknown key in the file raises: a typo that silently does nothing is how a run
    ends up not using the threshold it says it used.
    """
    config = read_yaml(get_config_dir() / PARAMETERS_FILENAME)
    unknown = sorted(set(config) - set(DEFAULTS))
    if unknown:
        raise KeyError(
            f"{PARAMETERS_FILENAME} has unknown parameter(s): {', '.join(unknown)}. "
            f"Known: {', '.join(sorted(DEFAULTS))}"
        )
    return {**DEFAULTS, **config}


def resolve(name: str, override=None):
    """One parameter: the CLI override when given, else the configured value."""
    if name not in DEFAULTS:
        raise KeyError(f"unknown parameter {name!r}. Known: {', '.join(sorted(DEFAULTS))}")
    return parameters()[name] if override is None else override


def extract_parameters(**overrides) -> dict:
    """The three `extract` parameters, with any supplied overrides applied."""
    return {
        key: resolve(key, overrides.get(key))
        for key in ("score_thresh", "presence_frac", "gap_limit")
    }


@lru_cache
def _models_config() -> dict:
    return read_yaml(get_config_dir() / MODELS_FILENAME)


@lru_cache
def model_root() -> Optional[Path]:
    """``model_root`` from ``models.local.yml``, or None when the file is absent."""
    value = read_yaml(get_config_dir() / MODELS_LOCAL_FILENAME).get("model_root")
    return Path(value) if value else None


def roles() -> list:
    """Every role name in ``models.yml``."""
    return sorted((_models_config().get("models") or {}))


def default_role() -> Optional[str]:
    return _models_config().get("default")


def resolve_model(model: Optional[str] = None, *, previous: bool = False) -> Path:
    """A model directory from a role name, an explicit path, or the configured default.

    - an existing path is returned as-is, so a one-off model needs no config entry;
    - a role resolves through ``models.yml``, and ``previous=True`` picks that role's
      pre-retraining variant;
    - a relative path resolves under `model_root`.

    Raises `FileNotFoundError` when the resolved directory does not exist, naming what
    was tried -- a missing model must not surface later as a sleap-track error.
    """
    config = _models_config()
    models = config.get("models") or {}
    name = model or config.get("default")
    if not name:
        raise ValueError(f"no model given and no `default` in {MODELS_FILENAME}")

    if name in models:
        entry = models[name]
        key = "previous" if previous else "path"
        raw = entry.get(key) if isinstance(entry, dict) else entry
        if not raw:
            raise KeyError(f"role {name!r} has no {key!r} in {MODELS_FILENAME}")
        role = name
    else:
        candidate = Path(name)
        if not candidate.is_absolute() and not candidate.exists():
            raise KeyError(
                f"{name!r} is neither a role in {MODELS_FILENAME} nor an existing path. "
                f"Roles: {', '.join(roles()) or '(none)'}"
            )
        raw, role = name, None

    path = Path(raw)
    if not path.is_absolute():
        root = model_root()
        if root is None:
            raise FileNotFoundError(
                f"{raw!r} is relative and no `model_root` is set. Add one to "
                f"{get_config_dir() / MODELS_LOCAL_FILENAME}, or give an absolute path."
            )
        path = root / path

    if not path.is_dir():
        raise FileNotFoundError(
            f"model directory not found for {name!r}: {path}"
            + (f" (role {role!r})" if role else "")
        )
    return path


# A trained model's config, newest format first. sleap-nn writes `training_config.yaml`
# beside a `best.ckpt`; the older TensorFlow models wrote `training_config.json` beside a
# `best_model.h5`. Both are in circulation, so both are looked for.
TRAINING_CONFIG_NAMES = ("training_config.yaml", "training_config.json")


def model_provenance(model: Optional[str] = None, *, previous: bool = False) -> dict:
    """``{role, path, training_config, training_config_md5}`` for the quality report.

    ``training_config`` names which file was hashed; both are None when the directory
    has neither.
    """
    config = _models_config()
    role = model if model in (config.get("models") or {}) else None
    if model is None:
        role = config.get("default")
    path = resolve_model(model, previous=previous)
    for name in TRAINING_CONFIG_NAMES:
        candidate = path / name
        if candidate.is_file():
            return {
                "role": role,
                "path": str(path),
                "training_config": name,
                "training_config_md5": hashlib.md5(candidate.read_bytes()).hexdigest(),
            }
    return {"role": role, "path": str(path),
            "training_config": None, "training_config_md5": None}


__all__ = [
    "DEFAULTS", "parameters", "resolve", "extract_parameters",
    "roles", "default_role", "model_root", "resolve_model", "model_provenance",
]
