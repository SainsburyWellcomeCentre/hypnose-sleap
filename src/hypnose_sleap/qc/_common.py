"""Fingerprinting core for the hypnose-sleap regression gate.

Hashes a session's saved SLEAP output so a restructuring step can be shown to change
no number. Three levels, fingerprinted independently:

- ``per_video`` (L1) -- ``.slp`` -> per-video parquet. Expected byte-identical.
- ``combined`` (L2)  -- per-video parquet + harp streams -> combined parquet. Measured at
  Phase 4 and found byte-identical, so asserted as one since.
- ``quality``  (L3)  -- the quality report ``.yml``. Checked for consistency, not identity.

Two sources: ``disk`` reads the saved baseline files, ``rederive`` runs the new
pipeline into a temp dir. Only ``disk`` works until `hypnose_sleap.extract` exists.
"""
from __future__ import annotations

import io
import hashlib
import json
import shutil
import sys
import tempfile
import contextlib
from pathlib import Path

import pandas as pd

# --- single import surface -------------------------------------------------
# Update ONLY these lines as modules move. The md5s they produce must not change.
try:
    from hypnose_sleap.extract import extract_session as _extract_session  # noqa: F401
    from hypnose_sleap.timestamps import combine_session as _combine_session  # noqa: F401
    REDERIVE_AVAILABLE = True
except Exception:
    _extract_session = None
    _combine_session = None
    REDERIVE_AVAILABLE = False
# ---------------------------------------------------------------------------


# Session discovery is a local glob, not `hypnose_helpers.io.layout`:
# - the gate must not import what it gates;
# - at Phase 0 it runs in `sleap-analysis`, where the helpers are not installed.
RESULTS_DIRNAME = "saved_analysis_results"
RAWDATA_DIRNAME = "rawdata"
SUBJECT_GLOB = "sub-{subject}_id-*"
SESSION_GLOB = "ses-*_date-{date}"

# Matched with `rglob`, so flat and `movement_analysis/`-grouped sessions both resolve.
PER_VIDEO_GLOB = "sleap_tracking_video*.parquet"
COMBINED_STEM = "*_combined_sleap_tracking_timestamps"
COMBINED_GLOB = f"{COMBINED_STEM}.parquet"
QUALITY_GLOB = "sleap_quality_sub-*.yml"
SLP_GLOB = "*.slp"

LEVELS = ("per_video", "combined", "quality")

# The engine that wrote every baseline parquet; enforced by `read_baseline`.
BASELINE_ENGINE = "fastparquet"

ABSENT = "ABSENT"


# --- naming ----------------------------------------------------------------

def _is_noise(path: Path) -> bool:
    """True for files the pipeline did not write: ``._`` forks and ``_OLD`` copies."""
    return path.name.startswith("._") or "_OLD" in path.stem


def _sorted_matches(root: Path, pattern: str) -> list[Path]:
    """Non-noise `rglob` matches under ``root``, ordered by file name."""
    return sorted((p for p in root.rglob(pattern) if not _is_noise(p)), key=lambda p: p.name)


# --- hashing ---------------------------------------------------------------

def _md5(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def _md5_file(path: Path) -> str:
    """md5 of a file's bytes, streamed in 1 MB chunks."""
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical(df: pd.DataFrame) -> pd.DataFrame:
    """Columns sorted, index reset -- the form every fingerprint is taken over.

    Column order and row labels cannot move a hash on their own; a changed value can.
    """
    return df.reindex(sorted(df.columns), axis=1).reset_index(drop=True)


def fingerprint_frame(df: pd.DataFrame) -> tuple[str, dict]:
    """``(overall md5, {column: md5})`` over the canonical CSV of a frame.

    - overall md5 is the pass/fail signal;
    - per-column md5s report which column moved on a mismatch.
    """
    df = canonical(df)
    overall = _md5(df.to_csv(index=False))
    per_col = {str(c): _md5(df[c].to_csv(index=False, header=False)) for c in df.columns}
    return overall, per_col


# --- reading ---------------------------------------------------------------

def read_baseline(path: Path) -> pd.DataFrame:
    """Read a saved parquet with `BASELINE_ENGINE`.

    Falls back to pandas' default engine when it is unavailable, and warns on stderr:
    fingerprints taken through a different reader are not comparable.
    """
    try:
        return pd.read_parquet(path, engine=BASELINE_ENGINE)
    except ImportError:
        print(
            f"  ! {BASELINE_ENGINE} unavailable; reading {path.name} with the default "
            f"engine. Fingerprints are NOT comparable with ones taken under "
            f"{BASELINE_ENGINE}.",
            file=sys.stderr,
        )
        return pd.read_parquet(path)


def read_quality(path: Path) -> dict:
    """A quality report ``.yml`` as a plain dict."""
    import yaml
    return yaml.safe_load(path.read_text()) or {}


# --- discovery -------------------------------------------------------------

def results_dir(derivatives_root, subjid, date) -> Path:
    """The ``saved_analysis_results`` directory for one session.

    - raises when the subject, the session or the directory is missing;
    - raises when two session directories carry the same date.
    """
    root = Path(derivatives_root)
    subject = str(subjid).zfill(3)
    subj_dirs = sorted(root.glob(SUBJECT_GLOB.format(subject=subject)))
    if not subj_dirs:
        raise FileNotFoundError(f"no subject directory for sub-{subject} under {root}")
    ses_dirs = sorted(subj_dirs[0].glob(SESSION_GLOB.format(date=date)))
    if not ses_dirs:
        raise FileNotFoundError(f"no session dated {date} under {subj_dirs[0]}")
    if len(ses_dirs) > 1:
        raise ValueError(
            f"sub-{subject} has {len(ses_dirs)} directories dated {date}:\n  "
            + "\n  ".join(str(d) for d in ses_dirs)
        )
    out = ses_dirs[0] / RESULTS_DIRNAME
    if not out.exists():
        raise FileNotFoundError(f"no {RESULTS_DIRNAME} directory in {ses_dirs[0]}")
    return out


def _nodes_of(df: pd.DataFrame) -> list[str]:
    """The skeleton nodes a per-video table carries, from its ``<node>_x`` columns."""
    return sorted(
        c[:-2] for c in df.columns
        if c.endswith("_x") and not c.startswith("centroid")
    )


# --- level fingerprints ----------------------------------------------------

def fingerprint_inputs(results: Path) -> dict:
    """``{slp filename: md5}`` for the session's prediction files.

    Not a level -- it is the input L1 claims to reproduce, so an L1 mismatch can be
    attributed to the code rather than to a changed ``.slp``.
    """
    return {p.name: _md5_file(p) for p in _sorted_matches(results, SLP_GLOB)}


def fingerprint_per_video(results: Path) -> dict:
    """L1: one entry per ``sleap_tracking_video*.parquet``, keyed by file name.

    - keyed by name, so an added or removed video reports as a file, not a shift;
    - carries ``nodes`` and ``centroid_nodes``, which L3 is checked against.
    """
    out = {}
    for path in _sorted_matches(results, PER_VIDEO_GLOB):
        df = read_baseline(path)
        overall, columns = fingerprint_frame(df)
        centroid_nodes = (
            str(df["centroid_nodes"].iloc[0])
            if "centroid_nodes" in df.columns and len(df)
            else None
        )
        out[path.name] = {
            "md5": overall,
            "columns": columns,
            "rows": int(len(df)),
            "nodes": _nodes_of(df),
            "centroid_nodes": centroid_nodes,
        }
    return out


def fingerprint_combined(results: Path) -> dict | str:
    """L2: the combined-timestamps parquet, or `ABSENT` when there is none.

    Two matches raise rather than picking the first -- an ambiguous tree is an error.
    """
    matches = _sorted_matches(results, COMBINED_GLOB)
    if not matches:
        return ABSENT
    if len(matches) > 1:
        raise ValueError(
            f"{len(matches)} combined parquets in {results}:\n  "
            + "\n  ".join(p.name for p in matches)
        )
    df = read_baseline(matches[0])
    overall, columns = fingerprint_frame(df)
    return {
        "file": matches[0].name,
        "md5": overall,
        "columns": columns,
        "rows": int(len(df)),
        "videos": sorted(df["video_file"].dropna().unique().tolist())
        if "video_file" in df.columns else [],
    }


def fingerprint_quality(results: Path) -> dict | str:
    """L3: the quality report ``.yml``, or `ABSENT` when there is none.

    `ABSENT` for every session until `quality.py` starts writing them; L3 is a
    consistency check (`check_quality_consistency`), never an identity check.
    """
    matches = _sorted_matches(results, QUALITY_GLOB)
    if not matches:
        return ABSENT
    if len(matches) > 1:
        raise ValueError(
            f"{len(matches)} quality reports in {results}:\n  "
            + "\n  ".join(p.name for p in matches)
        )
    report = read_quality(matches[0])
    return {
        "file": matches[0].name,
        "md5": _md5(json.dumps(report, sort_keys=True, default=str)),
        "centroid_nodes": report.get("centroid_nodes"),
        "model": report.get("model"),
        "parameters": report.get("parameters"),
    }


def check_discovery(results: Path) -> dict:
    """Whether `hypnose_behavior` locates the combined parquet in ``results``.

    Phase 4's gate. `qc/check_layout.py` asserts the same agreement against an empty
    file it writes itself; this asserts it against the real combined parquet, in the
    tree the gate has just built.
    """
    ours = _sorted_matches(results, COMBINED_GLOB)
    written = ours[0] if ours else None
    try:
        from hypnose_behavior.io.layout import find_tracking_file
    except Exception as exc:
        return {"written": written.name if written else None,
                "found": None, "agree": None, "error": str(exc)}

    found = find_tracking_file(results, COMBINED_STEM)
    found = Path(found) if found else None
    return {
        "written": written.name if written else None,
        "found": found.name if found else None,
        "agree": bool(written is not None and found == written),
    }


def check_combined_rows(fingerprint: dict) -> list[str]:
    """L2's shape assertion: the combined table has as many rows as its parts.

    Measured at Phase 0 on all four L2 fixtures (`docs/DECISIONS.md` section 4) -- the
    join adds columns, not rows. So a row-count delta is a change in the join, which is
    a different finding from the `time` values moving, and is worth separating.
    """
    combined = fingerprint.get("combined")
    per_video = fingerprint.get("per_video")
    if not isinstance(combined, dict) or not isinstance(per_video, dict):
        return []
    expected = sum(entry["rows"] for entry in per_video.values())
    if combined["rows"] == expected:
        return []
    return [f"{combined['file']} has {combined['rows']:,} rows; its {len(per_video)} "
            f"per-video table(s) sum to {expected:,}"]


def check_quality_consistency(fingerprint: dict) -> list[str]:
    """L3's assertion: the report's ``centroid_nodes`` matches every parquet beside it.

    Returns a list of complaints, empty when consistent. An absent report is not a
    complaint.
    """
    quality = fingerprint.get("quality")
    if quality == ABSENT or not isinstance(quality, dict):
        return []

    reported = quality.get("centroid_nodes")
    if isinstance(reported, list):
        reported = ";".join(reported)

    complaints = []
    for name, entry in sorted(fingerprint.get("per_video", {}).items()):
        actual = entry.get("centroid_nodes")
        if actual != reported:
            complaints.append(
                f"{quality.get('file')} says centroid_nodes={reported!r} "
                f"but {name} carries {actual!r}"
            )
    return complaints


# --- session fingerprint ---------------------------------------------------

def default_rawdata(derivatives) -> Path:
    """The rawdata root beside a derivatives root: ``E:\\derivatives`` -> ``E:\\rawdata``.

    Derived from the fixture's own root rather than read from the active profile, so the
    gate measures the tree `sessions.yml` names and needs neither ``HYPNOSE_*`` env vars
    nor a ``cache_clear()``.
    """
    return Path(derivatives).parent / RAWDATA_DIRNAME


@contextlib.contextmanager
def _measured(results: Path, subjid, date, *, rederive: bool, levels, rawdata):
    """The directory the fingerprinted levels are read from.

    ``rederive=False`` yields the saved results directory. Otherwise the session's
    ``.slp`` files are copied into a temp tree and the pipeline is run against that --
    `extract` and `combine` both write beside their input, so pointing them at the real
    tree would overwrite the baselines.

    Only ``.slp`` is staged: `extract` writes its parquets into the temp tree and
    `combine` reads them there, while the harp streams `combine` needs are read from the
    real ``rawdata``. Nothing is written outside the temp tree.
    """
    if not rederive:
        yield results
        return

    with tempfile.TemporaryDirectory(prefix="hyp_rederive_") as tmp:
        session_dir = results.parent
        deriv = Path(tmp) / "derivatives"
        staged = deriv / session_dir.parent.name / session_dir.name / RESULTS_DIRNAME
        staged.mkdir(parents=True)
        for slp in _sorted_matches(results, SLP_GLOB):
            shutil.copy2(slp, staged / slp.name)
        # The model that wrote these `.slp` is not recorded anywhere, so it is not
        # claimed: `model=None` writes UNKNOWN_MODEL into the quality report.
        with contextlib.redirect_stdout(io.StringIO()):
            _extract_session(subjid, date, model=None, derivatives=deriv, skip_empty=True)
            if "combined" in levels:
                _combine_session(subjid, date, derivatives=deriv, rawdata=rawdata)
        yield staged


def fingerprint_session(subjid, date, *, derivatives, levels=LEVELS, rederive=None,
                        rawdata=None) -> dict:
    """Fingerprint one session across the requested `levels`.

    - a level not in ``levels`` records as ``"NOT BASELINED"``, which is distinct from
      `ABSENT` -- sub-066 has no combined parquet, and neither may read as the other;
    - ``rederive`` defaults to `REDERIVE_AVAILABLE`: every level then comes from a fresh
      run into a temp tree rather than from the saved files;
    - ``rawdata`` is where `combine` reads its harp streams from, defaulting to the
      sibling of ``derivatives``. It is only ever read.
    """
    results = results_dir(derivatives, subjid, date)
    if rederive is None:
        rederive = REDERIVE_AVAILABLE
    if rawdata is None:
        rawdata = default_rawdata(derivatives)

    fingerprint: dict = {
        "results_dir": str(results),
        "inputs": fingerprint_inputs(results),
    }
    with _measured(results, subjid, date, rederive=rederive, levels=levels,
                   rawdata=rawdata) as measured:
        fingerprint["per_video"] = (
            fingerprint_per_video(measured) if "per_video" in levels else "NOT BASELINED"
        )
        fingerprint["combined"] = (
            fingerprint_combined(measured) if "combined" in levels else "NOT BASELINED"
        )
        fingerprint["quality"] = (
            fingerprint_quality(measured) if "quality" in levels else "NOT BASELINED"
        )
        fingerprint["discovery"] = (
            check_discovery(measured) if rederive and "combined" in levels
            else "NOT MEASURED"
        )
    return fingerprint


# --- reporting -------------------------------------------------------------

def diff_report(label: str, expected: dict, got: dict, indent: str = "      ") -> list[str]:
    """Added / removed / changed lines between two ``{name: md5}`` maps."""
    eset, gset = set(expected), set(got)
    added = sorted(gset - eset)
    removed = sorted(eset - gset)
    changed = sorted(k for k in (eset & gset) if expected[k] != got[k])
    lines = []
    if added:
        lines.append(f"{indent}+ added {label}: {', '.join(added)}")
    if removed:
        lines.append(f"{indent}- removed {label}: {', '.join(removed)}")
    if changed:
        lines.append(f"{indent}~ changed {label}: {', '.join(changed)}")
    if not lines:
        lines.append(f"{indent}(overall md5 differs but every {label} md5 matches "
                     f"-- likely row order / dtype / something not captured here)")
    return lines


# --- environment -----------------------------------------------------------

# Distribution name for each module whose version is recorded, where the two differ.
DISTRIBUTIONS = {"sleap_io": "sleap-io"}


def _version(module_name: str) -> str | None:
    """An installed package's version, or None if absent.

    Read from distribution metadata, not by importing -- ``import sleap`` pulls in torch.
    """
    from importlib.metadata import PackageNotFoundError, version
    try:
        return version(DISTRIBUTIONS.get(module_name, module_name))
    except PackageNotFoundError:
        return None
    except Exception:
        return None


def env_fingerprint() -> dict:
    """The six versions the md5s depend on, written to ``fixtures/env.json``.

    - ``sleap`` writes ``.slp`` and cannot move a number measured here;
    - ``sleap_io`` reads them in `extract` and is directly upstream of every L1 hash;
    - ``parquet_engine`` records both engines present, so a RED can be attributed.
    """
    engines = [name for name in ("fastparquet", "pyarrow") if _version(name) is not None]
    return {
        "python": sys.version.split()[0],
        "pandas": pd.__version__,
        "numpy": _version("numpy"),
        "parquet_engine": {
            "baseline": BASELINE_ENGINE,
            "available": engines,
            "fastparquet": _version("fastparquet"),
            "pyarrow": _version("pyarrow"),
        },
        "sleap_io": _version("sleap_io"),
        "sleap": _version("sleap"),
    }
