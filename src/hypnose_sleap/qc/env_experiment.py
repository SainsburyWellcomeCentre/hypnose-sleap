#!/usr/bin/env python
"""Phase 0.5: measure the environment change with the code held constant.

Re-derives each fixture session's per-video parquet from its ``.slp`` using the
repointed `sleap_utils.py`, into a throwaway derivatives tree, and compares the
fingerprints against ``fixtures/``.

- GREEN -- the environment is transparent; the restructure gates against ``fixtures/``.
- RED   -- a characterised environment delta, before any code moved. ``--write-newenv``
  records it as ``fixtures/<session>.newenv.json`` for the restructure to gate against.
- SKIP  -- ``sleap_utils.py`` is gone. Phase 7 deletes it, which retires this script:
  its subject is *old code*, and the answer it produced is `DECISIONS.md` §8. Check out
  a commit before the deletion to repeat it. `regression.py` is the standing check that
  the current code still re-derives the baselines.

The real derivatives tree is only read. Each session's ``.slp`` files are copied into a
temp tree and the extractor is pointed at that, because the extractor writes its parquet
beside its input and would otherwise overwrite the baseline.

Usage
-----
  python src/hypnose_sleap/qc/env_experiment.py
  python src/hypnose_sleap/qc/env_experiment.py 058:20260717
  python src/hypnose_sleap/qc/env_experiment.py --write-newenv
"""
from __future__ import annotations

import io
import sys
import json
import shutil
import argparse
import tempfile
import contextlib
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent
FIXTURES = HERE / "fixtures"
REPO = HERE.parents[2]

sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))          # sleap_utils.py still lives at the repo root

import _common  # noqa: E402

# The pipeline defaults every baseline was produced with.
PARAMS = {"score_thresh": 0.4, "presence_frac": 0.7, "gap_limit": 120}


def _extractor():
    """The repointed `sleap_labels_and_centroid`, imported late so ``--help`` works."""
    from sleap_utils import sleap_labels_and_centroid
    return sleap_labels_and_centroid


def _subject_available() -> bool:
    """Whether the pre-restructure module this experiment measures still exists.

    Phase 7 deletes it, which retires this script by design: it exists to compare *old
    code* across two environments, and there is no old code any more. Reporting that as
    RED would claim the environment stopped being transparent, which is the opposite of
    what a missing subject means.
    """
    return (REPO / "sleap_utils.py").is_file()


def _stage(results: Path, tmp: Path) -> Path:
    """Mirror a session's directory names under ``tmp`` and copy its ``.slp`` in.

    Returns the staged derivatives root, which is what the extractor is pointed at.
    """
    session_dir = results.parent
    subject_dir = session_dir.parent
    staged = tmp / "derivatives" / subject_dir.name / session_dir.name / _common.RESULTS_DIRNAME
    staged.mkdir(parents=True, exist_ok=True)
    for slp in _common._sorted_matches(results, _common.SLP_GLOB):
        shutil.copy2(slp, staged / slp.name)
    return tmp / "derivatives"


def _rederive(subjid, date, results: Path, tmp: Path) -> Path:
    """Run the extractor over a staged copy. Returns the staged results directory."""
    deriv = _stage(results, tmp)
    sink = io.StringIO()
    with contextlib.redirect_stdout(sink):
        _extractor()(
            subjid, int(date), base_dir=str(deriv), node_pool=None,
            skip_empty=True, **PARAMS
        )
    return next(deriv.rglob(_common.RESULTS_DIRNAME))


def _engine_agreement(path: Path):
    """True when both parquet engines hash a file identically; None if one is missing."""
    hashes = []
    for engine in ("fastparquet", "pyarrow"):
        try:
            hashes.append(_common.fingerprint_frame(pd.read_parquet(path, engine=engine))[0])
        except ImportError:
            return None
    return hashes[0] == hashes[1]


def _numeric_delta(old: pd.DataFrame, new: pd.DataFrame) -> list:
    """Per-column deltas between two versions of one per-video table.

    - reports row-count and column-set changes;
    - for shared numeric columns, max |delta| and how many values differ;
    - for everything else, how many values differ and the dtype change.
    """
    lines = []
    if len(old) != len(new):
        lines.append(f"rows {len(old)} -> {len(new)}")
    only_old = sorted(set(old.columns) - set(new.columns))
    only_new = sorted(set(new.columns) - set(old.columns))
    if only_old:
        lines.append(f"columns lost: {only_old}")
    if only_new:
        lines.append(f"columns gained: {only_new}")
    if len(old) != len(new):
        return lines

    for col in sorted(set(old.columns) & set(new.columns)):
        a = old[col].reset_index(drop=True)
        b = new[col].reset_index(drop=True)
        if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
            av = a.to_numpy(dtype=float)
            bv = b.to_numpy(dtype=float)
            both_nan = np.isnan(av) & np.isnan(bv)
            differ = ~both_nan & ~(av == bv)
            n = int(np.count_nonzero(differ))
            if n:
                finite = differ & np.isfinite(av) & np.isfinite(bv)
                worst = float(np.max(np.abs(av[finite] - bv[finite]))) if finite.any() else float("nan")
                nan_flips = int(np.count_nonzero(np.isnan(av) != np.isnan(bv)))
                extra = f", {nan_flips:,} NaN flips" if nan_flips else ""
                lines.append(f"{col}: {n:,} values differ, max |delta| = {worst:.6g}{extra}")
        else:
            n = int((a.astype(str) != b.astype(str)).sum())
            if n:
                lines.append(f"{col}: {n:,} values differ (non-numeric; {a.dtype} -> {b.dtype})")
    return lines


def _load_sessions(targets: set) -> list:
    config = yaml.safe_load((HERE / "sessions.yml").read_text())
    default_root = config.get("default_derivatives")
    out = []
    for s in config["sessions"]:
        entry = dict(s)
        entry.setdefault("derivatives", default_root)
        key = f"{str(entry['subjid']).zfill(3)}:{entry['date']}"
        if not targets or key in targets or f"{entry['subjid']}:{entry['date']}" in targets:
            out.append(entry)
    return out


def _print_env_table(baseline: dict, env: dict) -> None:
    print("Phase 0.5 -- code held constant, environment changed.\n")
    print(f"  {'':<14}{'baseline':<24}{'now':<24}")
    for key in ("python", "pandas", "numpy", "sleap_io", "sleap"):
        old, new = baseline.get(key), env.get(key)
        flag = "" if old == new else "   <-- changed"
        print(f"  {key:<14}{str(old):<24}{str(new):<24}{flag}")
    print(f"  {'engines':<14}{str(baseline['parquet_engine']['available']):<24}"
          f"{str(env['parquet_engine']['available']):<24}")
    print(f"\n  parameters: {PARAMS}\n")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write-newenv", action="store_true",
                    help="write fixtures/<session>.newenv.json from this environment")
    ap.add_argument("targets", nargs="*", help="optional 'subjid:date' keys")
    args = ap.parse_args(argv)

    env = _common.env_fingerprint()
    baseline = json.loads((FIXTURES / "env.json").read_text())
    _print_env_table(baseline, env)

    if not _subject_available():
        print("SKIP -- sleap_utils.py is gone (Phase 7), so there is no pre-restructure\n"
              "  code left to re-derive from. This experiment ran once and is recorded in\n"
              "  docs/DECISIONS.md section 8: 16/16 per-video parquets byte-identical.\n"
              "  To repeat it, check out a commit before the deletion. The ongoing check\n"
              "  that the current code still re-derives the baselines is qc/regression.py.")
        return 0

    red = 0
    for s in _load_sessions(set(args.targets)):
        subjid, date, label = s["subjid"], s["date"], s.get("label", "")
        key = f"{str(subjid).zfill(3)}:{date}"
        fixture = json.loads(
            (FIXTURES / f"sub-{str(subjid).zfill(3)}_date-{date}.json").read_text()
        )
        expected = fixture["per_video"]
        results = _common.results_dir(s["derivatives"], subjid, date)

        with tempfile.TemporaryDirectory(prefix="hyp_env_") as tmp_str:
            try:
                staged = _rederive(subjid, date, results, Path(tmp_str))
            except Exception as e:
                print(f"  [ERROR] {key} ({label}): {type(e).__name__}: {e}")
                red += 1
                continue
            got = _common.fingerprint_per_video(staged)

            for name in sorted(set(expected) | set(got)):
                if name not in got:
                    print(f"  [RED]   {key} {name}: not produced")
                    red += 1
                    continue
                if name not in expected:
                    print(f"  [RED]   {key} {name}: produced but not in the baseline")
                    red += 1
                    continue
                if expected[name]["md5"] == got[name]["md5"]:
                    print(f"  [green] {key} {name[:58]} ok ({got[name]['md5'][:8]})")
                    continue

                red += 1
                print(f"  [RED]   {key} {name}")
                print(f"          {expected[name]['md5'][:8]} -> {got[name]['md5'][:8]}")
                print(f"          both parquet engines agree on the new file: "
                      f"{_engine_agreement(next(staged.rglob(name)))}")
                old_df = _common.read_baseline(next(results.rglob(name)))
                new_df = _common.read_baseline(next(staged.rglob(name)))
                for line in _numeric_delta(_common.canonical(old_df), _common.canonical(new_df)):
                    print(f"          {line}")

            if args.write_newenv:
                payload = {
                    "subjid": fixture["subjid"], "date": fixture["date"],
                    "label": fixture["label"], "environment": env,
                    "parameters": PARAMS, "per_video": got,
                }
                out = FIXTURES / f"sub-{str(subjid).zfill(3)}_date-{date}.newenv.json"
                out.write_text(json.dumps(payload, indent=2))
                print(f"          wrote {out.name}")

    print()
    if red:
        print(f"PHASE 0.5 RED: {red} mismatch(es). The environment is NOT transparent.")
        print("Record the deltas in docs/DECISIONS.md and gate the restructure against "
              "the .newenv.json fixtures.")
        return 1
    print("PHASE 0.5 GREEN: the environment is transparent -- every per-video parquet "
          "re-derives byte-identically.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
