#!/usr/bin/env python
"""Golden-master regression for the hypnose-sleap restructure.

Fingerprints every session in ``sessions.yml`` at three levels and compares them
against ``fixtures/*.json``. Exit 0 = GREEN, 1 = RED.

- ``per_video`` (L1) -- must be byte-identical.
- ``combined``  (L2) -- must be byte-identical, and hold as many rows as its parts.
- ``quality``   (L3) -- checked for consistency with the parquets beside it.

Usage
-----
  # write baselines from the saved server output (optionally limit to sessions)
  python src/hypnose_sleap/qc/regression.py --generate
  python src/hypnose_sleap/qc/regression.py --generate 058:20260717

  # check against the baselines
  python src/hypnose_sleap/qc/regression.py

Sessions are ``subjid:date`` keys. Run in the env recorded in ``fixtures/env.json``.
Reads the derivatives tree read-only.
"""
from __future__ import annotations

import sys
import json
import argparse
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
FIXTURES = HERE / "fixtures"
REPO = HERE.parents[2]

# Importable without an install, so Phase 0 can run in `sleap-analysis`.
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "src"))

import _common  # noqa: E402


def _load_config() -> dict:
    with open(HERE / "sessions.yml") as fh:
        return yaml.safe_load(fh)


def _session_entries(config: dict) -> list[dict]:
    """Sessions with ``derivatives`` and ``levels`` defaulted from the file header."""
    default_root = config.get("default_derivatives")
    entries = []
    for session in config["sessions"]:
        entry = dict(session)
        entry.setdefault("derivatives", default_root)
        entry.setdefault("levels", list(_common.LEVELS))
        entries.append(entry)
    return entries


def _fixture_path(subjid, date) -> Path:
    return FIXTURES / f"sub-{str(subjid).zfill(3)}_date-{date}.json"


def _key(subjid, date) -> str:
    return f"{str(subjid).zfill(3)}:{date}"


def _select(sessions: list[dict], targets: set[str]) -> list[dict]:
    """Filter by ``subjid:date`` key; an empty target set means all of them."""
    if not targets:
        return sessions
    return [s for s in sessions
            if _key(s["subjid"], s["date"]) in targets
            or f"{s['subjid']}:{s['date']}" in targets]


def _source_banner() -> str:
    """One line naming what the fingerprints are taken from.

    A disk-vs-disk compare re-hashes the same files the fixtures were written from, so
    it proves the hashing is deterministic and nothing about the pipeline.
    """
    if _common.REDERIVE_AVAILABLE:
        return "SOURCE: rederive -- the new pipeline is run and its output fingerprinted."
    return ("SOURCE: disk -- re-hashing the saved baseline files. This checks the "
            "fingerprinting only; it is NOT a gate on the restructure.")


def generate(targets: set[str]) -> int:
    sessions = _select(_session_entries(_load_config()), targets)
    FIXTURES.mkdir(parents=True, exist_ok=True)
    print(f"Generating fixtures for {len(sessions)} session(s)"
          f"{' (filtered)' if targets else ''}...\n")
    failures = 0
    for s in sessions:
        subjid, date, label = s["subjid"], s["date"], s.get("label", "")
        try:
            # Always `disk`: a baseline is what the saved files say, never a re-run.
            fp = _common.fingerprint_session(
                subjid, date, derivatives=s["derivatives"], levels=s["levels"],
                rederive=False, rawdata=s.get("rawdata"),
            )
        except Exception as e:
            print(f"  [FAIL] {_key(subjid, date)} ({label}): {e}")
            failures += 1
            continue
        payload = {
            "subjid": str(subjid), "date": str(date), "label": label,
            "derivatives": str(s["derivatives"]), "levels": list(s["levels"]),
            **fp,
        }
        _fixture_path(subjid, date).write_text(json.dumps(payload, indent=2))
        n_videos = len(fp["per_video"]) if isinstance(fp["per_video"], dict) else 0
        combined = fp["combined"]
        combined_md5 = combined["md5"][:8] if isinstance(combined, dict) else combined
        print(f"  [OK]   {_key(subjid, date)} ({label}): "
              f"{len(fp['inputs'])} slp, {n_videos} per-video, combined={combined_md5}")
    (FIXTURES / "env.json").write_text(json.dumps(_common.env_fingerprint(), indent=2))
    print(f"\nWrote fixtures to {FIXTURES} (env.json recorded).")
    if failures:
        print(f"{failures} session(s) failed -- fix before relying on these fixtures.")
        return 1
    return 0


def _compare_inputs(expected: dict, got: dict, key: str, label: str) -> int:
    """Report changed ``.slp`` inputs. Not a RED -- it explains one."""
    if expected == got:
        return 0
    print(f"  [INPUT CHANGED] {key} ({label}): the .slp files differ from the baseline")
    for line in _common.diff_report("slp", expected, got):
        print(line)
    return 0


def _compare_per_video(expected: dict, got: dict, key: str, label: str) -> int:
    """L1. Every file must be present and byte-identical. Returns the RED count."""
    if not isinstance(expected, dict):
        print(f"  [skip]  {key} ({label}) per_video: not baselined")
        return 0

    red = 0
    for name in sorted(set(expected) | set(got)):
        if name not in got:
            print(f"  [RED]   {key} ({label}) per_video {name}: missing from output")
            red += 1
            continue
        if name not in expected:
            print(f"  [RED]   {key} ({label}) per_video {name}: not in the baseline")
            red += 1
            continue
        exp, act = expected[name], got[name]
        if exp["md5"] != act["md5"]:
            print(f"  [RED]   {key} ({label}) per_video {name}: "
                  f"expected {exp['md5'][:8]} got {act['md5'][:8]} "
                  f"({exp['rows']} -> {act['rows']} rows)")
            for line in _common.diff_report("column", exp["columns"], act["columns"]):
                print(line)
            red += 1
        else:
            print(f"  [green] {key} ({label}) per_video {name} ok ({act['md5'][:8]})")
    return red


def _compare_combined(expected, got, fingerprint: dict, key: str, label: str) -> int:
    """L2. Asserted byte-identical since Phase 4 measured the delta at zero.

    Two separate complaints, because they mean different things: the row count must
    equal the sum of the per-video tables (the join changed shape), and the md5 must
    match (the values moved). A md5 mismatch has one upstream cause this repo does not
    own -- `load_all_streams` -- so the RED names it.
    """
    red = 0
    for complaint in _common.check_combined_rows(fingerprint):
        print(f"  [RED]   {key} ({label}) combined: row count is not the sum of its parts")
        print(f"        {complaint}")
        red += 1

    if not isinstance(expected, dict):
        if expected == _common.ABSENT and isinstance(got, dict):
            print(f"  [RED]   {key} ({label}) combined: baseline is ABSENT, output has one")
            return red + 1
        print(f"  [skip]  {key} ({label}) combined: not baselined")
        return red
    if not isinstance(got, dict):
        print(f"  [RED]   {key} ({label}) combined: baseline has one, output is {got}")
        return red + 1
    if expected["md5"] == got["md5"]:
        print(f"  [green] {key} ({label}) combined ok ({got['md5'][:8]})")
        return red
    print(f"  [RED]   {key} ({label}) combined: expected {expected['md5'][:8]} "
          f"got {got['md5'][:8]} ({expected['rows']} -> {got['rows']} rows)")
    for line in _common.diff_report("column", expected["columns"], got["columns"]):
        print(line)
    print("        L2 was byte-identical at Phase 4. A mismatch is either this repo or "
          "hypnose_behavior.io.loaders.load_all_streams -- check fixtures/env.json first.")
    return red + 1


def _compare_discovery(got, key: str, label: str) -> int:
    """Phase 4's gate: `hypnose_behavior` finds the combined parquet we just wrote.

    An assertion about this run, not a comparison against a fixture -- the baselines
    were written from disk, where nothing re-derived a file to look for.
    """
    if not isinstance(got, dict):
        return 0
    if got.get("error"):
        print(f"  [skip]  {key} ({label}) discovery: hypnose_behavior not importable "
              f"({got['error']})")
        return 0
    if got["agree"]:
        print(f"  [green] {key} ({label}) discovery: find_tracking_file -> {got['found']}")
        return 0
    print(f"  [RED]   {key} ({label}) discovery: wrote {got['written']}, "
          f"hypnose_behavior.io.layout.find_tracking_file returned {got['found']}")
    return 1


def _compare_quality(expected, got, fingerprint: dict, key: str, label: str) -> int:
    """L3. Consistency with the parquets beside it. Returns the RED count."""
    complaints = _common.check_quality_consistency(fingerprint)
    if complaints:
        print(f"  [RED]   {key} ({label}) quality: inconsistent with its parquets")
        for line in complaints:
            print(f"        {line}")
        return len(complaints)
    if isinstance(got, dict):
        print(f"  [green] {key} ({label}) quality consistent ({got['file']})")
    elif isinstance(expected, dict):
        print(f"  [RED]   {key} ({label}) quality: baseline has a report, output has none")
        return 1
    else:
        print(f"  [skip]  {key} ({label}) quality: no report written yet")
    return 0


def compare(targets: set[str]) -> int:
    sessions = _select(_session_entries(_load_config()), targets)
    print(_source_banner())
    print(f"Checking {len(sessions)} session(s) against fixtures...\n")
    red = 0
    for s in sessions:
        subjid, date, label = s["subjid"], s["date"], s.get("label", "")
        key = _key(subjid, date)
        fpath = _fixture_path(subjid, date)
        if not fpath.exists():
            print(f"  [MISSING FIXTURE] {key} ({label}) -- run --generate first")
            red += 1
            continue
        expected = json.loads(fpath.read_text())
        try:
            got = _common.fingerprint_session(
                subjid, date, derivatives=s["derivatives"], levels=s["levels"],
                rawdata=s.get("rawdata"),
            )
        except Exception as e:
            print(f"  [ERROR] {key} ({label}): {e}")
            red += 1
            continue

        _compare_inputs(expected.get("inputs", {}), got["inputs"], key, label)
        red += _compare_per_video(expected.get("per_video"), got["per_video"], key, label)
        red += _compare_combined(expected.get("combined"), got["combined"], got, key, label)
        red += _compare_quality(expected.get("quality"), got["quality"], got, key, label)
        red += _compare_discovery(got.get("discovery"), key, label)

    print()
    if red:
        print(f"REGRESSION RED: {red} mismatch(es). See the lines above.")
        return 1
    print("REGRESSION GREEN: L1 and L2 byte-identical, L3 consistent.")
    if not _common.REDERIVE_AVAILABLE:
        print("  (disk source -- this GREEN says the fingerprinting is stable, "
              "not that the pipeline is.)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--generate", action="store_true",
                    help="write fixtures from the saved output (run before changes)")
    ap.add_argument("targets", nargs="*",
                    help="optional 'subjid:date' keys, e.g. 058:20260717; default: all")
    args = ap.parse_args()
    targets = set(args.targets)
    return generate(targets) if args.generate else compare(targets)


if __name__ == "__main__":
    raise SystemExit(main())
