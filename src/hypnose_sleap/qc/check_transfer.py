#!/usr/bin/env python
"""Assert the three things `transfer_sleap_results.ps1` got wrong stay fixed.

The Phase 5 gate compared `push --dry-run`'s plan against the script's on the real
tree and they matched file for file. What that comparison cannot see is *behaviour*:
the old script's plan was correct while its side effects were not. This checks the
side effects, against a temporary tree, so it runs anywhere.

Checks:

- planning creates nothing -- ``:132-134`` ran ``New-Item -Force`` on every destination
  directory *before* the ``if ($DryRun)`` at ``:136``, so ``-DryRun`` wrote 143 empty
  directories onto ceph on the tree measured at Phase 5;
- `PUSH_PATTERNS` covers the quality report -- ``:7`` lists the two tables in both
  extensions but not ``sleap_quality_sub-*.yml``, which Phase 3 added, so quality
  reports have never reached the server;
- `run_plan` refuses an existing destination without ``force`` -- ``:140`` passed
  ``Copy-Item -Force`` unconditionally, making every run an overwrite;
- every destination lands under the destination root.

Exit 0 = fixed, 1 = regressed.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from hypnose_sleap.io import transfer

# One session's worth of output, named the way `extract` and `combine` name it.
SESSION = ("sub-057_id-370", "ses-060_date-20260717")
OUTPUTS = (
    "sleap_tracking_video1_2026-07-17T12-27-01__VideoData_1904-01-04T00-00-00.parquet",
    "sleap_tracking_video2_2026-07-17T12-27-01__VideoData_1904-01-04T01-00-00.csv",
    "sub-057_ses-20260717_combined_sleap_tracking_timestamps.parquet",
    "sleap_quality_sub-057_ses-20260717.yml",
)
# Not output: must not be planned.
NOISE = ("notes.txt", "sleap_tracking_video1.parquet.bak")


def _build(root: Path) -> Path:
    """A derivatives tree holding one session's outputs plus two files to ignore."""
    results = root / SESSION[0] / SESSION[1] / "saved_analysis_results"
    movement = results / "movement_analysis"
    movement.mkdir(parents=True)
    for name in OUTPUTS:
        (movement / name).write_bytes(b"x" * 16)
    for name in NOISE:
        (movement / name).write_bytes(b"x" * 16)
    return results


def main() -> int:
    failures = []

    with tempfile.TemporaryDirectory(prefix="hyp_transfer_") as tmp:
        source_root = Path(tmp) / "derivatives"
        dest_root = Path(tmp) / "remote"
        _build(source_root)
        dest_root.mkdir()

        plan = transfer.plan_push(source_root, dest_root)

        # 1. Planning is read-only.
        created = list(dest_root.rglob("*"))
        ok = not created
        print(f"  {'ok  ' if ok else 'FAIL'} planning creates nothing "
              f"({len(created)} path(s) under the destination)")
        if not ok:
            failures.append("dry-run writes")

        # 2. The pattern set, including the quality report the old script missed.
        planned = {c.source.name for c in plan}
        missing = sorted(set(OUTPUTS) - planned)
        extra = sorted(planned & set(NOISE))
        ok = not missing and not extra
        print(f"  {'ok  ' if ok else 'FAIL'} plans {len(planned)}/{len(OUTPUTS)} outputs"
              f"{f', missing {missing}' if missing else ''}"
              f"{f', wrongly planned {extra}' if extra else ''}")
        if not ok:
            failures.append("patterns")

        quality = [c for c in plan if c.source.name.startswith("sleap_quality_sub-")]
        ok = bool(quality)
        print(f"  {'ok  ' if ok else 'FAIL'} quality report is in the pattern set "
              f"(transfer_sleap_results.ps1:7 omitted it)")
        if not ok:
            failures.append("quality report")

        # 3. Destinations stay under the destination root.
        try:
            transfer.check_destinations(plan, dest_root)
            print("  ok   every destination is under the destination root")
        except SystemExit as exc:
            print(f"  FAIL {exc}")
            failures.append("destinations")

        # 4. An existing destination is refused without force, and overwritten with it.
        transfer.run_plan(plan, force=False)
        first = plan[0].dest
        first.write_bytes(b"OLD")
        replanned = transfer.plan_push(source_root, dest_root)
        result = transfer.run_plan(replanned, force=False)
        ok = result["copied"] == 0 and result["skipped"] == len(replanned)
        print(f"  {'ok  ' if ok else 'FAIL'} re-running without --force copies nothing "
              f"(copied {result['copied']}, skipped {result['skipped']})")
        if not ok:
            failures.append("overwrite guard")

        ok = first.read_bytes() == b"OLD"
        print(f"  {'ok  ' if ok else 'FAIL'} the existing destination is untouched")
        if not ok:
            failures.append("overwrite guard")

        result = transfer.run_plan(transfer.plan_push(source_root, dest_root), force=True)
        ok = result["copied"] == len(replanned) and first.read_bytes() != b"OLD"
        print(f"  {'ok  ' if ok else 'FAIL'} --force overwrites (copied {result['copied']})")
        if not ok:
            failures.append("force")

    print()
    if failures:
        print(f"TRANSFER REGRESSED: {', '.join(sorted(set(failures)))}")
        return 1
    print("TRANSFER OK -- dry-run is read-only, quality reports are carried, "
          "overwrites need --force.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
