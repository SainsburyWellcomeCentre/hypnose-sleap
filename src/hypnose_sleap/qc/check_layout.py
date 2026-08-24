#!/usr/bin/env python
"""Assert this repo and `hypnose-behavior` agree about where SLEAP output lives.

`io/layout.py` names ``movement_analysis`` itself rather than importing it, so that
`extract` runs without `hypnose_behavior`. This is what stops the two copies drifting:
if they disagree, `hypnose_behavior.io.layout.find_tracking_file` stops finding what
`hypnose-sleap` writes, and nothing else would notice.

Checks:

- ``MOVEMENT_SUBFOLDER`` and ``RESULTS_DIRNAME`` are identical in both packages;
- ``SUBJECT_PATTERN`` is identical, so both walk the same subject directories;
- `find_tracking_file` locates a combined parquet written at our `write_path`.

Exit 0 = agree, 1 = drifted.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from hypnose_sleap.io import layout as ours


def main() -> int:
    try:
        from hypnose_behavior.io import layout as theirs
    except ImportError as e:
        print(f"SKIP -- hypnose_behavior is not installed ({e})")
        return 0

    failures = []
    for name in ("MOVEMENT_SUBFOLDER", "RESULTS_DIRNAME", "SUBJECT_PATTERN"):
        a, b = getattr(ours, name, None), getattr(theirs, name, None)
        status = "ok  " if a == b else "DRIFT"
        print(f"  {status} {name}: hypnose_sleap={a!r} hypnose_behavior={b!r}")
        if a != b:
            failures.append(name)

    # The behavioural consequence, asserted by calling the real reader.
    stem = "sub-057_ses-20260717_combined_sleap_tracking_timestamps"
    with tempfile.TemporaryDirectory(prefix="hyp_layout_") as tmp:
        results = Path(tmp) / ours.RESULTS_DIRNAME
        written = ours.write_path(results, f"{stem}.parquet")
        written.write_bytes(b"")
        found = theirs.find_tracking_file(results, "*_combined_sleap_tracking_timestamps")
        ok = found is not None and Path(found) == written
        print(f"  {'ok  ' if ok else 'DRIFT'} find_tracking_file locates our write_path"
              f" -> {found}")
        if not ok:
            failures.append("find_tracking_file")

    print()
    if failures:
        print(f"LAYOUT DRIFT: {', '.join(failures)}")
        return 1
    print("LAYOUT OK -- hypnose-behavior can find what hypnose-sleap writes.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
