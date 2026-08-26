#!/usr/bin/env python
"""Assert `annotate` draws what `annotate_videos_with_sleap_and_trials` drew.

Two clips of the same session, same window, same rotation, rendered by the old code and
the new one in the same environment -- the Phase 0.5 one-variable method, with the code
as the only variable.

The kept reference clips on the server cannot serve as the baseline: they were rendered
2026-07-07 under `sleap-analysis` (opencv 4.11.0, against 5.0.0 here) and, more
decisively, nothing recorded which time window produced them. So the baseline is
rendered fresh rather than assumed.

Compared, in the pre-registered order:

- frame count, dimensions and fps -- the container;
- decoded frames, pixel for pixel -- the strong form, available because both renders use
  one encoder;
- failing that, overlay *geometry*: the centroid marker's centre, and the pink odour and
  green reward text masks. That is what this module computes; encoding is opencv's.

Usage:  python -m hypnose_sleap.qc.check_annotate <old.mp4> <new.mp4>

Exit 0 = the port draws the same overlay, 1 = it moved.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np

# The three overlay colours, BGR as decoded, with a tolerance for encoder rounding.
MARKS = {
    "centroid": ((0, 0, 255), 90),
    "odor": ((90, 40, 255), 90),
    "reward": ((0, 220, 0), 90),
}

# How many evenly spaced frames to compare when the pixel-exact pass fails.
SAMPLE_FRAMES = 60


def _probe(path):
    import cv2

    cap = cv2.VideoCapture(str(path))
    meta = {
        "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "fps": round(float(cap.get(cv2.CAP_PROP_FPS)), 6),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    cap.release()
    return meta


def _mask_centroid(frame, color, tol):
    """The centre of mass of pixels near ``color``, and how many there were."""
    target = np.array(color, dtype=np.int16)
    diff = np.abs(frame.astype(np.int16) - target).sum(axis=2)
    mask = diff <= tol
    count = int(mask.sum())
    if count == 0:
        return None, 0
    ys, xs = np.nonzero(mask)
    return (float(xs.mean()), float(ys.mean())), count


def compare(old_path: Path, new_path: Path) -> int:
    import cv2

    failures = []

    old_meta, new_meta = _probe(old_path), _probe(new_path)
    for key in ("frames", "fps", "width", "height"):
        ok = old_meta[key] == new_meta[key]
        print(f"  {'ok  ' if ok else 'FAIL'} {key}: old={old_meta[key]} new={new_meta[key]}")
        if not ok:
            failures.append(key)
    if failures:
        print("\nANNOTATE MOVED: the container differs; not comparing frames.")
        return 1

    old_md5 = hashlib.md5(Path(old_path).read_bytes()).hexdigest()
    new_md5 = hashlib.md5(Path(new_path).read_bytes()).hexdigest()
    if old_md5 == new_md5:
        print(f"  ok   encoded bytes identical ({old_md5[:8]})")
        print("\nANNOTATE OK -- byte-identical, which is stronger than the gate asks.")
        return 0
    print(f"  note encoded bytes differ (old {old_md5[:8]}, new {new_md5[:8]}); "
          f"comparing decoded frames")

    cap_old, cap_new = cv2.VideoCapture(str(old_path)), cv2.VideoCapture(str(new_path))
    total = old_meta["frames"]
    step = max(1, total // SAMPLE_FRAMES)

    exact = 0
    compared = 0
    drift = {name: 0.0 for name in MARKS}
    mark_missing = {name: 0 for name in MARKS}

    for index in range(total):
        ok_old, frame_old = cap_old.read()
        ok_new, frame_new = cap_new.read()
        if not (ok_old and ok_new):
            break
        if np.array_equal(frame_old, frame_new):
            exact += 1
        if index % step:
            continue
        compared += 1
        for name, (color, tol) in MARKS.items():
            c_old, n_old = _mask_centroid(frame_old, color, tol)
            c_new, n_new = _mask_centroid(frame_new, color, tol)
            if (c_old is None) != (c_new is None):
                mark_missing[name] += 1
                continue
            if c_old is None:
                continue
            drift[name] = max(drift[name],
                              abs(c_old[0] - c_new[0]) + abs(c_old[1] - c_new[1]))
    cap_old.release()
    cap_new.release()

    ok = exact == total
    print(f"  {'ok  ' if ok else 'note'} decoded frames identical: {exact}/{total}")

    for name in MARKS:
        present_ok = mark_missing[name] == 0
        drift_ok = drift[name] <= 1.0
        status = "ok  " if (present_ok and drift_ok) else "FAIL"
        print(f"  {status} {name}: max centre drift {drift[name]:.2f} px, "
              f"{mark_missing[name]} presence mismatch(es) over {compared} sampled frames")
        if not (present_ok and drift_ok):
            failures.append(name)

    print()
    if failures:
        print(f"ANNOTATE MOVED: {', '.join(sorted(set(failures)))}")
        return 1
    print("ANNOTATE OK -- the overlay geometry is unchanged.")
    return 0


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 2:
        print(__doc__)
        return 2
    old_path, new_path = Path(argv[0]), Path(argv[1])
    for path in (old_path, new_path):
        if not path.is_file():
            print(f"missing clip: {path}")
            return 2
    print(f"Annotate check\n  old: {old_path}\n  new: {new_path}\n")
    return compare(old_path, new_path)


if __name__ == "__main__":
    sys.exit(main())
