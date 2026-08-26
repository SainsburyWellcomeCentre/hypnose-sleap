"""Copying data between the local machine and the server.

- ``fetch``: remote rawdata ``behav/`` tree -> local rawdata, same tree.
- ``push``: local derivatives tables and quality reports -> remote derivatives.
- Endpoints come from ``transfer: {remote, local}`` in ``configs/data_locations.yml``;
  ``--from`` / ``--to`` override either end.
- Planning and copying are separate: `plan_fetch` and `plan_push` only read, so
  ``--dry-run`` prints a plan without creating a single directory. `run_plan` is the
  only thing here that writes.
- `run_plan` refuses to overwrite an existing destination unless ``force``.

`fetch` takes the whole ``behav/<exp>/`` tree rather than the ``.avi`` alone: `combine`
reads ``VideoData/*.csv`` and the harp streams under ``Behavior/``, and their absence
degrades the ``time`` column instead of raising (`DECISIONS.md` §12). The extra streams
are 1.3 % of a session.
"""
from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from hypnose_sleap.io import layout, paths

# What `push` moves. The first four are `transfer_sleap_results.ps1:7`; each keeps its
# leading ``*`` because a combined table carries a ``sub-XXX_ses-YYYYMMDD_`` prefix.
# The fifth is new: Phase 3 added the quality report and the script predates it, so
# those reports have never reached the server.
PUSH_PATTERNS = (
    "*sleap_tracking_video*.parquet",
    "*sleap_tracking_video*.csv",
    "*combined_sleap_tracking_timestamps*.parquet",
    "*combined_sleap_tracking_timestamps*.csv",
    "sleap_quality_sub-*.yml",
)

# What `fetch` takes from a raw session: everything under `behav/`, both video and the
# harp streams `combine` needs.
FETCH_SUBDIR = "behav"


@dataclass(frozen=True)
class Copy:
    """One planned file copy, with what the destination already holds.

    `exists` and `same_size` are read while planning so ``--dry-run`` can say which
    copies would be refused without `--force`, rather than finding out mid-transfer.
    """

    source: Path
    dest: Path
    size: int
    exists: bool
    same_size: bool

    @property
    def label(self) -> str:
        if not self.exists:
            return "new"
        return "exists (same size)" if self.same_size else "exists (differs)"


def _plan_copy(source: Path, dest: Path) -> Copy:
    """One `Copy`, stat-ing both ends. Never creates anything."""
    size = source.stat().st_size
    exists = dest.exists()
    return Copy(
        source=source, dest=dest, size=size, exists=exists,
        same_size=bool(exists and dest.stat().st_size == size),
    )


def endpoints(source: Optional[str], dest: Optional[str], *, direction: str) -> dict:
    """The two profiles a transfer runs between, as ``{'source': ..., 'dest': ...}``.

    ``direction`` is ``"fetch"`` (remote -> local) or ``"push"`` (local -> remote); the
    defaults come from the ``transfer:`` block and either end may be overridden by name.
    Raises when an end is neither configured nor given, rather than guessing one.
    """
    configured = paths.transfer_endpoints()
    if direction == "fetch":
        wanted = {"source": source or configured["remote"], "dest": dest or configured["local"]}
    else:
        wanted = {"source": source or configured["local"], "dest": dest or configured["remote"]}

    resolved = {}
    for end, name in wanted.items():
        if not name:
            other = "remote" if (direction == "fetch") == (end == "source") else "local"
            raise SystemExit(
                f"`{direction}` has no {end} endpoint: `transfer.{other}` is unset in "
                f"{paths.PROFILES_FILENAME} and no override was given.\n"
                f"  Pass --{'from' if end == 'source' else 'to'} <profile>."
            )
        resolved[end] = paths.resolve_profile(name)
    return resolved


def _sessions(root: Path, args_subject, ses, date) -> list:
    """Sessions under one root matching the shared selectors, over every subject.

    A missing subject directory is skipped rather than raised: a transfer names subjects
    that may exist at only one end, which is the normal case for `fetch`.
    """
    sessions_layout = layout.layout_for(root, name="rawdata")
    if args_subject:
        from hypnose_helpers.io.selectors import parse_subjects
        subjids = parse_subjects(args_subject)
    else:
        subjids = [subjid for subjid, _ in sessions_layout.iter_subjects()]

    found = []
    for subjid in subjids:
        found.extend(
            sessions_layout.find_sessions(subjid, ses=ses, date=date, missing_ok=True)
        )
    return found


def _mirror(source_root: Path, dest_root: Path, source_file: Path) -> Path:
    """The destination for one source file, at the same path relative to its root."""
    return dest_root / source_file.relative_to(source_root)


def plan_fetch(source_root: Path, dest_root: Path, *,
               subject=None, ses=None, date=None) -> list:
    """Every file `fetch` would copy, remote rawdata -> local rawdata.

    The whole ``behav/`` tree of each matching session, not the ``.avi`` alone. Reads
    only: nothing is created until `run_plan`.
    """
    source_root, dest_root = Path(source_root), Path(dest_root)
    plan = []
    for session in _sessions(source_root, subject, ses, date):
        behav = Path(session.path) / FETCH_SUBDIR
        if not behav.is_dir():
            continue
        for item in sorted(behav.rglob("*")):
            if not item.is_file() or item.name.startswith("._"):
                continue
            plan.append(_plan_copy(item, _mirror(source_root, dest_root, item)))
    return plan


def plan_push(source_root: Path, dest_root: Path, *,
              subject=None, ses=None, date=None, patterns=PUSH_PATTERNS) -> list:
    """Every file `push` would copy, local derivatives -> remote derivatives.

    Walks sessions through the layout rather than the whole tree, so a stray directory
    beside the subjects contributes nothing. `find_outputs` rglobs, so a table written
    flat and one written into ``movement_analysis/`` are both picked up, each landing at
    the same relative path on the far side.
    """
    source_root, dest_root = Path(source_root), Path(dest_root)
    sessions_layout = layout.layout_for(source_root, name="derivatives")
    if subject:
        from hypnose_helpers.io.selectors import parse_subjects
        subjids = parse_subjects(subject)
    else:
        subjids = [subjid for subjid, _ in sessions_layout.iter_subjects()]

    seen, plan = set(), []
    for subjid in subjids:
        for session in sessions_layout.find_sessions(subjid, ses=ses, date=date, missing_ok=True):
            results = layout.results_dir(session)
            for pattern in patterns:
                for item in layout.find_outputs(results, pattern):
                    if item in seen:
                        continue
                    seen.add(item)
                    plan.append(_plan_copy(item, _mirror(source_root, dest_root, item)))
    return sorted(plan, key=lambda c: c.source)


def check_destinations(plan, dest_root: Path) -> None:
    """Assert every destination lies under ``dest_root``.

    A transfer is the one verb that writes outside the tree it read, so where it may
    write is checked against the resolved root rather than trusted to path arithmetic.
    """
    dest_root = Path(dest_root).resolve()
    outside = [c for c in plan if dest_root not in Path(c.dest).resolve().parents]
    if outside:
        raise SystemExit(
            f"{len(outside)} destination(s) fall outside {dest_root}:\n  "
            + "\n  ".join(str(c.dest) for c in outside[:5])
        )


def _human(num_bytes: int) -> str:
    value = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} GB"


def print_plan(plan, *, verb: str, source_root: Path, dest_root: Path, force: bool) -> None:
    """The plan, one line per file, and what `run_plan` would refuse."""
    print(f"\n{verb} --dry-run: {len(plan)} file(s), {_human(sum(c.size for c in plan))}")
    print(f"  from {source_root}\n  to   {dest_root}")
    for copy in plan:
        print(f"  {copy.label:20s} {copy.source} -> {copy.dest}")
    blocked = [c for c in plan if c.exists] if not force else []
    if blocked:
        print(f"\n  {len(blocked)} destination(s) already exist and would be skipped. "
              f"Pass --force to overwrite.")


def run_plan(plan, *, force: bool = False, verb: str = "copy") -> dict:
    """Execute a plan. The only writer here.

    Copies with `shutil.copy2`, creating each destination's parent as it goes -- so a
    plan that copies nothing creates nothing. An existing destination is skipped unless
    ``force``: `transfer_sleap_results.ps1:140` passed ``-Force`` unconditionally, which
    made every run an overwrite.
    """
    copied = skipped = 0
    for copy in plan:
        if copy.exists and not force:
            print(f"  skipped (exists): {copy.dest}")
            skipped += 1
            continue
        copy.dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(copy.source, copy.dest)
        print(f"  copied: {copy.source} -> {copy.dest}")
        copied += 1
    print(f"\nDone. Copied: {copied}. Skipped (exists): {skipped}.")
    return {"copied": copied, "skipped": skipped}


def transfer(direction: str, args) -> int:
    """`fetch` and `push`: resolve both ends, plan, then either print or copy."""
    ends = endpoints(args.source, args.dest, direction=direction)
    key = "rawdata" if direction == "fetch" else "derivatives"
    source_root, dest_root = ends["source"][key], ends["dest"][key]

    if not Path(source_root).exists():
        raise SystemExit(f"source root not found: {source_root}")

    planner = plan_fetch if direction == "fetch" else plan_push
    plan = planner(source_root, dest_root,
                   subject=args.subject, ses=args.ses, date=_join_dates(args.date))
    check_destinations(plan, dest_root)

    if args.dry_run:
        print_plan(plan, verb=direction, source_root=source_root, dest_root=dest_root,
                   force=getattr(args, "force", False))
        return 0

    if not plan:
        print(f"{direction}: nothing to copy.")
        return 0

    print(f"\n{direction}: {len(plan)} file(s), {_human(sum(c.size for c in plan))}")
    print(f"  from {source_root}\n  to   {dest_root}")
    run_plan(plan, force=getattr(args, "force", False), verb=direction)
    return 0


def _join_dates(values):
    """The date selector as one string, matching how the session verbs join it."""
    if not values:
        return None
    return values[0] if len(values) == 1 else ",".join(str(v) for v in values)


__all__ = [
    "PUSH_PATTERNS", "FETCH_SUBDIR", "Copy",
    "endpoints", "plan_fetch", "plan_push", "check_destinations",
    "print_plan", "run_plan", "transfer",
]
