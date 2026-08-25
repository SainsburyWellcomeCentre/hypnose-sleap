#!/usr/bin/env python
"""AST byte-identity check: proves a split was a move, not an edit.

Every definition in scope must reappear in the new files with a byte-identical source
segment, compared after ``textwrap.dedent`` so a helper lifted out of its enclosing
function still counts as a move. MISSING, DUPLICATED and CHANGED fail; ADDED is reported.

  python src/hypnose_sleap/qc/ast_move_check.py            # the Phase 3 set vs HEAD
  python .../ast_move_check.py --only get_video_frame_times # a later phase's set
  python .../ast_move_check.py --all                        # once sleap_utils.py is gone
  python .../ast_move_check.py --base 5126d88 --show-diff
"""
from __future__ import annotations

import ast
import sys
import argparse
import subprocess
import difflib
import textwrap
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]

# The pre-restructure module, and the package it is being carved into.
DEFAULT_OLD = "sleap_utils.py"
DEFAULT_NEW_DIR = "src/hypnose_sleap"

# Phase 3: the centroid pipeline. Three of these are nested in
# `sleap_labels_and_centroid` and are lifted to module level by the move.
DEFAULT_ONLY = (
    "_compute_session_centroid",
    "node_names_for_instance",
    "extract_points_and_scores",
    "infer_video_file_from_slp",
    "to_number",
)

GREEN = "\033[32m"
RED = "\033[31m"
YELLOW = "\033[33m"
DIM = "\033[2m"
RESET = "\033[0m"


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO))
    except ValueError:
        return str(path)


def _read_at_ref(ref: str, relpath: str) -> str:
    """Read a file's contents at a git ref. Raises SystemExit if it is not there."""
    proc = subprocess.run(
        ["git", "-C", str(REPO), "show", f"{ref}:{relpath}"],
        capture_output=True, text=True, encoding="utf-8",
    )
    if proc.returncode != 0:
        raise SystemExit(f"cannot read {relpath} at {ref}: {proc.stderr.strip()}")
    return proc.stdout


def _span(node: ast.AST, lines: list[str]) -> tuple[int, int]:
    """Full-line span of a definition, decorators included, 1-based inclusive."""
    start = node.lineno
    for dec in getattr(node, "decorator_list", []):
        start = min(start, dec.lineno)
    return start, node.end_lineno


def _walk_defs(body, nested: bool = False):
    """Yield ``(node, nested)`` for definitions in a body, descending into functions.

    - a def inside a function is a candidate for de-nesting, so it is yielded too;
    - a class is yielded but not descended into: its methods are part of its segment.
    """
    for node in body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            yield node, nested
            yield from _walk_defs(node.body, True)
        elif isinstance(node, ast.ClassDef):
            yield node, nested


def collect_defs(source: str, origin: str) -> dict[str, dict]:
    """Map every def/class in ``source`` to its dedented source segment.

    Dedenting is what lets a helper lifted out of its enclosing function compare
    equal; whitespace-only lines normalise to empty, which is the only leniency.

    Two top-level definitions of one name is an error. Two *nested* ones is not --
    ``sleap_utils.py`` defines ``resolve_deriv_root`` inside two different functions --
    so the name is marked ambiguous and refused only if a caller puts it in scope.
    """
    tree = ast.parse(source)
    lines = source.splitlines(keepends=True)
    out: dict[str, dict] = {}
    for node, nested in _walk_defs(tree.body):
        start, end = _span(node, lines)
        if node.name in out:
            first = out[node.name]
            if not nested and not first["nested"]:
                raise SystemExit(
                    f"{origin}: two top-level definitions named {node.name!r} "
                    f"(lines {first['line']} and {start})"
                )
            first["ambiguous"] = f"{origin}:{first['line']} and :{start}"
            continue
        out[node.name] = {
            "segment": textwrap.dedent("".join(lines[start - 1:end])),
            "origin": origin,
            "line": start,
            "kind": type(node).__name__,
            "nested": nested,
            "ambiguous": None,
        }
    return out


def collect_assignments(source: str) -> dict[str, str]:
    """Module-level constant assignments, name -> exact source segment.

    Split out from ``collect_defs`` because these are reported but never fail:
    an import rearrangement legitimately moves them and a split may need a
    constant in two places.
    """
    tree = ast.parse(source)
    lines = source.splitlines(keepends=True)
    out: dict[str, str] = {}
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        segment = "".join(lines[node.lineno - 1:node.end_lineno])
        for t in targets:
            if isinstance(t, ast.Name):
                out[t.id] = segment
    return out


def _diff(name: str, before: str, after: str, old_origin: str, new_origin: str) -> str:
    return "".join(difflib.unified_diff(
        before.splitlines(keepends=True),
        after.splitlines(keepends=True),
        fromfile=f"{old_origin}::{name}",
        tofile=f"{new_origin}::{name}",
        n=2,
    ))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="HEAD", help="git ref holding the pre-split file (default HEAD)")
    ap.add_argument("--old", action="append", default=None,
                    help="repo-relative path of a pre-split file, read at --base. Repeatable.")
    ap.add_argument("--new", action="append", default=None,
                    help="post-split file in the working tree. Repeatable.")
    ap.add_argument("--new-dir", default=None,
                    help="directory whose *.py files are the post-split files")
    ap.add_argument("--only", action="append", default=None,
                    help="definition name to assert. Repeatable. Default: the Phase 3 set.")
    ap.add_argument("--all", action="store_true",
                    help="assert every definition in --old, not just a phase's set")
    ap.add_argument("--show-diff", action="store_true", help="print a unified diff for each CHANGED body")
    ap.add_argument("--quiet", action="store_true", help="only print the verdict and failures")
    args = ap.parse_args(argv)

    olds = args.old or [DEFAULT_OLD]
    if args.new is None and args.new_dir is None:
        new_dir = REPO / DEFAULT_NEW_DIR
        news = sorted(p for p in new_dir.glob("*.py") if p.name != "__init__.py")
    else:
        news = [REPO / p if not Path(p).is_absolute() else Path(p) for p in (args.new or [])]
        if args.new_dir:
            d = Path(args.new_dir)
            d = d if d.is_absolute() else REPO / d
            news += sorted(p for p in d.glob("*.py") if p.name != "__init__.py")
    news = sorted(set(p.resolve() for p in news))

    # --- before ---------------------------------------------------------
    before: dict[str, dict] = {}
    before_consts: dict[str, str] = {}
    for relpath in olds:
        src = _read_at_ref(args.base, relpath)
        for name, info in collect_defs(src, f"{args.base}:{relpath}").items():
            if name in before:
                raise SystemExit(f"{name!r} defined in two --old files")
            before[name] = info
        before_consts.update(collect_assignments(src))

    # --- scope ----------------------------------------------------------
    scope = set(before) if args.all else set(args.only or DEFAULT_ONLY)
    unknown = sorted(scope - set(before))
    if unknown:
        raise SystemExit(
            f"--only names not defined in {', '.join(olds)} at {args.base}: "
            f"{', '.join(unknown)}"
        )
    ambiguous = sorted(n for n in scope if before[n]["ambiguous"])
    if ambiguous:
        raise SystemExit(
            "in scope but defined more than once, so a move cannot be proved:\n  "
            + "\n  ".join(f"{n}   {before[n]['ambiguous']}" for n in ambiguous)
        )

    # --- after ----------------------------------------------------------
    after: dict[str, list[dict]] = {}
    after_consts: dict[str, list[str]] = {}
    for path in news:
        if not path.exists():
            raise SystemExit(f"--new file does not exist: {path}")
        src = path.read_text(encoding="utf-8")
        for name, info in collect_defs(src, _rel(path)).items():
            after.setdefault(name, []).append(info)
        for name in collect_assignments(src):
            after_consts.setdefault(name, []).append(_rel(path))

    print(f"AST move check  base={args.base}")
    print(f"  before : {', '.join(olds)}  ({len(before)} definitions)")
    for p in news:
        print(f"  after  : {_rel(p)}")
    print(f"  scope  : {'every definition' if args.all else ', '.join(sorted(scope))}")
    print()

    missing, duplicated, changed, identical = [], [], [], []
    for name, info in sorted(before.items(), key=lambda kv: kv[1]["line"]):
        if name not in scope:
            continue
        homes = after.get(name, [])
        if not homes:
            missing.append(name)
            continue
        if len(homes) > 1:
            duplicated.append((name, [h["origin"] for h in homes]))
            continue
        home = homes[0]
        if home["segment"] == info["segment"]:
            identical.append((name, info, home))
        else:
            changed.append((name, info, home))

    # A destination file that already existed at --base contributes its own
    # definitions to `after`; those are not additions. Only names that exist in
    # neither the old file nor any destination-as-of-base are genuinely new.
    preexisting: set[str] = set()
    for path in news:
        relpath = _rel(path)
        proc = subprocess.run(
            ["git", "-C", str(REPO), "show", f"{args.base}:{relpath}"],
            capture_output=True, text=True, encoding="utf-8",
        )
        if proc.returncode == 0:
            preexisting |= set(collect_defs(proc.stdout, relpath))
    added = sorted(set(after) - set(before) - preexisting)

    if not args.quiet:
        width = max((len(n) for n in scope), default=0)
        for name, info, home in identical:
            note = "  (de-nested)" if info["nested"] and not home["nested"] else ""
            print(f"  {GREEN}identical{RESET}  {name:<{width}}  ->  {home['origin']}{note}")
        print()

    ok = True
    if missing:
        ok = False
        print(f"{RED}MISSING ({len(missing)}) -- in scope, in none of the new files:{RESET}")
        for name in missing:
            print(f"    {name}   (was {before[name]['origin']}:{before[name]['line']})")
        print()
    if duplicated:
        ok = False
        print(f"{RED}DUPLICATED ({len(duplicated)}) -- now defined in more than one new file:{RESET}")
        for name, origins in duplicated:
            print(f"    {name}   {', '.join(origins)}")
        print()
    if changed:
        ok = False
        print(f"{RED}CHANGED ({len(changed)}) -- source segment is not byte-identical:{RESET}")
        for name, info, home in changed:
            b, a = info["segment"], home["segment"]
            note = ""
            if b.split() == a.split():
                note = "  (whitespace only)"
            print(f"    {name}   {info['origin']}:{info['line']} -> {home['origin']}:{home['line']}"
                  f"   {len(b)} -> {len(a)} bytes{note}")
            if args.show_diff:
                print(_diff(name, b, a, info["origin"], home["origin"]))
        print()

    if added:
        print(f"{YELLOW}ADDED ({len(added)}) -- not present before the split (review, does not fail):{RESET}")
        for name in added:
            print(f"    {name}   {', '.join(h['origin'] for h in after[name])}")
        print()

    const_missing = sorted(set(before_consts) - set(after_consts))
    if const_missing and not args.quiet:
        print(f"{DIM}module constants not carried over (review, does not fail): "
              f"{', '.join(const_missing)}{RESET}")
        print()

    total = len(scope)
    print(f"  {len(identical)}/{total} definitions in scope byte-identical, "
          f"{len(changed)} changed, {len(missing)} missing, {len(duplicated)} duplicated")
    if ok:
        print(f"{GREEN}PASS{RESET} -- provably a move: every body in scope is byte-identical.")
    else:
        print(f"{RED}FAIL{RESET} -- this is not a pure move.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
