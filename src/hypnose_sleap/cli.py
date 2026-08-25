"""The ``hypnose-sleap`` command line: one entry point for every verb.

- ``fetch``    remote rawdata ``.avi``            -> local rawdata
- ``infer``    local ``.avi`` + a model role      -> ``.predictions.slp``
- ``extract``  ``.slp``                           -> per-video parquet + quality ``.yml``
- ``combine``  per-video parquet + harp streams   -> combined parquet
- ``run``      ``extract`` + ``combine`` in one process
- ``push``     local derivatives                  -> remote derivatives
- ``annotate`` ``.avi`` + combined parquet        -> annotated ``.mp4``

The whole loop runs on local disk: ``fetch`` -> ``infer`` -> ``run`` -> ``push``. Only
``fetch`` and ``push`` touch the server, and the writing verbs refuse a profile marked
``remote: true`` without ``--allow-remote``.

Nothing here deletes anything. Reclaiming local disk is manual.

Subject and date selectors are shared by every verb and parsed by
`hypnose_helpers.io.selectors`, so ``-s 57,58`` and ``-d 20260601-20260630`` mean the
same thing everywhere.

Handlers are imported inside their verb: `infer` needs sleap and torch, `combine` and
`annotate` need hypnose_behavior, and ``--help`` must work without any of them.

``extract``, ``combine`` and ``run`` share one session driver -- the same selection,
the same skip rules and the same success / failed / skipped tally that
``process_sleap_sessions`` printed. They differ only in which steps they run.
"""
from __future__ import annotations

import argparse
import re
import sys

VERBS = ("fetch", "infer", "extract", "combine", "run", "push", "annotate")

# What each session verb runs, in order.
STEPS = {
    "extract": ("extract",),
    "combine": ("combine",),
    "run": ("extract", "combine"),
}

# The output whose presence means the verb has already run here, named for the skip line.
DONE_LABEL = {
    "extract": "tracking output",
    "combine": "combined tracking file",
    "run": "combined tracking file",
}


def _add_session_selectors(parser: argparse.ArgumentParser) -> None:
    """The subject / date / session selectors every verb accepts."""
    parser.add_argument("-s", "--subject", action="append", default=None,
                        help="subject(s): 57, 057, sub-057, or 57,58,59. Repeatable.")
    parser.add_argument("-d", "--date", action="append", default=None,
                        help="date(s): YYYYMMDD, a comma list, or YYYYMMDD-YYYYMMDD. Repeatable.")
    parser.add_argument("--ses", default=None,
                        help="session number(s): 12, 12,14, or 12-20.")


def _add_dry_run(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dry-run", action="store_true",
                        help="list what would be done, then exit")


def _add_allow_remote(parser: argparse.ArgumentParser) -> None:
    """For verbs that write bulk output, which belongs on local disk until `push`."""
    parser.add_argument("--allow-remote", action="store_true",
                        help="permit writing to a profile marked `remote: true`")


def _add_extract_options(parser: argparse.ArgumentParser) -> None:
    """The centroid parameters and the model role, for the verbs that run `extract`.

    ``--model`` is optional but never guessed: omitting it records the quality report's
    provenance as unknown rather than claiming the configured default wrote the `.slp`.
    """
    parser.add_argument("-m", "--model", default=None,
                        help="model role (naive | eeg_surgery | eeg_headstage) or an "
                             "explicit path, recorded in the quality report")
    parser.add_argument("--score-thresh", type=float, default=None)
    parser.add_argument("--presence-frac", type=float, default=None)
    parser.add_argument("--gap-limit", type=int, default=None)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hypnose-sleap",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--profile", default=None,
                        help="data-location profile to use instead of the active one")
    sub = parser.add_subparsers(dest="verb", metavar="VERB")

    p = sub.add_parser("fetch", help="copy remote rawdata .avi to the local tree")
    _add_session_selectors(p)
    _add_dry_run(p)
    p.add_argument("--from", dest="source", default=None, help="override the remote end")
    p.add_argument("--to", dest="dest", default=None, help="override the local end")

    p = sub.add_parser("infer", help="run sleap-track over a session's videos")
    _add_session_selectors(p)
    _add_dry_run(p)
    _add_allow_remote(p)
    p.add_argument("-m", "--model", default=None,
                   help="model role (naive | eeg_surgery | eeg_headstage) or an explicit path")
    p.add_argument("-bz", "--batch-size", type=int, default=None,
                   help="sleap-track batch size; defaults to configs/parameters.yml")

    p = sub.add_parser("extract", help=".slp -> per-video parquet + quality report")
    _add_session_selectors(p)
    _add_dry_run(p)
    _add_allow_remote(p)
    _add_extract_options(p)
    p.add_argument("--recompute", action="store_true",
                   help="re-extract sessions that already have output")

    p = sub.add_parser("combine", help="per-video parquet + harp streams -> combined parquet")
    _add_session_selectors(p)
    _add_dry_run(p)
    _add_allow_remote(p)
    p.add_argument("--recompute", action="store_true")

    p = sub.add_parser("run", help="extract then combine, in one process")
    _add_session_selectors(p)
    _add_dry_run(p)
    _add_allow_remote(p)
    _add_extract_options(p)
    p.add_argument("--recompute", action="store_true")

    p = sub.add_parser("push", help="copy local derivatives to the server")
    _add_session_selectors(p)
    _add_dry_run(p)
    p.add_argument("--from", dest="source", default=None, help="override the local end")
    p.add_argument("--to", dest="dest", default=None, help="override the remote end")
    p.add_argument("--force", action="store_true",
                   help="overwrite destination files that already exist")

    p = sub.add_parser("annotate", help="render an annotated overlay video")
    _add_session_selectors(p)
    p.add_argument("--rotate", type=int, default=0, choices=(0, 90, 180, 270))
    p.add_argument("--window", action="append", default=None,
                   help="clip to START-END relative to the video start, e.g. 0:00:00-0:10:00")
    p.add_argument("--video", action="append", default=None,
                   help="1-based video number(s) to render; default all")

    return parser


# --- session selection -----------------------------------------------------

def _join(values) -> str | None:
    """One selector string from a repeatable flag, or None if it was never given."""
    if not values:
        return None
    return values[0] if len(values) == 1 else ",".join(str(v) for v in values)


def _requested_dates(value):
    """The dates a selector names explicitly, or None when it is a range or absent.

    A range describes dates that may legitimately not exist; a *named* date with no
    session directory is worth reporting, which is what `process_sleap_sessions` did.
    """
    if not value or re.search(r"(?<=\d)-(?=\d)", value):
        return None
    from hypnose_helpers.io.selectors import parse_dates
    return parse_dates(value)


def _subjects(sessions_layout, args) -> list:
    """Subject numbers the selectors name, or every subject in the tree."""
    if args.subject:
        from hypnose_helpers.io.selectors import parse_subjects
        return parse_subjects(args.subject)
    return [subjid for subjid, _ in sessions_layout.iter_subjects()]


def _roots(profile) -> dict:
    """The rawdata and derivatives roots a verb works under.

    None means "the configured root", which is what every handler already defaults to.
    ``--profile`` resolves a named profile instead, without making it the active one.
    """
    if profile is None:
        return {"rawdata": None, "derivatives": None}
    from hypnose_sleap.io import paths
    resolved = paths.resolve_profile(profile)
    return {"rawdata": resolved["rawdata"], "derivatives": resolved["derivatives"]}


def _already_done(results, steps):
    """The output whose presence means this session has been processed, or None.

    The last step's output: with `combine` in the steps that is the combined table,
    otherwise the per-video tables `extract` writes.
    """
    from hypnose_sleap.io import layout
    if "combine" in steps:
        return layout.find_combined_table(results)
    tables = layout.find_tracking_tables(results)
    return tables[0] if tables else None


# --- the session driver ----------------------------------------------------

def _process_one(session, results, slp_count, *, steps, args, roots) -> dict:
    """Run one session's steps, returning its status, reason and message block."""
    from hypnose_sleap.io import layout

    subj, date_str = int(session.subjid), session.date
    messages = [f"\nSubject {subj:02d} Date {date_str} - Processing SLEAP Output:"]

    try:
        if "extract" in steps:
            from hypnose_sleap.extract import extract_session
            extracted = extract_session(
                subj, date_str, model=args.model, derivatives=roots["derivatives"],
                skip_empty=True, score_thresh=args.score_thresh,
                presence_frac=args.presence_frac, gap_limit=args.gap_limit,
            )
            messages.append(f"Centroid Processing: found {slp_count} video(s) to process.")
            messages.append(f"        Successfully processed {len(extracted['outputs'])} videos.")

        if "combine" in steps:
            from hypnose_sleap.timestamps import combine_session
            timestamp_found = len(layout.find_tracking_tables(results))
            combined = combine_session(
                subj, date_str,
                derivatives=roots["derivatives"], rawdata=roots["rawdata"],
            )
            matched_videos = (int(combined["video_file"].nunique())
                              if combined is not None and not combined.empty else 0)
            saved_flag = layout.find_combined_table(results) is not None
            messages.append(f"Timestamp Processing: found {timestamp_found} video(s) to process.")
            messages.append(f"         Successfully matched {matched_videos} video(s) to sleap "
                            f"files, combined sleap file saved = {saved_flag}")

        return {"status": "success", "reason": "", "messages": messages}
    except FileNotFoundError as exc:
        return {"status": "skipped", "reason": str(exc),
                "messages": messages + [f"  Skipped: {exc}"]}
    except Exception as exc:
        return {"status": "failed", "reason": str(exc),
                "messages": messages + [f"  Failed: {exc}"]}


def _run_sessions(args, verb: str) -> int:
    """`extract`, `combine` and `run`: select, skip, process, tally.

    Returns 1 if any session failed. A skipped session is not a failure -- most runs
    sweep a whole subject and expect to step over sessions already done.
    """
    from hypnose_sleap.io import layout, paths

    steps = STEPS[verb]
    paths.require_local(verb, allow_remote=args.allow_remote, profile=args.profile)
    roots = _roots(args.profile)
    sessions_layout = layout.layout_for(roots["derivatives"])

    if "extract" in steps and args.model is None:
        print("No --model given: the quality report records provenance as unknown.\n"
              "  Pass --model naive | eeg_surgery | eeg_headstage, or a path, to record it.\n")

    summary: dict = {}
    tasks: list = []
    requested = _requested_dates(_join(args.date))

    for subjid in _subjects(sessions_layout, args):
        summary[subjid] = {"success": [], "failed": [], "skipped": []}

        if sessions_layout.subject_dir(subjid, missing_ok=True) is None:
            summary[subjid]["skipped"].append(("ALL", "Subject directory not found"))
            print(f"Subject {subjid:02d}: no subject directory found under {sessions_layout.root}")
            continue

        sessions = sessions_layout.find_sessions(subjid, ses=args.ses, date=_join(args.date))
        found_dates = {s.date for s in sessions}
        for date_str in requested or []:
            if date_str not in found_dates:
                summary[subjid]["skipped"].append((date_str, "Session directory not found"))

        if not sessions:
            print(f"Subject {subjid:02d}: no matching dates to process")
            continue

        for session in sessions:
            date_str = session.date
            results = layout.results_dir(session)

            if not results.exists():
                summary[subjid]["skipped"].append((date_str, "Results directory not found"))
                print(f"Subject {subjid:02d} Date {date_str}: results directory missing, skipping")
                continue

            if _already_done(results, steps) is not None and not args.recompute:
                reason = f"Existing {DONE_LABEL[verb]}, skipping directory"
                summary[subjid]["skipped"].append((date_str, reason))
                print(f"Subject {subjid:02d} Date {date_str} - {reason} (recompute=False)")
                continue

            slp_files = layout.find_outputs(results, "*.slp") if "extract" in steps else []
            if "extract" in steps and not slp_files:
                summary[subjid]["skipped"].append((date_str, "No .slp files found"))
                print(f"Subject {subjid:02d} Date {date_str}: no .slp files, skipping")
                continue

            tasks.append((session, results, len(slp_files)))

    if args.dry_run:
        print(f"\n{verb} --dry-run: {len(tasks)} session(s) would run {' + '.join(steps)}:")
        for session, results, slp_count in tasks:
            print(f"  sub-{int(session.subjid):03d} date-{session.date}   {slp_count} .slp"
                  f"   -> {layout.movement_dir(results)}")
        return 0

    for task in tasks:
        session = task[0]
        result = _process_one(*task, steps=steps, args=args, roots=roots)
        summary[int(session.subjid)][result["status"]].append((session.date, result["reason"]))
        for line in result["messages"]:
            print(line)

    print("\nSummary:")
    failed = 0
    for subj, stats in summary.items():
        total = len(stats["success"]) + len(stats["failed"]) + len(stats["skipped"])
        failed += len(stats["failed"])
        print(f"  {subj}:")
        print(f"    Successful: {len(stats['success'])}/{total}")
        print(f"    Failed: {len(stats['failed'])}/{total}")
        print(f"    Skipped: {len(stats['skipped'])}/{total}")

    return 1 if failed else 0


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.verb is None:
        parser.print_help()
        return 1

    if args.verb in STEPS:
        return _run_sessions(args, args.verb)

    raise SystemExit(
        f"`{args.verb}` is not implemented yet -- hypnose-sleap is mid-restructure.\n"
        f"See docs/restructure-plan.md for which phase lands it."
    )


if __name__ == "__main__":
    sys.exit(main())
