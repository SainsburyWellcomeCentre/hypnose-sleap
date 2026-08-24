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
"""
from __future__ import annotations

import argparse
import sys

VERBS = ("fetch", "infer", "extract", "combine", "run", "push", "annotate")


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
    p.add_argument("--score-thresh", type=float, default=None)
    p.add_argument("--presence-frac", type=float, default=None)
    p.add_argument("--gap-limit", type=int, default=None)
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


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.verb is None:
        parser.print_help()
        return 1

    raise SystemExit(
        f"`{args.verb}` is not implemented yet -- hypnose-sleap is mid-restructure.\n"
        f"See docs/restructure-plan.md for which phase lands it."
    )


if __name__ == "__main__":
    sys.exit(main())
