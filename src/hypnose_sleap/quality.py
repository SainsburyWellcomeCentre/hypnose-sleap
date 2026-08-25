"""Per-node tracking quality: a DataFrame for comparing models, and a ``.yml`` beside the parquet.

`session_node_stats` counts presence and confidence per skeleton node straight from a
session's ``.slp`` files. `write_report` records those alongside the node set `extract`
actually used, as ``movement_analysis/sleap_quality_sub-XXX_ses-YYYYMMDD.yml``.

    from hypnose_sleap.quality import sleap_node_quality_report
    sleap_node_quality_report(57, 20260717)
    sleap_node_quality_report(57, date="20260601-20260630")
"""
from __future__ import annotations

import gc
from pathlib import Path

import numpy as np
import pandas as pd
import sleap_io
import yaml

from hypnose_sleap import parameters
from hypnose_sleap.io import layout

QUALITY_FILENAME = "sleap_quality_sub-{subject}_ses-{date}.yml"

# What is recorded when the caller cannot say which model wrote the `.slp`. Nothing in
# a `.slp` records it, and the configured default would be a guess, not a fact.
UNKNOWN_MODEL = {
    "role": None, "path": None, "training_config": None, "training_config_md5": None,
}


def _plain(value):
    """A float YAML can render: NaN becomes None, everything else rounds to 4 dp."""
    value = float(value)
    return None if np.isnan(value) else round(value, 4)


def session_node_stats(slp_files, *, score_thresh, presence_frac, verbose=True):
    """Presence and confidence per skeleton node, over one session's ``.slp`` files.

    Returns ``(records, n_frames, n_occupied)``, one record per node. ``n_occupied``
    counts frames with any node present *before* the confidence gate, which is not the
    denominator `extract` selects on -- see `write_report`.
    """
    nodes = None
    present_cnt = None      # coords present per node
    gated_cnt = None        # coords present AND score >= thresh
    score_sum = None
    scores_by_node = None
    n_frames = 0
    n_occupied = 0

    for slp_path in slp_files:
        labels = None
        try:
            labels = sleap_io.load_slp(str(slp_path))
            if nodes is None:
                nodes = [n.name for n in labels.skeletons[0].nodes]
                k = len(nodes)
                present_cnt = np.zeros(k, dtype=np.int64)
                gated_cnt = np.zeros(k, dtype=np.int64)
                score_sum = np.zeros(k, dtype=float)
                scores_by_node = [[] for _ in range(k)]

            arr = np.asarray(labels.numpy(return_confidence=True), dtype=float)
            if arr.ndim == 4:  # (frames, instances, nodes, 3) -> rows of (nodes, 3)
                arr = arr.reshape(arr.shape[0] * arr.shape[1], arr.shape[2], arr.shape[3])
        except Exception as exc:
            if verbose:
                print(f"  ⚠️ Failed to read {slp_path.name}: {exc}; skipping")
            continue
        finally:
            del labels
            gc.collect()

        x, y, s = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
        present = ~np.isnan(x) & ~np.isnan(y)
        gated = present & (s >= score_thresh)
        n_frames += arr.shape[0]
        n_occupied += int(present.any(axis=1).sum())
        present_cnt += present.sum(axis=0)
        gated_cnt += gated.sum(axis=0)
        for j in range(len(nodes)):
            sj = s[present[:, j], j]
            if sj.size:
                score_sum[j] += float(np.nansum(sj))
                scores_by_node[j].append(sj)

    if not nodes or n_frames == 0:
        return [], n_frames, n_occupied

    records = []
    for j, name in enumerate(nodes):
        all_scores = np.concatenate(scores_by_node[j]) if scores_by_node[j] else np.array([])
        pcnt = int(present_cnt[j])
        gcnt = int(gated_cnt[j])
        if all_scores.size:
            p10, p25, p50, p75, p90 = (float(v) for v in np.percentile(all_scores, [10, 25, 50, 75, 90]))
            avg = float(all_scores.mean())
        else:
            p10 = p25 = p50 = p75 = p90 = avg = float("nan")
        pres_occ_gated = 100 * gcnt / n_occupied if n_occupied else 0.0
        records.append({
            "node": name,
            "n_frames": n_frames,
            "n_occupied": n_occupied,
            "pres_pct_all": 100 * pcnt / n_frames if n_frames else 0.0,
            "pres_pct_occ": 100 * pcnt / n_occupied if n_occupied else 0.0,
            "pres_pct_occ_gated": pres_occ_gated,
            "avg_score": avg,
            "score_p10": p10,
            "score_p25": p25,
            "score_p50": p50,
            "score_p75": p75,
            "score_p90": p90,
            "selected": pres_occ_gated >= 100 * presence_frac,
        })
    return records, n_frames, n_occupied


def _print_session(subject: str, date: str, records: list, n_frames: int, n_occupied: int) -> None:
    """The per-session table `sleap_node_quality_report` prints."""
    print(f"\n=== {subject} date-{date}  ({n_frames:,} frames, {n_occupied:,} occupied) ===")
    print(f"{'node':<14}{'pres%all':>9}{'pres%occ':>9}{'gated%occ':>10}"
          f"{'avg':>7}{'p10':>7}{'p50':>7}{'p90':>7}  sel")
    for r in records:
        print(f"{r['node']:<14}{r['pres_pct_all']:9.1f}{r['pres_pct_occ']:9.1f}"
              f"{r['pres_pct_occ_gated']:10.1f}{r['avg_score']:7.2f}"
              f"{r['score_p10']:7.2f}{r['score_p50']:7.2f}{r['score_p90']:7.2f}"
              f"   {'[x]' if r['selected'] else '[ ]'}")


def sleap_node_quality_report(subjid, date=None, *, derivatives=None,
                              score_thresh=None, presence_frac=None,
                              verbose=True) -> pd.DataFrame:
    """Per-node quality for one subject across one or more sessions.

    - ``date`` takes a single date, a list, a comma string or an inclusive
      ``YYYYMMDD-YYYYMMDD`` range; None means every session the subject has;
    - one row per (date, node), with the columns `session_node_stats` produces plus
      ``subject`` and ``date``;
    - ``selected`` here is the report's own verdict at these thresholds, which is not
      what `extract` gated on -- see `write_report`.
    """
    params = parameters.extract_parameters(
        score_thresh=score_thresh, presence_frac=presence_frac
    )
    sessions = layout.layout_for(derivatives).find_sessions(subjid, date=date)
    if not sessions:
        raise FileNotFoundError(
            f"No matching sessions for {layout.normalize_subjid(subjid)} "
            f"under {layout.layout_for(derivatives).root}"
        )

    records: list = []
    for session in sessions:
        slp_files = layout.find_outputs(layout.results_dir(session), "*.slp")
        if not slp_files:
            if verbose:
                print(f"{session.date}: no .slp files, skipping")
            continue
        rows, n_frames, n_occupied = session_node_stats(
            slp_files, score_thresh=params["score_thresh"],
            presence_frac=params["presence_frac"], verbose=verbose,
        )
        if not rows:
            continue
        subject = f"sub-{int(session.subjid):03d}"
        records.extend({"subject": int(session.subjid), "date": session.date, **r}
                       for r in rows)
        if verbose:
            _print_session(subject, session.date, rows, n_frames, n_occupied)

    return pd.DataFrame(records)


def write_report(results, *, subjid, date, slp_files, selection, params, model) -> Path:
    """Write one session's quality ``.yml`` beside its parquets. Returns the path.

    - ``centroid_nodes``, and each node's ``selected``, come from `extract`'s selection,
      so the file agrees with the ``centroid_nodes`` column in every parquet beside it;
    - ``pres_pct_occ_gated`` and ``selection_pct`` have different denominators: the
      first divides by frames with any node present, the second by the gated occupancy
      `extract` selects on. They can disagree on a borderline node;
    - ``model`` may be None, recorded as `UNKNOWN_MODEL`.
    """
    rows, n_frames, n_occupied = session_node_stats(
        slp_files, score_thresh=params["score_thresh"],
        presence_frac=params["presence_frac"], verbose=False,
    )
    centroid_nodes = list(selection["centroid_nodes"])
    selection_pct = selection["selection_pct"]

    report = {
        "subject": f"sub-{int(subjid):03d}",
        "date": str(date),
        "model": parameters.model_provenance(model) if model is not None else dict(UNKNOWN_MODEL),
        "parameters": {k: params[k] for k in ("score_thresh", "presence_frac", "gap_limit")},
        "frames": {
            "total": int(n_frames),
            "occupied": int(n_occupied),
            "occupied_gated": int(selection["occupied_frames"]),
        },
        "nodes": {
            r["node"]: {
                "pres_pct_occ": _plain(r["pres_pct_occ"]),
                "pres_pct_occ_gated": _plain(r["pres_pct_occ_gated"]),
                "score_p50": _plain(r["score_p50"]),
                "selection_pct": _plain(selection_pct.get(r["node"], 0.0)),
                "selected": r["node"] in centroid_nodes,
            }
            for r in rows
        },
        "centroid_nodes": centroid_nodes,
    }

    path = layout.write_path(
        results, QUALITY_FILENAME.format(subject=f"{int(subjid):03d}", date=date)
    )
    path.write_text(yaml.safe_dump(report, sort_keys=False), encoding="utf-8")
    return path


__all__ = [
    "QUALITY_FILENAME", "UNKNOWN_MODEL",
    "session_node_stats", "sleap_node_quality_report", "write_report",
]
