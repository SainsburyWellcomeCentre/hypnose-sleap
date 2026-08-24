# Regression gate

Proves a restructuring step changed no number.

```bash
# check the current code against the baselines
python src/hypnose_sleap/qc/regression.py

# one session only
python src/hypnose_sleap/qc/regression.py 058:20260717

# rewrite the baselines from the saved server output
python src/hypnose_sleap/qc/regression.py --generate
```

Exit 0 = GREEN, 1 = RED.

## Levels

| level | input | expectation |
| --- | --- | --- |
| L1 `per_video` | `.slp` -> per-video parquet | byte-identical |
| L2 `combined` | + timestamps from `load_all_streams` | measured, reported as a delta |
| L3 `quality` | the quality report `.yml` | consistent with the parquets beside it |

- L1 is the real gate: pure `sleap_io` + `pandas`, no behavioural data.
- L2's input is not frozen. The saved `time` column came from a pre-2.0.0
  `load_all_streams`, so a delta is recorded in `docs/DECISIONS.md`, not chased to zero.
- L3 has no baseline; nothing wrote a report before. It asserts the report's
  `centroid_nodes` equals the `centroid_nodes` column in every parquet beside it.

## Sources

- `disk` — re-hash the saved baseline files. Checks that the fingerprinting is
  deterministic; **not** a gate on the pipeline. Used by `--generate`, and by a compare
  while `hypnose_sleap.extract` does not exist.
- `rederive` — run the new pipeline and fingerprint its output. The actual gate; turns
  on automatically once `extract.py` lands.

Which one ran is printed at the top of every compare.

## Fixtures

`fixtures/sub-XXX_date-YYYYMMDD.json` holds, per session:

- `inputs` — md5 of each `.slp`. Not a level; it makes an L1 mismatch attributable to
  the code rather than to a changed input.
- `per_video` / `combined` — overall md5 of the canonical CSV (columns sorted, index
  reset) plus a per-column md5, with row count, node set and `centroid_nodes`.
- `quality` — `ABSENT` until `quality.py` writes reports.

Never parquet bytes: pyarrow's file metadata is not deterministic.

`fixtures/env.json` records python, pandas, numpy, both parquet engines, `sleap_io` and
`sleap`. Baselines were written under python 3.11.14 / pandas 2.3.3 / numpy 1.26.4 /
fastparquet 2025.12.0 / sleap-io 0.5.7 / sleap 1.5.2, and are read back with
fastparquet — the engine that wrote them.

Measured 2026-08-24: all five fixtures hash identically under fastparquet and pyarrow in
one interpreter, so the fixtures survive the environment switch.

## Sessions

`sessions.yml`, keyed `subjid:date`. Each entry may override `derivatives` and `levels`;
both default from the file header. Chosen for coverage of: both skeletons, a node below
`presence_frac`, and a multi-`behav/` session whose two folders hold videos with the
same basename.
