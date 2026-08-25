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
| L2 `combined` | + timestamps from `load_all_streams` | byte-identical, and rows = sum of parts |
| L3 `quality` | the quality report `.yml` | consistent with the parquets beside it |

- L1 is the cleanest gate: pure `sleap_io` + `pandas`, no behavioural data.
- L2 was measured at Phase 4 and came back byte-identical on all four L2 fixtures, so it
  is asserted rather than reported (`docs/DECISIONS.md` §12). Its input is still not
  frozen — `load_all_streams` lives in `hypnose-behavior` — so an L2 RED names that
  function as a candidate cause and points at `env.json`.
- L2 carries a second, separate assertion: the combined row count equals the sum of its
  per-video tables. Split from the md5 so "the join changed shape" and "the values
  moved" are never reported as the same finding.
- L3 has no baseline; nothing wrote a report before. It asserts the report's
  `centroid_nodes` equals the `centroid_nodes` column in every parquet beside it.
- `discovery` is not a level: it asserts `hypnose_behavior.io.layout.find_tracking_file`
  locates the combined parquet the gate just wrote. Only measured when re-deriving.

## Sources

- `disk` — re-hash the saved baseline files. Checks that the fingerprinting is
  deterministic; **not** a gate on the pipeline. Always used by `--generate`.
- `rederive` — run the new pipeline and fingerprint its output. The actual gate; on
  automatically since `extract.py` landed.

Which one ran is printed at the top of every compare. `rederive` stages the session's
`.slp` into a temp tree and runs `extract_session` and then `combine_session` there,
because both write beside their input — the real derivatives tree is only ever read.

Only `.slp` is staged: `extract` writes its parquets into the temp tree and `combine`
reads them there. The harp streams `combine` needs come from the real `rawdata`, taken
as the sibling of the fixture's `derivatives` root (override per session with a
`rawdata:` key). Nothing is written outside the temp tree, and no `HYPNOSE_*` env var or
`cache_clear()` is involved — both roots are passed explicitly.

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

## AST move check

```bash
python src/hypnose_sleap/qc/ast_move_check.py             # the Phase 3 set vs HEAD
python src/hypnose_sleap/qc/ast_move_check.py --show-diff # why something drifted
```

Complements `regression.py`: the regression proves five sessions still produce the same
numbers, this proves no moved body drifted at all — including in a branch those sessions
never take. Every definition in scope must reappear with a byte-identical source segment,
compared after `dedent` so a helper lifted out of its enclosing function still counts as
a move.

`--only NAME` names what a phase moved; the default is Phase 3's five. Later phases pass
their own set, and `--all` — every definition in `sleap_utils.py` — is what Phase 7 runs
once the file is gone.
