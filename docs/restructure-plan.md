# Restructure plan — `sleap-hypnose` → `hypnose-sleap`

Repo root holds 2 445 lines of shell, PowerShell and one 1 631-line Python module. No package, no
config, no gate. Target: a `src/`-layout package resolving data locations through
`hypnose_helpers`, with a regression gate proving the move changed no number. Measured on `14d93b3`.

**Doc style, here and in `DECISIONS.md`:** bullets, one fact each, no prose around the point.

---

## Baseline

- Baseline = the `.parquet` files already saved on the server, re-derived from the same `.slp`.
- No copy of the old pipeline runs anywhere. `sleap_utils.py:9-12` imports the pre-rename
  `hypnose` 1.0.0 package; both machines have pulled `hypnose-behavior` 2.0.0 over its source.
  `sleap_analysis.ipynb`: 11 of 22 cells the same.
- Old code is recoverable in 4 import lines (Phase 0.5) — but repointed code in a new env is not
  what the server holds, so it is not a baseline.

## Inventory

| File | Lines | Fate |
| --- | ---: | --- |
| `sleap_utils.py` | 1 631 | split into 6 modules under `src/hypnose_sleap/` |
| `run_sleap_inference_local_windows.sh` | 269 | → `infer` verb (the only one in use) |
| `run_sleap_inference_local.sh` | 126 | delete — differs only in the conda hook and two defaults |
| `run_sleap_inference.sh` | 141 | delete — HPC copy of the same walk |
| `submit_sleap_inference.sh` | 84 | keep, thinned to a SLURM wrapper around the shared CLI |
| `transfer_sleap_results.ps1` | 147 | → `push` verb |
| `convert-avi.ps1` | 47 | unchanged, moves to `scripts/` |
| `sleap_analysis.ipynb` | 22 cells | trimmed to the 6 that call the pipeline |
| `models/` | 45 MB | untracked (Q1) |
| `labels_160_images_2_videos.v001.slp` | 71 MB | untracked (Q1) |

- Six independent implementations of the `rawdata/sub-*/ses-*/behav/*/VideoData/*.avi` walk (4
  bash, 1 Python, 1 PowerShell) plus 2 date-range parsers. `hypnose_helpers.io.{layout,selectors}`
  replaces all of them.
- `.gitignore:24` reads `.slp`, which matches a file *named* `.slp`, not `*.slp` — that is why the
  71 MB label file is tracked.
- `.DS_Store` is matched by `.gitignore:17` but was committed before the rule existed → needs
  `git rm --cached`.

Changed upstream in `56dad6b` / `14d93b3` (from the PC), after the first draft of this plan:

- `transfer_sleap_results.ps1` is now a PowerShell **profile function**
  (`function transfer_sleap_results`, `PositionalBinding=$false`, `return` instead of `exit`);
  body unchanged. `-Sub` / `-Date` now take comma-separated values.
- `-s` in the inference script takes several subjects in one flag (`-s 57 58 59`).
- The notebook gained two `sleap_node_quality_report` calls (cells 2 and 8).
- The inference script now activates **`sleap-gpu`**, hard-failing, via a hardcoded
  `/c/ProgramData/Miniconda3/etc/profile.d/conda.sh`. That is a different env from
  `sleap-analysis`, which runs the notebook. **Confirm before Phase 0.5 whether the PC really has
  two envs** — if inference and extraction are already split, `hypnose-sleap` replaces both or
  only the analysis one, and that changes what the new `scripts/run_inference.sh` activates.

## Environment

- All work on the analysis PC: every gate after Phase 1 needs the `.slp`, the saved parquets, `E:`
  and `Z:`.
- Build one env, `hypnose-sleap`, running the whole pipeline. Not a gamble: 16 of
  `hypnose-behavior`'s 17 dependencies are already in `sleap-analysis`; only `pyarrow` and the
  3.12 interpreter are missing.

| | `sleap-analysis` (current) | `hypnose-sleap` (target) |
| --- | --- | --- |
| Python | 3.11.14 | 3.12 |
| sleap | 1.5.2 | ≥1.5.5 — the version the models were trained in |
| sleap-io | 0.5.7 | newest the sleap pin allows |
| parquet | fastparquet | pyarrow **+** fastparquet, to read baselines with the engine that wrote them |
| behaviour | `hypnose-analysis==1.0.0`, source gone | `-e hypnose-helpers`, `-e hypnose-behavior` 2.0.0 |

- `extract` / `combine` stay separate verbs — now structural, not forced by an interpreter pin.
  `run` does both in one process. Free insurance for the next env that cannot host both.
- Never generate fixtures on the Mac: `sleap-2` has sleap-io 0.6.4 (vs 0.5.7) and pyarrow (vs
  fastparquet). Both move values, not just formatting.

### Two version axes

- `sleap` (+ `sleap-nn`, torch) runs `sleap-track` in `infer`. Must match the models (trained
  1.5.5). Affects only `.slp` files not yet written.
- `sleap_io` reads `.slp` in `extract`. Affects the numbers the gate measures.
- ⇒ a SLEAP upgrade is not a gate risk; a `sleap_io` bump is. Phase 3 records which of the three
  fallbacks in `extract_points_and_scores` 0.5.7 actually takes, so the next bump is measured.

### Environment files

- Commit `sleap-analysis-environment.yml` as `environment.lock.yml` — the record of the env that
  produced every baseline parquet. Strip the `prefix:` line and the `hypnose-analysis==1.0.0` pip
  line (local install; pip would look on PyPI and fail).
- Hand-write `environment.yml` for the new env: name `hypnose-sleap`, `python=3.12`, `sleap`,
  `sleap-io`, `pyarrow`, `fastparquet`, `pandas`, `numpy<2`, `h5py`, `opencv`, `pyyaml`, `tqdm`,
  `harp-python`, `swc-aeon`, `-e ../hypnose-helpers`, `-e ../hypnose-behavior`, `-e .`.
- GPU torch wheels come from a custom index → README, not either file.

---

## Decisions

### Q1 — models stay out of the repo

- `models/` (45 MB) and the label `.slp` (71 MB) are **tracked**; deleting them from the tree does
  not shrink a clone.
- Delete + gitignore anyway. History rewrite is available (5 commits, all superseded) but
  force-pushes over every clone — not recommended; `git clone --filter=blob:none` covers the cost.
- `configs/models.yml` (committed): role → path. `configs/models.local.yml` (git-ignored):
  `model_root`. Absolute path wins; relative resolves under `model_root`.
- Roles: `naive`, `eeg_surgery`, `neuropixel_surgery`. `--model naive`, or `--model <path>` for a
  one-off.
- `260305_143707.single_instance.n=340` is a training timestamp; call sites want a role name.
- Every run records the resolved model dir + md5 of its `training_config.json` in the quality
  `.yml`. Nothing in the current parquets says which of the three models produced them.

### Q2 — one inference entry point, in Python

- The two local scripts differ only in the conda hook and two defaults; the HPC one only in SLURM
  submission. What all three spend their lines on is the session walk.
- `hypnose_helpers` installs into the SLEAP env (py ≥3.9; numpy/pandas/matplotlib/pyyaml already
  present) → the CLI runs inside it and still uses the family selectors.
- `scripts/run_inference.sh` = conda activate + `python -m hypnose_sleap.cli infer "$@"`. 5 lines.
- `scripts/submit_inference.sh` = SLURM wrapper → same CLI, so HPC and PC stop drifting.
- The PowerShell profile-function ergonomics stay available where wanted: a one-line function
  calling the CLI gives `transfer_sleap_results`-style invocation without a second implementation.

### Q3 — one active profile, two named transfer endpoints

- No second `active` key: it means "where my data is" family-wide, and `--show` would start
  disagreeing with `hypnose-behavior` on the same machine.
- `configs/data_locations.yml` gains `transfer: {remote: server-windows, local: local_1}`.
- `fetch` = remote rawdata → local rawdata (**the whole `behav/<exp>/` tree**, not just
  `.avi`). `push` = local derivatives → remote derivatives. `--from` / `--to` override
  either end.
- Measured at Phase 4: `.avi` is 98.7 % of a session (15.8 GB), every harp stream
  together 1.3 % (212 MB). `combine` needs `VideoData/*.csv` and `Behavior/` and never
  opens an `.avi`, so an `.avi`-only fetch would leave `combine` unable to run — and
  `load_all_streams` degrades quietly rather than raising when `Behavior/` is missing.
  Copying the extra 1.3 % is what keeps "only `fetch` and `push` touch the server" true.
- `push` prints its plan and refuses to overwrite an existing destination file without `--force`.
- The active profile stays what `infer` / `extract` / `combine` read and write.

---

## Target layout

```
hypnose-sleap/
├── configs/
│   ├── data_locations.yml         profiles + transfer: {remote, local}   (committed)
│   ├── data_locations.local.yml   active: <profile>                      (git-ignored)
│   ├── models.yml                 role -> model path                     (committed)
│   ├── models.local.yml           model_root for this machine            (git-ignored)
│   └── parameters.yml             score_thresh, presence_frac, gap_limit, batch_size
├── docs/                          restructure-plan.md, DECISIONS.md
├── notebooks/sleap_analysis.ipynb
├── scripts/                       run_inference.sh, submit_inference.sh, convert-avi.ps1
└── src/hypnose_sleap/
    ├── io/{paths,layout,transfer}.py
    ├── parameters.py     parameters.yml + models.yml resolution
    ├── inference.py      video enumeration + sleap-track
    ├── extract.py        .slp -> per-video parquet (centroid pipeline)
    ├── timestamps.py     per-video parquet -> combined parquet   [hypnose_behavior]
    ├── quality.py        per-node presence/confidence -> .yml
    ├── annotate.py       overlay video                            [hypnose_behavior]
    ├── cli.py            fetch / infer / extract / combine / push / annotate / run
    └── qc/               README.md, sessions.yml, _common.py, regression.py,
                          ast_move_check.py, fixtures/
```

## Verbs

| verb | in | out |
| --- | --- | --- |
| `fetch` | remote rawdata `behav/` tree | local rawdata, same tree |
| `infer` | local `.avi` + model role | `movement_analysis/<behav>__<video>.predictions.slp` |
| `extract` | `.slp` | per-video `.parquet` + quality `.yml` |
| `combine` | per-video `.parquet` + harp streams | one combined `.parquet` per session |
| `push` | local derivatives `.parquet` / `.yml` | remote derivatives |
| `annotate` | `.avi` + combined `.parquet` + streams | annotated `.mp4` |

### Everything heavy happens on local disk

- The loop is `fetch` → `infer` → `run` → `push` → `clean`. Only `fetch` and `push` touch
  the server.
- One deliberate exception, off by default: `combine` / `run` take `--rawdata-from
  <profile>` so harp streams can be read straight off the server when local `rawdata`
  has been deleted to reclaim disk. `combine_session(rawdata=...)` already takes the
  override — Phase 5 only exposes it. Reading is never the default: it would make the
  heavy loop depend on the network to save 1.3 % of local disk.
- Server profiles carry `remote: true` in `configs/data_locations.yml`. `infer`, `extract`,
  `combine` and `run` call `io.paths.require_local()` and refuse such a profile without
  `--allow-remote`. Reads are never blocked.
- The two repos legitimately sit on different profiles on the same machine —
  `hypnose-sleap` on `local_1`, `hypnose-behavior` on `server-windows` — because one
  writes bulk intermediates and the other reads finished results. The guard is what keeps
  that from being a footgun rather than a convention.
- Outputs go to `saved_analysis_results/movement_analysis/`
  (= `hypnose_behavior.io.layout.MOVEMENT_SUBFOLDER`). Closes the `sleap-hypnose` item in
  `hypnose-behavior/docs/TODO.md`. That is a *naming* agreement — which drive it lands on
  is the active profile's business, and is local until `push`.
- Reads use `rglob` — flat and grouped sessions both resolve. Writes create the parent.
- Filenames unchanged, including `sub-045_ses-20260219_combined_sleap_tracking_timestamps.parquet`
  (a date in the `ses-` slot). The consumer matches by suffix so a rename is safe, but not while
  the fixtures depend on it.
- `.slp` stays local: the current transfer copies only parquet/csv, and re-running inference is
  cheaper than storing them. `extract` still reads `.slp` from the active profile's derivatives, so
  older sessions whose `.slp` reached the mount still work.

### No deletion, anywhere

- **`rawdata/` is read-only.** It holds the only copy of every recorded video.
- **No verb deletes anything.** No `clean` verb, and no deletion call in the package.
- Rejected: a `clean` verb behind an "is this path local?" test. Such a test is a
  per-platform heuristic, and a server mounted as `Z:` — or under `/Volumes`, or `/mnt` —
  and then added as a profile defeats it. The convenience saved is one manual delete; the
  failure mode is losing irreplaceable video.
- Reclaiming local disk is manual: delete `rawdata/` on `D:` / `E:` by hand.
- `push` composes its destination from `get_derivatives_root()` only. That is what keeps
  it out of rawdata — `hypnose_helpers` resolves the two roots separately, so reaching
  rawdata takes a deliberate call to the wrong function.

### Quality report

- `sleap_node_quality_report` already computes everything and only prints; the notebook now calls
  it by hand per session (cells 2 and 8). Add a writer →
  `movement_analysis/sleap_quality_sub-XXX_ses-YYYYMMDD.yml`, emitted by `extract` so the report
  is produced with the parquet rather than asked for afterwards.
- Contents: `model {role, path, training_config_md5}`; `parameters {score_thresh, presence_frac,
  gap_limit}`; `frames {total, occupied}`; per node `{pres_pct_occ, pres_pct_occ_gated, score_p50,
  selected}`; `centroid_nodes`.
- `centroid_nodes` must equal the `centroid_nodes` column in the parquet — asserted by the gate.

---

## QC gate

Mirrors `hypnose-behavior/src/hypnose_behavior/qc/`: `sessions.yml`, `_common.py`,
`regression.py [--generate]`, `fixtures/*.json` + `env.json`, exit 0 GREEN / 1 RED.

Four differences:

1. `--generate` reads the saved files (via `rglob`), not a run of old code.
2. Fingerprint = md5 of the canonical CSV (columns sorted, index reset) plus a per-column md5.
   Never parquet bytes — pyarrow metadata is not deterministic.
3. Three levels, three different input stabilities.
4. AST move check — copy `ast_move_check.py`. `_compute_session_centroid`,
   `extract_points_and_scores`, `to_number` and `infer_video_file_from_slp` must reappear with
   byte-identical source.

| level | what | expectation |
| --- | --- | --- |
| L1 `per_video` | `.slp` → per-video parquet | **byte-identical.** Pure `sleap_io` + `pandas`; the actual gate on the restructure |
| L2 `combined` | + timestamps from `load_all_streams` | measured and explained, not assumed |
| L3 `quality` | the new `.yml` | no baseline exists; asserted *consistent* with the parquet beside it |

- L2's input is not frozen: the saved `time` column came from a pre-v2.0.0 `load_all_streams`.
  First measurement — join new against saved on `(video_file, frame)`, report row-count delta and
  max |Δtime|. Zero → L2 becomes a byte-identity check. Non-zero → the value goes in
  `DECISIONS.md` and L2 stays a reported delta.

`env.json` records six fields: python, pandas, numpy, **parquet engine**, **sleap_io**, sleap.

- Baselines were produced under py 3.11.14 / pandas 2.3.3 / numpy 1.26.4 / fastparquet 2025.12.0
  (no pyarrow) / sleap-io 0.5.7 / sleap 1.5.2.
- Read a baseline with the engine that wrote it — otherwise a dtype round-trip enters the
  fingerprint and a RED cannot separate "the code changed" from "the reader did".
- After Phase 0.5 `env.json` records the new env; `environment.lock.yml` keeps the old one.

Fixture sessions — two, each needing `.slp` + per-video parquets + combined parquet present:

- one multi-`behav/` session → exercises the `<behav>__<video>` prefix and video ordering;
- one session with a node failing `presence_frac` → exercises node selection and the fallback at
  `sleap_utils.py:457`;
- which sessions qualify has to be checked on the PC.

---

## Phases

One per session; each gate green before the next.

**0 — capture the baseline.** In the current `sleap-analysis` env, before anything moves. Write
`qc/sessions.yml`, `_common.py`, `regression.py --generate`, `fixtures/env.json`.
*Gate:* `--generate` twice → identical fixtures; every fixture parquet non-empty with the expected
node set; one baseline read with fastparquet and with pyarrow → same canonical-CSV md5 (decides
whether the fixtures survive the env switch).

Done 2026-08-24.

**0.5 — env change, code held constant.** Never measure the env and the restructure together.

- Build `hypnose-sleap`.
- Repoint only the 4 imports in `sleap_utils.py`: `hypnose.io.paths` →
  `hypnose_behavior.io.paths`; `hypnose.trial_classification.classification_utils` →
  `hypnose_behavior.io.loaders`; `hypnose.metric_analysis.metrics_utils` →
  `hypnose_behavior.io.load_results`; `hypnose.utils.helpers` →
  `hypnose_behavior.utils.helpers`. No other edit.
- Run `regression.py` in the new env.
- GREEN → the env is transparent. Adopt it, retire `sleap-analysis`, later gates unchanged.
- RED → a characterised env delta before any code moved. Record the numbers in `DECISIONS.md`;
  write `fixtures/<session>.newenv.json` from old code in the new env; the restructure gates
  against that, and the disk fixture stays as the record of what the server holds.
- Tolerance is decided now: no "close enough" tier. A moved float is accepted once, in writing,
  with a new baseline — or it is a bug.
- A non-zero delta also means the server tree (extracted with 0.5.7) no longer matches new runs.
  Re-extraction is cheap in compute but needs the `.slp`, which live on `E:` and were never
  pushed. Check what survives during Phase 0.

  Done 2026-08-24.

**1 — skeleton + rename.** GitHub rename (`Joschua21/sleap_analysis` → `hypnose-sleap`; consider
moving to `SainsburyWellcomeCentre/`), `git remote set-url`, local directory rename.
`pyproject.toml`, `src/hypnose_sleap/` stubs, both environment files, `.gitignore` (`*.slp`,
`models/`, `configs/*.local.yml`, `*.egg-info/`), `git rm --cached` on `models/`, the label `.slp`
and both `.DS_Store`.
*Gate:* `-e ../hypnose-helpers`, `-e ../hypnose-behavior` and `-e .` all install; all three imports
work; `hypnose-sleap --help` runs; `regression.py` still reads its fixtures.

Done 2026-08-24.

**2 — paths, layout, parameters.** `io/paths.py`
(`DataLocations(config_dir=configs, env_prefix="HYPNOSE")`), `io/layout.py` (rawdata/derivatives
`SessionLayout`, `subject_pattern="{subject}_id-*"`, movement paths),
`configs/{data_locations,models,parameters}.yml`, `parameters.py`.
*Gate:* `hypnose-set-data-location --list/--show` works from the repo root; `find_sessions` returns
the same session list the old bash glob did — capture that list first.

Done 2026-08-24

**3 — extract.** `sleap_labels_and_centroid` → `extract.py`, numerics untouched, paths via layout,
outputs to `movement_analysis/`. `sleap_node_quality_report` → `quality.py` plus the `.yml` writer.
Record which points fallback `sleap_io` takes.
*Gate:* L1 GREEN; `ast_move_check` clean; L3 consistency holds.

Done 2026-08-25.

**4 — combine.** `get_video_frame_times` + `add_timestamps_to_sleap_tracking` → `timestamps.py`,
imports repointed, lazy, `rglob` discovery. `process_sleap_sessions` → the `run` verb, same summary
output.
*Gate:* L2 measured and recorded; `hypnose_behavior.io.layout.find_tracking_file` finds the
combined file in the temp derivatives dir — assert by calling it.

Done 2026-08-25. L2 measured **zero** — byte-identical on all four L2 fixtures — so per the
pre-registered rule it is now a byte-identity check, and Risk 3 is closed by measurement.
`DECISIONS.md` §12.

**5 — infer, fetch, push.** `inference.py`, `io/transfer.py`. Four bash scripts → two thin
wrappers; `transfer_sleap_results.ps1` retires. Plus `--rawdata-from` on `combine` / `run`,
and `require_local` on `infer` (the flag exists, nothing calls it yet).
*Gate:* `infer --dry-run` lists the same videos the old script processed (capture first);
`push --dry-run` plans the same file set as `transfer_sleap_results.ps1 -DryRun` on the same
filters, and every destination it names is under `get_derivatives_root()`.

Decided before starting:

- `fetch` copies the whole `behav/<exp>/` tree (Q3), not `.avi` only.
- `push`'s pattern set **must gain `sleap_quality_sub-*.yml`** — Phase 3 added that file and
  `transfer_sleap_results.ps1:7` predates it, so quality reports reach nobody today.
- `infer` writes `.slp` through `layout.write_path`, i.e. grouped into `movement_analysis/`
  like Phase 3 and 4. The old script writes flat (`:238`); `find_outputs` rglobs, so both
  resolve and old sessions keep working.
- **The env question from §7 must be answered first.** `sleap-gpu` has torch but none of the
  three hypnose packages, and it is python 3.11.14 while `pyproject.toml` pins
  `>=3.12,<3.13` — so `hypnose_sleap` cannot be installed there as written. Either relax
  the pin, or install the GPU stack into `hypnose-sleap`. This blocks `infer`, not `fetch`
  or `push`.
- `set_float32_matmul_precision('high')` at `:246` runs in a **separate** python process and
  therefore never applied to `sleap-track`. Porting it "correctly" would change float32
  matmul on Ampere+ and move predictions. Preserve the no-op, or change it deliberately and
  re-baseline — never silently.

**6 — annotate, notebook, README.** `annotate.py` with repointed imports and `find_tracking_file`.
Notebook trimmed to the pipeline calls plus video creation; cells 10–19 (harp-stream debugging)
move to `hypnose-behavior/notebooks/` or go; cell 20 (`.slp` frame inspection via h5py) stays — it
is a SLEAP question, and is the natural seed for a `peek` verb.
*Gate:* one clip renders end to end; frame count and a sample of frame hashes match a kept
reference clip if one exists.

**7 — cleanup.** Delete `sleap_utils.py`, `models/`, the label `.slp`, `__pycache__/`. Strike the
`sleap-hypnose` item from `hypnose-behavior/docs/TODO.md`. Open `docs/DECISIONS.md`.

---

## Risks

1. ~~`hypnose_behavior` in the SLEAP env~~ — closed by the 3.12 rebuild. Open instead: does
   `sleap_io` ≥0.6 extract what 0.5.7 did (Phase 0.5).
2. `sleap_io` drift is real and now deliberate — 0.5.7 wrote every baseline. Every future bump
   repeats the Phase 0.5 one-variable experiment.
3. ~~L2 may never be byte-identical, for reasons in `hypnose-behavior` (its standing caveat).
   Budget for documenting a delta, not chasing it to zero.~~ — closed at Phase 4: the delta
   measured zero on all four L2 fixtures, and L2 is now asserted. Open instead: `load_all_streams`
   lives in another repo, so an L2 RED is not necessarily this repo's doing (`DECISIONS.md` §12).
4. Fixture availability — sessions keeping both `.slp` and saved parquets must be found on the PC.
   If none, fall back to re-running inference on one session with a pinned model; weaker, since it
   stops proving the new code reproduces what downstream reads.
5. Model provenance for existing results is unrecoverable. The quality `.yml` fixes it going
   forward only.
