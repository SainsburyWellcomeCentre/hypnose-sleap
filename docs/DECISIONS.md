# Decisions

Measurements and choices made during the restructure. Bullets, one fact each.

---

## 1 — Baseline environment (Phase 0, 2026-08-24)

- The analysis PC carries **two** SLEAP envs: `sleap-gpu` (inference) and
  `sleap-analysis` (extraction / notebook). The plan's open question is answered:
  inference and extraction are already split.
- `sleap-analysis` matches what the plan recorded: python 3.11.14, pandas 2.3.3,
  numpy 1.26.4, fastparquet 2025.12.0, **no pyarrow**, sleap-io 0.5.7, sleap 1.5.2.
- `hypnose` 1.0.0 is absent from every env on the machine, so `sleap_utils.py` cannot
  run. The baseline is the saved `.parquet` files, as planned — it could not have been
  re-derived.
- Captured verbatim as `environment.lock.yml`, minus `prefix:` and the
  `hypnose-analysis==1.0.0` pip line.

## 2 — Parquet engine does not move the fingerprint

- Gate: read each fixture parquet with fastparquet and with pyarrow in **one**
  interpreter (python 3.11.14, pandas 2.3.3, pyarrow 25.0.1 installed to a throwaway
  `--target` dir), so the engine is the only variable.
- Result: **9/9 files hash identically.** One per-video and one combined parquet per
  fixture session.
- ⇒ the Phase 0 fixtures survive the environment switch. A RED after Phase 0.5 is
  attributable to `sleap_io` or to the code, not to the reader.
- `read_baseline` still pins fastparquet. The measurement licenses the switch; it does
  not make the pin pointless, since a future pandas may not agree.

## 3 — Fixture sessions

- Five, all confirmed present with `.slp` + per-video parquet on disk.

| session | model role | skeleton | node dropped | videos / behav | levels |
| --- | --- | --- | --- | --- | --- |
| 057:20260717 | eeg_surgery | tail_tip | `tail_tip` | 2 / 1 | L1 L2 L3 |
| 058:20260717 | eeg_surgery | tail_tip | `tail_tip` | 4 / 1 | L1 L2 L3 |
| 059:20260717 | eeg_surgery | tail_tip | `tail_tip` | 3 / 1 | L1 L2 L3 |
| 066:20260723 | eeg_headstage | headstage | `headtstage_base` | 4 / 1 | L1 L3 |
| 057:20260616 | eeg_surgery | tail_tip | `tail_tip` | 3 / **2** | L1 L2 L3 |

- 066 lives on `D:\derivatives` and has no combined parquet, so it is baselined at L1
  and L3 only. `levels` in `sessions.yml` records that; a `NOT BASELINED` level is
  distinct from an `ABSENT` one and does not read as agreement with it.
- 057:20260616 is the only multi-`behav/` case. Its two behav folders each contain a
  video named `...1904-01-07T15-00-00.avi`, so it is what proves the `<behav>__` prefix
  and the video ordering survive the move.
- Node selection is exercised on every fixture: exactly one node falls below
  `presence_frac` in each, and `centroid_nodes` is identical across a session's videos.

## 4 — L2 adds columns, not rows

- Measured over the four L2 fixtures: the combined parquet's row count **equals** the
  sum of its per-video parquets' rows, in all four cases (273,042 / 626,936 / 284,034 /
  350,869).
- ⇒ `add_timestamps_to_sleap_tracking` is a pure left join at these sessions. Any
  row-count delta at Phase 4 is a change, not the join's nature.

## 5 — Model roles

- Three, from `../sleap-models/sleap_v1.5.5/models/MODEL_EXPLANATION.txt`:

| role | model directory |
| --- | --- |
| `naive` | `260305_143707.single_instance.n=340` |
| `eeg_surgery` | `260528_150609...n=580_260529_104135...n=590_260717_162525...n=710` |
| `eeg_headstage` | `260529_144352...n=325_260601_140335...n=466_260602_141703...n=541_260810_104904...n=801` |

- The plan named the third role `neuropixel_surgery`. **It is `eeg_headstage`**; there
  is no neuropixel model.
- `eeg_surgery` and `eeg_headstage` each have an older variant without the
  brighter-image retraining. `models.yml` names the current one per role; an explicit
  path covers a one-off against an older one.
- The two skeletons are a property of the role, not of a session: `eeg_surgery` tracks
  `center_head / tail_base / tail_tip`, `eeg_headstage` tracks
  `tailbase / headstage_top / headtstage_base`. Node selection has to stay
  skeleton-agnostic.
- `models/` in this repo held neither — two 2025 models used for nothing current.
  Untracked in Phase 1, deleted in Phase 7.

## 6 — Provenance is not recoverable for existing results

- Nothing in the saved parquets records which model produced them. The role in the
  table above is inferred from the skeleton, which distinguishes `eeg_headstage` from
  the other two but not `naive` from `eeg_surgery`.
- The quality `.yml` fixes this going forward only, by recording the resolved model dir
  and the md5 of its training config — `training_config.yaml` for these sleap-nn models;
  see section 10.

## 7 — The new environment

- `hypnose-sleap`: python 3.12.14, pandas 3.0.5, numpy 1.26.4, pyarrow 25.0.0,
  fastparquet 2026.5.0, sleap 1.5.2, sleap-io 0.5.7. All three sibling packages
  installed editable.
- **There is no sleap 1.5.5.** PyPI goes 1.5.0, 1.5.1, 1.5.2, then 1.6.0. conda tops out
  at 1.4.1/py37. Both existing envs run 1.5.2; `sleap_v1.5.5` is a folder name in
  `../sleap-models`, not a version. The plan's `sleap >=1.5.5` is unsatisfiable.
- sleap 1.5.2 supports `>=3.11,<3.14` and ships a pure-python wheel, so the current
  SLEAP stack runs on 3.12 unchanged.
- **`sleap-io` is pinned to `==0.5.7`.** sleap 1.5.2 pins it as a floor only
  (`sleap-io[all]>=0.5.7`), so an unpinned install resolves to 0.9.2. Holding it keeps
  the highest-risk variable out of the environment switch; bumping it is its own
  one-variable experiment, per Risk 2.
- **`infer` does not work in this env yet.** `sleap-track.exe` is installed but its
  backend is not: torch, sleap-nn, lightning and kornia are all absent, because sleap
  1.5.2 does not declare them. `sleap-gpu` carries torch 2.9.1+cu126 from a custom
  index. Phase 5 either installs that stack here or keeps `infer` in `sleap-gpu`.

## 8 — Phase 0.5: the environment is transparent

- Method: copy each fixture session's `.slp` into a temp tree, run the repointed
  `sleap_utils.sleap_labels_and_centroid` against it at the pipeline defaults, and
  fingerprint what it wrote. The real derivatives tree is only read.
- Only the four import lines in `sleap_utils.py` were edited — `git diff --numstat`
  reports exactly `4 4`.
- Variables that moved: python 3.11.14 -> 3.12.14, pandas 2.3.3 -> 3.0.5, fastparquet
  2025.12.0 -> 2026.5.0, pyarrow added. Held: numpy 1.26.4, sleap-io 0.5.7, sleap 1.5.2.
- **Result: 16/16 per-video parquets re-derive byte-identically** across all five
  sessions, both skeletons.
- Reading the baselines under the new environment is also unchanged: `regression.py`
  is GREEN there, including the four L2 combined parquets.
- ⇒ **no `.newenv.json` fixtures are needed.** The restructure gates against the Phase 0
  fixtures directly, and a RED from here on is the code.
- The pandas 2 -> 3 jump moving nothing is the notable part: the centroid pipeline uses
  `groupby`, `reindex`, `interpolate` and `nanmean`, and none changed under pandas 3 at
  these inputs.

## 9 — Phase 4 hazard: `add_timestamps_to_sleap_tracking` cannot be redirected

- It takes no `base_dir`. It resolves `get_data_root() / "rawdata"`, derives
  `../derivatives` from that, and writes the combined parquet into the real tree
  (`sleap_utils.py:580-581`, `:752-754`).
- `sleap_labels_and_centroid` *does* take `base_dir`, which is what made Phase 0.5 safe.
- ⇒ any L2 gate must redirect with `HYPNOSE_DATA_ROOT` / `HYPNOSE_DERIVATIVES_ROOT` plus
  `cache_clear()`, the way `hypnose_behavior/qc/_common.py` does. Running it as-is would
  overwrite the L2 baselines.

## 10 — Phase 2: the shared resolvers replace the six walks

- `io/paths.py` wraps `hypnose_helpers.io.paths.DataLocations` over `configs/`, with
  `env_prefix: HYPNOSE` and profile names matching `hypnose-behavior`, so both repos
  resolve the same tree and `--show` cannot disagree.
- `io/layout.py` binds two `SessionLayout`s with `subject_pattern="{subject}_id-*"` and
  owns the one `behav/*/VideoData/*.avi` walk (`session_videos`) and the one
  `<behav>__<video>` naming rule (`video_key`).
- **Gate: identical to the old bash glob.** `find_sessions` over `E:\rawdata` returns
  **175 sessions across 7 subjects**, and `session_videos` enumerates **545 videos** —
  exactly what `run_sleap_inference_local_windows.sh`'s `find` + `for TS_DIR in
  "$BEHAV_DIR"/*T*` walk produced. No session added, dropped or reordered.
- `local_2` (`D:\rawdata` / `D:\derivatives`) added as a profile, since sub-066 lives there.
- `transfer: {remote: server-windows, local: local_1}` added per Q3. No second `active`
  key; `resolve_profile(name)` resolves a named endpoint alongside the active profile.

### `MOVEMENT_SUBFOLDER` is duplicated, not imported

- `io/layout.py` defines `movement_analysis` itself rather than importing it from
  `hypnose_behavior`, so `extract` runs without that package.
- `qc/check_layout.py` is what stops the two copies drifting: it asserts
  `MOVEMENT_SUBFOLDER`, `RESULTS_DIRNAME` and `SUBJECT_PATTERN` are equal in both
  packages, and — the part that matters — that
  `hypnose_behavior.io.layout.find_tracking_file` actually locates a file written at our
  `write_path`. Currently green.
- This closes the `sleap-hypnose` item in `hypnose-behavior/docs/TODO.md`.

### The models are sleap-nn, not TensorFlow

- Every directory in `../sleap-models/sleap_v1.5.5/models` holds `best.ckpt` +
  `training_config.yaml` + `initial_config.yaml`. The old TF layout
  (`best_model.h5` + `training_config.json`) is what the retired `models/` in this repo
  had, and section 5's provenance plan named the wrong file.
- `model_provenance` looks for `training_config.yaml` then `training_config.json`, and
  records which it hashed. Current md5s: `naive` `3b66ccc8`, `eeg_surgery` `b6f86d0a`,
  `eeg_headstage` `69f5b20e`.
- A model directory name is a concatenation of training run timestamps
  (`..n=580_..n=590_..n=710`); it records the retraining chain, not three models.

### Bulk writes are fenced to local disk

- Server profiles carry `remote: true`. `io.paths.require_local(verb)` raises unless the
  active profile is local or `--allow-remote` is passed; `infer`, `extract`, `combine` and
  `run` take the flag. Reads are never blocked.
- The key is ignored by `hypnose_helpers`, so the profiles stay interchangeable with
  `hypnose-behavior`'s.
- Measured on this machine: `hypnose-sleap` is on `local_1`, `hypnose-behavior` is on
  `server-windows`. That divergence is correct — one writes bulk intermediates, the other
  reads finished results — but nothing enforced it, and a single
  `hypnose-set-data-location server-windows` here would have sent 60 MB `.slp` files to
  ceph. The guard makes it a deliberate act.
- `find_tracking_file` agreeing with our `write_path` is a **naming** agreement only.
  Which drive the file lands on is the active profile's business.

### No deletion, and no machinery to police it

- `rawdata/` holds the only copy of every recorded video. Nothing here deletes anything.
- **No `clean` verb.** Rejected because its guard would be an "is this path local?" test,
  which is a per-platform heuristic: a server mounted as `Z:`, or under `/Volumes`, or
  `/mnt`, then added as a profile, defeats it. The convenience saved is one manual
  delete; the failure mode is losing irreplaceable video.
- Local disk is reclaimed by hand: delete `rawdata/` on `D:` / `E:` when it fills.
- **Also rejected: an `io/safety.py` write-guard module and AST checks to prove the
  absence of deletion.** `hypnose_helpers` resolves rawdata and derivatives as separate
  roots, so `push` composing its destination from `get_derivatives_root()` is already
  what keeps it out of rawdata. A module plus two gates to enforce what one function call
  determines is weight this repo does not need — it does one contained job.
- The guard that stays is `io.paths.require_local()`: twenty lines in an existing file,
  addressing a hazard that actually exists (writing 60 MB `.slp` files to the server).

### Config gotcha: UNC paths need two backslashes in YAML

- `server-windows` must read `'\\ceph-gw02...'`. A single leading backslash parses
  cleanly, resolves to a valid-looking `Path`, and silently is not the share.
- Now byte-identical to `hypnose-behavior/configs/data_locations.yml`, asserted by
  comparing the parsed values rather than the file text.

## 11 — Phase 3: the centroid pipeline moved, de-nested

- `sleap_labels_and_centroid` → `extract.extract_session`; `sleap_node_quality_report`
  → `quality.py`, plus the `.yml` writer.
- **L1 GREEN: 16/16 per-video parquets re-derive byte-identically** across all five
  sessions. L3 consistent on all five. L2 unchanged (still read from disk).
- `regression.py` now prints `SOURCE: rederive`. `_common.fingerprint_session` gained
  the branch behind that banner: it stages the session's `.slp` into a temp tree and
  runs `extract_session` there, because extract writes beside its input. `--generate`
  pins `rederive=False` — a baseline is what the saved files say, never a re-run.

### Four helpers de-nested, byte-identically

- `extract_points_and_scores`, `to_number`, `infer_video_file_from_slp` and
  `node_names_for_instance` were nested in `sleap_labels_and_centroid`; only
  `resolve_deriv_root` closed over the signature, and `io/layout.py` replaces it.
- They close over one variable, `nan = float("nan")`, now a module constant.
- `_compute_session_centroid` was already top-level and moved unchanged.
- All five assembled into `extract.py` by script, not by retyping — the AST check
  proves the result, it does not protect the transcription.

### `ast_move_check.py` needed three changes to express this gate

- **Nested collection.** It collected top-level defs only, so three of the four
  required names were invisible on the "before" side. It now descends into functions
  (not into classes: a method is part of its class's segment).
- **Dedented comparison.** De-nesting shifts every line left 4, so segments are
  compared after `textwrap.dedent`. That also normalises whitespace-only lines —
  `sleap_utils.py:228` is 8 spaces — and is the only leniency. A re-wrapped or
  re-indented body is still CHANGED.
- **`--only` scope.** `sleap_utils.py` empties over four phases, so the eleven
  definitions still awaiting Phases 4-6 would all read as MISSING. `--only` names what
  a phase moved (default: the Phase 3 five); `--all` is what Phase 7 runs.
- Ambiguity is refused, not resolved: `resolve_deriv_root` is nested in **two**
  functions (`:200` and `:1337`), so the name is marked ambiguous and fails only if
  something puts it in scope. Two *top-level* definitions of one name still raise.
- It also read files with the locale encoding. `sleap_utils.py` is UTF-8 (`✓`, `⚠️`),
  so on this machine every read died in cp1252 before comparing anything; `git show`
  and `read_text` are both pinned to UTF-8 now.

### `sleap_io` 0.5.7 takes the first branch every time

- Measured on `sub-057_date-20260717`: instances are `PredictedInstance`,
  `inst.numpy()` returns an `(n_nodes, 2)` `ndarray`, and `inst.points[idx][1]` yields
  a `float64` confidence.
- ⇒ the `points_array` / `points` coordinate fallbacks and the `point_confidences`
  score fallback are **dead code at 0.5.7**, as is every `as_xy` branch after the first.
- That is the measurement Risk 2 wants: a `sleap_io` bump is a regression risk exactly
  because it could move which branch fires, and now there is a recorded starting point.

### The quality report and extract count occupied frames differently

- `extract` (`sleap_utils.py:442`) counts a frame occupied when some node is present
  **and** above `score_thresh`. The report (`:1581`) counts it occupied when some node
  is present, **ungated**, then divides gated counts by that.
- Different denominators, so the two can disagree on a borderline node. Measured on
  057:20260717, `center_head`: 98.954 % (report) vs 98.9681 % (extract); occupancy
  273,003 gated vs 273,042 ungated, a 39-frame gap.
- **Not unified.** Extract's denominator cannot move without breaking L1, so
  unification would have to change the report — which is the number the notebook has
  been reading to compare models, and changing it inside a move phase is two variables
  at once.
- Unified instead on *authority*: the `.yml`'s `centroid_nodes`, and each node's
  `selected`, come from extract's selection, so L3's assertion holds by construction
  rather than by coincidence. Both fractions are written side by side
  (`pres_pct_occ_gated` and `selection_pct`), and the report's own `selected` column
  stays in the DataFrame for interactive use without entering the file.

### Model provenance is required, and never guessed

- `extract_session(model=...)` is a **required** keyword that may be None. None records
  `quality.UNKNOWN_MODEL` (all four fields null).
- `parameters.model_provenance(None)` falls through to `models.yml`'s `default`, so a
  re-extraction of an old `.slp` would otherwise write a confident, wrong provenance
  into the one file meant to fix §6. Requiring the argument makes it a decision.
- The regression harness passes `model=None`: nothing records which model wrote the
  baseline `.slp` files.

### Smaller notes

- Session discovery moved to `io/layout.py`. `find_session` raises on an ambiguous date
  where the old `sorted(...)[0]` picked the first; `.slp` discovery uses `find_outputs`,
  which rglobs (so `movement_analysis/`-grouped `.slp` resolve) and skips `._` forks.
- `_resolve_deriv_root`, `_available_sessions` and `_normalize_date_arg` were used only
  by `sleap_node_quality_report` and are **not ported** — `find_sessions(date=...)`
  already takes a value, list, comma string or `A-B` range.
- `sleap_node_quality_report` is not a byte-identical move: its per-session arithmetic
  is `quality.session_node_stats`, and the wrapper is discovery plus printing. The
  arithmetic is unchanged; it is not in the required-identical set.
- `write_report` re-reads every `.slp`, so `extract` reads each one twice — once via
  `load_slp` for the table, once via `labels.numpy()` for the stats. Accepted: the
  alternative is a second implementation of the presence counts, computed over a
  different row universe, which is a numerics change dressed as an optimisation.
- `require_local()` is **not** called in `extract.py`. It guards the active profile,
  which is meaningless when `derivatives=` is passed explicitly; it belongs on the CLI
  verb, wired with the rest of the CLI.
- `df.to_parquet(output_path, index=False)` keeps its default engine — pyarrow, in this
  env. That is exactly what Phase 0.5 measured, and §2 established the engine does not
  move the fingerprint.

## 12 — Phase 4: combine moved, and L2 measured zero

- `get_video_frame_times` + `add_timestamps_to_sleap_tracking` →
  `timestamps.combine_session`; `process_sleap_sessions` → the `run` verb in `cli.py`.
- **L2 is byte-identical on all four L2 fixtures.** Re-derived md5s `580319c1` /
  `29a770d6` / `c771fcce` / `44b17a42` equal the Phase 0 baselines exactly.
- Row counts equal the §4 sums to the row: 273,042 / 626,936 / 284,034 / 350,869. The
  join still adds columns, not rows.
- ⇒ **the delta the plan budgeted for is zero.** The pre-2.0.0 `load_all_streams` that
  wrote the saved `time` column and the 2.0.0 one agree at these sessions.
- Per the plan's pre-registered rule ("Zero → L2 becomes a byte-identity check"), L2 is
  now a **RED**, not a reported delta. Risk 3 is closed by measurement rather than by
  documentation.
- L2's input is still not frozen — `load_all_streams` lives in another repo. So the RED
  names it, and points at `fixtures/env.json`, rather than asserting the cause is here.
- L2 also carries a separate shape assertion (`check_combined_rows`): the combined row
  count must equal the sum of the per-video tables. Split from the md5 so "the join
  changed shape" and "the values moved" cannot be confused for each other.

### `combine` took four redirects, not one

- **`_find_tracking_files` globbed flat** (`sleap_utils.py:45-53`), so it found zero
  tracking files for any session Phase 3 had extracted into `movement_analysis/`. Only
  the flat baselines still resolved. `layout.find_tracking_tables` rglobs.
- **No `base_dir`** (§9): `combine_session` now takes `derivatives=` and `rawdata=`, the
  same override shape `extract_session` has. The two roots stay independent, which is
  how `hypnose_helpers` resolves them — the gate writes into a temp derivatives tree
  while reading the real rawdata.
- ⇒ §9's hazard is closed: the QC harness needs **no `HYPNOSE_*` env vars and no
  `cache_clear()`**. `_common.default_rawdata` derives the raw root from the fixture's
  own derivatives root, so the gate measures the tree `sessions.yml` names.
- **Wrote flat** (`:752-754`) → `layout.write_path`, so output is grouped like Phase 3's.
  Filename unchanged and fixture-matched.
- **A seventh copy of the behav walk** (`:608`, `:614-615`) → `layout.session_experiments`,
  derived from `session_videos` rather than from a second glob. It matched experiment
  folders by a `\d{4}-\d{2}-\d{2}T...` regex; the folders are now whatever holds video.
  Equivalent in output — a folder with no video contributes no frame times either way.

### Two output layouts can coexist, so the reader picks

- Nothing deletes, so re-extracting a baselined session leaves the flat parquets beside
  the new grouped ones and the same stem matches twice.
- `layout.find_tables` resolves it: parquet over `.csv`, then `movement_analysis/` over
  flat. Same rule `hypnose_behavior.io.layout.find_tracking_file` already applies.
- Without it `combine` would have joined each video twice and doubled the row count —
  which is exactly what `check_combined_rows` would have caught.

### Three helpers moved byte-identically

- `get_video_frame_times`, `_read_table` and `_peek_video_file` are 3/3 identical under
  `ast_move_check --only`. No AST requirement was set for this phase; they qualified.
- `get_video_frame_times` stayed identical because the lazy `hypnose_behavior` import is
  a **named module-level shim** — `timestamps.load_all_streams` — rather than an import
  statement inserted into the body. It doubles as the single import surface.
- `combine_session` is not a byte-identical move: four redirects is what the phase was.

### `fetch` must copy the whole `behav/` tree — measured, not assumed

- Nothing in this repo has ever copied rawdata. `fetch` is a Phase 5 stub; the full tree
  on `E:` was put there by something else (datashuttle, or by hand). The inference script
  only *detects* it (`run_sleap_inference_local_windows.sh:152-154`).
- ⇒ the gate passing is not evidence that the design produces the tree it needs.
- Measured on `057:20260717`: `.avi` 15,809.5 MB (**98.7 %**), `Behavior/` 87.1 MB,
  `Olfactometer0/1` 106.4 MB, `VideoData/*.csv` 11.8 MB, rest 6.7 MB — every non-video
  stream together is **212 MB, 1.3 %** of the session.
- `combine` never opens an `.avi`: `load_video` globs `VideoData/VideoData_*.csv`
  (`hypnose_behavior/io/readers.py:121-124`), and the sync needs `Behavior/`.
- An `.avi`-only fetch therefore breaks `combine` **quietly**: `load_all_streams` catches a
  missing heartbeat, prints `Failed to load heartbeat`, and returns an empty timestamp
  mapping — a combined parquet with a degraded `time` column, not an error.
- ⇒ `fetch` copies the whole `behav/<exp>/` tree. +1.3 % transfer to keep "only `fetch` and
  `push` touch the server" true.
- **Rejected: `combine` reading harp streams off the server by default.** It saves 1.3 % of
  local disk next to 15.8 GB of video already held, and costs network-dependence on the
  main loop. It would also pull ~200 MB to use ~99 MB, since `load_all_streams` is
  all-or-nothing and `get_video_frame_times` only uses `video_data` plus the heartbeat.
  Narrowing which streams it loads is not available: `get_video_frame_times` is
  byte-identical today, and changing the sync path is what L2 exists to catch.
- Kept as an override instead: `--rawdata-from <profile>` on `combine` / `run`, for
  re-combining after local `rawdata` has been deleted. The plumbing already exists —
  `combine_session(rawdata=...)` — so Phase 5 only exposes it.

### `find_tracking_file` is now asserted against a real file

- `qc/check_layout.py` writes an empty file at our `write_path` and asserts the reader
  finds it. That proves the naming agreement, not that the pipeline produces a file.
- `regression.py` now also asserts it against the **combined parquet the gate just
  re-derived**, in the temp tree (`_common.check_discovery`). Green on all four L2
  fixtures. That is the Phase 4 gate the plan asked for.

### The CLI verbs are wired, and `require_local` is finally called

- `extract`, `combine` and `run` share one session driver in `cli.py`, keeping
  `process_sleap_sessions`' message text, skip rules and success/failed/skipped tally.
  `run` prints byte-identical output to the old wrapper; the other two print the lines
  for the steps they ran.
- `--model` is on `extract` and `run`. Omitting it is legal and prints a notice: the
  quality report records `UNKNOWN_MODEL` rather than claiming the configured default.
- `require_local(verb, allow_remote=..., profile=...)` is called by all three, and gained
  a `profile` argument so `--profile server-windows` is refused for the profile it would
  actually write to, not for the active one.
- `--profile` resolves a named profile into `derivatives=` / `rawdata=` overrides. It
  never calls `set_active`: choosing a profile for one command must not rewrite the
  machine's `data_locations.local.yml`.

## 13 — Phase 5: infer, fetch, push; five scripts retired

Done 2026-08-26. Both gates GREEN, captured before any code was written.

### The gate

- **`infer --dry-run` = the old walk, exactly.** 545 videos on both sides, 0 only-in-old,
  0 only-in-new, and all 545 `.slp` leaf names identical. The only difference is the
  directory: all 545 move from flat `saved_analysis_results/` into `movement_analysis/`,
  which is the Phase 3 grouping. The baseline was captured by lifting the discovery half
  of `run_sleap_inference_local_windows.sh` verbatim into a scratch script with
  `sleap-track` removed, so it is the script's own walk, not a reimplementation of it.
  545 also matches §10's independent count.
- **`push --dry-run` = the old plan, exactly.** 296 files on both sides, 0 only-in-old,
  0 only-in-new. Every destination is under the resolved destination root, and every
  destination mirrors its source's relative path.
- Both baselines live in the session scratch directory, not committed: they are
  measurements of a tree, not fixtures of a computation.

**Capture gotcha.** The first `push` baseline read 293, not 296. `Out-File` wraps at the
host's width, and three planned paths collided once truncated at ~93 characters.
`ForEach-Object { $_.ToString() } | Out-File -Width 8192` fixes it. `Write-Host` also
bypasses the pipeline entirely — the plan only reaches a file through the information
stream (`6>&1`). A baseline that silently loses three rows is worse than no baseline.

### `infer` runs `sleap-track` as a subprocess, and neither environment moved

§7 left this open: `sleap-gpu` has torch but none of the three hypnose packages, and it
is python 3.11.14 against a `>=3.12,<3.13` pin. Both options named there were rejected:

- installing the hypnose packages into `sleap-gpu` needs the pin relaxed *and* would
  downgrade its numpy 2.2.6, because `pyproject.toml` pins `numpy<2` — a working GPU
  environment perturbed to gain a CLI;
- installing torch + sleap-nn + lightning + kornia into `hypnose-sleap` is ~3 GB into
  the exact environment every Phase 0–4 baseline was measured in.

Taken instead: **the CLI runs in `hypnose-sleap` and calls `sleap-gpu`'s
`sleap-track.exe` as a subprocess** — which is what the bash script did, so it is the
faithful port rather than a workaround. Nothing was installed, no pin changed, and
`--dry-run` runs in the baselined environment. Resolution order is
`HYPNOSE_SLEAP_TRACK` > `configs/inference.local.yml` > `sleap-track` on `PATH`.

Measured: the two environments agree on both variables that move predictions —
sleap 1.5.2 and sleap-io 0.5.7 in each. They differ only in the torch stack, numpy
(2.2.6 vs 1.26.4) and pandas (2.3.3 vs 3.0.5), none of which `sleap-track` reads.

### The float32 no-op is preserved, structurally

`:246` ran `python -c "import torch; torch.set_float32_matmul_precision('high')"` in a
**separate process**, so it never reached `sleap-track` and never moved a prediction. It
is not set in the port either. This machine is an RTX 4090 — Ada, so Ampere+ — where
setting it would enable TF32 for float32 matmul and move every prediction. That is a
re-baselining decision, not a port.

Running `sleap-track` as a subprocess makes the no-op structural rather than a comment:
there is no in-process torch for the setting to leak into. Its one working effect was
the `&&` — abort the video when torch is unimportable — and that role now belongs to
`sleap_track_executable`, which fails once, up front, naming the environment.

### `DEFAULT_MODEL` was dead, and the port says so

`:19` pointed into `sleap-models/sleap_v1.5.5_new_model/`, which does not exist, so the
script only ever worked with an explicit `-m`. The port defaults to the `naive` role in
`models.yml`, which resolves the same leaf name (`260305_143707.single_instance.n=340`)
under `sleap_v1.5.5/models/`, which does exist — confirmed by the dry-run printing the
resolved path. This fixes a dead default rather than reproducing it.

### `push` had three defects the plan comparison cannot see

The plan matched file for file while the behaviour did not. `qc/check_transfer.py`
asserts the behaviour against a temp tree, so it runs anywhere:

- **`-DryRun` wrote to ceph.** `:132-134` ran `New-Item -Force` on every destination
  directory *before* the `if ($DryRun)` at `:136`. Measured: **143 empty directories, 0
  files** created by a single dry run over the real tree. The port separates `plan_push`
  (reads only) from `run_plan` (the only writer), so a dry run creates nothing —
  asserted, not asserted-by-inspection.
- **Quality reports reached nobody.** `:7` lists both tables in both extensions but not
  `sleap_quality_sub-*.yml`, which Phase 3 added. `PUSH_PATTERNS` adds it. This does not
  disturb the gate: there are currently 0 such files on `E:`, so both sides planned 296.
  The first `extract` run after this lands them, and they will now be carried.
- **Every copy was an overwrite.** `:140` passed `Copy-Item -Force` unconditionally. The
  port refuses an existing destination unless `--force`, and `--dry-run` labels each row
  `new` / `exists (same size)` / `exists (differs)` so the refusals are visible before
  the transfer. On the real tree 3 of the 296 already exist on the server.

Not ported, deliberately: `:124`'s `Write-Host "Skipped (" + ... + "):"` passes three
arguments rather than concatenating, and `$matches` at `:80` shadows PowerShell's
automatic `$Matches` used at `:101`/`:110` — surviving only because `-match` reassigns it
first. Both are cosmetic in a file that no longer exists.

### `infer` joins the shared driver; it needed three seams, not a fork

`_run_sessions` now drives four verbs. The three seams, each expressed as data or one
guard rather than as scattered conditionals:

- `SELECT_FROM` — `infer` selects sessions from **rawdata** (videos live there); the
  others from derivatives.
- `layout.mirror_session` — a rawdata session's derivatives twin, composed from the two
  directory names because on a first run it does not exist yet. This is what the old
  `${DERIV_DIR}/${SUBJ_DIR_NAME}/${SESS_DIR_NAME}` relied on too. The `results.exists()`
  skip is therefore not applied to `infer`, which creates it.
- `_already_done` — gained an `infer` branch, and takes the session now, not just the
  results directory. `infer` is done when **every** video has a `.slp`, not when one
  does: a session that gained a video after its first run is not done. Skipping is also
  per video inside `session_plan`, so a partial session costs only what is missing —
  the old script re-ran every video every time.

`require_local` is now called by `infer` too, via the shared driver — confirmed by
`--profile server-windows infer` exiting 1 with the refusal. That closes the flag Phase 4
left on the parser with nothing calling it.

### `--rawdata-from` is a CLI-only change

`combine_session(rawdata=...)` already existed (§12), so the verb only exposes it.
`_roots` takes `args` rather than a profile name and layers `--rawdata-from` over
`--profile`, redirecting the read end alone.

### Five scripts retired, two of them already broken

`run_sleap_inference_local_windows.sh` → `inference.py` + `infer`.
`transfer_sleap_results.ps1` → `io/transfer.py` + `fetch` / `push`.

The plan's premise that the other two "differ only in the conda hook and two defaults"
held for `run_sleap_inference_local.sh` but **not** for `run_sleap_inference.sh`, which
could not have run as written:

- it passes `--batch-size 1` and `--peak-threshold 0.5`; `sleap-track` accepts
  `--batch_size` and `--peak_threshold`, with underscores;
- its `basename $(dirname ...)` chain is one level short, so `SUBJ_DIR` resolves to the
  *session* directory and `SESS_DIR` to `behav` — it would write to
  `derivatives/ses-NNN_date-YYYYMMDD/behav/saved_analysis_results`;
- it writes `${BASENAME%.*}.predictions.slp` with no `<behav>__` prefix, so two videos
  from different `behav/` folders collide.

`submit_sleap_inference.sh` was not in the plan's delete list but sources
`run_sleap_inference.sh` at `:8` and `:50`, so deleting one orphans the other. Both went:
the HPC path loads `module load SLEAP/2024-08-14`, a different SLEAP entirely, where
`hypnose-sleap` is not installed — repointing it would have produced a script that still
could not work. Restoring the cluster path is fresh work, not a port, and both files
remain in git history.

`README.md` documented only the deleted scripts and was rewritten around the verbs.

### Carried into Phase 6

`annotate` has no `--allow-remote` and does not call `require_local`, so it will render
an `.mp4` onto ceph if that is the active profile. It is the one unguarded writer left.
