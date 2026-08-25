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
