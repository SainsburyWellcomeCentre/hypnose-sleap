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
  and the md5 of its `training_config.json`.
