# hypnose-sleap

SLEAP pose tracking for hypnose experiments: inference, centroid extraction, timestamp
alignment and annotated video, behind one command line.

```
hypnose-sleap fetch    remote rawdata behav/ tree     -> local rawdata
hypnose-sleap infer    local .avi + a model role      -> .predictions.slp
hypnose-sleap extract  .slp                           -> per-video parquet + quality .yml
hypnose-sleap combine  per-video parquet + harp       -> combined parquet
hypnose-sleap run      extract + combine, one process
hypnose-sleap push     local derivatives              -> remote derivatives
hypnose-sleap annotate .avi + combined parquet        -> annotated .mp4
```

The whole loop runs on local disk: `fetch` -> `infer` -> `run` -> `push`. Only `fetch`
and `push` touch the server, and the writing verbs refuse a profile marked
`remote: true` without `--allow-remote`. Nothing deletes anything.

Every verb takes the same selectors, and every verb takes `--dry-run`. A session can be
named three interchangeable ways, each accepting one value, a comma list, or an
inclusive `A-B` range:

| flag | means | examples |
| --- | --- | --- |
| `-s` / `--subject` | subject | `57`, `057`, `sub-057`, `57,58` |
| `-d` / `--date` | session date | `20260717`, `20260717,20260718`, `20260601-20260630` |
| `--ses` | the number on the directory — stable, what you quote in a lab book | `12`, `12,14`, `12-20` |
| `--index` | the subject's gap-free chronological rank, 1..N — comparable across animals recorded months apart | `3`, `1,2,3`, `1-9` |

Supplying several intersects them: `--ses 12-20 --index 1-9` is the sessions that are
both. `--index` is ranked within the tree being read, so `fetch` counts the server's
sessions and `push` the local ones.

## Quick start

```bash
conda env create -f environment.yml
conda activate hypnose-sleap
hypnose-set-data-location local_1     # or server-windows; --list to see them
hypnose-sleap --help
```

`configs/data_locations.yml` holds the profiles and the `transfer:` endpoints;
`configs/models.yml` maps model roles to directories; `configs/parameters.yml` holds the
pipeline defaults every saved baseline was produced with. The three `*.local.yml` files
beside them are per-machine and git-ignored.

## Inference

`infer` runs `sleap-track` as a subprocess. Its backend (torch, sleap-nn, lightning,
kornia) lives in the `sleap-gpu` environment, not in `hypnose-sleap`, so point at it
once in `configs/inference.local.yml`:

```yaml
sleap_track: 'C:\Users\HarrisLab\.conda\envs\sleap-gpu\Scripts\sleap-track.exe'
```

`HYPNOSE_SLEAP_TRACK` overrides it, and `sleap-track` on `PATH` is the fallback — which
is what an already-activated `sleap-gpu` shell gets for free.

```bash
hypnose-sleap infer -s 57 --dry-run           # every video, and where its .slp lands
hypnose-sleap infer -s 57 -d 20260717         # one session
hypnose-sleap infer -s 57 -m eeg_headstage    # a model role, or an explicit path
hypnose-sleap infer -s 57 -bz 32              # override the configured batch size
hypnose-sleap infer -s 57 --recompute         # re-run videos that already have a .slp
```

Output goes to `<derivatives>/sub-XXX_id-YYY/ses-NNN_date-YYYYMMDD/saved_analysis_results/movement_analysis/<behav>__<video>.predictions.slp`.
The `<behav>__` prefix keeps identical video basenames from different `behav/` folders
apart, which happens whenever a session was restarted within the same hour. Videos that
already have a `.slp` are skipped unless `--recompute`.

The model defaults to the `naive` role in `configs/models.yml` rather than to a path.

## Transfer

`fetch` copies the whole `behav/<exp>/` tree, not the `.avi` alone: `combine` reads
`VideoData/*.csv` and the harp streams under `Behavior/`, and their absence degrades the
`time` column silently instead of raising. The extra streams are 1.3 % of a session.

`push` copies the per-video tables, the combined table and the quality report. Both
plan before they write, so `--dry-run` creates nothing, and `push` refuses to overwrite
an existing destination without `--force`.

```bash
hypnose-sleap fetch --dry-run -s 57 -d 20260717
hypnose-sleap fetch -s 57 -d 20260717
hypnose-sleap push --dry-run                       # the whole local derivatives tree
hypnose-sleap push -s 45,46,47
hypnose-sleap push -d 20260213,20260217-20260220
hypnose-sleap push -s 45 -d 20260213-20260220 --force
```

Endpoints come from the `transfer:` block in `configs/data_locations.yml`; `--from` and
`--to` override either end by profile name.

## Extract, combine, run

```bash
hypnose-sleap run -s 57 -m eeg_headstage        # extract then combine
hypnose-sleap extract -s 57 --recompute
hypnose-sleap combine -s 57 --rawdata-from server-windows
```

`--model` is optional but never guessed: omitting it records the quality report's
provenance as unknown rather than claiming the configured default wrote the `.slp`.

`--rawdata-from <profile>` reads the harp streams from another profile — for
re-combining after local `rawdata` has been deleted. It is never the default: the main
loop stays off the network.

## Annotate

Centroid overlay, the active odour near the poke port, and reward markers after each
supply-port pulse.

```bash
hypnose-sleap annotate --dry-run -s 57
hypnose-sleap annotate -s 57 -d 20260717 --rotate 90 --window 0:05:00-0:07:00
hypnose-sleap annotate -s 57 -d 20260717 --video 1 --video 2 --mark 11:33:11
```

`--window` is relative to the video's first frame and repeatable; each window writes its
own `.mp4` beside the session's tables. A re-render overwrites a clip of the same name.

The overlay font is resolved, not hard-coded: the platform font directories are searched
for Arial, DejaVu, Liberation, FreeSans or Noto, then matplotlib's bundled DejaVuSans,
then Pillow's default. Set `overlay_font` in `configs/parameters.yml` or
`HYPNOSE_OVERLAY_FONT` only to pin a particular face.

## Convert videos from .avi to .mp4 for SLEAP (model training and labeling)

On the Analysis PC, PowerShell loads the conversion script via the PowerShell profile.

- To configure or change the path, run `notepad $PROFILE` and update the path to the
  file. Refresh with `. $PROFILE` or restart PowerShell.
- Once configured, the function is available from anywhere in PowerShell.
- Run `convert-avitomp4 "file\path\to\file.avi"`. Works with single or multiple files.
- Outputs are written to `E:\videos_sleap_models`.
- Encoding uses NVIDIA NVENC (h264_nvenc, preset p7, CQ18).

## Quality control

```bash
python -m hypnose_sleap.qc.regression        # L1/L2 byte-identity against the baselines
python -m hypnose_sleap.qc.check_layout      # hypnose-behavior can find what we write
python -m hypnose_sleap.qc.check_transfer    # dry-run is read-only, overwrites need --force
python -m hypnose_sleap.qc.ast_move_check    # the moved numerics are provably a move
python -m hypnose_sleap.qc.check_annotate OLD.mp4 NEW.mp4   # two renders draw the same overlay
```

`docs/restructure-plan.md` is the plan; `docs/DECISIONS.md` records what was measured
and why each choice was made.
