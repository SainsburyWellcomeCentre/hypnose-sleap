"""Video enumeration and ``sleap-track``.

- Walks ``rawdata/sub-*/ses-*/behav/*/VideoData/*.avi`` through `layout.session_videos`,
  the one implementation of that walk.
- Resolves the model from a role name (`parameters.py`) or an explicit path.
- Writes ``movement_analysis/<behav>__<video>.predictions.slp``, the ``<behav>__``
  prefix keeping identical basenames from different behav folders apart.
- A video whose ``.slp`` already exists is skipped unless ``--recompute``.
- ``--dry-run`` lists the videos it would process and exits, touching nothing.

`sleap-track` runs as a subprocess, exactly as it did under bash. The backend it needs
(torch, sleap-nn, lightning, kornia) lives in the ``sleap-gpu`` environment, not in
``hypnose-sleap``; `sleap_track_executable` resolves which one to call, so the CLI runs
in the analysis environment and only inference crosses over.

Two things the old script did that are deliberately not reproduced:

- ``run_sleap_inference_local_windows.sh:246`` ran
  ``python -c "import torch; torch.set_float32_matmul_precision('high')"`` in a
  *separate* process, so it never reached ``sleap-track`` and never changed a
  prediction. It is not set here either. Setting it would enable TF32 for float32
  matmul on Ampere and later -- this machine is an RTX 4090 -- and move every
  prediction, which is a re-baselining decision, not a port (`DECISIONS.md` §13).
  Its only working effect was to abort the video when torch was unimportable; that
  role now belongs to `sleap_track_executable`, which fails once, up front, naming
  the environment.
- ``:19``'s ``DEFAULT_MODEL`` pointed into ``sleap-models/sleap_v1.5.5_new_model/``,
  which no longer exists, so the script only ran with an explicit ``-m``. The default
  here is the ``naive`` role from ``models.yml``, which resolves the same leaf name
  under ``sleap_v1.5.5/models/``. This fixes a dead default rather than reproducing it.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Optional

from hypnose_helpers.io.paths import read_yaml

from hypnose_sleap import parameters
from hypnose_sleap.io import layout, paths

# Per-machine location of the GPU environment's `sleap-track`, git-ignored like the
# other `.local.yml` files.
INFERENCE_LOCAL_FILENAME = "inference.local.yml"

# The env var that overrides it, following the `HYPNOSE_*` convention in `io/paths.py`.
SLEAP_TRACK_ENV = "HYPNOSE_SLEAP_TRACK"

SLP_SUFFIX = ".predictions.slp"


def slp_name(video: Path) -> str:
    """The ``.slp`` filename for one video: `layout.video_key` with the extension swapped.

    ``<behav>__<video>.avi`` -> ``<behav>__<video>.predictions.slp``, which is what
    `extract.infer_video_file_from_slp` reads back.
    """
    key = layout.video_key(video)
    return key[: -len(".avi")] + SLP_SUFFIX if key.endswith(".avi") else key + SLP_SUFFIX


def sleap_track_executable() -> Path:
    """The ``sleap-track`` to run, or raise naming how to configure one.

    Resolution order: ``HYPNOSE_SLEAP_TRACK`` > ``sleap_track`` in
    ``configs/inference.local.yml`` > ``sleap-track`` on ``PATH``. The last is what an
    already-activated ``sleap-gpu`` shell gets for free.

    The executable is not probed for a working backend -- that costs a torch import per
    call. A missing backend surfaces as the first video's failure, with sleap's own
    message.
    """
    override = paths._env_path(SLEAP_TRACK_ENV)
    if override:
        candidate = Path(override)
        if not candidate.exists():
            raise SystemExit(f"{SLEAP_TRACK_ENV} points at a missing file: {candidate}")
        return candidate

    configured = read_yaml(paths.get_config_dir() / INFERENCE_LOCAL_FILENAME).get("sleap_track")
    if configured:
        candidate = Path(str(configured))
        if not candidate.exists():
            raise SystemExit(
                f"`sleap_track` in {paths.get_config_dir() / INFERENCE_LOCAL_FILENAME} "
                f"points at a missing file: {candidate}"
            )
        return candidate

    found = shutil.which("sleap-track")
    if found:
        return Path(found)

    raise SystemExit(
        "no `sleap-track` found. Inference needs the GPU environment (torch, sleap-nn, "
        "lightning, kornia), which `hypnose-sleap` does not carry.\n"
        f"  Point at it in {paths.get_config_dir() / INFERENCE_LOCAL_FILENAME}:\n"
        "      sleap_track: 'C:\\Users\\<you>\\.conda\\envs\\sleap-gpu\\Scripts\\sleap-track.exe'\n"
        f"  or set {SLEAP_TRACK_ENV}, or run from an activated sleap-gpu shell."
    )


def session_plan(session, results, *, recompute: bool = False) -> list:
    """``[(video, slp_path)]`` for one raw session, in `layout.session_videos` order.

    That order is what numbers the downstream ``sleap_tracking_video<N>`` outputs, so it
    is the walk order rather than an accident of the filesystem. Videos whose ``.slp``
    already exists are dropped unless ``recompute`` -- the old script re-ran every video
    every time.

    The ``.slp`` path is computed, not created: planning never touches disk.
    """
    plan = []
    for video in layout.session_videos(session):
        destination = layout.movement_dir(results) / slp_name(video)
        existing = layout.find_outputs(results, slp_name(video))
        if existing and not recompute:
            continue
        plan.append((video, destination))
    return plan


def track_video(video: Path, destination: Path, *, model: Path, batch_size: int,
                executable: Optional[Path] = None) -> bool:
    """Run ``sleap-track`` over one video. True when it succeeded.

    The argument order is ``run_sleap_inference_local_windows.sh:247`` unchanged:
    ``sleap-track <video> -m <model> -o <out> --batch_size <n>``. Note the underscore --
    ``--batch-size`` is not a flag sleap-track accepts, which is one of the reasons
    ``run_sleap_inference.sh`` could not have worked.
    """
    executable = Path(executable or sleap_track_executable())
    destination.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(executable), str(video),
        "-m", str(model),
        "-o", str(destination),
        "--batch_size", str(batch_size),
    ]
    return subprocess.run(command).returncode == 0


def infer_session(session, results, *, model=None, batch_size=None,
                  recompute: bool = False, executable: Optional[Path] = None) -> dict:
    """Run inference over one session's videos, returning per-video outcomes.

    Never raises for a failed video: the old script reported each failure and carried on
    to the next, because one unreadable file should not cost the rest of the night.
    """
    model_path = parameters.resolve_model(model)
    batch = parameters.resolve("batch_size", batch_size)
    executable = Path(executable or sleap_track_executable())

    plan = session_plan(session, results, recompute=recompute)
    outputs, failed = [], []
    for video, destination in plan:
        print(f"-> Processing: {video}")
        if track_video(video, destination, model=model_path, batch_size=batch,
                       executable=executable):
            print(f"   Completed: {video.name} -> {destination}")
            outputs.append(destination)
        else:
            print(f"   FAILED: {video.name}")
            failed.append(video)

    return {"planned": len(plan), "outputs": outputs, "failed": failed,
            "model": model_path, "batch_size": batch}


__all__ = [
    "INFERENCE_LOCAL_FILENAME", "SLEAP_TRACK_ENV", "SLP_SUFFIX",
    "slp_name", "sleap_track_executable", "session_plan", "track_video", "infer_session",
]
