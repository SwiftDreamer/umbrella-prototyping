"""Headless render: runs a scene module's Isaac Lab environment and writes an mp4.

One-shot (starts Isaac, renders, exits):
    /workspace/isaaclab/isaaclab.sh -p /project/render.py \
        /project/scenes/drone-example.py --out /project/out.mp4

worker.py keeps Isaac running and calls render() per request instead.

A scene module defines `EnvCfg` (a DirectRLEnvCfg), `Env` (the DirectRLEnv built from
it) and optionally `policy(env, obs) -> actions`; without a policy every action is zero.
Recording uses Isaac Lab's own path, as `isaaclab play --video` does: a headless Kit
visualizer is the camera, aimed by the env's `sim.default_visualizer_cfg`, and a
VideoRecorderCfg writes the clip. The recorder needs MoviePy and imageio-ffmpeg, which
the image leaves out; `simproto render` installs them into /project/.packages.
"""

import argparse
import importlib
import importlib.util
import pathlib
import shutil
import sys
import tempfile
from types import ModuleType

PROJECT = pathlib.Path(__file__).resolve().parent
# Kit runs headless with camera rendering; `video` keeps the viewport rendering that the
# Kit visualizer captures and fails at startup if MoviePy is missing
LAUNCH_ARGS = {
    "headless": True,
    "enable_cameras": True,
    "video": True,
    "require_kit": True,
}
# env steps before recording, so the RTX renderer has warmed up (as in Isaac Lab's
# video recording tutorial)
WARMUP_STEPS = 5


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(exit_on_error=False)
    result.add_argument("scene", help="path to a .py file defining EnvCfg and Env")
    result.add_argument("--out", default="/project/out.mp4")
    result.add_argument("--seconds", type=float, default=3.0)
    result.add_argument("--width", type=int, default=640)
    result.add_argument("--height", type=int, default=480)
    result.add_argument(
        "--eye",
        type=float,
        nargs=3,
        help="camera position; overrides the env's default view",
    )
    result.add_argument(
        "--target", type=float, nargs=3, help="point the camera looks at"
    )
    return result


def is_project_code(location: str) -> bool:
    """True for the project's own files, but not packages under /project/.packages."""
    path = pathlib.Path(location).resolve()
    if not path.is_relative_to(PROJECT):
        return False
    top = path.relative_to(PROJECT).parts[:1]
    return not (top and top[0].startswith("."))


def load_scene(path: str) -> ModuleType:
    """Imports the scene fresh, so a long-running worker picks up edited files."""
    # drop project modules from earlier requests, e.g. helpers a scene imports. This
    # includes packages: `scenes/` has no __init__.py (no __file__, only __path__),
    # and a cached package would hand back its old submodules as attributes.
    for name, module in list(sys.modules.items()):
        if name in ("__main__", __name__):
            continue
        locations = [getattr(module, "__file__", None)]
        try:  # some third-party modules set __path__ to non-path objects
            locations += [str(p) for p in getattr(module, "__path__", None) or []]
        except TypeError:
            pass
        if any(isinstance(loc, str) and is_project_code(loc) for loc in locations):
            del sys.modules[name]
    # Compile from source every time. Right after rsync replaces a file, the GPU
    # node's NFS cache can still report the old file's mtime and size, which makes
    # Python reuse the old __pycache__ bytecode. An empty cache dir avoids that.
    if sys.pycache_prefix:
        shutil.rmtree(sys.pycache_prefix, ignore_errors=True)
    sys.pycache_prefix = tempfile.mkdtemp(prefix="scene-pycache-")
    importlib.invalidate_caches()  # see files created since the last request
    spec = importlib.util.spec_from_file_location("scene", path)
    assert spec is not None and spec.loader is not None, f"cannot load {path}"
    scene = importlib.util.module_from_spec(spec)
    # registered before it runs, as import does: dataclasses (configclass) look up the
    # defining module in sys.modules
    sys.modules[spec.name] = scene
    spec.loader.exec_module(scene)
    return scene


def render(args: argparse.Namespace) -> int:
    """Renders one scene on a fresh stage; needs Kit running (see main())."""
    import torch
    from isaaclab.app import validate_launch
    from isaaclab.envs.utils.video_recorder_cfg import VideoRecorderCfg
    from isaaclab.sim import create_new_stage
    from isaaclab_visualizers.kit import KitVisualizerCfg

    # isolate requests: the env builds on the current stage, so start an empty one
    create_new_stage()
    scene = load_scene(args.scene)
    if not hasattr(scene, "EnvCfg") or not hasattr(scene, "Env"):
        raise TypeError(
            f"{args.scene} must define EnvCfg and Env, see scenes/drone-example.py"
        )
    env_cfg = scene.EnvCfg()

    # the camera: a headless Kit viewer, which takes eye/lookat from the env's
    # sim.default_visualizer_cfg unless --eye/--target set them here
    viewer = KitVisualizerCfg(
        headless=True, window_width=args.width, window_height=args.height
    )
    if args.eye:
        viewer.eye = tuple(args.eye)
    if args.target:
        viewer.lookat = tuple(args.target)
    env_cfg.sim.visualizer_cfgs = [viewer]

    # one frame per env step; the recorder plays them at 1 / step_dt, i.e. real time
    steps = max(1, round(args.seconds / (env_cfg.sim.dt * env_cfg.decimation)))
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    clips = pathlib.Path(tempfile.mkdtemp(prefix=".clips-", dir=out.parent))
    env_cfg.video_recorders = [
        VideoRecorderCfg(
            source="visualizer:kit",
            output_dir=str(clips),
            video_length=steps,
            step_offset=WARMUP_STEPS,
        )
    ]
    # resolves the config as launch_simulation would, and rejects runtimes (e.g. OVRTX)
    # that can't share this Kit process
    validate_launch(env_cfg, dict(LAUNCH_ARGS))

    try:
        env = scene.Env(env_cfg)
        try:
            policy = getattr(scene, "policy", None)
            obs, _ = env.reset()
            with torch.inference_mode():
                for _ in range(WARMUP_STEPS + steps):
                    if policy is None:
                        actions = torch.zeros(env.action_space.shape, device=env.device)
                    else:
                        actions = policy(env, obs)
                    obs, *_ = env.step(actions)
        finally:
            env.close()  # also flushes the recorder
        # the recorder logs write errors instead of raising them
        clip = clips / "clip_0000.mp4"
        if not clip.exists():
            raise RuntimeError(
                "the video recorder wrote no clip; see its [VideoRecorder] log lines"
            )
        clip.replace(out)
    finally:
        shutil.rmtree(clips, ignore_errors=True)
    return steps


def main() -> None:
    args = parser().parse_args()
    # Kit must start before anything imports omni/isaacsim modules
    from isaaclab.app import launch_simulation

    with launch_simulation(None, dict(LAUNCH_ARGS)):
        count = render(args)
        print(f"[render] wrote {count} frames to {args.out}", flush=True)


if __name__ == "__main__":
    main()
