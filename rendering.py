"""Renders project scenes headlessly in a session's job; see the README.

Three places take part:

- your machine (client): mirrors the local project into the session's project
  directory, asks the login node to render, keeps the log and downloads the clip;
- the login node (run, start_worker, stop_worker, fetch): starts project/worker.py,
  which keeps Isaac loaded in the job, or runs project/render.py once;
- the GPU node (queue): hands a request to the worker and waits for its result
  there, because the login node's NFS cache can hide new files for up to ~20 s.

Containers start through runtime.py, so they get the session's container setup.
"""

from __future__ import annotations

import dataclasses
import json
import pathlib
import re
import subprocess
import sys
import tempfile
import time
import uuid
from typing import Callable

import projectsync

ISAACLAB = "/workspace/isaaclab/isaaclab.sh"
HEARTBEAT_AGE = 30  # seconds; an older worker.alive means the worker is gone
# MoviePy and its ffmpeg backend for Isaac Lab's video recorder, which the image
# leaves out. --no-deps: the image has their other dependencies (numpy, imageio,
# pillow, requests, tqdm), and copies in .packages would shadow them.
VIDEO_PACKAGES = (
    "moviepy==1.0.3",
    "decorator>=4.0.2,<5",
    "proglog<=1.0.0",
    "imageio-ffmpeg>=0.2.0",
)
NAME = re.compile(r"[A-Za-z0-9_.-]+")


@dataclasses.dataclass
class Job:
    """A session's running allocation, seen from the login node."""

    root: pathlib.Path  # simproto installation
    session: pathlib.Path
    job: str
    project: pathlib.Path  # mounted at /project

    @property
    def queue(self) -> pathlib.Path:
        return self.project / ".queue"

    @property
    def renders(self) -> pathlib.Path:
        return self.project / "renders"

    def container(self, command: list[str]) -> list[str]:
        """Runs command in the session's container setup, beside the session."""
        runtime = str(self.root / "runtime.py")
        return [*self.step(), runtime, str(self.session), "--", *command]

    def node(self, arguments: list[str]) -> list[str]:
        """Runs a render step of this launcher on the job's node."""
        launcher = str(self.root / "simproto")
        return [*self.step(), launcher, "--remote", "_render", *arguments]

    def step(self) -> list[str]:
        return ["srun", f"--jobid={self.job}", "--overlap", "python3"]


def alive(queue: pathlib.Path) -> bool:
    """True while the worker touches its heartbeat file."""
    try:
        return time.time() - (queue / "worker.alive").stat().st_mtime < HEARTBEAT_AGE
    except FileNotFoundError:
        return False


def install_video_packages(job: Job) -> None:
    """Installs MoviePy into the project's .packages once."""
    if (job.project / ".packages/moviepy").is_dir():
        return
    print("[render] installing MoviePy into the project's .packages (once)", flush=True)
    subprocess.run(
        job.container(
            [ISAACLAB, "-p", "-m", "pip", "install", "--quiet", "--no-deps"]
            + ["--target", "/project/.packages", *VIDEO_PACKAGES]
        ),
        check=True,
    )


def start_worker(job: Job) -> int:
    """Starts project/worker.py in the job unless it runs, and waits until it loads."""
    job.queue.mkdir(exist_ok=True)
    if alive(job.queue):
        print("[render] worker already running", flush=True)
    else:
        install_video_packages(job)
        for name in ("stop", "worker.alive", "worker.ready"):
            (job.queue / name).unlink(missing_ok=True)
        job.renders.mkdir(exist_ok=True)
        with (job.renders / "worker.log").open("w") as log:
            # its own session keeps it running after this SSH connection closes
            subprocess.Popen(
                job.container([ISAACLAB, "-p", "/project/worker.py"]),
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=log,
                start_new_session=True,
            )
        # the heartbeat starts with Python, before Isaac loads
        deadline = time.time() + 120
        while not alive(job.queue) and time.time() < deadline:
            time.sleep(2)
    # loading takes about a minute with a warm shader cache, minutes on a cold node
    while not (job.queue / "worker.ready").exists():
        if not alive(job.queue):
            print(
                f"[render] worker exited during startup; see {job.renders}/worker.log",
                file=sys.stderr,
            )
            return 4
        time.sleep(2)
    print("[render] worker ready", flush=True)
    return 0


def stop_worker(project: pathlib.Path) -> int:
    (project / ".queue").mkdir(exist_ok=True)
    (project / ".queue/stop").touch()
    print("[render] worker asked to stop", flush=True)
    return 0


def run(job: Job, scene: str, name: str, options: list[str]) -> int:
    """Renders project/scene to renders/<name>.mp4, through the worker if one runs."""
    projectsync.check_path(scene)
    if not NAME.fullmatch(name):
        raise ValueError(f"Invalid render name: {name!r}")
    job.renders.mkdir(exist_ok=True)
    # a failed render must not leave the previous clip behind to be downloaded
    (job.renders / f"{name}.mp4").unlink(missing_ok=True)
    install_video_packages(job)
    arguments = [f"/project/{scene}", "--out", f"/project/renders/{name}.mp4", *options]
    if not alive(job.queue):
        print(
            "[render] no worker; starting Isaac for this render "
            "(render --worker avoids this)",
            flush=True,
        )
        return subprocess.call(
            job.container([ISAACLAB, "-p", "/project/render.py", *arguments])
        )
    if not (job.queue / "worker.ready").exists():
        print("[render] worker still starting; request queued", flush=True)
    request = json.dumps({"args": arguments})
    identifier = f"{name}-{uuid.uuid4().hex[:8]}"
    result = subprocess.run(
        job.node(["queue", str(job.project), identifier, request]),
        stdout=subprocess.PIPE,
        text=True,
    )
    if result.returncode:
        print(
            f"[render] worker exited; see {job.renders}/worker.log",
            file=sys.stderr,
        )
        return 5
    data = json.loads(result.stdout)
    sys.stdout.write(data.get("output", ""))  # the scene's print() output
    if not data["ok"]:
        print(data["error"], file=sys.stderr)
        return 1
    print(
        "[render] wrote %(frames)d frames to %(out)s in %(seconds).1fs (worker)" % data,
        flush=True,
    )
    return 0


def queue(project: pathlib.Path, identifier: str, request: str) -> int:
    """On the GPU node: hands a request to the worker and prints its result."""
    if not NAME.fullmatch(identifier):
        raise ValueError(f"Invalid request ID: {identifier!r}")
    directory = project / ".queue"
    temporary = directory / f"{identifier}.tmp"
    temporary.write_text(request)
    temporary.rename(directory / f"{identifier}.request.json")
    result = directory / f"{identifier}.result.json"
    while not result.exists():
        if not alive(directory):
            (directory / f"{identifier}.request.json").unlink(missing_ok=True)
            return 5
        time.sleep(0.1)
    sys.stdout.write(result.read_text())
    result.unlink()
    return 0


def fetch(project: pathlib.Path, name: str) -> int:
    """Writes renders/<name>.mp4 to stdout once the login node sees it."""
    if not NAME.fullmatch(name):
        raise ValueError(f"Invalid render name: {name!r}")
    path = project / "renders" / f"{name}.mp4"
    # written on the GPU node; the login node's NFS cache may not list it yet
    deadline = time.time() + 30
    while not path.is_file():
        if time.time() > deadline:
            raise ValueError(f"{path} does not exist")
        time.sleep(1)
    with path.open("rb") as clip:
        while chunk := clip.read(1 << 20):
            sys.stdout.buffer.write(chunk)
    sys.stdout.buffer.flush()
    return 0


def client(
    remote: Callable[[list[str]], list[str]],
    project: pathlib.Path,
    session: str | None,
    scene: str | None,
    options: list[str],
    worker: bool = False,
    stop: bool = False,
) -> int:
    """On your machine: mirrors the project, renders and downloads the clip.

    remote turns launcher arguments into an SSH command that runs them on Umbrella.
    """

    def step(name: str, *arguments: str) -> list[str]:
        # --session goes before the step: everything after it is the step's arguments
        selection = ["--session", session] if session else []
        return remote(["_render", *selection, name, *arguments])

    if stop:
        return subprocess.call(step("stop"))
    project = project.resolve()
    if not project.is_dir():
        raise ValueError(f"Local project directory does not exist: {project}")
    relative = None
    if not worker:
        if scene is None:
            raise ValueError("Name a scene file, or use --worker or --stop-worker")
        path = pathlib.Path(scene).resolve()
        if not path.is_file() or path.suffix != ".py":
            raise ValueError(f"Scene must be a .py file: {scene}")
        if not path.is_relative_to(project):
            raise ValueError(f"Scene must be inside {project}")
        relative = path.relative_to(project).as_posix()

    with subprocess.Popen(
        step("sync"),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
    ) as process:
        assert process.stdin is not None and process.stdout is not None
        try:
            summary = projectsync.push(project, process.stdin, process.stdout)
        except (OSError, RuntimeError):
            process.kill()
            raise RuntimeError(
                "Could not mirror the project; see the message above. If it says "
                "invalid choice: '_render', run install to update the launcher."
            ) from None
    if process.returncode:
        raise RuntimeError("Could not mirror the project; see the message above")
    print(
        f"[render] mirrored the project: {summary['sent']} file(s) sent, "
        f"{summary['deleted']} deleted",
        flush=True,
    )
    if relative is None:
        return subprocess.call(step("worker"))

    name = relative.removesuffix(".py").replace("/", "__")
    renders = project / "renders"
    renders.mkdir(exist_ok=True)
    log = renders / f"{name}.log"
    # the full output goes to the log; the [render] lines are shown as they arrive
    command = step("run", relative, name, *options)
    with (
        log.open("wb") as output,
        subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
        ) as process,
    ):
        assert process.stdout is not None
        for line in process.stdout:
            output.write(line)
            output.flush()
            if line.startswith(b"[render]"):
                print(line.decode(errors="replace").rstrip(), flush=True)
    text = log.read_text(errors="replace")
    code = process.returncode
    # one-shot Isaac can exit 0 although the scene raised, so require the result line
    if not code and "[render] wrote" not in text:
        code = 1
    if code:
        print(f"--- failed (exit {code}); last lines of {log}:", file=sys.stderr)
        print("\n".join(text.splitlines()[-25:]), file=sys.stderr)
        return code

    clip = renders / f"{name}.mp4"
    with tempfile.NamedTemporaryFile(dir=renders, suffix=".part", delete=False) as part:
        code = subprocess.call(step("fetch", name), stdout=part)
    partial = pathlib.Path(part.name)
    if code:
        partial.unlink()
        return code
    partial.replace(clip)
    print(clip)
    return 0
