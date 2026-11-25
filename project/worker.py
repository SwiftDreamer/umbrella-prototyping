"""Keeps Isaac loaded and renders requests dropped into /project/.queue.

Started by `simproto render --worker`. Protocol, all files in .queue/:
  <id>.request.json   {"args": [render.py arguments]}; written atomically by the
                      client
  <id>.result.json    {"ok": true, "frames": N, "out": path} or
                      {"ok": false, "error": traceback}
  worker.alive        touched every few seconds by a thread while the process runs,
                      including during startup and long renders; the client checks
                      its age
  worker.ready        written once Isaac has loaded
  stop                ask the worker to exit
Requests sent while the worker is still starting wait in the queue. The worker exits
after WORKER_IDLE seconds (default 1800) without requests, so a forgotten worker stops
occupying the GPU. Restart it after editing render.py or worker.py; scene files are
reloaded on every request.
"""

import contextlib
import io
import json
import os
import pathlib
import sys
import threading
import time
import traceback
from typing import TextIO

QUEUE = pathlib.Path("/project/.queue")
IDLE = float(os.environ.get("WORKER_IDLE", 1800))
HEARTBEAT = 5.0
ALIVE = QUEUE / "worker.alive"
READY = QUEUE / "worker.ready"


def beat(stop: threading.Event) -> None:
    # a thread, because Isaac's main loop can block for minutes (shader compilation)
    while not stop.wait(HEARTBEAT):
        ALIVE.touch()


QUEUE.mkdir(exist_ok=True)
READY.unlink(missing_ok=True)
ALIVE.touch()
stopping = threading.Event()
threading.Thread(target=beat, args=(stopping,), daemon=True).start()

import render  # noqa: E402  (stdlib only at import; Isaac imports happen per request)


class Tee(io.TextIOBase):
    """Copies writes to the worker log and keeps them for the request's result."""

    def __init__(self, stream: TextIO) -> None:
        self.stream = stream
        self.buffer_ = io.StringIO()

    def write(self, text: str) -> int:
        self.stream.write(text)
        return self.buffer_.write(text)

    def flush(self) -> None:
        self.stream.flush()


def handle(request: pathlib.Path) -> None:
    stem = request.name.removesuffix(".request.json")
    data = json.loads(request.read_text())
    request.unlink()
    started = time.time()
    # the scene's print() output goes back to the client's local .log as well
    output = Tee(sys.stdout)
    try:
        with contextlib.redirect_stdout(output):
            args = render.parser().parse_args(data["args"])
            frames = render.render(args)
        result = {
            "ok": True,
            "frames": frames,
            "out": args.out,
            "seconds": time.time() - started,
        }
    except (Exception, SystemExit):  # a broken scene must not kill the worker
        result = {"ok": False, "error": traceback.format_exc()}
    result["output"] = output.buffer_.getvalue()
    temporary = QUEUE / f"{stem}.result.tmp"
    temporary.write_text(json.dumps(result))
    temporary.rename(QUEUE / f"{stem}.result.json")
    print(f"[worker] {stem}: {'ok' if result['ok'] else 'failed'}", flush=True)


def main() -> None:
    from isaaclab.app import launch_simulation

    # Kit stays up for the worker's lifetime; each request builds and closes its own env
    with launch_simulation(None, dict(render.LAUNCH_ARGS)):
        import omni.kit.app

        app = omni.kit.app.get_app()
        READY.touch()
        print("[worker] ready", flush=True)
        last_request = time.time()
        while time.time() - last_request < IDLE and not (QUEUE / "stop").exists():
            requests = sorted(
                QUEUE.glob("*.request.json"), key=lambda p: p.stat().st_mtime
            )
            if requests:
                handle(requests[0])
                last_request = time.time()
            else:
                app.update()  # keep Kit's event loop serviced while idle
                time.sleep(0.2)
        stopping.set()
        for path in (ALIVE, READY, QUEUE / "stop"):
            path.unlink(missing_ok=True)
        print("[worker] exiting", flush=True)


main()
