"""Tests render's argument forwarding and the worker queue handshake."""

import importlib.machinery
import importlib.util
import io
import json
import os
import pathlib
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock

import rendering

loader = importlib.machinery.SourceFileLoader(
    "render_launcher", str(pathlib.Path(__file__).resolve().parents[1] / "simproto")
)
spec = importlib.util.spec_from_loader(loader.name, loader)
launcher = importlib.util.module_from_spec(spec)
loader.exec_module(launcher)


class ArgumentTest(unittest.TestCase):
    def test_scene_options_reach_render_py_unchanged(self) -> None:
        options = ["--seconds", "4", "--eye", "1", "-1", "1.1", "--session", "x"]
        args = launcher.parser().parse_args(
            ["render", "--session", "abc", "project/scenes/drone.py", *options]
        )
        self.assertEqual((args.session, args.scene), ("abc", "project/scenes/drone.py"))
        self.assertEqual(args.options, options)

    def test_client_steps_parse_back_with_session_and_options(self) -> None:
        sent: list[list[str]] = []
        run = mock.MagicMock()
        run.__enter__.return_value = run
        run.stdout = [b"[render] wrote 240 frames to /project/renders/x.mp4\n"]
        run.returncode = 0
        with tempfile.TemporaryDirectory() as temporary:
            project = pathlib.Path(temporary)
            (project / "scenes").mkdir()
            (project / "scenes/drone.py").write_text("")
            options = ["--seconds", "4", "--session", "x"]
            with (
                mock.patch.object(
                    rendering.projectsync,
                    "push",
                    return_value={"sent": 0, "deleted": 0},
                ),
                mock.patch.object(rendering.subprocess, "Popen", return_value=run),
                mock.patch.object(rendering.subprocess, "call", return_value=1),
                mock.patch.object(sys, "stdout", io.StringIO()),
            ):
                rendering.client(
                    lambda arguments: sent.append(arguments) or [],
                    project,
                    "abc",
                    str(project / "scenes/drone.py"),
                    options,
                )
        steps = [launcher.parser().parse_args(arguments) for arguments in sent]
        self.assertEqual([args.step for args in steps], ["sync", "run", "fetch"])
        self.assertEqual({args.session for args in steps}, {"abc"})
        self.assertEqual(
            steps[1].arguments, ["scenes/drone.py", "scenes__drone", *options]
        )
        self.assertEqual(steps[2].arguments, ["scenes__drone"])

    def test_scene_outside_the_project_is_rejected_before_connecting(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            project = pathlib.Path(temporary) / "project"
            project.mkdir()
            outside = pathlib.Path(temporary) / "scene.py"
            outside.write_text("")
            remote = mock.Mock()
            with self.assertRaisesRegex(ValueError, "inside"):
                rendering.client(remote, project, None, str(outside), [])
            remote.assert_not_called()


class QueueTest(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.project = pathlib.Path(temporary.name)
        self.queue = self.project / ".queue"
        self.queue.mkdir()
        (self.queue / "worker.alive").touch()

    def test_request_reaches_worker_and_result_is_printed_once(self) -> None:
        result = {"ok": True, "frames": 240, "out": "/project/renders/a.mp4"}

        def worker() -> None:
            # answers like project/worker.py: consume the request, publish the result
            while True:
                requests = list(self.queue.glob("*.request.json"))
                if requests:
                    break
                time.sleep(0.01)
            request = requests[0]
            self.assertEqual(json.loads(request.read_text()), {"args": ["x"]})
            request.unlink()
            stem = request.name.removesuffix(".request.json")
            (self.queue / f"{stem}.result.json").write_text(json.dumps(result))

        thread = threading.Thread(target=worker)
        thread.start()
        output = io.StringIO()
        with mock.patch.object(sys, "stdout", output):
            code = rendering.queue(self.project, "scenes__a-1", '{"args": ["x"]}')
        thread.join(timeout=10)
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(output.getvalue()), result)
        self.assertEqual(list(self.queue.glob("scenes__a-1.*")), [])

    def test_dead_worker_withdraws_the_request(self) -> None:
        stale = time.time() - rendering.HEARTBEAT_AGE - 5
        os.utime(self.queue / "worker.alive", (stale, stale))
        self.assertEqual(rendering.queue(self.project, "scenes__a-2", "{}"), 5)
        self.assertEqual(list(self.queue.glob("scenes__a-2.*")), [])


if __name__ == "__main__":
    unittest.main()
