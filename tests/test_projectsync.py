"""Tests mirroring a project through the same pipe protocol render uses over SSH."""

import os
import pathlib
import tempfile
import threading
import unittest

import projectsync


def write(path: pathlib.Path, text: str, mtime: int = 1_700_000_000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    os.utime(path, (mtime, mtime))


class SyncTest(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.local = pathlib.Path(temporary.name) / "local"
        self.remote = pathlib.Path(temporary.name) / "remote"
        self.local.mkdir()
        self.remote.mkdir()

    def sync(self) -> dict:
        """Runs both ends over OS pipes, as ssh connects them."""
        up_read, up_write = os.pipe()
        down_read, down_write = os.pipe()
        failures: list[BaseException] = []

        def remote_side() -> None:
            with open(up_read, "rb") as source, open(down_write, "wb") as sink:
                try:
                    projectsync.receive(self.remote, source, sink)
                except BaseException as error:
                    failures.append(error)

        thread = threading.Thread(target=remote_side)
        thread.start()
        summary: dict = {}
        local_failure = None
        try:
            with open(up_write, "wb") as sink, open(down_read, "rb") as source:
                summary = projectsync.push(self.local, sink, source)
        except (OSError, RuntimeError) as error:
            local_failure = error
        thread.join(timeout=10)
        # the remote side's error explains a local broken pipe
        if failures:
            raise failures[0]
        if local_failure:
            raise local_failure
        return summary

    def test_mirrors_changes_in_place_and_keeps_cluster_outputs(self) -> None:
        write(self.local / "render.py", "same")
        write(self.remote / "render.py", "same")
        write(self.local / "scenes/drone.py", "new version", mtime=1_700_000_100)
        write(self.remote / "scenes/drone.py", "old version")
        write(self.local / "scenes/new/helper.py", "added")
        write(self.remote / "scenes/removed.py", "gone locally")
        write(self.remote / "old/nested/file.py", "gone locally")
        write(self.local / ".venv/lib/site.py", "local tooling")
        write(self.local / "scenes/__pycache__/drone.pyc", "bytecode")
        cluster_outputs = {
            ".packages/moviepy/__init__.py": "installed",
            "renders/scenes__drone.mp4": "clip",
            "logs/rsl_rl/model.pt": "checkpoint",
            ".queue/worker.alive": "",
        }
        for path, text in cluster_outputs.items():
            write(self.remote / path, text)
        inode = (self.remote / "scenes/drone.py").stat().st_ino

        self.assertEqual(self.sync(), {"sent": 2, "deleted": 2})

        drone = self.remote / "scenes/drone.py"
        self.assertEqual(drone.read_text(), "new version")
        self.assertEqual(drone.stat().st_ino, inode)
        self.assertEqual(int(drone.stat().st_mtime), 1_700_000_100)
        self.assertEqual((self.remote / "scenes/new/helper.py").read_text(), "added")
        self.assertFalse((self.remote / "scenes/removed.py").exists())
        self.assertFalse((self.remote / "old").exists())
        self.assertFalse((self.remote / ".venv").exists())
        self.assertFalse((self.remote / "scenes/__pycache__").exists())
        for path, text in cluster_outputs.items():
            self.assertEqual((self.remote / path).read_text(), text, path)
        # unchanged files keep their times, so the next sync sends nothing
        self.assertEqual(self.sync(), {"sent": 0, "deleted": 0})

    def test_shorter_content_is_truncated(self) -> None:
        write(self.local / "scene.py", "short", mtime=1_700_000_100)
        write(self.remote / "scene.py", "a much longer old version")
        self.sync()
        self.assertEqual((self.remote / "scene.py").read_text(), "short")

    def test_unsafe_paths_are_rejected_before_deleting(self) -> None:
        write(self.remote / "keep.py", "remote file")
        for path in ("../outside.py", ".packages/moviepy/x.py", "/etc/passwd"):
            with self.subTest(path=path):
                request = '{"files": {"%s": [1, 1]}, "directories": []}\n' % path
                with (
                    tempfile.TemporaryFile() as source,
                    tempfile.TemporaryFile() as sink,
                ):
                    source.write(request.encode())
                    source.seek(0)
                    with self.assertRaisesRegex(ValueError, "Unsafe project path"):
                        projectsync.receive(self.remote, source, sink)
                self.assertEqual((self.remote / "keep.py").read_text(), "remote file")


if __name__ == "__main__":
    unittest.main()
