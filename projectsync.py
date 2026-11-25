"""Mirrors a project directory to Umbrella, like rsync -a --inplace --delete.

Both ends are this module, so it needs only the standard library and works from
Windows. The local side sends its file list; the remote side answers with the files
it lacks or holds a different version of (by size and modification time, rsync's
quick check), deletes what the local side no longer has, then reads the requested
files as a tar stream and writes them in place.
"""

from __future__ import annotations

import json
import os
import pathlib
import shutil
import tarfile
from typing import IO

# never sent and never deleted, at any depth: local tooling, and what the cluster
# side creates (installed packages, render queue and output, Isaac Lab training logs)
EXCLUDED = frozenset(
    {
        ".venv",
        ".lsp",
        ".packages",
        ".queue",
        "renders",
        "logs",
        "__pycache__",
        "pyrightconfig.json",
    }
)


def manifest(root: pathlib.Path) -> dict:
    """Lists files as {path: [size, mtime seconds]} and directories, by POSIX path."""
    files: dict[str, list[int]] = {}
    directories: list[str] = []
    for directory, children, names in os.walk(root):
        children[:] = sorted(name for name in children if name not in EXCLUDED)
        base = pathlib.Path(directory)
        relative = base.relative_to(root)
        directories += [(relative / name).as_posix() for name in children]
        for name in names:
            if name in EXCLUDED:
                continue
            stat = (base / name).stat()
            files[(relative / name).as_posix()] = [stat.st_size, int(stat.st_mtime)]
    return {"files": files, "directories": directories}


def check_path(path: str) -> str:
    """Rejects paths that could leave the project or touch excluded directories."""
    parts = path.split("/")
    if (
        not path
        or path.startswith("/")
        or "\\" in path
        or any(part in ("", ".", "..") or part in EXCLUDED for part in parts)
    ):
        raise ValueError(f"Unsafe project path: {path!r}")
    return path


def push(root: pathlib.Path, to_remote: IO[bytes], from_remote: IO[bytes]) -> dict:
    """Local side: sends the file list, then the files the remote asks for."""
    to_remote.write(json.dumps(manifest(root)).encode() + b"\n")
    to_remote.flush()
    line = from_remote.readline()
    if not line:
        raise RuntimeError("The remote side ended before answering")
    needed = json.loads(line)
    with tarfile.open(fileobj=to_remote, mode="w|") as archive:
        for path in needed:
            archive.add(root / check_path(path), arcname=path, recursive=False)
    to_remote.close()
    line = from_remote.readline()
    if not line:
        raise RuntimeError("The remote side ended before confirming the files")
    return json.loads(line)


def receive(root: pathlib.Path, from_local: IO[bytes], to_local: IO[bytes]) -> dict:
    """Remote side: makes root match the local file list and returns what changed."""
    local = json.loads(from_local.readline())
    files: dict[str, list[int]] = {
        check_path(path): entry for path, entry in local["files"].items()
    }
    directories = {check_path(path) for path in local["directories"]}
    current = manifest(root)
    needed = sorted(path for path in files if current["files"].get(path) != files[path])
    to_local.write(json.dumps(needed).encode() + b"\n")
    to_local.flush()

    # delete first, so a file can take the place of a directory and the reverse
    deleted = [path for path in current["files"] if path not in files]
    for path in deleted:
        (root / path).unlink()
    # deepest first; directories still holding excluded files stay, as with rsync
    for path in sorted(set(current["directories"]) - directories, reverse=True):
        try:
            (root / path).rmdir()
        except OSError:
            pass
    for path in sorted(directories):
        (root / path).mkdir(parents=True, exist_ok=True)

    pending = set(needed)
    with tarfile.open(fileobj=from_local, mode="r|") as archive:
        for member in archive:
            if member.name not in pending or not member.isfile():
                raise ValueError(f"Unexpected file in the upload: {member.name!r}")
            source = archive.extractfile(member)
            assert source is not None
            target = root / member.name
            write_in_place(target, source)
            mtime = files[member.name][1]
            os.utime(target, (mtime, mtime))
            pending.remove(member.name)
    if pending:
        raise ValueError(f"The upload ended without {len(pending)} file(s)")
    summary = {"sent": len(needed), "deleted": len(deleted)}
    to_local.write(json.dumps(summary).encode() + b"\n")
    to_local.flush()
    return summary


def write_in_place(path: pathlib.Path, source: IO[bytes]) -> None:
    """Overwrites an existing file's content without replacing the file.

    Keeping the inode lets the GPU node's NFS client revalidate the file on open;
    a replaced file can briefly be served from its cache in the old version.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "r+b" if path.is_file() else "wb") as target:
        shutil.copyfileobj(source, target)
        target.truncate()
