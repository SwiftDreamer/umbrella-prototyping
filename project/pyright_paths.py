"""Builds a merged module tree so editors resolve Isaac Sim's extension modules.

Isaac Sim installs each extension in its own folder (for example
isaacsim/extsDeprecated/isaacsim.core.api/isaacsim/core/api) and adds those folders
to sys.path at runtime. Listing them as pyright extraPaths isn't enough: the
installed isaacsim/__init__.py makes `isaacsim` a regular package, which pyright
prefers over the extensions' namespace folders. So this symlinks everything into
one tree, .lsp/, and points pyrightconfig.json at it.
Run after `uv sync`:  uv run python pyright_paths.py
"""

import json
import pathlib
import shutil
import sys

PROJECT = pathlib.Path(__file__).resolve().parent
SITE = PROJECT / ".venv/lib/python3.12/site-packages"
TREE = PROJECT / ".lsp"
# extension trees; kit/dev holds templates, not importable modules
ROOTS = ("isaacsim/exts", "isaacsim/extsDeprecated", "isaacsim/kit/extscore")
SKIP = {"config", "docs", "data", "tests", "__pycache__"}
# install directories under isaacsim/ itself; skipped only there, since `kit` is also a
# module name (omni.kit.app)
INSTALL_DIRS = frozenset({"exts", "extsDeprecated", "kit"})


def merge(
    source: pathlib.Path, target: pathlib.Path, skip: frozenset[str] = frozenset()
) -> None:
    """Links source's children into target, merging directories that both provide.

    `skip` names children of source alone to leave out, on top of SKIP everywhere.
    """
    target.mkdir(parents=True, exist_ok=True)
    for child in source.iterdir():
        if child.name in SKIP or child.name in skip:
            continue
        destination = target / child.name
        if not destination.exists() and not destination.is_symlink():
            destination.symlink_to(child)
        elif child.is_dir() and destination.is_dir():
            if destination.is_symlink():  # turn an earlier link into a real merged dir
                linked = destination.resolve()
                destination.unlink()
                merge(linked, destination)
            merge(child, destination)


def main() -> int:
    if not SITE.is_dir():
        print(f"{SITE} missing; run `uv sync` first", file=sys.stderr)
        return 1
    shutil.rmtree(TREE, ignore_errors=True)
    merge(
        SITE / "isaacsim", TREE / "isaacsim", INSTALL_DIRS
    )  # SimulationApp and the package root
    merge(SITE / "isaacsim/kit/kernel/py", TREE)  # carb
    extensions = [
        p.parent.parent
        for root in ROOTS
        for p in (SITE / root).glob("*/config/extension.toml")
    ]
    for extension in sorted(extensions):
        merge(extension, TREE)
    config = {
        "venvPath": ".",
        "venv": ".venv",
        "pythonVersion": "3.12",
        "typeCheckingMode": "standard",
        "exclude": [".venv", ".lsp", "renders"],
        "extraPaths": [".lsp"],
    }
    (PROJECT / "pyrightconfig.json").write_text(json.dumps(config, indent=2) + "\n")
    print(f".lsp: merged {len(extensions)} extensions")
    return 0


if __name__ == "__main__":
    sys.exit(main())
