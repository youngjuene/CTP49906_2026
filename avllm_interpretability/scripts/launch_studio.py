"""Prepare the embedded view BEFORE starting a compatible Studio server.

Run in an already provisioned Python/GPU environment. This does not install
packages, change torch, or activate an already running hosting server. Hosts
can use --prepare-only as a filesystem preparation hook before their own launch.
"""

import argparse
import ast
import asyncio
from importlib import metadata
import json
import os
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from src.studio_bundle import restore_studio_bundle


def prepare_notebook(notebook):
    """Read literal bundle data; never import or execute the uploaded notebook."""
    notebook = Path(notebook).resolve()
    tree = ast.parse(notebook.read_text(encoding="utf-8"), filename=str(notebook))
    cells = [node for node in tree.body
             if isinstance(node, ast.FunctionDef) and node.name == "studio_files"]
    if len(cells) != 1:
        raise ValueError("Expected one generated studio_files cell")
    payloads = [node.value for node in cells[0].body if isinstance(node, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "_payload" for t in node.targets)]
    calls = [node for node in ast.walk(cells[0]) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Name) and node.func.id == "_restore_studio_bundle"]
    if len(payloads) != 1 or len(calls) != 1 or len(calls[0].args) != 3:
        raise ValueError("Expected the generated literal Studio bundle")
    try:
        payload = ast.literal_eval(payloads[0])
        digest = ast.literal_eval(calls[0].args[2])
    except (ValueError, TypeError, SyntaxError) as error:
        raise ValueError("Studio payload and digest must be literals") from error
    if not isinstance(payload, str) or not isinstance(digest, str):
        raise ValueError("Studio payload and digest must be string literals")
    return restore_studio_bundle(notebook.parent, payload, digest)


def check_runtime():
    """The Studio release patches a specific marimo frontend/server version."""
    for package, expected in (("marimo", "0.25.0"), ("marimo-studio", "0.2.3")):
        try:
            actual = metadata.version(package)
        except metadata.PackageNotFoundError:
            actual = "missing"
        if actual != expected:
            raise RuntimeError(f"Requires {package}=={expected}; found {actual}. "
                               "Provision a compatible server environment before launching.")


async def validate_and_build(notebook):
    from marimo_studio.authoring import open_workspace

    workspace = open_workspace(notebook)
    report = await workspace.validate(view="explore", level="static")
    if not report.ok:
        raise RuntimeError(json.dumps(report.to_dict(), ensure_ascii=False, default=str))
    return await workspace.view("explore").build(profile="production")


def server_command(notebook, mode, port):
    # Same interpreter and GPU stack; run mode otherwise defaults to no token.
    return [sys.executable, "-m", "marimo", mode, str(notebook),
            "--no-sandbox", "--token", "--headless", "--host", "127.0.0.1", "--port", str(port)]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("notebook", nargs="?", type=Path,
                        default=PROJECT / "CTP49906_avllm_molab_kr.py")
    parser.add_argument("--prepare-only", action="store_true",
                        help="Restore files only; does not verify or activate the host extension")
    parser.add_argument("--replay", action="store_true", help="Saved examples without GPU inference")
    parser.add_argument("--mode", choices=("edit", "run"), default="edit")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)
    notebook = args.notebook.resolve()
    try:
        if not args.prepare_only:
            check_runtime()
            if not 1 <= args.port <= 65535:
                raise ValueError("Port must be between 1 and 65535")
            if not args.replay:
                import torch
                if not torch.cuda.is_available():
                    raise RuntimeError("No CUDA GPU in this environment. Use --replay for saved examples.")
        result = prepare_notebook(notebook)
        print(json.dumps({"files_prepared": result, "server_started": False}, ensure_ascii=False), flush=True)
        if args.prepare_only:
            return 0
        asyncio.run(validate_and_build(notebook))
    except (ValueError, OSError, RuntimeError, ImportError) as error:
        parser.exit(1, f"Studio startup failed: {error}\n")
    # Be explicit even when the shell previously launched a replay session.
    os.environ["CTP49906_REPLAY"] = "1" if args.replay else "0"
    command = server_command(notebook, args.mode, args.port)
    os.execv(sys.executable, command)


if __name__ == "__main__":
    main()
