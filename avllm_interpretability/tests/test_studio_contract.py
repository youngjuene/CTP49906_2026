"""End-to-end contracts for the Korean marimo Studio integration."""

from __future__ import annotations

import ast
import json
import shutil
import subprocess
import sys
import textwrap
from html.parser import HTMLParser
from pathlib import Path

import pytest

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised only on Python < 3.11
    tomllib = None


PROJECT = Path(__file__).resolve().parents[1]
NOTEBOOK = PROJECT / "CTP49906_avllm_molab_kr.py"
VIEW_ROOT = PROJECT / "studio" / "ctp49906-kr"
EXPLORE = VIEW_ROOT / "explore"


def _notebook_source() -> str:
    return NOTEBOOK.read_text(encoding="utf-8")


def test_running_notebook_setup_also_provisions_studio_files():
    from marimo._ast.load import load_app

    app = load_app(str(NOTEBOOK))
    cells = [data.cell for data in app._cell_manager.cell_data() if data.cell is not None]
    setup = next(cell for cell in cells if "PROJECT_DIR" in cell.defs)
    assert "studio_bundle_status" in setup.refs
    bundle = next(cell for cell in cells if "studio_bundle_status" in cell.defs)
    assert "mo" in bundle.refs
    assert "PROJECT_DIR" not in bundle.refs  # no bootstrap cycle


def _notebook_tree() -> ast.Module:
    return ast.parse(_notebook_source(), filename=str(NOTEBOOK))


def _pep723_document() -> dict:
    if tomllib is None:
        pytest.skip("PEP 723 TOML parsing needs Python 3.11 tomllib")
    lines = _notebook_source().splitlines()
    try:
        start = lines.index("# /// script")
        end = lines.index("# ///", start + 1)
    except ValueError as exc:
        raise AssertionError("Korean notebook is missing its PEP 723 script block") from exc
    body = []
    for line in lines[start + 1 : end]:
        assert line.startswith("#"), line
        body.append(line[1:].removeprefix(" "))
    return tomllib.loads("\n".join(body))


class ProjectionParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.stack: list[tuple[str, dict[str, str]]] = []
        self.shell_depth: int | None = None
        self.cells: list[dict[str, object]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr = {key: value or "" for key, value in attrs}
        parent_cell = next((item for item in reversed(self.stack) if item[0] == "marimo-cell"), None)
        inside_shell = self.shell_depth is not None
        if tag == "div" and attr.get("id") == "app-shell":
            assert self.shell_depth is None, "expected one #app-shell"
            self.shell_depth = len(self.stack)
            inside_shell = True
        if tag == "marimo-cell":
            self.cells.append(
                {
                    "name": attr.get("name"),
                    "inside_shell": inside_shell,
                    "nested_in": parent_cell[1].get("name") if parent_cell else None,
                }
            )
        self.stack.append((tag, attr))

    def handle_endtag(self, tag: str) -> None:
        for index in range(len(self.stack) - 1, -1, -1):
            if self.stack[index][0] == tag:
                del self.stack[index:]
                break
        if self.shell_depth is not None and len(self.stack) <= self.shell_depth:
            self.shell_depth = None


def _projection_parser() -> ProjectionParser:
    parser = ProjectionParser()
    parser.feed((EXPLORE / "index.html").read_text(encoding="utf-8"))
    parser.close()
    return parser


def _projected_names() -> list[str]:
    names = [cell["name"] for cell in _projection_parser().cells]
    assert all(isinstance(name, str) and name for name in names)
    return [str(name) for name in names]


@pytest.fixture(scope="module")
def loaded_notebook_cell_names() -> set[str]:
    marimo = pytest.importorskip("marimo", reason="Studio notebook contract needs marimo")
    assert marimo
    from marimo._ast.load import load_app

    app = load_app(str(NOTEBOOK))
    names = set()
    for data in app._cell_manager.cell_data():
        if data.cell is not None and data.cell.name:
            names.add(data.cell.name)
    return names


def test_pep723_studio_config_resolves_to_tracked_view_root() -> None:
    config = _pep723_document()["tool"]["marimo-studio"]

    assert config["default"] == "explore"
    assert config["runtime"] == "server"
    assert config["preserve_session"] is True
    assert (NOTEBOOK.parent / config["view_root"]).resolve() == VIEW_ROOT
    assert (VIEW_ROOT / config["default"] / "view.toml").is_file()


def test_projected_html_cells_exist_once_inside_the_loaded_notebook(
    loaded_notebook_cell_names: set[str],
) -> None:
    parser = _projection_parser()
    names = [str(cell["name"]) for cell in parser.cells]

    assert "verdict_submit_handler" in names
    assert len(names) == len(set(names))
    assert all(cell["inside_shell"] for cell in parser.cells)
    assert all(cell["nested_in"] is None for cell in parser.cells)
    assert set(names) <= loaded_notebook_cell_names


def test_studio_navigation_preserves_mounted_projection_hosts(tmp_path: Path) -> None:
    if shutil.which("node") is None:
        pytest.skip("node is required for Studio navigation behavior simulation")

    harness = tmp_path / "studio-nav-check.mjs"
    harness.write_text(
        textwrap.dedent(
            f"""
            import fs from "node:fs";
            import vm from "node:vm";

            class ClassList {{
              constructor(owner) {{ this.owner = owner; this.values = new Set(owner.className.split(/\\s+/).filter(Boolean)); }}
              toggle(name, force) {{ force ? this.values.add(name) : this.values.delete(name); this.owner.className = Array.from(this.values).join(" "); }}
              contains(name) {{ return this.values.has(name); }}
            }}

            class Element {{
              constructor(tag, attrs = {{}}) {{
                this.tagName = tag.toUpperCase();
                this.id = attrs.id || "";
                this.className = attrs.class || "";
                this.dataset = attrs.dataset || {{}};
                this.children = [];
                this.parentNode = null;
                this.listeners = {{}};
                this.attributes = new Map();
                this.classList = new ClassList(this);
                this.tabIndex = attrs.tabIndex ?? 0;
                this.hidden = false;
                this.scrollTop = 37;
                this.focused = false;
              }}
              appendChild(child) {{ child.parentNode = this; this.children.push(child); return child; }}
              addEventListener(type, fn) {{ (this.listeners[type] ||= []).push(fn); }}
              click() {{ for (const fn of this.listeners.click || []) fn({{ currentTarget: this }}); }}
              keydown(key) {{ for (const fn of this.listeners.keydown || []) fn({{ key, currentTarget: this, preventDefault() {{ this.prevented = true; }} }}); }}
              focus() {{ this.focused = true; }}
              setAttribute(name, value) {{ this.attributes.set(name, String(value)); }}
              getAttribute(name) {{ return this.attributes.get(name); }}
              toggleAttribute(name, force) {{ force ? this.attributes.set(name, "") : this.attributes.delete(name); if (name === "hidden") this.hidden = Boolean(force); }}
            }}

            const all = [];
            function el(tag, attrs) {{ const node = new Element(tag, attrs); all.push(node); return node; }}

            const shell = el("div", {{ id: "app-shell", dataset: {{ panel: "explore", method: "diversity" }} }});
            const panelButtons = ["explore", "guide", "evidence"].map((name) => el("button", {{ dataset: {{ panelTarget: name }} }}));
            const panels = ["explore", "guide", "evidence"].map((name) => el("section", {{ class: "panel", dataset: {{ panel: name }} }}));
            const methodButtons = ["diversity", "band", "tf"].map((name) => el("button", {{ dataset: {{ methodTarget: name }} }}));
            const methodPanels = ["diversity", "band", "tf"].map((name) => el("section", {{ class: "method-panel", dataset: {{ method: name }} }}));
            for (const node of [...panelButtons, ...panels, ...methodButtons, ...methodPanels]) shell.appendChild(node);
            const hosts = ["studio_status", "diversity_form", "band_form", "tf_form", "verdict_submit_handler"].map((name, index) => {{
              const host = el("marimo-cell", {{ dataset: {{ hostName: name }} }});
              panels[index % panels.length].appendChild(host);
              return host;
            }});
            const before = hosts.map((host) => host.parentNode);

            globalThis.document = {{
              getElementById(id) {{ return id === "app-shell" ? shell : all.find((node) => node.id === id) || null; }},
              querySelectorAll(selector) {{
                if (selector === "[data-panel-target]") return panelButtons;
                if (selector === ".panel[data-panel]") return panels;
                if (selector === "[data-method-target]") return methodButtons;
                if (selector === ".method-panel[data-method]") return methodPanels;
                return [];
              }},
            }};

            vm.runInThisContext(fs.readFileSync({json.dumps(str(EXPLORE / "main.js"))}, "utf8"));
            panelButtons[2].click();
            methodButtons[2].click();
            panelButtons[1].keydown("ArrowLeft");
            methodButtons[0].keydown("End");

            const after = hosts.map((host) => host.parentNode);
            if (before.some((parent, index) => parent !== after[index])) throw new Error("projection host was remounted");
            if (shell.dataset.panel !== "explore") throw new Error(`unexpected panel ${{shell.dataset.panel}}`);
            if (shell.dataset.method !== "tf") throw new Error(`unexpected method ${{shell.dataset.method}}`);
            if (!panels[0].classList.contains("is-active") || !panels[1].hidden || !panels[2].hidden) throw new Error("panel visibility is inconsistent");
            if (!methodPanels[2].classList.contains("is-active") || !methodPanels[0].hidden || !methodPanels[1].hidden) throw new Error("method visibility is inconsistent");
            """
        ),
        encoding="utf-8",
    )

    subprocess.run(["node", str(harness)], cwd=PROJECT, check=True)


def test_studio_navigation_uses_native_hosts_without_notebook_setters() -> None:
    source = (EXPLORE / "main.js").read_text(encoding="utf-8")
    tree = ast.parse(_notebook_source(), filename=str(NOTEBOOK))
    notebook_setters = {
        node.name
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name.startswith("set_")
    }

    assert "marimo-cell" not in source
    assert ".value" not in source
    assert not any(name in source for name in notebook_setters)


def test_embedded_studio_bundle_matches_tracked_sources() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/build_studio_bundle.py", "--check"],
        cwd=PROJECT,
        text=True,
        capture_output=True,
        check=True,
    )

    assert "Studio bundle matches tracked sources" in result.stdout


def _studio_files_function() -> ast.FunctionDef:
    for node in _notebook_tree().body:
        if isinstance(node, ast.FunctionDef) and node.name == "studio_files":
            node.decorator_list = []
            return node
    raise AssertionError("missing generated studio_files cell")


class FakeMo:
    def __init__(self, location: Path) -> None:
        self.location = location
        self.callouts: list[tuple[object, str]] = []

    def notebook_location(self) -> str:
        return str(self.location)

    def md(self, text: str) -> str:
        return text

    def callout(self, body: object, *, kind: str) -> tuple[object, str]:
        self.callouts.append((body, kind))
        return body, kind


def test_actual_generated_studio_files_cell_is_single_file_and_preserves_edits(
    tmp_path: Path,
) -> None:
    function = _studio_files_function()
    module = ast.Module(body=[function], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace: dict[str, object] = {}
    exec(compile(module, str(NOTEBOOK), "exec"), namespace)  # noqa: S102 -- trusted notebook AST, fake I/O
    studio_files = namespace["studio_files"]

    fake_mo = FakeMo(tmp_path)
    first_status = studio_files(fake_mo)[0]
    index = tmp_path / "studio" / "ctp49906-kr" / "explore" / "index.html"
    manifest = tmp_path / "studio" / "ctp49906-kr" / ".bundle-manifest.json"

    assert set(first_status["written"]) == {
        "explore/AGENTS.md",
        "explore/index.html",
        "explore/main.js",
        "explore/style.css",
        "explore/view.toml",
    }
    assert index.read_text(encoding="utf-8") == (EXPLORE / "index.html").read_text(encoding="utf-8")
    assert manifest.is_file()
    assert fake_mo.callouts[-1][1] == "info"

    second_status = studio_files(fake_mo)[0]
    assert second_status["written"] == []

    index.write_text("student-edited view", encoding="utf-8")
    conflict_status = studio_files(fake_mo)[0]
    assert "error" in conflict_status
    assert "local edits" in conflict_status["error"]
    assert index.read_text(encoding="utf-8") == "student-edited view"
