"""Exercise the real Studio replay side pane through an existing editor server.

Requires Playwright and Chromium. This checks the actual iframe and native
controls, preserving their DOM identity across tab changes. It does not claim
Molab hosting or live-GPU qualification.
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import expect, sync_playwright


def presentation(page, timeout=60):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        for frame in page.frames:
            if "/_marimo-studio/presentation/" in frame.url:
                try:
                    if frame.locator("#app-shell").count():
                        return frame
                except PlaywrightError:
                    continue  # Studio replaces the preview frame during activation.
        page.wait_for_timeout(200)
    raise AssertionError("Studio presentation did not open beside the notebook")


def verify(url, output, browser_executable=None, notebook=None):
    output.mkdir(parents=True, exist_ok=True)
    report = {"url": url, "viewports": [], "errors": []}
    with sync_playwright() as p:
        options = {"headless": True}
        if browser_executable:
            options["executable_path"] = browser_executable
        browser = p.chromium.launch(**options)
        report["browser"] = browser.version
        page = browser.new_page(viewport={"width": 1440, "height": 900})
        page.on("pageerror", lambda error: report["errors"].append(str(error)))
        page.goto(url, wait_until="domcontentloaded")
        if notebook:
            deadline = time.monotonic() + 30
            while page.frame(name="marimo-studio-editor") is None:
                assert time.monotonic() < deadline, "Studio's native editor did not connect"
                page.wait_for_timeout(100)
            editor = page.frame(name="marimo-studio-editor")
            editor.locator("body").wait_for()
            query = parse_qs(urlparse(editor.url).query)
            assert Path(query["file"][0]).resolve() == notebook.resolve()
            # An attached frame can precede its kernel connection. Use the
            # documented session discovery before sending execution commands.
            deadline = time.monotonic() + 30
            while True:
                listing = subprocess.run(
                    [sys.executable, "-m", "marimo", "pair", "notebook", "list", "--url", url],
                    text=True, capture_output=True, check=True, timeout=10,
                )
                notebooks = json.loads(listing.stdout).get("notebooks", [])
                sessions = [
                    candidate["id"] for item in notebooks
                    if Path(item["path"]).resolve() == notebook.resolve()
                    for candidate in item.get("sessions", [])
                ]
                # Studio's frame routing ID can precede/refer to an older
                # connection; the public pairing registry is authoritative.
                if len(sessions) == 1:
                    session = sessions[0]
                    break
                assert time.monotonic() < deadline, "No unambiguous kernel session for the fixture"
                page.wait_for_timeout(200)
            code = '''import os
import marimo._code_mode as cm
assert os.environ.get("CTP49906_REPLAY") == "1", "Initial verification requires a replay server"
async with cm.get_context() as ctx:
    for cid, cell in ctx.cells.items():
        if cell.status == "stale":
            ctx.run_cell(cid)
'''
            result = subprocess.run(
                [sys.executable, "-m", "marimo", "pair", "execute", "--url", url,
                 "--file", str(notebook.resolve()), "--session", session, "--code-file", "-"],
                input=code, text=True, capture_output=True, check=False, timeout=90,
            )
            assert result.returncode == 0, result.stdout + result.stderr
            execution = json.loads(result.stdout)
            assert execution.get("success"), execution
            report["initial_setup"] = "Executed real notebook cells through marimo pair in replay mode"
            page.reload(wait_until="domcontentloaded")
        frame = presentation(page)
        frame.wait_for_selector('html[data-marimo-studio-state="ready"]', timeout=60000)
        assert page.frame(name="marimo-studio-editor") is not None
        assert "재생" in frame.locator('marimo-cell[name="studio_status"]').inner_text()
        diagnostics = frame.evaluate("window.marimoStudio.diagnostics()")
        assert not [d for d in diagnostics if d.get("severity") == "error"], diagnostics

        # Keep real host identity and an unsubmitted draft across every panel.
        frame.evaluate("window.__testFormHost = document.querySelector('marimo-cell[name=diversity_form]')")
        prompt = frame.locator('marimo-cell[name="diversity_form"] input[type="text"]:not([inputmode="numeric"])[placeholder=""]')
        prompt.fill("검증용 초안 — 제출하지 않음")
        prompt.press("Tab")
        for panel in ("guide", "evidence", "explore"):
            frame.locator(f'[data-panel-target="{panel}"]').click()
        assert prompt.input_value() == "검증용 초안 — 제출하지 않음"
        assert frame.evaluate("window.__testFormHost === document.querySelector('marimo-cell[name=diversity_form]')")

        for method, target in (("diversity", "diversity_form"), ("band", "band_form"), ("tf", "tf_form")):
            frame.locator(f'[data-method-target="{method}"]').click()
            run = frame.locator(f'marimo-cell[name="{target}"]').get_by_role("button", name="▶")
            expect(run).to_have_count(1)
            expect(run).to_be_disabled()
        # The classroom release adds a submitted caption budget. Ensure the
        # compact template actually mounts it, rather than only declaring it.
        tf_form = frame.locator('marimo-cell[name="tf_form"]')
        advanced = tf_form.locator("summary:visible")
        advanced.click()
        expect(tf_form.locator('[role="slider"][aria-valuemax="128"]')).to_be_visible()
        advanced.click()
        frame.locator('[data-method-target="diversity"]').click()

        for width, height in ((1440, 900), (1280, 800)):
            page.set_viewport_size({"width": width, "height": height})
            page.wait_for_timeout(300)
            bounds = frame.evaluate("""() => ({
                width: innerWidth, height: innerHeight,
                bodyWidth: document.body.scrollWidth,
                bodyHeight: document.body.scrollHeight,
                shellHeight: document.querySelector('#app-shell').getBoundingClientRect().height
            })""")
            assert bounds["bodyWidth"] <= bounds["width"] + 1, bounds
            assert bounds["bodyHeight"] <= bounds["height"] + 1, bounds
            for method, target in (("diversity", "diversity_form"), ("band", "band_form"), ("tf", "tf_form")):
                frame.locator(f'[data-method-target="{method}"]').click()
                run = frame.locator(f'marimo-cell[name="{target}"]').get_by_role("button", name="▶")
                expect(run).to_be_in_viewport(ratio=1)
            frame.locator('[data-method-target="diversity"]').click()
            screenshot = output / f"studio-split-{width}.png"
            page.screenshot(path=str(screenshot))
            report["viewports"].append({"window": [width, height], "pane": bounds, "screenshot": str(screenshot)})

        assert not report["errors"], report["errors"]
        report["diagnostics"] = frame.evaluate("window.marimoStudio.diagnostics()")
        assert not [d for d in report["diagnostics"] if d.get("severity") == "error"]
        report["passed"] = True
        browser.close()
    (output / "browser-report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:2786")
    parser.add_argument("--output", type=Path, default=Path("/tmp/ctp-studio-browser"))
    parser.add_argument("--browser-executable")
    parser.add_argument("--notebook", type=Path, help="Run initial cells of this disposable replay notebook before checking its view")
    args = parser.parse_args()
    print(json.dumps(verify(args.url, args.output, args.browser_executable, args.notebook), ensure_ascii=False, indent=2))
