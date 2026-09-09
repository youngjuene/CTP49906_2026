import os
import shutil
import subprocess
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]


def _write_executable(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")
    path.chmod(0o755)


def _fixture(tmp_path: Path, *, public_ip: str = "143.248.249.7", port: str = "8888") -> Path:
    root = tmp_path / "fixture"
    scripts = root / "scripts"
    bin_dir = root / ".tmpbin"
    venv_bin = root / ".venv" / "bin"
    scripts.mkdir(parents=True)
    bin_dir.mkdir()
    venv_bin.mkdir(parents=True)
    shutil.copy2(REPO / "scripts" / "class.sh", scripts / "class.sh")
    (root / ".env").write_text(
        f"ATLAS_PUBLIC_IP={public_ip}\nATLAS_PORT={port}\n", encoding="utf-8"
    )
    (root / ".run").mkdir()

    _write_executable(
        bin_dir / "systemctl",
        """#!/usr/bin/env bash
echo "systemctl $*" >> "$ATLAS_TEST_ROOT/systemctl.log"
exit 0
""",
    )
    _write_executable(
        bin_dir / "curl",
        """#!/usr/bin/env bash
echo "curl $*" >> "$ATLAS_TEST_ROOT/curl.log"
if [[ "$ATLAS_TEST_HEALTH" == "ok" ]]; then
  printf '{"ok":true,"protocol":1}\\n'
  exit 0
fi
exit 7
""",
    )
    _write_executable(
        bin_dir / "sleep",
        """#!/usr/bin/env bash
exit 0
""",
    )
    _write_executable(
        bin_dir / "seq",
        """#!/usr/bin/env bash
printf '1\\n2\\n'
""",
    )
    _write_executable(
        venv_bin / "python",
        """#!/usr/bin/env bash
if [[ "$1" == "scripts/make_qr.py" ]]; then
  echo "qr($2,$3)" > "$ATLAS_TEST_ROOT/fake-qr.log"
  mkdir -p "$(dirname "$3")"
  printf 'PNG' > "$3"
  exit 0
fi
exit 99
""",
    )
    return root


def _run_class(root: Path, command: str, *, health: str = "ok") -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{root / '.tmpbin'}:{env['PATH']}",
            "ATLAS_TEST_ROOT": str(root),
            "ATLAS_TEST_HEALTH": health,
        }
    )
    return subprocess.run(
        ["bash", "scripts/class.sh", command],
        cwd=root,
        env=env,
        text=True,
        capture_output=True,
        timeout=5,
    )


def test_start_uses_managed_https_service_and_prints_numeric_class_url(tmp_path):
    root = _fixture(tmp_path)

    result = _run_class(root, "start")

    assert result.returncode == 0, result.stderr + result.stdout
    assert "학생 공유 주소: https://143.248.249.7:8888" in result.stdout
    assert "관리자: https://143.248.249.7:8888/admin" in result.stdout
    assert "각 참가자에게 본인의 고정 개인 접근 코드만 따로 전달하세요." in result.stdout
    assert "trycloudflare" not in result.stdout
    assert "cloudflared" not in result.stdout
    assert (root / ".run" / "qr.png").read_text() == "PNG"
    assert "start feedback-atlas.service feedback-atlas-renew.timer" in (
        root / "systemctl.log"
    ).read_text(encoding="utf-8")
    assert "https://143.248.249.7:8888/healthz" in (
        root / "curl.log"
    ).read_text(encoding="utf-8")


def test_start_refuses_to_print_url_when_https_healthcheck_fails(tmp_path):
    root = _fixture(tmp_path)

    result = _run_class(root, "start", health="fail")

    assert result.returncode != 0
    assert "HTTPS startup failed" in result.stderr
    assert "학생 공유 주소" not in result.stdout
    assert not (root / ".run" / "qr.png").exists()


def test_url_command_prints_qr_for_running_https_service_without_starting_it(tmp_path):
    root = _fixture(tmp_path, public_ip="192.0.2.10", port="8443")

    result = _run_class(root, "url")

    assert result.returncode == 0, result.stderr + result.stdout
    assert "학생 공유 주소: https://192.0.2.10:8443" in result.stdout
    assert not (root / "systemctl.log").exists()
    assert "qr(https://192.0.2.10:8443,.run/qr.png)" in (
        root / "fake-qr.log"
    ).read_text(encoding="utf-8")


def test_url_command_refuses_when_https_service_is_not_ready(tmp_path):
    root = _fixture(tmp_path)

    result = _run_class(root, "url", health="fail")

    assert result.returncode != 0
    assert "HTTPS server is not ready" in result.stderr
    assert "학생 공유 주소" not in result.stdout


def test_status_reports_systemd_status_and_health(tmp_path):
    root = _fixture(tmp_path)

    result = _run_class(root, "status")

    assert result.returncode == 0
    assert '{"ok":true,"protocol":1}' in result.stdout
    assert "--user --no-pager status feedback-atlas.service" in (
        root / "systemctl.log"
    ).read_text(encoding="utf-8")
    assert "-fsS --max-time 3 https://143.248.249.7:8888/healthz" in (
        root / "curl.log"
    ).read_text(encoding="utf-8")


def test_stop_stops_only_the_managed_service(tmp_path):
    root = _fixture(tmp_path)

    result = _run_class(root, "stop")

    assert result.returncode == 0
    assert "stop feedback-atlas.service" in (root / "systemctl.log").read_text(
        encoding="utf-8"
    )
    assert not (root / "curl.log").exists()


def test_unknown_command_prints_usage_and_does_not_start_anything(tmp_path):
    root = _fixture(tmp_path)

    result = _run_class(root, "tunnel")

    assert result.returncode == 2
    assert "Usage: scripts/class.sh {start|url|status|stop}" in result.stderr
    assert not (root / "systemctl.log").exists()
    assert not (root / "curl.log").exists()
