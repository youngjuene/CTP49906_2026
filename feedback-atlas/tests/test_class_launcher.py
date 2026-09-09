import os
import shutil
import subprocess
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]


def _write_executable(path: Path, text: str) -> None:
    path.write_text(text)
    path.chmod(0o755)


def _fixture(tmp_path: Path, *, server_running: bool = False) -> Path:
    root = tmp_path / "fixture"
    scripts = root / "scripts"
    bin_dir = root / ".tmpbin"
    venv_bin = root / ".venv" / "bin"
    scripts.mkdir(parents=True)
    bin_dir.mkdir()
    venv_bin.mkdir(parents=True)
    shutil.copy2(REPO / "scripts" / "class.sh", scripts / "class.sh")
    (root / "roster.csv").write_text("student_id,name\ns1,S1\n")
    if server_running:
        (root / "uvicorn_started").touch()

    _write_executable(
        bin_dir / "pgrep",
        """#!/usr/bin/env bash
root="$ATLAS_TEST_ROOT"
if [[ "$*" == *"uvicorn --factory src.server:create_app"* ]]; then
  [[ -f "$root/uvicorn_started" ]] && echo 1111
elif [[ "$*" == *"cloudflared tunnel --url http://localhost:"* ]]; then
  [[ -f "$root/cf_started" ]] && echo 2222
fi
""",
    )
    _write_executable(
        bin_dir / "curl",
        """#!/usr/bin/env bash
if [[ "$ATLAS_TEST_HEALTH" == "ok" ]]; then
  printf '{"opinions":0,"connections":0}\\n'
  exit 0
fi
exit 7
""",
    )
    _write_executable(
        bin_dir / "cloudflared",
        """#!/usr/bin/env bash
touch "$ATLAS_TEST_ROOT/cf_started"
echo "https://fixture.trycloudflare.com"
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
if [[ "$1" == "-m" && "$2" == "uvicorn" ]]; then
  echo "[fake-uvicorn] args:$*" >> "$ATLAS_TEST_ROOT/fake-uvicorn.log"
  touch "$ATLAS_TEST_ROOT/uvicorn_started"
  exit 0
elif [[ "$1" == "scripts/make_qr.py" ]]; then
  echo "qr($1,$2,$3)" > "$ATLAS_TEST_ROOT/fake-qr.log"
  mkdir -p "$(dirname "$3")"
  printf 'PNG' > "$3"
  exit 0
fi
exit 99
""",
    )
    return root


def _run_class(root: Path, *, health: str = "ok") -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{root / '.tmpbin'}:{env['PATH']}",
            "ATLAS_ADMIN_CODE": "teacher-secret",
            "ATLAS_TEST_ROOT": str(root),
            "ATLAS_TEST_HEALTH": health,
        }
    )
    return subprocess.run(
        ["bash", "scripts/class.sh", "start"],
        cwd=root,
        env=env,
        text=True,
        capture_output=True,
        timeout=5,
    )


def test_healthy_new_server_starts_tunnel_and_prints_class_url(tmp_path):
    root = _fixture(tmp_path)

    result = _run_class(root, health="ok")

    assert result.returncode == 0, result.stderr + result.stdout
    assert "학생 공유 주소" in result.stdout
    assert "https://fixture.trycloudflare.com" in result.stdout
    assert (root / "uvicorn_started").exists()
    assert (root / "cf_started").exists()
    assert (root / ".run" / "qr.png").read_text() == "PNG"


def test_alive_new_server_without_health_fails_before_tunnel(tmp_path):
    root = _fixture(tmp_path)

    result = _run_class(root, health="fail")

    assert result.returncode != 0
    assert "서버가 /healthz에 응답하지 않아 터널을 열지 않습니다" in result.stdout
    assert "학생 공유 주소" not in result.stdout
    assert not (root / "cf_started").exists()
    assert not (root / ".run" / "url.txt").exists()


def test_already_running_unhealthy_server_refuses_tunnel(tmp_path):
    root = _fixture(tmp_path, server_running=True)

    result = _run_class(root, health="fail")

    assert result.returncode != 0
    assert "서버는 이미 돌고 있습니다" in result.stdout
    assert "이미 실행 중인 서버가 /healthz에 응답하지 않아 터널을 열지 않습니다" in result.stdout
    assert "학생 공유 주소" not in result.stdout
    assert not (root / "cf_started").exists()
    assert not (root / ".run" / "url.txt").exists()
