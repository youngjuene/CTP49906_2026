import csv
import base64
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.create_access_codes import generate
from src.access import AccessControl


def digest(code: str) -> str:
    return hashlib.sha256(code.encode("utf-8")).hexdigest()


def b64decode(text: str) -> bytes:
    return base64.urlsafe_b64decode((text + "=" * (-len(text) % 4)).encode("ascii"))


def write_hashes(path: Path, data: dict[str, str]) -> None:
    path.write_text(json.dumps(data), encoding="utf-8")


def test_authenticate_accepts_canonical_identity_and_rejects_bad_inputs(tmp_path):
    path = tmp_path / "access.json"
    write_hashes(path, {"kim.seoyeon": digest("correct horse"), "ta.seowoo": digest("blue")})
    access = AccessControl.from_path(path)

    assert access.authenticate(" KIM.SEOYEON ", "correct horse") == "kim.seoyeon"
    assert access.authenticate("kim.seoyeon", "wrong") is None
    assert access.authenticate("missing", "correct horse") is None
    assert access.authenticate("missing", "wrong") is None


def test_authenticate_code_returns_the_canonical_owner(tmp_path):
    path = tmp_path / "access.json"
    write_hashes(path, {"kim.seoyeon": digest("correct horse"), "ta.seowoo": digest("blue")})
    access = AccessControl.from_path(path)

    assert access.authenticate_code("correct horse") == "kim.seoyeon"
    assert access.authenticate_code("blue") == "ta.seowoo"


@pytest.mark.parametrize("code", ["", " ", "wrong", object(), "x" * 257])
def test_authenticate_code_rejects_malformed_or_unknown_codes(tmp_path, code):
    path = tmp_path / "access.json"
    write_hashes(path, {"kim.seoyeon": digest("correct horse")})
    access = AccessControl.from_path(path)

    assert access.authenticate_code(code) is None


def test_hashes_property_is_read_only_copy(tmp_path):
    path = tmp_path / "access.json"
    write_hashes(path, {"kim.seoyeon": digest("code")})
    access = AccessControl.from_path(path)

    with pytest.raises(TypeError):
        access.hashes["kim.seoyeon"] = digest("other")
    assert access.authenticate("kim.seoyeon", "code") == "kim.seoyeon"


def test_tokens_verify_tampering_expiry_and_length(tmp_path):
    now = 1000
    path = tmp_path / "access.json"
    write_hashes(path, {"kim.seoyeon": digest("code")})
    access = AccessControl.from_path(path, clock=lambda: now)

    token = access.issue("kim.seoyeon")
    assert access.verify(token) == "kim.seoyeon"
    body = json.loads(b64decode(token.split(".", 1)[0]))
    assert body["v"] != digest("code")
    assert access.verify(token + "x") is None
    assert access.verify("x" * 3000) is None

    expired = AccessControl.from_path(path, clock=lambda: now + 8 * 60 * 60 + 1)
    expired._signing_key = access._signing_key
    assert expired.verify(token) is None


def test_reload_preserves_key_and_revokes_rotated_users(tmp_path):
    path = tmp_path / "access.json"
    write_hashes(path, {"kim.seoyeon": digest("old"), "ta.seowoo": digest("same")})
    access = AccessControl.from_path(path, clock=lambda: 100)
    kim_token = access.issue("kim.seoyeon")
    seowoo_token = access.issue("ta.seowoo")

    write_hashes(path, {"kim.seoyeon": digest("new"), "ta.seowoo": digest("same")})
    access.reload(path)

    assert access.verify(kim_token) is None
    assert access.verify(seowoo_token) == "ta.seowoo"
    assert access.authenticate("kim.seoyeon", "old") is None
    assert access.authenticate("kim.seoyeon", "new") == "kim.seoyeon"


def test_reload_keeps_previous_valid_credentials_when_new_hashes_are_invalid(tmp_path):
    path = tmp_path / "access.json"
    write_hashes(path, {"kim.seoyeon": digest("old"), "ta.seowoo": digest("blue")})
    access = AccessControl.from_path(path, clock=lambda: 100)
    token = access.issue("kim.seoyeon")

    write_hashes(path, {"kim.seoyeon": digest("duplicate"), "ta.seowoo": digest("duplicate")})
    with pytest.raises(ValueError):
        access.reload(path)

    assert access.authenticate_code("old") == "kim.seoyeon"
    assert access.verify(token) == "kim.seoyeon"


def test_restarted_access_control_accepts_the_same_fixed_code(tmp_path):
    path = tmp_path / "access.json"
    write_hashes(path, {"kim.seoyeon": digest("fixed personal code")})

    assert AccessControl.from_path(path).authenticate_code("fixed personal code") == "kim.seoyeon"
    assert AccessControl.from_path(path).authenticate_code("fixed personal code") == "kim.seoyeon"


def test_submission_owner_context_is_stable_private_and_scoped(tmp_path):
    path = tmp_path / "access.json"
    hashes = {"kim.seoyeon": digest("first private code"), "ta.seowoo": digest("second private code")}
    write_hashes(path, hashes)
    first = AccessControl.from_path(path)
    restarted = AccessControl.from_path(path)
    owner = first.submission_owner("kim.seoyeon", scope="classroom")
    assert owner == restarted.submission_owner("kim.seoyeon", scope="classroom")
    assert owner != first.submission_owner("kim.seoyeon", scope="demo")
    assert owner != first.submission_owner("ta.seowoo", scope="classroom")
    assert "kim.seoyeon" not in owner
    assert hashes["kim.seoyeon"] not in owner
    assert len(owner) >= 32
    write_hashes(path, {**hashes, "kim.seoyeon": digest("rotated code")})
    restarted.reload(path)
    assert owner != restarted.submission_owner("kim.seoyeon", scope="classroom")


@pytest.mark.parametrize(
    "data",
    [
        [],
        {},
        {"": digest("code")},
        {"Kim": digest("a"), " kim ": digest("b")},
        {"kim": digest("same"), "lee": digest("same")},
        {"kim": "not-a-sha256"},
        {"kim": digest("code").upper()[:-1] + "Z"},
    ],
)
def test_access_file_validation_rejects_invalid_maps(tmp_path, data):
    path = tmp_path / "access.json"
    path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ValueError):
        AccessControl.from_path(path)


def test_cli_generation_writes_private_files_without_printing_codes(tmp_path, capsys):
    roster = tmp_path / "roster.csv"
    roster.write_text(
        "id,display_name,role\n"
        "kim.seoyeon,김서연,student\n"
        "ta.seowoo,서우,ta\n",
        encoding="utf-8",
    )
    hashes_path = tmp_path / ".run" / "access-codes.json"
    out_path = tmp_path / ".run" / "access-codes.csv"

    count, _, _ = generate(roster, hashes_path, out_path)
    captured = capsys.readouterr()

    assert count == 2
    assert captured.out == ""
    assert hashes_path.stat().st_mode & 0o777 == 0o600
    assert out_path.stat().st_mode & 0o777 == 0o600
    assert hashes_path.parent.stat().st_mode & 0o777 == 0o700

    rows = list(csv.DictReader(out_path.read_text(encoding="utf-8").splitlines()))
    hashes = json.loads(hashes_path.read_text(encoding="utf-8"))
    assert set(hashes) == {"kim.seoyeon", "ta.seowoo"}
    for row in rows:
        code = row["access_code"]
        assert len(code) >= 22
        assert code not in captured.out
        assert digest(code) == hashes[row["id"]]


def test_cli_main_refuses_overwrite_and_does_not_print_codes(tmp_path):
    roster = tmp_path / "roster.csv"
    roster.write_text("id,display_name,role\nkim.seoyeon,김서연,student\n", encoding="utf-8")
    hashes_path = tmp_path / ".run" / "access-codes.json"
    out_path = tmp_path / ".run" / "access-codes.csv"

    cmd = [
        sys.executable,
        "scripts/create_access_codes.py",
        "--roster",
        str(roster),
        "--hashes",
        str(hashes_path),
        "--out",
        str(out_path),
    ]
    first = subprocess.run(cmd, cwd=Path(__file__).resolve().parents[1], text=True,
                           capture_output=True, check=True)
    rows = list(csv.DictReader(out_path.read_text(encoding="utf-8").splitlines()))
    code = rows[0]["access_code"]
    assert code not in first.stdout
    assert str(hashes_path) in first.stdout
    assert str(out_path) in first.stdout

    second = subprocess.run(cmd, cwd=Path(__file__).resolve().parents[1], text=True,
                            capture_output=True)
    assert second.returncode != 0
    assert code not in second.stdout
    assert code not in second.stderr
    assert "refusing to rotate credentials" in second.stderr
