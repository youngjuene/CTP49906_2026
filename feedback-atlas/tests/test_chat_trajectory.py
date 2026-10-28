"""Saved claude.ai conversations become ordered, attributable sentence trajectories.

The defect this guards against is a silent one: a conversation whose turns are
attributed to the wrong speaker, or put in the wrong order, still renders as a
perfectly plausible path. So the page structure the converter relies on is pinned
here with a small synthetic share page -- data-testid roles, a <time> beside (not
inside) the student's message, and the furniture that must be dropped -- rather
than trusted to keep working.

Also covered: the build lays out sentences but not status lines, the build file
is private on disk, and /api/chats serves it to anyone with the link.

Run:  python -m pytest feedback-atlas/tests/test_chat_trajectory.py
"""
import json
import stat
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # -> feedback-atlas/

from scripts.build_chat_atlas import main as build  # noqa: E402
from scripts.chat_to_trajectory import (  # noqa: E402
    convert, merge_short, parse_conversation, split_long, split_sentences)
from src.roster import Roster  # noqa: E402

ROSTER_CSV = (
    "id,display_name,role\n"
    "target1,대상1,student\n"
    "writer1,작성자1,observer\n"
)

# The shape of a saved share page, reduced to what the converter reads.
PAGE = """<!doctype html><html><body>
<nav data-testid="sidebar-recents"><a>Someone else's chat title</a></nav>
<div data-testid="transcript-list">
  <div data-testid="user-message"><p>죽음이란 무엇일까?</p></div>
  <div data-testid="user-message-actions"><time datetime="2026-10-17T03:47:35.560Z">10월 17일</time></div>
  <div data-testid="assistant-message">
    <h3 class="sr-only select-none">Claude 응답: 첫 문단을 반복하는 제목입니다.</h3>
    <button data-testid="tool-status-pill">웹 검색됨&#xe027;</button>
    <p>죽음은 관점에 따라 다르게 정의됩니다. 생물학적으로는 생명 활동이 멈추는 것입니다.<a class="group/tag relative">huggingface.co +3&#xe015;</a></p>
    <p>좋은 질문이에요.</p>
    <p>철학적으로는 에피쿠로스의 말이 자주 인용됩니다.</p>
    <pre><code>## Phase 1: this prompt block must not reach the map</code></pre>
    <div data-testid="message-actions"><time datetime="2026-10-17T03:47:48.196Z">10월 17일</time></div>
  </div>
  <div data-testid="file-thumbnail"><h4>공유 시 숨겨지는 파일</h4></div>
  <div data-testid="user-message"><p>slide 1<br>김서연<br>이 결과를 해석해 줄래?</p></div>
  <div data-testid="user-message-actions"><time datetime="2026-10-22T05:50:58.096Z">4일 전</time></div>
  <div data-testid="assistant-message">
    <p class="font-base">인터랙티브 시각 자료는 공유 시 숨겨짐</p>
    <p>결과는 데이터셋이 의도대로 고쳐졌다는 뜻이에요.</p>
    <div data-testid="message-actions"><time datetime="2026-10-22T05:59:03.219Z">4일 전</time></div>
  </div>
</div></body></html>"""


@pytest.fixture
def roster(tmp_path):
    path = tmp_path / "roster.csv"
    path.write_text(ROSTER_CSV, encoding="utf-8")
    return path


def _page(tmp_path, name="target1__w1__01.html", html=PAGE):
    chats = tmp_path / "chats"
    chats.mkdir(exist_ok=True)
    page = chats / name
    page.write_text(html, encoding="utf-8")
    return page


def _rows(tmp_path, roster):
    rows, _ = convert(_page(tmp_path), Roster.from_path(roster), min_chars=20, max_chars=1000)
    return rows


def test_speakers_come_from_the_page_not_from_guessing():
    messages = parse_conversation(PAGE)
    assert [m["speaker"] for m in messages] == ["student", "ai", "student", "ai"]


def test_every_message_gets_its_own_time_including_the_students():
    """A prompt's <time> sits beside the message, not inside it."""
    times = [m["timestamp"] for m in parse_conversation(PAGE)]
    assert times == ["2026-10-17T03:47:35.560Z", "2026-10-17T03:47:48.196Z",
                     "2026-10-22T05:50:58.096Z", "2026-10-22T05:59:03.219Z"]


def test_furniture_is_dropped_and_status_lines_are_kept(tmp_path, roster):
    rows = _rows(tmp_path, roster)
    texts = [r["text"] for r in rows]
    joined = " ".join(texts)
    assert "Claude 응답" not in joined                 # screen-reader heading
    assert "huggingface.co" not in joined              # citation chip
    assert "Phase 1" not in joined                     # code block
    assert "Someone else's" not in joined              # the saver's own sidebar
    assert "" not in joined and "" not in joined
    status = [(r["speaker"], r["text"]) for r in rows if r["kind"] == "status"]
    assert status == [("ai", "웹 검색됨"), ("student", "공유 시 숨겨지는 파일"),
                      ("ai", "인터랙티브 시각 자료는 공유 시 숨겨짐")]


def test_order_turns_and_metadata(tmp_path, roster):
    rows = _rows(tmp_path, roster)
    assert [r["seq"] for r in rows] == list(range(len(rows)))
    assert {r["turn"] for r in rows} == {1, 2}
    assert all(r["target_id"] == "target1" and r["week"] == 1 for r in rows)
    assert all(r["conversation"] == "target1__w1__01" for r in rows)
    # A prompt and its reply share a turn.
    turn2 = [r["speaker"] for r in rows if r["turn"] == 2]
    assert turn2[0] == "student" and turn2[-1] == "ai"


def test_short_fragments_join_a_neighbour(tmp_path, roster):
    rows = _rows(tmp_path, roster)
    texts = [r["text"] for r in rows if r["kind"] == "sentence"]
    assert "좋은 질문이에요." not in texts
    assert any(t.startswith("좋은 질문이에요. 철학적으로는") for t in texts)
    # Across a paragraph break too: "slide 1" and "김서연" are lines of their own.
    assert any(t.startswith("slide 1 김서연") for t in texts)


def test_sentence_splitting_rules():
    assert split_sentences("첫째예요. 둘째예요!\n셋째") == ["첫째예요.", "둘째예요!", "셋째"]
    assert split_sentences("유사도는 0.73이에요. 끝.") == ["유사도는 0.73이에요.", "끝."]
    assert split_sentences("e.g. this stays whole.") == ["e.g. this stays whole."]
    assert merge_short(["짧다.", "이 문장은 충분히 긴 문장입니다."], 10) == ["짧다. 이 문장은 충분히 긴 문장입니다."]
    assert merge_short(["이 문장은 충분히 긴 문장입니다.", "끝."], 10) == ["이 문장은 충분히 긴 문장입니다. 끝."]
    assert all(len(c) <= 30 for c in split_long("가나다 " * 40, 30))


@pytest.mark.parametrize("name, why", [
    ("target1.html", "expected"),
    ("target1__w9__01.html", "week 9"),
    ("writer1__w1__01.html", "not a student"),
    ("nobody__w1__01.html", "not a student"),
])
def test_files_that_cannot_be_placed_are_refused(tmp_path, roster, name, why):
    with pytest.raises(ValueError, match=why):
        convert(_page(tmp_path, name), Roster.from_path(roster), min_chars=20, max_chars=1000)


def test_a_page_without_messages_is_refused(tmp_path, roster):
    with pytest.raises(ValueError, match="no messages"):
        convert(_page(tmp_path, html="<html><body>empty</body></html>"),
                Roster.from_path(roster), min_chars=20, max_chars=1000)


def test_build_lays_out_sentences_and_pins_status_lines(tmp_path, roster):
    page = _page(tmp_path)
    out = tmp_path / "atlas.json"
    assert build(["--chats", str(page.parent), "--roster", str(roster), "--out", str(out),
                  "--cache", str(tmp_path / "cache.db"), "--fake-embedder"]) == 0
    assert stat.S_IMODE(out.stat().st_mode) == 0o600
    data = json.loads(out.read_text(encoding="utf-8"))
    points = data["points"]
    assert [c["id"] for c in data["conversations"]] == ["target1__w1__01"]
    assert data["targets"] == [{"id": "target1", "display_name": "대상1"}]
    assert len({p["id"] for p in points}) == len(points)
    by_seq = {p["seq"]: p for p in points}
    for p in points:
        if p["kind"] != "status":
            continue
        # Pinned just off the next sentence of the same conversation.
        nxt = next(by_seq[s] for s in range(p["seq"] + 1, len(points))
                   if by_seq[s]["kind"] != "status")
        assert abs(p["x"] - nxt["x"]) < 0.1 and abs(p["y"] - nxt["y"]) < 0.1


# --- /api/chats ---------------------------------------------------------------

@pytest.fixture
def client(tmp_path, roster):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from src.config import load_config
    from src.server import create_app

    cfg = load_config({
        "ATLAS_ADMIN_CODE": "admin-code-for-tests",
        "ATLAS_ROSTER": str(roster),
        "ATLAS_DB": str(tmp_path / "atlas.db"),
        "ATLAS_CHATS": str(tmp_path / "atlas.json"),
        "ATLAS_UNSAFE_FAKE_EMBEDDER": "1",
        "ATLAS_SKIP_WARMUP": "1",
        "ATLAS_ALLOW_INSECURE_HTTP": "1",
    })
    with TestClient(create_app(cfg)) as c:
        yield c


def test_chats_are_open_to_anyone_with_the_link(client, tmp_path):
    (tmp_path / "atlas.json").write_text('{"points":[]}', encoding="utf-8")
    ok = client.get("/api/chats")
    assert ok.status_code == 200 and ok.json() == {"points": []}
    assert ok.headers["cache-control"] == "no-store"


def test_a_rebuild_is_served_without_a_restart(client, tmp_path):
    (tmp_path / "atlas.json").write_text('{"points":[]}', encoding="utf-8")
    assert client.get("/api/chats").json() == {"points": []}
    (tmp_path / "atlas.json").write_text('{"points":[{"id":"a#0"}]}', encoding="utf-8")
    assert client.get("/api/chats").json() == {"points": [{"id": "a#0"}]}


def test_no_chats_yet_is_a_clear_404(client):
    r = client.get("/api/chats")
    assert r.status_code == 404 and r.json()["code"] == "NO_CHATS"


def test_the_chat_map_is_the_home_page_and_the_opinion_map_moved_to_map(client):
    home = client.get("/")
    assert home.status_code == 200 and 'src="chats.js"' in home.text
    opinions = client.get("/map")
    assert opinions.status_code == 200 and 'src="bootstrap.js"' in opinions.text
    # The old addresses keep working.
    assert 'src="chats.js"' in client.get("/chats.html").text
    assert 'src="bootstrap.js"' in client.get("/index.html").text
