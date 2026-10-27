#!/usr/bin/env python
"""Turn a saved claude.ai share page into an ordered list of sentences.

The input is the share page saved from a browser ("Webpage, Complete"), not a
copy-paste. A pasted conversation carries no speaker labels, and in practice the
obvious heuristics fail: students write multi-paragraph prompts, quote the
model's paragraphs back at it, and the model pastes `## Phase 1` prompts that
look like structure. The saved page marks every message with
data-testid="user-message" or "assistant-message", and every message carries a
<time datetime=...> in UTC, so who said what, and when, is read rather than
guessed.

The file name carries what the page cannot:

    .run/chats/<target_id>__w<week>__<nn>.html     e.g. kim.seoyeon__w1__01.html

target_id must be a student on the roster and week one of 1..4. A file that does
not match is refused, not guessed at -- a conversation filed against the wrong
project is invisible once it is on the map.

Output is one JSONL row per unit, in conversation order:

    conversation  file stem, e.g. "kim.seoyeon__w1__01"
    target_id     from the file name
    week          from the file name (every unit of a conversation shares it)
    turn          1-based exchange number; a student prompt and the reply to it
                  share one turn
    speaker       "student" | "ai"
    kind          "sentence" | "status"   (status: "웹 검색됨", "공유 시 숨겨지는 파일", ...)
    seq           0-based position in the conversation; the trajectory order
    text          the unit, verbatim apart from whitespace
    timestamp     the message's own UTC time

What is dropped, and why:
  * code blocks -- the pasted Claude Code prompts. They are long English
    instructions, often repeated near-verbatim, and would dominate the map.
  * citation chips ("huggingface.co +3") -- link titles, not the model's words.
  * the screen-reader heading each reply starts with -- it repeats the first
    paragraph.
Status lines are kept: they mark where the model searched or ran tools, which is
part of the conversation's shape.

Sentences shorter than --min-chars are merged into the next sentence of the same
paragraph ("좋은 질문이에요." on its own is a point with no meaning), or into the
previous one when they come last.

    uv run --python .venv/bin/python scripts/chat_to_trajectory.py \\
        .run/chats/kim.seoyeon__w1__01.html --roster roster.csv --out .run/chats/build

This writes <stem>.jsonl and a human-readable <stem>.preview.md next to it, and
touches no database.
"""

import argparse
import json
import re
import sys
from html.parser import HTMLParser
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models import WEEKS  # noqa: E402
from src.roster import Roster  # noqa: E402
from src.textnorm import normalize_text  # noqa: E402

FILENAME = re.compile(r"^(?P<target>[^_].*?)__w(?P<week>\d+)__(?P<nn>\d+)$")

# Icon glyphs: claude.ai draws its icons with a private-use font, and they come
# through as text ("웹 검색됨").
_PRIVATE_USE = re.compile(r"[-]")

# A sentence ends at . ! ? or … followed by whitespace. Decimals ("0.73") have no
# whitespace after the point and survive; the abbreviations below would not.
_SENTENCE_END = re.compile(r"(?<=[.!?…])\s+(?=\S)")
_ABBREVIATIONS = ("e.g.", "i.e.", "vs.", "etc.", "cf.", "Fig.", "fig.", "Dr.", "Mr.", "Ms.")

_VOID = frozenset({
    "area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta",
    "source", "track", "wbr", "path", "circle", "rect", "line", "polyline",
    "polygon", "use", "stop",
})
_BLOCKS = frozenset({"p", "li", "h1", "h2", "h3", "h4", "h5", "h6"})


class _Node:
    __slots__ = ("tag", "attrs", "parent", "kids")

    def __init__(self, tag, attrs, parent):
        self.tag, self.attrs, self.parent, self.kids = tag, dict(attrs), parent, []

    def walk(self):
        for kid in self.kids:
            if isinstance(kid, _Node):
                yield kid
                yield from kid.walk()

    def testid(self):
        return self.attrs.get("data-testid") or ""

    def classes(self):
        return self.attrs.get("class") or ""

    def inside(self, tag):
        node = self.parent
        while node is not None:
            if node.tag == tag:
                return True
            node = node.parent
        return False


class _TreeBuilder(HTMLParser):
    """Enough of a DOM for a saved page. The standard library only, on purpose:
    this runs on the class machine, whose environment is pinned."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.root = _Node("#root", [], None)
        self._cur = self.root

    def handle_starttag(self, tag, attrs):
        node = _Node(tag, attrs, self._cur)
        self._cur.kids.append(node)
        if tag not in _VOID:
            self._cur = node

    def handle_endtag(self, tag):
        node = self._cur
        while node is not None and node.tag != tag:
            node = node.parent
        if node is not None and node.parent is not None:
            self._cur = node.parent

    def handle_data(self, data):
        self._cur.kids.append(data)


def _text(node) -> str:
    """Visible text, with <br> as a line break and citation chips left out."""
    parts = []

    def collect(n):
        for kid in n.kids:
            if isinstance(kid, str):
                parts.append(kid)
            elif kid.tag == "br":
                parts.append("\n")
            elif "group/tag" in kid.classes():
                continue                    # citation chip: "huggingface.co +3"
            else:
                collect(kid)

    collect(node)
    return _PRIVATE_USE.sub("", "".join(parts))


def _clean(text: str) -> str:
    return " ".join(text.split())


def split_sentences(text: str) -> list[str]:
    """Line breaks first, then sentence ends. Students write outlines with no
    final punctuation ("slide 1", "Idea"), so a line is a boundary of its own."""
    out = []
    for line in text.split("\n"):
        line = _clean(line)
        if not line:
            continue
        pieces = _SENTENCE_END.split(line)
        merged = []
        for piece in pieces:
            if merged and merged[-1].endswith(_ABBREVIATIONS):
                merged[-1] = f"{merged[-1]} {piece}"
            else:
                merged.append(piece)
        out.extend(merged)
    return out


def merge_short(sentences: list[str], min_chars: int) -> list[str]:
    out, carry = [], ""
    for sentence in sentences:
        sentence = f"{carry} {sentence}".strip() if carry else sentence
        if len(sentence) < min_chars:
            carry = sentence
            continue
        out.append(sentence)
        carry = ""
    if carry:
        if out:
            out[-1] = f"{out[-1]} {carry}"
        else:
            out.append(carry)
    return out


def split_long(text: str, max_chars: int) -> list[str]:
    """A unit over the store's limit is cut at the last space before it."""
    out = []
    while len(text) > max_chars:
        cut = text.rfind(" ", 0, max_chars)
        if cut <= 0:
            cut = max_chars
        out.append(text[:cut].strip())
        text = text[cut:].strip()
    if text:
        out.append(text)
    return out


def _assistant_units(message):
    """(kind, text) in reading order."""
    for node in message.walk():
        testid = node.testid()
        if testid in ("tool-status-pill", "file-card-block"):
            yield "status", _clean(_text(node))
        elif node.tag in _BLOCKS:
            if "sr-only" in node.classes() or node.inside("pre"):
                continue
            if node.tag == "p" and node.inside("li"):
                continue                    # the <li> already carries it
            if node.tag == "li" and any(k.tag in ("p", "li") for k in node.walk()):
                continue                    # its own <p>/<li> children carry it
            if node.inside("button"):
                continue
            text = _text(node)
            if "font-base" in node.classes().split():
                # "인터랙티브 시각 자료는 공유 시 숨겨짐": a notice, not prose.
                yield "status", _clean(text)
            else:
                yield "prose", text


def _student_units(message):
    for node in message.walk():
        if node.tag == "p" and not node.inside("button"):
            yield "prose", _text(node)


def parse_conversation(html: str) -> list[dict]:
    """[{speaker, timestamp, units:[(kind, text)], code_chars}] in order."""
    builder = _TreeBuilder()
    builder.feed(html)
    messages, pending_files = [], []
    for node in builder.root.walk():
        testid = node.testid()
        if node.tag == "time" and node.attrs.get("datetime"):
            # Each message's first <time> follows it: inside a reply's action bar,
            # but beside a prompt rather than inside it.
            if messages and messages[-1]["timestamp"] is None:
                messages[-1]["timestamp"] = node.attrs["datetime"]
        elif testid == "file-thumbnail":
            # Sits beside, not inside, the prompt it was attached to.
            label = _clean(_text(node)) or "첨부 파일"
            pending_files.append(label)
        elif testid in ("user-message", "assistant-message"):
            student = testid == "user-message"
            units = list(_student_units(node) if student else _assistant_units(node))
            if student and pending_files:
                units = [("status", label) for label in pending_files] + units
                pending_files = []
            code_chars = sum(len(_clean(_text(n))) for n in node.walk() if n.tag == "pre")
            messages.append({
                "speaker": "student" if student else "ai",
                "timestamp": None,
                "units": units,
                "code_chars": code_chars,
            })
    return messages


def _segments(units, min_chars):
    """(kind, pieces) runs. Prose between two status lines is merged as one run,
    so a one-line paragraph ("김서연") joins its neighbour instead of standing
    alone; a status line is never merged into prose."""
    prose = []
    for kind, text in units:
        if kind == "status":
            if prose:
                yield "sentence", merge_short(prose, min_chars)
                prose = []
            if normalize_text(text):
                yield "status", [text]
        else:
            prose.extend(split_sentences(text))
    if prose:
        yield "sentence", merge_short(prose, min_chars)


def build_rows(messages, *, conversation, target_id, week, min_chars, max_chars):
    rows, turn = [], 0
    for message in messages:
        if message["speaker"] == "student" or turn == 0:
            turn += 1
        for unit_kind, pieces in _segments(message["units"], min_chars):
            for piece in pieces:
                for chunk in split_long(normalize_text(piece), max_chars):
                    if not chunk:
                        continue
                    rows.append({
                        "conversation": conversation,
                        "target_id": target_id,
                        "week": week,
                        "turn": turn,
                        "speaker": message["speaker"],
                        "kind": unit_kind,
                        "seq": len(rows),
                        "text": chunk,
                        "timestamp": message["timestamp"],
                    })
    return rows


def _preview(rows, messages, conversation) -> str:
    lines = [f"# {conversation}", ""]
    counts = {}
    for row in rows:
        key = (row["speaker"], row["kind"])
        counts[key] = counts.get(key, 0) + 1
    lines.append("| speaker | kind | units |")
    lines.append("|---|---|---|")
    for (speaker, kind), n in sorted(counts.items()):
        lines.append(f"| {speaker} | {kind} | {n} |")
    dropped = sum(m["code_chars"] for m in messages)
    lines += ["", f"Code blocks dropped: {dropped} characters.", ""]
    current = None
    for row in rows:
        if (row["turn"], row["speaker"]) != current:
            current = (row["turn"], row["speaker"])
            who = "학생" if row["speaker"] == "student" else "AI"
            lines += ["", f"## turn {row['turn']} · {who} · {row['timestamp']}", ""]
        mark = "· " if row["kind"] == "status" else ""
        lines.append(f"{row['seq']:>4}  {mark}{row['text']}")
    return "\n".join(lines) + "\n"


def convert(path: Path, roster: Roster, *, min_chars: int, max_chars: int):
    match = FILENAME.match(path.stem)
    if path.suffix.lower() != ".html" or match is None:
        raise ValueError(
            f"{path.name}: expected <target_id>__w<week>__<nn>.html, "
            "e.g. kim.seoyeon__w1__01.html")
    week = int(match["week"])
    if week not in WEEKS:
        raise ValueError(f"{path.name}: week {week} is not one of {WEEKS}")
    entry = roster.resolve(match["target"])
    if entry is None or entry.role != "student":
        raise ValueError(f"{path.name}: {match['target']!r} is not a student on the roster")

    messages = parse_conversation(path.read_text(encoding="utf-8"))
    if not messages:
        raise ValueError(
            f"{path.name}: no messages found. Was it saved as \"Webpage, Complete\" "
            "after the whole conversation had loaded?")
    if not any(m["speaker"] == "ai" for m in messages):
        raise ValueError(f"{path.name}: no model replies found")
    rows = build_rows(messages, conversation=path.stem, target_id=entry.id, week=week,
                      min_chars=min_chars, max_chars=max_chars)
    return rows, messages


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("inputs", nargs="+", type=Path, help="saved .html share pages")
    ap.add_argument("--roster", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True, help="output directory")
    ap.add_argument("--min-chars", type=int, default=20)
    ap.add_argument("--max-chars", type=int, default=1000)
    args = ap.parse_args(argv)

    roster = Roster.from_path(args.roster)
    args.out.mkdir(parents=True, exist_ok=True)
    failed = 0
    for path in args.inputs:
        try:
            rows, messages = convert(path, roster, min_chars=args.min_chars,
                                     max_chars=args.max_chars)
        except (OSError, ValueError) as exc:
            print(f"  refused {exc}", file=sys.stderr)
            failed += 1
            continue
        jsonl = args.out / f"{path.stem}.jsonl"
        preview = args.out / f"{path.stem}.preview.md"
        jsonl.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                         encoding="utf-8")
        preview.write_text(_preview(rows, messages, path.stem), encoding="utf-8")
        turns = max(r["turn"] for r in rows)
        print(f"{path.name}: {len(rows)} units over {turns} turns -> {jsonl}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
