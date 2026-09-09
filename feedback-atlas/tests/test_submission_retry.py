"""Durable submit receipts for websocket retry.

A classroom phone can send an opinion, lose the ack frame, reconnect, and replay
the same nonce. The retry must return the original opinion id without inserting a
second point, even after a server restart.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # -> feedback-atlas/

from src.models import new_opinion  # noqa: E402
from src.store import Store  # noqa: E402
from src.textnorm import text_hash  # noqa: E402


def _store(tmp_path):
    store = Store(str(tmp_path / "atlas.db"))
    store.migrate()
    return store


def _op(reviewer_id="writer1", text="확인 응답이 끊겨도 한 번만 저장합니다."):
    return new_opinion(
        reviewer_id=reviewer_id,
        target_id="target1",
        text=text,
        source="human",
        week=2,
        now="2026-09-08T04:11:22.000Z")


def test_replayed_nonce_after_restart_returns_original_opinion_id(tmp_path):
    store = _store(tmp_path)
    first = _op()
    inserted, opinion_id = store.insert_opinion_with_receipt(
        first, text_hash(first.text), "phone-ack-1")
    assert (inserted, opinion_id) == (True, first.id)
    store.close()

    reopened = _store(tmp_path)
    replay = _op(text="same nonce must not store this newer body")
    assert reopened.submission_receipt("writer1", "phone-ack-1") == first.id
    inserted, opinion_id = reopened.insert_opinion_with_receipt(
        replay, text_hash(replay.text), "phone-ack-1")

    assert (inserted, opinion_id) == (False, first.id)
    assert [op.id for op in reopened.all_opinions()] == [first.id]
    assert reopened.all_opinions()[0].text == first.text


def test_same_nonce_from_another_reviewer_is_a_distinct_submission(tmp_path):
    store = _store(tmp_path)
    a = _op(reviewer_id="writer1", text="첫 번째 작성자의 의견입니다.")
    b = _op(reviewer_id="writer2", text="두 번째 작성자의 의견입니다.")

    assert store.insert_opinion_with_receipt(a, text_hash(a.text), "same-phone-nonce") == (
        True, a.id)
    assert store.insert_opinion_with_receipt(b, text_hash(b.text), "same-phone-nonce") == (
        True, b.id)

    assert {op.id for op in store.all_opinions()} == {a.id, b.id}
    assert store.submission_receipt("writer1", "same-phone-nonce") == a.id
    assert store.submission_receipt("writer2", "same-phone-nonce") == b.id


def test_repeated_delivery_does_not_duplicate_or_replace_opinion(tmp_path):
    store = _store(tmp_path)
    first = _op(text="첫 전송 본문입니다.")
    second = _op(text="재전송 때 브라우저에 남아 있던 다른 본문입니다.")

    assert store.insert_opinion_with_receipt(first, text_hash(first.text), "n-1") == (
        True, first.id)
    for _ in range(3):
        assert store.insert_opinion_with_receipt(second, text_hash(second.text), "n-1") == (
            False, first.id)

    opinions = store.all_opinions()
    assert len(opinions) == 1
    assert opinions[0].id == first.id
    assert opinions[0].text == first.text


def test_empty_or_malformed_nonce_keeps_backward_compatible_insert_path(tmp_path):
    store = _store(tmp_path)
    a = _op(text="옛 클라이언트 첫 의견입니다.")
    b = _op(text="옛 클라이언트 두 번째 의견입니다.")

    assert store.insert_opinion_with_receipt(a, text_hash(a.text), "") == (True, a.id)
    assert store.insert_opinion_with_receipt(b, text_hash(b.text), None) == (True, b.id)

    assert len(store.all_opinions()) == 2
    assert store.submission_receipt("writer1", "") is None
    assert store.submission_receipt("writer1", None) is None
