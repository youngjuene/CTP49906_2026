"""Durable state, and the cache key that keeps two models apart. CPU only.

The defect: a model swap left several hundred stale 768-dimensional vectors in
the cache next to the new 384-dimensional ones. Nothing raised at the swap. The
projection crashed later, on a ragged vstack, with a message that pointed at
numpy rather than at the swap.

So the cache is keyed by (text_hash, cache_key) where cache_key describes the
whole vector space -- model *and* prompt convention, since the same weights with a
different prompt produce an incomparable space. The other half of this file is
the NFC normalisation: without it a Mac-typed sentence and the same sentence from
a phone are different cache rows, the cache never hits, and every restart
re-encodes the whole semester.

Run:  python -m pytest feedback-atlas/tests/test_store_sqlite.py
  or:  python feedback-atlas/tests/test_store_sqlite.py
"""
import sys
import unicodedata
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # -> feedback-atlas/

from src.config import MODEL_REGISTRY, EmbedderSpec, cache_key  # noqa: E402
from src.models import new_opinion  # noqa: E402
from src.store import Store, blob_to_vector, vector_to_blob  # noqa: E402
from src.textnorm import text_hash  # noqa: E402

GEMMA = cache_key(MODEL_REGISTRY["embeddinggemma"])
E5 = cache_key(MODEL_REGISTRY["e5-small-ko"])


def _store(tmp_path, name="atlas.db"):
    s = Store(str(tmp_path / name))
    s.migrate()
    return s


def _op(text="좋았어요."):
    return new_opinion(reviewer_id="kang.minsu", target_id="kim.seoyeon", text=text)


def test_a_float32_vector_round_trips_bit_exactly(tmp_path):
    """Models emit float64 as often as float32. Storing whichever arrives would
    make the blob width depend on the model, and the fixed-width read back would
    return garbage rather than fail."""
    store = _store(tmp_path)
    op = _op()
    store.insert_opinion(op, text_hash(op.text))
    v64 = np.random.default_rng(0).normal(size=768)  # float64 on purpose
    store.put_embeddings(GEMMA, 768, {text_hash(op.text): v64})
    back = store.get_embeddings([text_hash(op.text)], GEMMA)[text_hash(op.text)]
    assert back.dtype == np.float32
    assert np.array_equal(back, v64.astype(np.float32))


def test_a_vector_stored_under_one_model_is_not_returned_for_another(tmp_path):
    """The ragged-vstack defect, at its source."""
    store = _store(tmp_path)
    op = _op()
    store.insert_opinion(op, text_hash(op.text))
    store.put_embeddings(GEMMA, 768, {text_hash(op.text): np.zeros(768)})
    assert store.get_embeddings([text_hash(op.text)], E5) == {}
    assert [t for _, t in store.missing_embeddings(E5)] == [op.text]
    assert store.missing_embeddings(GEMMA) == []


def test_changing_only_the_prompt_convention_invalidates_the_cache():
    """Same weights, same dimension, different prompt -- and therefore a different,
    incomparable vector space. Keying on model_id alone would silently mix them,
    and the map would look plausible while being meaningless.
    """
    a = EmbedderSpec(model_id="m/x", dim=768, prompt_name="Clustering")
    b = EmbedderSpec(model_id="m/x", dim=768, query_prefix="query: ")
    c = EmbedderSpec(model_id="m/x", dim=768, raw_input=True)
    assert len({cache_key(a), cache_key(b), cache_key(c)}) == 3


def test_the_same_sentence_typed_on_a_mac_reuses_the_cached_vector(tmp_path):
    """NFD versus NFC. Miss this and the cache never hits for half the room."""
    store = _store(tmp_path)
    nfc = "숨이 막혔어요."
    nfd = unicodedata.normalize("NFD", nfc)
    assert nfc != nfd
    store.put_embeddings(GEMMA, 4, {text_hash(nfc): np.arange(4)})
    assert text_hash(nfd) in store.get_embeddings([text_hash(nfd)], GEMMA)


def test_two_people_writing_the_same_sentence_share_one_cached_vector(tmp_path):
    """Embeddings are keyed by text, not by opinion id: '좋았어요' from thirty
    people is one row and one encode, not thirty."""
    store = _store(tmp_path)
    a, b = _op("좋았어요."), _op("  좋았어요.  ")
    store.insert_opinion(a, text_hash(a.text))
    store.insert_opinion(b, text_hash(b.text))
    assert a.id != b.id
    assert len(store.missing_embeddings(GEMMA)) == 1


def test_swapping_a_model_and_swapping_back_costs_nothing(tmp_path):
    """put_embeddings adds rows under a new key rather than replacing the old
    ones, so an instructor who tries KURE mid-semester and reverts does not pay
    for a second full re-encode."""
    store = _store(tmp_path)
    op = _op()
    store.insert_opinion(op, text_hash(op.text))
    store.put_embeddings(GEMMA, 768, {text_hash(op.text): np.zeros(768)})
    store.put_embeddings(E5, 384, {text_hash(op.text): np.ones(384)})
    assert store.missing_embeddings(GEMMA) == []
    assert store.missing_embeddings(E5) == []
    assert store.get_embeddings([text_hash(op.text)], GEMMA)[text_hash(op.text)].shape == (768,)


def test_a_vector_whose_width_does_not_match_the_declared_dim_raises(tmp_path):
    """Fail loudly at the read, where the cause is still visible, rather than
    later inside numpy."""
    try:
        blob_to_vector(vector_to_blob(np.zeros(384)), 768)
    except ValueError as e:
        assert "384" in str(e) and "768" in str(e)
    else:
        raise AssertionError("a width mismatch must raise")


def test_a_vector_contradicting_the_declared_dim_is_refused_on_write(tmp_path):
    """Found by QA. The width mismatch used to be caught at read time, hundreds of
    opinions later, with nothing in the message to say which model swap caused it.
    Rejecting on write keeps the model and its registry entry in scope."""
    store = _store(tmp_path)
    try:
        store.put_embeddings(GEMMA, 768, {"abc123": np.zeros(384)})
    except ValueError as e:
        assert "384" in str(e) and "768" in str(e)
    else:
        raise AssertionError("a vector contradicting dim must not be stored")
    assert store.get_embeddings(["abc123"], GEMMA) == {}


def test_the_same_opinion_submitted_twice_is_not_two_points(tmp_path):
    """A retry over a flaky tunnel is one opinion."""
    store = _store(tmp_path)
    op = _op()
    assert store.insert_opinion(op, text_hash(op.text)) is True
    assert store.insert_opinion(op, text_hash(op.text)) is False
    assert len(store.all_opinions()) == 1


def test_opinions_and_coordinates_survive_a_restart(tmp_path):
    """PRD 6. Coordinates are persisted too, so a restart does not rotate the map:
    the previous layout is still there to anchor the next fit against."""
    store = _store(tmp_path)
    # Explicit, distinct timestamps: the ordering contract is what is under test,
    # and leaning on wall-clock resolution to separate three inserts in a tight
    # loop would make this pass or fail by how fast the machine is.
    ops = [new_opinion(reviewer_id="kang.minsu", target_id="kim.seoyeon",
                       text=f"의견 {i}", now=f"2026-09-07T04:11:{20 + i:02d}.000Z")
           for i in range(3)]
    for op in ops:
        store.insert_opinion(op, text_hash(op.text))
    store.put_coords({ops[0].id: (1.5, -2.5)}, layout_rev=7)
    store.close()

    reopened = _store(tmp_path)
    restored = reopened.all_opinions()
    assert len(restored) == 3
    assert {o.id for o in restored} == {o.id for o in ops}
    # Ordering is part of the contract: the projection builds its matrix in this
    # order, so a restart that reordered rows would permute the whole layout.
    assert restored == sorted(restored, key=lambda o: (o.timestamp, o.id))
    assert restored[0].text == ops[0].text
    assert reopened.all_coords()[ops[0].id] == (1.5, -2.5)


def test_the_database_is_in_wal_mode(tmp_path):
    """So a read during an in-flight write does not block. (Note for deployment:
    WAL is unreliable on NFS/SMB -- the db belongs on local disk.)"""
    store = _store(tmp_path)
    assert store._db.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"


def test_coordinates_are_upserted_not_duplicated(tmp_path):
    store = _store(tmp_path)
    op = _op()
    store.insert_opinion(op, text_hash(op.text))
    store.put_coords({op.id: (1.0, 1.0)}, 1)
    store.put_coords({op.id: (2.0, 2.0)}, 2)
    assert store.all_coords() == {op.id: (2.0, 2.0)}


def test_a_database_written_by_a_future_schema_refuses_to_open(tmp_path):
    """Better than reading rows with the wrong shape and reporting nonsense."""
    store = _store(tmp_path)
    store.set_meta("schema_version", "99")
    store.close()
    try:
        _store(tmp_path)
    except RuntimeError as e:
        assert "99" in str(e)
    else:
        raise AssertionError("a future schema version must not be opened")


def test_get_embeddings_handles_more_hashes_than_sqlites_variable_limit(tmp_path):
    """SQLite caps host parameters per statement (999 in older builds), so the
    lookup chunks. A semester's worth of text goes through this path on restart."""
    store = _store(tmp_path)
    vectors = {f"{i:064x}": np.full(4, i, dtype=np.float32) for i in range(1500)}
    store.put_embeddings(GEMMA, 4, vectors)
    got = store.get_embeddings(list(vectors), GEMMA)
    assert len(got) == 1500
    assert got[f"{7:064x}"][0] == 7


if __name__ == "__main__":
    import tempfile

    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    fails = 0
    for fn in fns:
        try:
            if fn.__code__.co_varnames[: fn.__code__.co_argcount] == ("tmp_path",):
                with tempfile.TemporaryDirectory() as d:
                    fn(Path(d))
            else:
                fn()
            print("PASS", fn.__name__)
        except Exception as e:  # noqa: BLE001
            fails += 1
            print("FAIL", fn.__name__, "->", type(e).__name__, e)
    print(f"\n{len(fns) - fails} passed, {fails} failed")
    sys.exit(1 if fails else 0)
