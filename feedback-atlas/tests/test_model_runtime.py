"""A character tokenizer intentionally stresses limits without model weights."""
import asyncio
import threading
import time
import numpy as np
import pytest
from src.embedder import HashEmbedder


class CountedEmbedder(HashEmbedder):
    def __init__(self):
        super().__init__(dim=8)
        self.calls=[]
    def token_counts(self,text,purpose):
        return len(text),len(text)+300  # prepared limit binds before content limit
    def encode_task(self,texts,purpose):
        self.calls.append((purpose,list(texts),threading.get_ident()))
        return self.encode(texts)
    def task_identity(self,purpose):
        return "counted:"+purpose


def test_worker_serializes_both_tasks_and_does_not_block_event_loop():
    from src.model_runtime import ModelRuntime
    async def check():
        runtime=ModelRuntime(CountedEmbedder())
        active=0
        highest=0
        threads=set()
        ticks=0
        def work():
            nonlocal active,highest
            active+=1
            highest=max(highest,active)
            threads.add(threading.get_ident())
            time.sleep(.02)
            runtime.encode_units(["a"])
            runtime.encode_similarity(["b"])
            active-=1
        async def heartbeat():
            nonlocal ticks
            for _ in range(10):
                await asyncio.sleep(.002)
                ticks+=1
        await asyncio.gather(*(runtime.run(work) for _ in range(4)),heartbeat())
        runtime.close()
        assert highest==1 and len(threads)==1 and ticks==10
        assert threading.get_ident() not in threads
    asyncio.run(check())


def test_model_methods_refuse_event_loop_thread():
    from src.model_runtime import ModelRuntime
    runtime=ModelRuntime(CountedEmbedder())
    with pytest.raises(RuntimeError,match="worker"):
        runtime.encode_units(["a"])
    runtime.close()


def test_windows_account_for_real_prompt_overhead_and_pool_normalized_vectors():
    from src.model_runtime import ModelRuntime
    embedder=CountedEmbedder()
    runtime=ModelRuntime(embedder)
    text="x"*4000
    async def check():
        windows=await runtime.run(runtime.text_windows,text,"clustering")
        output=await runtime.run(runtime.encode_units,[text,"short"])
        return windows,output
    windows,output=asyncio.run(check())
    runtime.close()
    assert "".join(window for window,weight in windows)==text
    assert all(len(window)<=1792 and len(window)+300<=2048 for window,weight in windows)
    assert np.allclose(np.linalg.norm(output,axis=1),1)
    expected=np.average(embedder.encode([w for w,_ in windows]),axis=0,weights=[n for _,n in windows])
    expected/=np.linalg.norm(expected)
    assert np.allclose(output[0],expected)
    assert np.array_equal(output[1],embedder.encode(["short"])[0])
    assert all(len(texts)<=32 for _,texts,_ in embedder.calls)


def test_hash_embedder_works_explicitly_without_claiming_tokenizer_quality():
    from src.model_runtime import ModelRuntime
    runtime=ModelRuntime(HashEmbedder(dim=8))
    result=asyncio.run(runtime.run(runtime.encode_units,["한글 👩🏽‍💻"]))
    runtime.close()
    assert result.shape==(1,8)
    assert "hash" in runtime.unit_cache_key
    assert runtime.similarity_key != runtime.unit_cache_key


def test_cache_identity_records_pooling_policy_and_prompt():
    from src.model_runtime import ModelRuntime
    a=ModelRuntime(CountedEmbedder())
    b=ModelRuntime(CountedEmbedder(),content_token_limit=1000)
    assert a.unit_cache_key != b.unit_cache_key
    assert "pool" in a.unit_cache_key
    assert a.unit_cache_key != a.similarity_key
    a.close(); b.close()


class CharacterTokenizer:
    def __call__(self,text,*,add_special_tokens=True,truncation=False):
        assert truncation is False
        return {"input_ids":list(range(len(text)+(2 if add_special_tokens else 0)))}


class RecordedModel:
    prompts={"Clustering":"cluster: ","STS":"similarity: "}
    default_prompt_name="Clustering"
    max_seq_length=2048
    tokenizer=CharacterTokenizer()
    def __init__(self):
        self.calls=[]
    def encode(self,texts,**kwargs):
        self.calls.append((texts,kwargs))
        return np.ones((len(texts),8),dtype=np.float32)/np.sqrt(8)


def loaded_stub():
    from src.config import EmbedderSpec
    from src.embedder import SentenceTransformerEmbedder
    embedder=SentenceTransformerEmbedder.__new__(SentenceTransformerEmbedder)
    embedder.spec=EmbedderSpec(model_id="test:gemma",dim=8,prompt_name="Clustering")
    embedder.dim=8
    embedder.model_id=embedder.spec.model_id
    embedder._model=RecordedModel()
    return embedder


def test_similarity_uses_supported_explicit_prompt_without_mutating_default():
    embedder=loaded_stub()
    embedder.encode_task(["a"],"similarity")
    embedder.encode_task(["b"],"clustering")
    assert embedder._model.calls[0][1]["prompt_name"]=="STS"
    assert embedder._model.calls[1][1]["prompt_name"]=="Clustering"
    assert embedder._model.default_prompt_name=="Clustering"
    assert embedder.token_counts("abc","similarity")== (3,len("similarity: abc")+2)


def test_unknown_similarity_prompt_fails_instead_of_silently_using_clustering():
    embedder=loaded_stub()
    embedder._model.prompts={"Clustering":"cluster: "}
    with pytest.raises(RuntimeError,match="similarity"):
        embedder.encode_task(["a"],"similarity")


def test_direct_task_encode_refuses_silent_truncation():
    embedder=loaded_stub()
    with pytest.raises(ValueError,match="token"):
        embedder.encode_task(["x"*2200],"similarity")
    assert not embedder._model.calls


def test_run_sync_uses_same_worker_and_nested_call_cannot_deadlock():
    from src.model_runtime import ModelRuntime
    runtime=ModelRuntime(CountedEmbedder())
    def nested():
        return runtime.run_sync(threading.get_ident)
    sync_thread=runtime.run_sync(nested)
    async_thread=asyncio.run(runtime.run(threading.get_ident))
    runtime.close()
    assert sync_thread==async_thread
    assert sync_thread != threading.get_ident()


def test_cache_identity_separates_fake_vector_dimensions():
    from src.model_runtime import ModelRuntime
    narrow=ModelRuntime(HashEmbedder(dim=8))
    wide=ModelRuntime(HashEmbedder(dim=16))
    try:
        assert narrow.unit_cache_key != wide.unit_cache_key
    finally:
        narrow.close(); wide.close()


def test_real_evaluator_long_probe_actually_requires_multiple_model_windows():
    from scripts.evaluate_segmentation import evaluate_long_unit
    from src.model_runtime import ModelRuntime
    from src.segmentation import SegmentationPolicy
    runtime=ModelRuntime(CountedEmbedder())
    try:
        result=runtime.run_sync(evaluate_long_unit,runtime,SegmentationPolicy())
    finally:
        runtime.close()
    assert result['unsplit_prepared_tokens'] > 2048
    assert result['raw_codepoints'] <= 20000
    assert result['internal_windows'] >= 2
    assert result['visible_units'] == 1
    assert result['max_content_tokens'] <= 1792
    assert result['max_prepared_tokens'] <= 2048
    assert result['internal_partition_exact']
    assert result['pooling_matches_independent_weighted_mean']
    assert result['finite']
    assert result['norm'] == pytest.approx(1,abs=1e-5)
