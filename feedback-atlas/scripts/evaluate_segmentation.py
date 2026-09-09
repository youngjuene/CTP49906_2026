#!/usr/bin/env python3
"""Reproducible real-model evaluation. Synthetic expectations are not human labels.

Calibration computes each prompt/context combination once, then scores frozen
threshold candidates against the same scores. Held-out evaluation does not tune.
"""
import argparse
import asyncio
from dataclasses import asdict, replace
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import resolve_spec
from src.embedder import SentenceTransformerEmbedder
from src.model_runtime import ModelRuntime
from src.segmentation import boundary_scores, candidate_spans, load_policy, split_feedback, validate_spans
from src.submissions import FeedbackSpan


def summarize(rows, cases):
    required = hits = forbidden = cuts = singles = preserved = covered = 0
    for row,case in zip(rows,cases):
        predicted = {end for _,end in row.get("spans",[]) if end < len(case['raw_text'])}
        expected = set(case['required'])
        disallowed = set(case['forbidden'])
        required += len(expected); hits += len(predicted & expected)
        forbidden += len(disallowed); cuts += len(predicted & disallowed)
        singles += int(case['single_topic'])
        preserved += int(case['single_topic'] and len(row.get('spans',[]))==1)
        covered += int(row.get('coverage_ok',False))
    latency = [row['latency_ms'] for row in rows]
    return dict(cases=len(cases),required_boundaries=required,required_hits=hits,
                required_recall=hits/required if required else None,
                forbidden_boundaries=forbidden,forbidden_cuts=cuts,
                forbidden_cut_rate=cuts/forbidden if forbidden else None,
                single_topic_cases=singles,single_topic_preserved=preserved,
                single_topic_preservation=preserved/singles if singles else None,
                exact_coverage_cases=covered,failures=sum('error' in row for row in rows),
                latency_ms_p50=float(np.percentile(latency,50)),
                latency_ms_p95=float(np.percentile(latency,95)))


def row_from_scores(case,candidates,scores,policy,latency):
    boundaries=[0]+[s.end for s,score in zip(candidates,scores) if score<policy.threshold]+[len(case['raw_text'])]
    spans=tuple(FeedbackSpan(a,b) for a,b in zip(boundaries,boundaries[1:]))
    row=dict(case_id=case['case_id'],spans=[[s.start,s.end] for s in spans],latency_ms=latency,
             scores=scores,candidate_boundaries=[s.end for s in candidates[:-1]])
    try:
        validate_spans(case['raw_text'],spans)
        row['coverage_ok']=True
    except ValueError as exc:
        row.update(coverage_ok=False,error=str(exc))
    return row


def chonkie_probe(runtime):
    """Probe actual stock adapter without permitting known truncated inference."""
    from chonkie import SemanticChunker
    from chonkie.embeddings import SentenceTransformerEmbeddings
    embedder=runtime.embedder
    adapter=SentenceTransformerEmbeddings(embedder._model)
    original_default=getattr(embedder._model,'default_prompt_name',None)
    output={'version':version('chonkie'),'shared_model_object':adapter.model is embedder._model,
            'default_prompt_name':original_default,'max_seq_length':embedder.max_seq_length,
            'prompts':embedder._model.prompts,'stock_encode_has_explicit_task_prompt':False}
    original_encode=adapter.embed_batch
    observed=[]
    def guarded(texts):
        for text in texts:
            content,total=embedder.token_counts(text,'similarity')
            observed.append({'content_tokens':content,'prepared_tokens':total})
            if content>1792 or total>2048:
                raise ValueError('evaluation guard: stock adapter would encode oversized window')
        return original_encode(texts)
    adapter.embed_batch=guarded
    chunker=SemanticChunker(embedding_model=adapter,threshold=.01,chunk_size=1792,
                            similarity_window=1,min_characters_per_sentence=1,skip_window=0)
    raw='  조명이 편안했습니다. 조명이 편안했습니다.\n\n음악이 너무 큽니다. 볼륨을 낮추세요.  '
    chunks=chunker.chunk(raw)
    output['short_offsets_exact']=all(raw[c.start_index:c.end_index]==c.text for c in chunks) and ''.join(c.text for c in chunks)==raw
    long='조명의 밝기와 색이 천천히 변해서 편안했습니다 ' * 400
    try:
        chunker.chunk(long+'. 다른 문장입니다. 마지막 문장입니다.')
        output['oversized_probe']='accepted'
    except ValueError as exc:
        output['oversized_probe']=str(exc)
    output['largest_observed_content_tokens']=max(row['content_tokens'] for row in observed)
    output['largest_observed_prepared_tokens']=max(row['prepared_tokens'] for row in observed)
    # Safely use runtime embeddings to isolate Chonkie's output length policy.
    adapter.embed_batch=lambda texts: list(runtime.encode_similarity(texts))
    repeated='조명이 편안했습니다. ' * 400
    chunks=chunker.chunk(repeated)
    output['coherent_repeated_case_units_with_chunk_size1792']=len(chunks)
    output['default_prompt_unchanged']=getattr(embedder._model,'default_prompt_name',None)==original_default
    return output


def evaluate_long_unit(runtime, policy):
    """Supplemental synthetic stress input, independent of held-out labels.

    Grow using the actual tokenizer, not codepoint length. This must exercise
    pooling even if a corpus's longest string still fits in one model window.
    """
    text = "조명의 따뜻한 색과 부드러운 변화가 만드는 편안한 분위기"
    while runtime._counts(text, "clustering")[1] <= 2048:
        text = text + " " + text
        if len(text) > 20000:
            raise AssertionError("could not construct over-context probe within intake limit")
    content, prepared = runtime._counts(text, "clustering")
    windows = runtime.text_windows(text, "clustering")
    split = split_feedback(text, runtime, policy)
    vector = runtime.encode_units([text])

    # Independent reference: encode bounded substrings directly, then pool with
    # the content-token weights. Do not reuse the runtime's pooled result.
    encoder = getattr(runtime.embedder, "encode_task", None)
    raw_vectors = []
    for start in range(0, len(windows), runtime.batch_size):
        batch = [window for window, _ in windows[start:start + runtime.batch_size]]
        raw_vectors.append(encoder(batch, "clustering") if encoder else runtime.embedder.encode(batch))
    reference = np.average(np.vstack(raw_vectors), axis=0, weights=[weight for _, weight in windows])
    reference /= np.linalg.norm(reference)
    counts = [runtime._counts(window, "clustering") for window, _ in windows]
    result = {
        "input_origin": "supplemental-synthetic-stress-no-human-labels",
        "raw_codepoints": len(text), "unsplit_content_tokens": content,
        "unsplit_prepared_tokens": prepared, "internal_windows": len(windows),
        "visible_units": len(split.spans),
        "max_content_tokens": max(count[0] for count in counts),
        "max_prepared_tokens": max(count[1] for count in counts),
        "internal_partition_exact": "".join(window for window, _ in windows) == text,
        "pooling_matches_independent_weighted_mean": bool(np.allclose(vector[0], reference, atol=1e-5)),
        "finite": bool(np.isfinite(vector).all()), "norm": float(np.linalg.norm(vector[0])),
    }
    assert prepared > 2048 and len(windows) >= 2
    assert result["max_content_tokens"] <= 1792 and result["max_prepared_tokens"] <= 2048
    assert result["visible_units"] == 1 and result["internal_partition_exact"]
    assert result["pooling_matches_independent_weighted_mean"] and result["finite"]
    assert abs(result["norm"] - 1) <= 1e-5
    return result


async def evaluate(args):
    path=Path(args.cases)
    all_cases=[json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
    cases=[case for case in all_cases if case['split']==args.split]
    if not cases:
        raise ValueError('no cases for selected split')
    if args.calibrate and args.split!='calibration':
        raise ValueError('calibration may never read held-out cases')
    policy=load_policy(args.policy)
    started=time.perf_counter()
    embedder=SentenceTransformerEmbedder(resolve_spec('embeddinggemma'),device=args.device)
    runtime=ModelRuntime(embedder)
    load_ms=(time.perf_counter()-started)*1000
    try:
        await runtime.run(runtime.encode_similarity,['모델 준비를 확인합니다.'])
        result={'corpus_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                'split':args.split,'policy':asdict(policy),'algorithm_key':policy.fingerprint(runtime),
                'model_load_ms':load_ms,'model_identity':embedder.task_identity('similarity'),
                'unit_cache_key':runtime.unit_cache_key,'similarity_key':runtime.similarity_key,
                'annotation_origins':sorted({c['annotation_origin'] for c in cases}),
                'human_quality_gate_passed':False,'human_quality_gate_reason':'No two-reader human judgments supplied.',
                'versions':{p:version(p) for p in ['numpy','sentence-transformers','transformers']}}
        baseline=[]
        for case in cases:
            before=time.perf_counter(); spans=candidate_spans(case['raw_text'])
            row={'case_id':case['case_id'],'spans':[[s.start,s.end] for s in spans],
                 'latency_ms':(time.perf_counter()-before)*1000,'coverage_ok':True}
            # Baseline can exceed automatic unit limit; report it as a failure.
            try: validate_spans(case['raw_text'],spans)
            except ValueError as exc: row.update(error=str(exc),coverage_ok=False)
            baseline.append(row)
        result['structure_baseline']={'metrics':summarize(baseline,cases),'cases':baseline}
        if args.calibrate:
            candidates=[]
            for prompt in ('similarity','clustering'):
                for width in (1,2):
                    prepared=[]
                    local=replace(policy,prompt=prompt,context_candidates=width)
                    for case in cases:
                        before=time.perf_counter()
                        spans,scores=await runtime.run(boundary_scores,case['raw_text'],runtime,local)
                        prepared.append((spans,scores,(time.perf_counter()-before)*1000))
                    for threshold in (.6,.7,.8,.9):
                        variant=replace(local,threshold=threshold)
                        rows=[row_from_scores(case,*data[:2],variant,data[2]) for case,data in zip(cases,prepared)]
                        metrics=summarize(rows,cases)
                        # Fixed before looking at heldout: balance recall, forbidden
                        # cuts and preservation. This is synthetic screening only.
                        score=(metrics['required_recall'] or 0)-(metrics['forbidden_cut_rate'] or 0)+.5*(metrics['single_topic_preservation'] or 0)-metrics['failures']
                        candidates.append({'policy':asdict(variant),'metrics':metrics,'selection_score':score,'cases':rows})
            result['candidates']=candidates
            winner=max(candidates,key=lambda item:item['selection_score'])
            result['selected_policy']=winner['policy']
        else:
            rows=[]
            for case in cases:
                before=time.perf_counter()
                try:
                    split=await runtime.run(split_feedback,case['raw_text'],runtime,policy)
                    row={'case_id':case['case_id'],'spans':[[s.start,s.end] for s in split.spans],
                         'coverage_ok':True}
                except ValueError as exc:
                    row={'case_id':case['case_id'],'spans':[],'coverage_ok':False,'error':str(exc)}
                row['latency_ms']=(time.perf_counter()-before)*1000
                rows.append(row)
            result['semantic']={'metrics':summarize(rows,cases),'cases':rows}
        if args.chonkie_probe:
            result['chonkie_probe']=await runtime.run(chonkie_probe,runtime)
        result['long_unit_vector_check']=await runtime.run(evaluate_long_unit,runtime,policy)
        Path(args.output).parent.mkdir(parents=True,exist_ok=True)
        Path(args.output).write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        print(json.dumps({key:value for key,value in result.items() if key in ('selected_policy','chonkie_probe','long_unit_vector_check')},ensure_ascii=False))
        if 'semantic' in result: print(json.dumps(result['semantic']['metrics']))
        print('Evaluation saved to',args.output)
    finally:
        runtime.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases',required=True)
    parser.add_argument('--split',choices=('calibration','heldout'),required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--policy',required=True)
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--calibrate',action='store_true')
    parser.add_argument('--chonkie-probe',action='store_true')
    asyncio.run(evaluate(parser.parse_args()))
