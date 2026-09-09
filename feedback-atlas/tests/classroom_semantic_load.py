"""Opt-in live v2 load measurement, temporary data only; fake or cached real CPU model."""
import asyncio
import json
import os
from pathlib import Path
import socket
import threading
import time

import httpx
import websockets

from src.config import load_config
from src.models import new_opinion
from src.protocol import PROTOCOL_VERSION
from src.server import create_app
from src.store import Store
from src.textnorm import text_hash

CODE = 'temporary-semantic-load'


def percentile(values, pct):
    ordered = sorted(values)
    return ordered[min(len(ordered)-1, round((len(ordered)-1)*pct))] if ordered else 0


def long_feedback(i):
    topics = ['조명이 따뜻해서 공간이 편안합니다. 색의 대비가 부드럽습니다. ',
              '음악이 너무 커서 대화를 듣기 어렵습니다. 음량을 낮추면 좋겠습니다. ',
              '입구의 안내 표지판이 잘 보이지 않습니다. 이동 경로를 표시하면 좋겠습니다. ',
              '재료의 질감이 자연스럽고 촉감이 좋습니다. 재활용 소재도 잘 어울립니다. ']
    return f'참가 의견 {i}.\n\n' + '\n\n'.join(t * 20 for t in topics)


async def frame(ws, wanted, timeout=180):
    async with asyncio.timeout(timeout):
        while True:
            result = json.loads(await ws.recv())
            if result.get('t') == 'ping':
                await ws.send(json.dumps({'t':'pong'}))
            elif result.get('t') == 'error':
                raise AssertionError(result)
            elif result.get('t') in wanted:
                return result


async def scenario(base, count, seed_count, texts, timeout):
    sockets = []
    heartbeat_samples = []
    started = time.monotonic()
    async def connect(identity):
        ws = await websockets.connect(base + '/ws', max_size=16*1024*1024)
        await ws.send(json.dumps(dict(t='hello', protocol=PROTOCOL_VERSION, id=identity)))
        hello = await frame(ws, {'hello_ok'})
        snap = await frame(ws, {'snapshot'})
        assert len(snap['points']) == seed_count
        return ws, hello
    heartbeat, _ = await connect('writer0')
    stop = asyncio.Event()
    async def ping():
        while not stop.is_set():
            tick = time.monotonic()
            await heartbeat.send(json.dumps({'t':'ping'}))
            await frame(heartbeat, {'pong'})
            heartbeat_samples.append(time.monotonic()-tick)
            await asyncio.sleep(.05)
    beat = asyncio.create_task(ping())
    try:
        for i in range(count):
            ws, _ = await connect(f'writer{i}'); sockets.append(ws)
        send_times = {}
        async def submit(i, ws):
            cap = ('cap-%04d-' % i) * 5
            envelope = dict(t='submit', nonce=f'load-{i}', owner_capability=cap,
                context_revision=0, target_id='target1', week=2, source='ai' if i%2 else 'human', text=texts[i])
            tick=time.monotonic(); await ws.send(json.dumps(envelope,ensure_ascii=False))
            receipt=await frame(ws, {'submission_accepted'})
            ack=time.monotonic()-tick; sid=receipt['submission_id']; send_times[sid]=tick
            # Explicit immutable replay must not create a second parent.
            await ws.send(json.dumps(envelope,ensure_ascii=False))
            replay=await frame(ws, {'submission_accepted'})
            assert replay['submission_id'] == sid
            return sid, ack
        accepted = await asyncio.gather(*(submit(i,ws) for i,ws in enumerate(sockets)))
        ready = {}
        ready_done = asyncio.Event()
        async def wait_ready(i, ws, sid):
            # Subscribe once; the server pushes every processing-state transition.
            # Keep receiving after this owner's result is ready, as browsers do.
            await ws.send(json.dumps(dict(t='submission_status', submission_id=sid,
                  owner_capability=('cap-%04d-' % i)*5, request_id=f'load-status-{i}')))
            while True:
                result=await frame(ws, {'submission_detail'}, timeout)
                if result['submission_id'] != sid:
                    continue
                assert result['raw_text'] == texts[i]
                assert result['state'] != 'failed', result['error_code']
                if result['state'] == 'ready':
                    assert ''.join(u['text'] for u in result['units']) == texts[i]
                    assert all(u['source'] == ('ai' if i%2 else 'human') for u in result['units'])
                    ready[sid] = (time.monotonic()-send_times[sid], len(result['units']))
                    if len(ready) == count: ready_done.set()
                    break
            while not ready_done.is_set():
                try:
                    message = json.loads(await asyncio.wait_for(ws.recv(), .5))
                    if message.get('t') == 'ping':
                        await ws.send(json.dumps({'t':'pong'}))
                except asyncio.TimeoutError:
                    pass
        await asyncio.gather(*(wait_ready(i,ws,accepted[i][0]) for i,ws in enumerate(sockets)))
        reader, _ = await connect_final(base)
        try:
            snap = await frame(reader, {'snapshot'})
            public = snap['points']; expected=seed_count+sum(n for _,n in ready.values())
            assert len(public) == expected and len({p['id'] for p in public}) == expected
            assert len({p['submission_id'] for p in public}) == seed_count+count
            assert not any('reviewer_id' in p or 'owner_capability' in p or 'raw_text' in p for p in public)
            import math
            assert all(math.isfinite(p['x']) and math.isfinite(p['y']) for p in public)
            snapshot_bytes=len(json.dumps(snap,ensure_ascii=False).encode())
        finally:
            await reader.close()
        return dict(participants=count,seed_units=seed_count,original_codepoints=len(texts[0]),
            resulting_units=expected,snapshot_bytes=snapshot_bytes,ack_p50_ms=percentile([a for _,a in accepted],.5)*1000,
            ack_p95_ms=percentile([a for _,a in accepted],.95)*1000,
            ready_p50_s=percentile([d for d,_ in ready.values()],.5),
            ready_p95_s=percentile([d for d,_ in ready.values()],.95),
            ready_max_s=max(d for d,_ in ready.values()),
            heartbeat_p95_ms=percentile(heartbeat_samples,.95)*1000,
            total_s=time.monotonic()-started, all_originals_exact=True, nonce_replay_unique=True)
    finally:
        stop.set(); await beat
        await heartbeat.close()
        await asyncio.gather(*(ws.close() for ws in sockets))


async def connect_final(base):
    ws=await websockets.connect(base+'/ws',max_size=16*1024*1024)
    await ws.send(json.dumps(dict(t='hello',protocol=PROTOCOL_VERSION,id='writer0')))
    return ws,await frame(ws,{'hello_ok'})


def run_measurement(tmp_path, *, seed_count=1000, count=30, long=True):
    import uvicorn
    real=os.getenv('ATLAS_SEMANTIC_LOAD_REAL') == '1'
    roster=tmp_path/'roster.csv'
    roster.write_text('id,display_name,role\ntarget1,대상,student\n'+''.join(f'writer{i},참가{i},auditor\n' for i in range(count)))
    db=tmp_path/'atlas.db'; store=Store(str(db));store.migrate()
    for i in range(seed_count):
        op=new_opinion(reviewer_id=f'writer{i%count}', target_id='target1',text=f'기존 의견 {i}: 공간의 색과 질감을 관찰했습니다.',source='human')
        store.insert_opinion(op,text_hash(op.text))
    store.close()
    cfg=load_config(dict(ATLAS_ADMIN_CODE=CODE,ATLAS_DB=str(db),ATLAS_ROSTER=str(roster),
        ATLAS_UNSAFE_FAKE_EMBEDDER='0' if real else '1',ATLAS_DEVICE='cpu',
        ATLAS_ALLOW_UNREVIEWED_SEMANTICS='1',ATLAS_DISABLE_VIEWER='1'))
    with socket.socket() as sock:
        sock.bind(('127.0.0.1',0)); port=sock.getsockname()[1]
    app=create_app(cfg);server=uvicorn.Server(uvicorn.Config(app,host='127.0.0.1',port=port,log_level='warning',access_log=False))
    thread=threading.Thread(target=server.run,daemon=True);tick=time.monotonic();thread.start()
    timeout=float(os.getenv('ATLAS_SEMANTIC_LOAD_TIMEOUT','240'))
    try:
        deadline=tick+timeout
        while time.monotonic()<deadline:
            try:
                health=httpx.get(f'http://127.0.0.1:{port}/healthz',timeout=1).json()
                if health['ok']:break
            except (httpx.HTTPError,ValueError): pass
            time.sleep(.1)
        else: raise AssertionError('temporary load server failed startup')
        startup=time.monotonic()-tick
        texts=[long_feedback(i) if long else f'짧은 의견 {i}' for i in range(count)]
        if os.getenv('ATLAS_SEMANTIC_LOAD_MAXIMUM') == '1':
            texts=[(text*((20000//len(text))+1))[:20000] for text in texts]
        metrics=asyncio.run(asyncio.wait_for(scenario(f'ws://127.0.0.1:{port}',count,seed_count,texts,timeout),timeout))
        import numpy as np
        norms=np.linalg.norm(app.state.atlas.state._vectors,axis=1)
        assert np.isfinite(norms).all() and np.allclose(norms,1,atol=2e-4)
        metrics['vectors_finite_normalized']=True
        metrics.update(model='embeddinggemma-cpu' if real else 'fake',startup_s=startup,semantic_quality_approved=False)
        print(json.dumps(metrics,ensure_ascii=False))
        output=os.getenv('ATLAS_SEMANTIC_LOAD_OUTPUT')
        if output:Path(output).write_text(json.dumps(metrics,ensure_ascii=False,indent=2)+'\n')
        assert metrics['ack_p95_ms'] <= 250
        assert metrics['heartbeat_p95_ms'] <= 100
        assert metrics['ready_max_s'] <= (120 if os.getenv('ATLAS_SEMANTIC_LOAD_MAXIMUM') == '1' else 30)
        return metrics
    finally:
        server.should_exit=True;thread.join(timeout=30)
        assert not thread.is_alive(),'temporary load server did not stop'


def test_semantic_classroom_burst(tmp_path):
    run_measurement(tmp_path)
