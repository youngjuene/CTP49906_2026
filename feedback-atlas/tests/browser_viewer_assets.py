"""Generated worker assets and optional real GPU viewer; no production data."""
from pathlib import Path
import re
import pytest
from browser_viewer import live_server, browser


def test_emitted_worker_urls_are_served_as_javascript_and_worker_starts(live_server, browser):
    bundle=(Path(__file__).resolve().parents[1]/'web/vendor/embedding-atlas.js').read_text()
    paths=re.findall(r'new URL\("([^\"]+\.worker-[^\"]+\.js)", import.meta.url\)',bundle)
    assert len(set(paths))==3, 'three emitted worker URLs must remain relative to the bundle'
    assert all(path.startswith('assets/') for path in paths)
    page=browser.new_page()
    try:
        page.goto(live_server.url)
        results=page.evaluate('''async paths=>Promise.all(paths.map(async path=>{
          const url=new URL(path,new URL('/vendor/embedding-atlas.js',location.href));
          const res=await fetch(url); return {url:url.href,status:res.status,mime:res.headers.get('content-type')};
        }))''',paths)
        assert all('/vendor/assets/' in r['url'] and r['status']==200 and 'javascript' in r['mime'] for r in results)
        cluster=next(r['url'] for r in results if 'clustering.worker-' in r['url'])
        assert page.evaluate('''url=>new Promise((resolve,reject)=>{
          const worker=new Worker(url,{type:'module'});
          const timer=setTimeout(()=>{worker.terminate();reject(Error('worker did not initialize'))},12000);
          worker.onerror=e=>{clearTimeout(timer);worker.terminate();reject(Error(e.message))};
          worker.onmessage=e=>{if(e.data.ready){clearTimeout(timer);worker.terminate();resolve(true)}};
        })''',cluster)
    finally:page.close()


def test_real_gpu_bundle_starts_its_clustering_worker(live_server, browser):
    page=browser.new_page()
    try:
        page.goto(live_server.url)
        capable=page.evaluate("async()=> (await import('/viewer.js')).rendererAvailable()")
        if not capable:
            pytest.skip('No compatible WebGPU adapter: full analysis rendering remains unverified and classroom toggle disabled')
        with page.expect_response(lambda r:'clustering.worker-' in r.url,timeout=20000) as response:
            page.evaluate('''async()=>{
              const node=document.createElement('div');node.style.cssText='height:800px;width:1100px';
              document.body.append(node);await(await import('/viewer.js')).mount(node);
            }''')
        assert '/vendor/assets/' in response.value.url and response.value.status==200
    finally:page.close()
