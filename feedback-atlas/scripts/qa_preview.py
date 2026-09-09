"""Temporary synthetic localhost preview; never opens a classroom database."""
import argparse
from pathlib import Path
import sys
import tempfile
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.config import load_config
from src.models import new_opinion
from src.store import Store
from src.textnorm import text_hash
from src.server import create_app
import uvicorn

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--port',type=int,default=18126)
args=parser.parse_args()
with tempfile.TemporaryDirectory(prefix='atlas-qa-preview-') as folder:
    root=Path(folder);roster=root/'roster.csv';db=root/'atlas.db'
    roster.write_text('id,display_name,role\ntarget1,빛과 공간,student\ntarget2,소리의 방,student\nwriter1,참가자1,auditor\nwriter2,참가자2,auditor\n')
    store=Store(str(db));store.migrate()
    for i,text in enumerate(['따뜻한 조명 덕분에 공간이 편안했습니다.','음량이 커서 대화를 듣기 어려웠습니다.','입구의 표지판이 조금 더 선명하면 좋겠습니다.','재활용 소재의 질감이 작품과 잘 어울립니다.']):
        op=new_opinion(reviewer_id='writer1',target_id='target1' if i<2 else 'target2',text=text,source='human' if i%2 else 'ai',week=1 if i<2 else 2)
        store.insert_opinion(op,text_hash(op.text))
    store.close()
    cfg=load_config(dict(ATLAS_ADMIN_CODE='qa-preview',ATLAS_ROSTER=str(roster),ATLAS_DB=str(db),
        ATLAS_UNSAFE_FAKE_EMBEDDER='1',ATLAS_SKIP_WARMUP='1'))
    print('Synthetic QA preview; participant writer1 / writer2; admin qa-preview',flush=True)
    uvicorn.run(create_app(cfg),host='127.0.0.1',port=args.port,log_level='warning',access_log=False)
