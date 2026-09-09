from datetime import datetime,timedelta,timezone
from src.submissions import FeedbackSpan,SplitResult
from test_submission_store import opened, request, make_actual_v1
from test_server_routes import client


def test_revision_history_includes_original_and_corrected_target_and_week(tmp_path):
    db=opened(tmp_path);req=request();sid=db.accept_submission(req).submission_id
    spans=(FeedbackSpan(0,len(req.raw_text)),)
    db.stage_revision(sid,1,SplitResult(spans,'test'),('human',))
    db.correct_submission(sid,1,'change-target',req.reviewer_id,req.owner_capability,spans,('ai',),'target2',4)
    history=db.submission_detail(sid,req.reviewer_id,req.owner_capability)['history']
    assert [(h['target_id'],h['week']) for h in history]==[('target1',2),('target2',4)]


def test_queue_diagnostics_report_age_and_failure_stage_without_identity(tmp_path):
    db=opened(tmp_path);a=db.accept_submission(request()).submission_id
    b=db.accept_submission(request(nonce='failed')).submission_id
    old=(datetime.now(timezone.utc)-timedelta(seconds=90)).isoformat()
    with db.transaction():db._db.execute('UPDATE submissions SET timestamp=? WHERE id=?',(old,a))
    db.job_started(b,1);db.job_failed(b,1,transient=False)
    result=db.queue_diagnostics()
    assert result['oldest_pending_seconds']>=89
    assert result['failed_by_stage']=={'splitting':1,'embedding':0}
    assert 'writer1' not in str(result)


def test_admin_counts_distinguish_units_originals_and_submitters(client):
    atlas=client.app.state.atlas
    req=request(raw_text='첫 의견. 둘째 의견.')
    sid=atlas.store.accept_submission(req).submission_id
    cuts=(FeedbackSpan(0,6),FeedbackSpan(6,len(req.raw_text)))
    ids=atlas.store.stage_revision(sid,1,SplitResult(cuts,'test'),('human','ai'))
    atlas.store.commit_publication(atlas.store.data_rev,{sid:1},{ids[0]:(0,0),ids[1]:(1,1)},1)
    atlas.runtime.run_sync(atlas.state.load)
    assert atlas.state.admin_snapshot()['counts']=={'units':2,'submissions':1,'submitters':1}
    assert 'counts' not in atlas.state.participant_snapshot()


def test_migration_backup_is_private(tmp_path):
    from src.store import Store
    path=make_actual_v1(tmp_path)
    db=Store(str(path));db.migrate();db.close()
    assert (path.parent/(path.name+'.pre-v2-backup')).stat().st_mode & 0o777 == 0o600
