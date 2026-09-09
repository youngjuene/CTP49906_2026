"""Recoverable SQLite jobs and one model executor; publish whole revisions."""
import asyncio
import logging

from src.atlas_state import RecomputeInput, RecomputeLoop
from src.hub import Channel
from src.segmentation import load_policy, split_feedback
from src.submissions import SubmissionError

log = logging.getLogger('atlas')


class SubmissionWorker:
    def __init__(self, state, hub, runtime, *, policy=None):
        self.state, self.hub, self.runtime = state, hub, runtime
        self.policy = policy or load_policy()
        self._wake = asyncio.Event()
        self._stop = False
        self._task = None
        self._reload = False
        self.in_flight = 0
        self.passes = 0

    def request(self, *, reload=False):
        self._reload = self._reload or reload
        self._wake.set()

    def start(self):
        self._task = asyncio.create_task(self.run())
        return self._task

    async def aclose(self):
        self._stop = True
        self._wake.set()
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        # A cancelled coroutine does not stop a model thread. Join before closing DB.
        await asyncio.to_thread(self.runtime.close)

    async def run(self):
        self.state.store.recover_jobs()
        while not self._stop:
            jobs = self.state.store.pending_jobs()
            if jobs or self._reload:
                reload = self._reload
                self._reload = False
                await self.process_wave(jobs, reload=reload)
                continue
            self._wake.clear()
            try:
                await asyncio.wait_for(self._wake.wait(), 0.5)
            except asyncio.TimeoutError:
                pass

    async def notify(self, sid):
        await self.hub.notify_submission(sid, self.state.store)
        await self.hub.broadcast(Channel.ADMIN, {"t": "processing", "processing": self.state.store.processing_counts(), "diagnostics": self.state.store.queue_diagnostics()})

    async def broadcast_snapshot(self):
        await self.hub.broadcast(Channel.PARTICIPANT, self.state.participant_snapshot())
        await self.hub.broadcast(Channel.ADMIN, self.state.admin_snapshot())
        self.state.mark_all_sent()

    async def refresh_published(self):
        """Immediate removal, without waiting for any in-flight model result."""
        state = self.state
        state.opinions = state.store.all_opinions()
        state.rev = state.store.data_rev
        state.server_coords = state.store.all_coords()
        active = {o.id for o in state.opinions}
        state.neighbors_by_id = {oid: {'ids': [i for i in nb['ids'] if i in active],
            'distances': [d for i, d in zip(nb['ids'], nb['distances']) if i in active]}
            for oid, nb in state.neighbors_by_id.items() if oid in active}
        # Until rebuilt, the old matrix may contain withdrawn units.
        state._vector_row = {}; state._row_ids = []; state._vectors = None
        state.refresh_mosaic()
        await self.broadcast_snapshot()

    async def process_wave(self, jobs, *, reload=False):
        state, store = self.state, self.state.store
        pending = {}
        for job in jobs:
            sid, revision = job['id'], job['revision']
            try:
                staged = store.job_started(sid, revision)
                await self.notify(sid)
                if not staged:
                    result = await self.runtime.run(split_feedback, job['raw_text'], self.runtime, self.policy)
                    store.stage_revision(sid, revision, result, (job['source'],) * len(result.spans))
                pending[sid] = revision
                await self.notify(sid)
            except SubmissionError as exc:
                if exc.code != 'REVISION_CONFLICT':
                    store.job_failed(sid, revision, exc.code, transient=False)
                    await self.notify(sid)
            except Exception:
                log.exception('semantic processing failed for submission %s', sid)
                store.job_failed(sid, revision, transient=True)
                await self.notify(sid)
        if not pending and not reload:
            return
        expected = store.data_rev
        candidate = store.staged_opinions(pending)
        snap = RecomputeInput(expected, tuple(candidate), dict(state.server_coords))
        self.in_flight += 1
        try:
            result = state.metadata_revision(snap) if state.rev == expected else None
            if result is None:
                result = await self.runtime.run(state.recompute, snap)
        finally:
            self.in_flight -= 1
        self.passes += 1
        if result.error:
            state.last_error = result.error
            for sid, revision in pending.items():
                store.job_failed(sid, revision)
                await self.notify(sid)
            if reload:
                # A reload is recoverable even when no new parent arrives.
                await asyncio.sleep(1)
                self._reload = True
            return
        try:
            new_rev = store.commit_publication(expected, pending, result.coords, state.layout_rev + 1)
        except SubmissionError:
            # A correction/withdrawal won. Never apply the stale result.
            for sid, revision in pending.items():
                store.job_deferred(sid, revision)
                await self.notify(sid)
            self._reload = True
            return
        old_ids = {o.id for o in state.opinions}
        state.opinions = store.all_opinions()
        state.rev = new_rev
        result.rev = new_rev
        state.last_error = None
        state.commit(result, persist=False)
        if not pending or old_ids - {o.id for o in state.opinions}:
            await self.broadcast_snapshot()
        else:
            await RecomputeLoop.broadcast(self, result)
        for sid in pending:
            await self.notify(sid)
