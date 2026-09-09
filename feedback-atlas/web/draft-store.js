/* Only this module owns durable browser drafts and immutable delivery records. */
export function randomToken() {
  const bytes = crypto.getRandomValues(new Uint8Array(32));
  return btoa(String.fromCharCode(...bytes)).replaceAll('+', '-').replaceAll('/', '_').replace(/=+$/, '');
}
export const newVersion = () => crypto.randomUUID();
export const draftKey = (dataset, identity) => JSON.stringify([dataset, identity.normalize('NFKC').trim().replace(/\s+/g, ' ').toLowerCase()]);
const STORES = ['drafts', 'outbox', 'receipts', 'leases'];
let database;
function open() {
  if (!database) database = new Promise((resolve, reject) => {
    const req = indexedDB.open('feedback-atlas-v2', 1);
    req.onupgradeneeded = () => STORES.forEach(name => req.result.createObjectStore(name, {keyPath:'key'}));
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
    req.onblocked = () => reject(Error('기기 저장소가 다른 탭에서 사용 중입니다.'));
  });
  return database;
}
async function transaction(names, callback) {
  const db = await open();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(names, 'readwrite');
    let value;
    tx.oncomplete = () => resolve(value);
    tx.onabort = tx.onerror = () => reject(tx.error || Error('기기 저장 실패'));
    callback(Object.fromEntries(names.map(n => [n, tx.objectStore(n)])), result => { value = result; });
  });
}
export function saveDraft(draft, expectedVersion) {
  return transaction(['drafts'], (s, done) => {
    const req=s.drafts.get(draft.key);
    req.onsuccess=()=>{
      const old=req.result;
      if(expectedVersion !== undefined && old?.version !== draft.version && (old?.version || null) !== expectedVersion) {done(false);return;}
      s.drafts.put(draft);done(true);
    };
  }).then(saved=>{if(!saved)throw Object.assign(Error('다른 탭에서 초안이 바뀌었습니다. 이 원문을 내려받은 뒤 새로고침해 주세요.'),{code:'DRAFT_CONFLICT'});});
}
export async function readScope(scope) {
  return transaction(STORES, (s, done) => {
    const result = {}; let remaining = 3;
    for (const name of ['drafts', 'outbox', 'receipts']) {
      const req = s[name].getAll();
      req.onsuccess = () => {
        result[name] = req.result.filter(row => row.scope === scope || row.key === scope);
        if (!--remaining) done(result);
      };
    }
  });
}
export function enqueue(draft, frame) {
  const item = {key: `${draft.key}:${frame.nonce}`, scope:draft.key, frame, version:draft.version};
  return transaction(['drafts', 'outbox'], (s, done) => {
    const req=s.drafts.get(draft.key);
    req.onsuccess=()=>{
      if(req.result && req.result.version !== draft.version) {done(false);return;}
      const pending=s.outbox.getAll();
      pending.onsuccess=()=>{
        if(pending.result.some(row=>row.scope===draft.key)) {done('QUEUE_BUSY');return;}
        s.drafts.put(draft);s.outbox.add(item);done(true);
      };
    };
  }).then(saved=>{if(saved==='QUEUE_BUSY')throw Object.assign(Error('다른 탭에서 이 원문을 전송 중입니다. 저장 결과를 확인합니다.'),{code:'QUEUE_BUSY'});if(!saved)throw Object.assign(Error('다른 탭에서 초안이 바뀌었습니다.'),{code:'DRAFT_CONFLICT'});return item;});
}
export function acknowledge(item, receipt) {
  return transaction(['drafts', 'outbox', 'receipts', 'leases'], (s, done) => {
    s.receipts.put({key: `${item.scope}:${receipt.submission_id}`, scope:item.scope,
      ...receipt, owner_capability:item.frame.owner_capability});
    s.outbox.delete(item.key); s.leases.delete(item.key);
    const req = s.drafts.get(item.scope);
    req.onsuccess = () => {
      if (req.result?.version === item.version) {
        const draft = {...req.result, text:'', dirty:false, version:newVersion()};
        s.drafts.put(draft); done(draft);
      }
    };
  });
}
export const updateOutbox = item => transaction(['outbox'], s => s.outbox.put(item));
export function replaceRejectedEnvelope(item, frame) {
  const replacement={key:`${item.scope}:${frame.nonce}`,scope:item.scope,version:item.version,frame};
  return transaction(['outbox','leases'],(s,done)=>{
    const req=s.outbox.get(item.key);
    req.onsuccess=()=>{
      if(!req.result?.blocked) {done(false);return;}
      s.outbox.delete(item.key);s.leases.delete(item.key);s.outbox.add(replacement);done(true);
    };
  }).then(replaced=>{if(!replaced)throw Error('다른 탭에서 전송 상태가 바뀌었습니다. 저장 결과를 먼저 확인해 주세요.');return replacement;});
}
export const removeOutbox = key => transaction(['outbox', 'leases'], s => {s.outbox.delete(key); s.leases.delete(key);});
export function claimLease(key, owner) {
  return transaction(['leases'], (s, done) => {
    const req = s.leases.get(key);
    req.onsuccess = () => {
      const old = req.result, now = Date.now();
      const claimed = !old || old.owner === owner || old.expires < now;
      if (claimed) s.leases.put({key, owner, expires:now + 12000});
      done(claimed);
    };
  });
}
export function clearScope(scope) {
  return transaction(STORES, s => {
    for (const name of STORES) {
      const cursor = s[name].openCursor();
      cursor.onsuccess = () => {
        const row = cursor.result;
        if (!row) return;
        if (row.value.scope === scope || row.key === scope || String(row.key).startsWith(scope+':')) row.delete();
        row.continue();
      };
    }
  });
}
