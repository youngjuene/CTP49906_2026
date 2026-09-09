/* Apple's Embedding Atlas, mounted on the live corpus.
 *
 * The viewer is a Mosaic application: it is not handed data, it issues SQL and
 * expects a database to answer. So this file is mostly plumbing between three
 * things that already exist -- the vendored component in web/vendor, the DuckDB
 * relation the server stands up in src/mosaic_db.py, and the websocket that
 * already tells this page when the corpus changed.
 *
 * Three things here are worth knowing:
 *
 * 1. **The connector is ours, not `restConnector`.** Mosaic ships one that would
 *    otherwise do exactly this job. It sends `credentials: "omit"` and a fixed
 *    body, so there is no way to carry the admin code except in the URL -- and a
 *    URL carrying the admin code lands in tunnel logs, proxy logs and browser
 *    history, which is the reason /api/export.csv is a POST too. A connector is
 *    an object with a `query` method, so writing one costs fifteen lines and
 *    keeps the code in the body.
 *
 * 2. **Refresh is explicit.** Mosaic caches aggressively and has no notion of a
 *    table changing underneath it, which is correct for its usual job of exploring
 *    a file that is not moving. Here the corpus grows mid-session, so every frame
 *    that changes the map also drops the query cache and re-runs the connected
 *    clients. The server updates DuckDB before it broadcasts, so by the time this
 *    fires the relation is already the new one.
 *
 * 3. **It can refuse to load, and that is handled.** The renderer is WebGPU-only
 *    since 0.24.0 -- the WebGL2 fallback was removed in that release -- so on a
 *    laptop without it the embedding view draws nothing. This module reports that
 *    up front rather than mounting a blank canvas, and web/app.js keeps the
 *    hand-written scatter for exactly that case.
 */

import { apiUrl } from "./app-context.js";

const VENDOR = "./vendor/embedding-atlas.js";
const TABLE = "dataset";

let mod = null;          // the vendored bundle, imported once and only if needed
let coordinator = null;
let instance = null;
let mountedInto = null;
let generation = 0;
let refreshState = null;

/* ---------------- capability ---------------- */

/** Whether this browser can draw the embedding view at all.
 *
 * This mirrors what the renderer itself demands, in the same order, because
 * anything less produces the exact failure it exists to prevent: a canvas that
 * mounts, reports no error to the page, and stays empty.
 *
 * Upstream's `requestWebGPUDevice` needs three things. It checks for
 * `navigator.gpu`, `requestAdapter` and `wgslLanguageFeatures`; it acquires an
 * adapter; and then it requests a device, trying seven descriptors of decreasing
 * buffer size. **Every one of those descriptors asks for `shader-f16`**, so an
 * adapter without that feature fails all seven and the renderer gives up.
 *
 * The feature check is the part that is easy to leave out and cannot be. Probing
 * only as far as the adapter was how this was written first, and it passes on
 * exactly the machines that then fail: software adapters like SwiftShader, and
 * older integrated drivers, hand out an adapter happily and have no f16 support.
 * `features.has` is asked instead of requesting a real device, because a device
 * acquired for a probe is a device the renderer then has to be given or denied.
 */
export async function rendererAvailable() {
  if (typeof navigator === "undefined" || !navigator.gpu) return false;
  if (!navigator.gpu.requestAdapter || !navigator.gpu.wgslLanguageFeatures) {
    return false;
  }
  try {
    const adapter = await navigator.gpu.requestAdapter();
    return Boolean(adapter?.features?.has("shader-f16"));
  } catch {
    return false;
  }
}

/* ---------------- the connector ---------------- */

/* Credentials are functions rather than values because the page can change
 * channel after this was built, and captured credentials would keep querying the
 * previous relation. */
function connector({ getCode, getToken, onError }, decodeIPC) {
  return {
    async query(query) {
      try {
        const code = getCode();
        const token = getToken();
        const headers = { "content-type": "application/json" };
        if (!code && token) headers.authorization = `Bearer ${token}`;
        const res = await fetch(apiUrl("/data/query"), {
          method: "POST",
          headers,
          body: JSON.stringify({
            type: query.type ?? "arrow",
            // Mosaic passes SQL objects as often as strings; the endpoint wants text.
            sql: String(query.sql),
            ...(code ? { code } : {}),
          }),
        });
        if (!res.ok) {
          let detail = `HTTP ${res.status}`;
          try {
            const body = await res.json();
            if (body?.error) detail = body.error;
            else if (body?.message) detail = body.message;
          } catch { /* a non-JSON error body; the status is all there is */ }
          throw new Error(`query failed: ${detail}`);
        }
        if (query.type === "exec") return undefined;
        if (query.type === "arrow") return decodeIPC(await res.arrayBuffer());
        return await res.json();
      } catch (err) {
        onError?.(err);
        throw err;
      }
    },
  };
}

/* ---------------- mounting ---------------- */

async function load() {
  if (!mod) mod = await import(VENDOR);
  return mod;
}

/** Mount the viewer into `container`. Idempotent; safe to call on every toggle. */
export async function mount(container, { getCode, getToken, onError, hideTargetId = false, colorScheme = "light" } = {}) {
  if (instance && mountedInto === container) {
    refreshState.onError = onError;
    instance.update({ colorScheme });
    return instance;
  }
  const mounting = ++generation;
  const { EmbeddingAtlas, Coordinator, decodeIPC, defaultCharts } = await load();
  if (mounting !== generation) return null;

  dispose();

  const activeCoordinator = new Coordinator(undefined, { preagg: { enabled: false } });
  coordinator = activeCoordinator;
  const reportError = (error) => {
    if (coordinator === activeCoordinator && generation === mounting) refreshState?.onError?.(error);
  };
  refreshState = { coordinator: activeCoordinator, reportError, onError, running: null, again: false };
  activeCoordinator.databaseConnector(connector({
    getCode: getCode ?? (() => null),
    getToken: getToken ?? (() => null),
    onError: reportError,
  }, decodeIPC));

  const chartConfig = hideTargetId ? {
    exclude: ["target_id"],
    table: { columns: ["id", "target_name", "text", "source", "week", "timestamp",
                      "x", "y", "neighbors",
                      ...(getCode?.() ? ["reviewer_id", "reviewer_name"] : [])] },
  } : undefined;
  const charts = await defaultCharts({
    coordinator: activeCoordinator, table: TABLE, id: "id",
    projection: { x: "x", y: "y", text: "text", neighbors: "neighbors" },
    config: chartConfig,
  });
  if (coordinator !== activeCoordinator || generation !== mounting) return null;

  const classroomCharts = charts
    .filter(chart => chart.type !== "predicates")
    .map(chart => chart.type === "embedding"
      ? { ...chart, title: "Embedding view", data: { ...chart.data, category: "source" } }
      : chart);
  classroomCharts.splice(1, 0, { type: "content-viewer", title: "Selected feedback", field: "text" });

  instance = new EmbeddingAtlas(container, {
    coordinator: activeCoordinator,
    data: {
      table: TABLE,
      // The opinion id. Also what the neighbours column holds, which is the
      // contract: "ids should be an array of row ids as given by the id column".
      id: "id",
      projection: { x: "x", y: "y" },
      // Drives the tooltip and the search box.
      text: "text",
      // Precomputed on the server from the same k-NN graph UMAP was fitted on,
      // so the neighbours listed here are the ones that shaped the layout.
      neighbors: "neighbors",
    },
    // Keep normal charts and cross-filtering, without the SQL-expression panel
    // in either the participant or administrator's initial workspace.
    initialState: { charts: Object.fromEntries(classroomCharts
      .map((chart, index) => [String(index + 1), chart])) },
    defaultChartsConfig: chartConfig,
    colorScheme,
  });
  mountedInto = container;
  return instance;
}

/** Re-read the relation. Called whenever a frame changed the corpus. */
export function refresh() {
  const job = refreshState;
  if (!job) return Promise.resolve();
  if (job.running) {
    job.again = true;
    return job.running;
  }
  // clients:false so the charts the instructor has arranged survive the reload;
  // cache:true because every cached aggregate was computed against the old rows.
  job.running = (async () => {
    do {
      job.again = false;
      const active = job.coordinator;
      if (coordinator !== active) return;
      active.__atlasDataVersion = (active.__atlasDataVersion ?? 0) + 1;
      active.clear({ clients: false, cache: true });
      await active.__atlasRefreshCategories?.();
      if (coordinator !== active) return;
      await Promise.all([...(active.clients ?? [])].map(client => {
        const predicate = client.filterBy?.predicate?.(client);
        const query = client.query?.(predicate);
        return query == null ? undefined : active.requestQuery(client, query);
      }));
    } while (job.again && refreshState === job);
  })().catch(job.reportError).finally(() => { job.running = null; });
  return job.running;
}

export function setColorScheme(colorScheme) {
  if (instance) instance.update({ colorScheme });
}

export function cancelPendingMount() {
  if (!instance) destroy();
}

export function destroy() {
  generation++;
  dispose();
}

function dispose() {
  const previous = instance;
  const previousCoordinator = coordinator;
  instance = null;
  coordinator = null;
  refreshState = null;
  mountedInto = null;
  try {
    previous?.destroy();
  } catch { /* already torn down */ }
  previousCoordinator?.clear({ clients: true, cache: true });
}
