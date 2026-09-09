/* What web/ is allowed to reach for out of the vendored bundle.
 *
 * An explicit surface rather than `export *`. Re-exporting everything would put
 * the whole viewer package -- React and Svelte wrappers included -- behind one
 * import, and the first thing to go wrong would be a 6 MB bundle nobody meant to
 * ship. Each name below is used in web/viewer.js.
 */

// The viewer: table, linked charts, cross-filtering, search, and the embedding
// view inside it. This is the component embedding-atlas's own CLI mounts.
//
// `EmbeddingView` is deliberately not re-exported. It is the renderer on its own,
// it ships in the same chunk as the viewer so exporting it would cost nothing,
// and web/ has no use for it: the hand-written scatter covers the map-only case
// and does it with two encoding channels instead of one.
export { EmbeddingAtlas, defaultCharts } from "embedding-atlas";

// Mosaic. The coordinator is what the viewer queries through; decodeIPC turns
// the Arrow IPC bytes our /data/query endpoint returns back into a table, which
// is the one piece a custom connector has to do for itself. Selection and Param
// are not re-exported: they are for building charts by hand, and every chart here
// is one the viewer creates for itself.
export { Coordinator, decodeIPC } from "@uwdata/mosaic-core";
