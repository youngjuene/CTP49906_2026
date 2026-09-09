import { defineConfig } from "vite";
import { readonlyCategoryTransformPlugin } from "./readonly-category-transform.js";

/* Builds Apple's viewer, its renderer and Mosaic into ../web/vendor as plain ES
 * modules with every dependency inlined.
 *
 * This exists because the two constraints look incompatible and are not. The
 * frontend is served as plain ES modules from web/ with no bundler and no CDN --
 * deliberate, for a tool whose failing network is the thing it has to survive --
 * and `embedding-atlas` cannot be dropped in like that: its viewer chunk imports
 * @uwdata/mosaic-core and @uwdata/mosaic-sql as bare specifiers, which no browser
 * resolves on its own.
 *
 * So the bundler runs here, once, at development time, and commits its output.
 * Nothing at class time needs node, npm or the network. Rebuild with
 * `npm install && npm run build` from this directory after changing a version.
 */
export default defineConfig({
  // The bundle is served from /vendor/ and /demo/vendor/, not the origin root.
  base: "./",
  plugins: [readonlyCategoryTransformPlugin()],
  build: {
    outDir: "../web/vendor",
    emptyOutDir: true,
    // Chrome 113 and Safari 17 are the floors for WebGPU, which the renderer
    // requires anyway; targeting lower would inflate the bundle to support
    // browsers that cannot draw the view regardless.
    target: "es2022",
    minify: true,
    sourcemap: false,
    lib: {
      entry: "src/bundle.js",
      formats: ["es"],
      fileName: () => "embedding-atlas.js",
    },
    rollupOptions: {
      // Nothing is external. Every bare import has to be resolved now, because
      // there is no import map or node_modules at run time.
      external: [],
      output: { chunkFileNames: "chunks/[name]-[hash].js" },
    },
  },
  // The component spawns workers for clustering, search and embedding, via
  // `new Worker(new URL(...), { type: "module" })`. They must stay ES modules or
  // their own imports break once emitted.
  worker: { format: "es" },
});
