import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { spawnSync } from "node:child_process";
import {
  transformEmbeddingAtlasChunk,
  readonlyCategoryTransformPlugin,
} from "../readonly-category-transform.js";

const chunkUrl = new URL("../node_modules/embedding-atlas/dist/chunk-BYfXPSbk.js", import.meta.url);
const indexUrl = new URL("../node_modules/embedding-atlas/dist/index.js", import.meta.url);
const chunk = readFileSync(chunkUrl, "utf8");
const index = readFileSync(indexUrl, "utf8");
const plugin = readonlyCategoryTransformPlugin();
const transformed = plugin.transform(chunk, chunkUrl.pathname).code;

// Validate the actual plugin output as an ES module, not just copied strings.
const parsed = spawnSync(process.execPath, ["--check", "--input-type=module"], {
  input: transformed, encoding: "utf8", maxBuffer: 8 * 1024 * 1024,
});
assert.equal(parsed.status, 0, parsed.stderr);
assert.equal(plugin.transform(index, indexUrl.pathname).code, index);
assert.equal(plugin.transform("const untouched = true;", "/src/app.js"), null);

// Version-preserving upstream changes must also fail closed. Do not silently
// ship the old mutation path when dependency packaging or helpers change.
assert.throws(() => plugin.transform(chunk + "\n// upstream drift", chunkUrl.pathname), /hash changed/);
assert.throws(() => plugin.transform(index + "\n// upstream drift", indexUrl.pathname), /hash changed/);
assert.throws(() => plugin.transform(chunk.replace("async function W6", "async function renamed"), chunkUrl.pathname), /hash changed/);
assert.throws(() => transformEmbeddingAtlasChunk(chunk.replace("...r.category == null", "...r.changed == null")), /missing/);

const start = transformed.indexOf("async function W6");
const end = transformed.indexOf("function bPe", start);
assert.ok(start >= 0 && end > start);
assert.doesNotMatch(transformed.slice(start, end), /\.exec\s*\(|ALTER TABLE|ADD COLUMN/);
// SQL/category semantics, empty data, and live refresh are covered by the real
// bundle tests in tests/browser_viewer_contract.py against the running server.
