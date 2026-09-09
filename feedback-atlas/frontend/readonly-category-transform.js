import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";

const PACKAGE_NAME = "embedding-atlas";
const EXPECTED_VERSION = "0.24.0";
const CHUNK_ID = "chunk-BYfXPSbk.js";
const EXPECTED_HASHES = {
  "dist/chunk-BYfXPSbk.js": "b40517455b6f0b18b4071e69cd62b7b577aeecaef408fc5d522bd4e0c559b988",
  "dist/index.js": "43be80cfeb3e04fce3c1b9dc26bd9ab4983d0149d628613ae7bae45c58a51acd",
};

function sha256(text) {
  return createHash("sha256").update(text).digest("hex");
}

function fail(message) {
  throw new Error(`[feedback-atlas readonly-category-transform] ${message}`);
}

function assertPackagePinned() {
  const pkg = JSON.parse(readFileSync(new URL("./node_modules/embedding-atlas/package.json", import.meta.url), "utf8"));
  if (pkg.version !== EXPECTED_VERSION) {
    fail(`expected ${PACKAGE_NAME} ${EXPECTED_VERSION}, found ${pkg.version}`);
  }
}

function assertExpectedHash(relativePath, code) {
  const actual = sha256(code);
  if (actual !== EXPECTED_HASHES[relativePath]) {
    fail(`${relativePath} hash changed: expected ${EXPECTED_HASHES[relativePath]}, found ${actual}`);
  }
}

function replaceOnce(code, search, replacement, label) {
  const first = code.indexOf(search);
  if (first < 0) fail(`missing ${label}`);
  if (code.indexOf(search, first + search.length) >= 0) {
    fail(`ambiguous ${label}`);
  }
  return code.slice(0, first) + replacement + code.slice(first + search.length);
}

export const readonlyCategoryAdapter = `
function faReadOnlyCategoryExpr(e) {
\treturn typeof e == "string" ? m.column(e) : e;
}
function faReadOnlyCategoryCacheKey(e, t) {
\treturn \`embedding/category/\${e}/\${t}\`;
}
function faRegisterReadOnlyCategoryRefresh(e, t) {
\tif (e == null) return () => {};
\tlet n = e.__feedbackAtlasCategoryRefreshers ??= new Set();
\te.__atlasRefreshCategories ??= async () => {
\t\tawait Promise.all(Array.from(e.__feedbackAtlasCategoryRefreshers ?? [], (e) => e()));
\t};
\tn.add(t);
\treturn () => {
\t\tn.delete(t);
\t\tif (n.size === 0) delete e.__atlasRefreshCategories;
\t};
}
`;

export const adaptedCategoryFunctions = `async function yPe(e, t, n, r) {
\tif (n == null) return null;
\tlet i = faReadOnlyCategoryExpr(n), [a] = Array.from(await e.query(m.Query.describe(m.Query.from(t).select({ value: i }))));
\tif (a == null) return null;
\tlet o = Zo(a.column_type);
\treturn o == "string" ? await W6(e, t, n, 10, r) : o == "number" || o == "Date" ? o == "number" && Qo(a.column_type) ? await G6(e, t, n, r) : await ss(e, t, n) <= 10 ? await W6(e, t, n, 10, r) : await G6(e, t, n, r) : null;
}
async function W6(e, t, n, r, i) {
\tlet a = faReadOnlyCategoryExpr(n), o = m.cast(a, "TEXT"), s = Array.from(await e.query(m.Query.from(t).select({
\t\tvalue: o,
\t\tcount: m.count()
\t}).where(m.not(m.isNull(o))).groupby(o).orderby(m.desc(m.count())).limit(r))), c = s.length, l = s.length + 1, u = (e) => s.length > 0 ? m.sql\`CASE \${o}
\t\t\${s.map(({ value: t }, n) => m.sql\`WHEN \${m.literal(t)} THEN \${m.literal(n)}\`).join(" ")}
\t\tELSE (CASE WHEN \${a} IS NULL THEN \${m.literal(e)} ELSE \${m.literal(c)} END) END\` : m.sql\`CASE WHEN \${a} IS NULL THEN \${m.literal(e)} ELSE \${m.literal(c)} END\`, d = u(l), f = Array.from(await e.query(m.Query.from(t).select({
\t\tindex: d,
\t\tcount: m.cast(m.count(), "INT")
\t}).groupby(d))), p = /* @__PURE__ */ new Map();
\tfor (let e of f) p.set(e.index, e.count);
\tlet h = p.get(c) ?? 0, g = p.get(l) ?? 0, _ = bPe(i, s.length), v = s.map(({ value: e }, t) => ({
\t\tlabel: e,
\t\tcolor: _[t],
\t\tpredicate: m.eq(o, m.literal(e)),
\t\tcount: p.get(t) ?? 0
\t}));
\tif (h > 0) {
\t\tlet { otherCategoryCount: r } = (await e.query(m.sql\`
\t\t\tSELECT COUNT(DISTINCT(\${o})) AS otherCategoryCount
\t\t\tFROM \${t}
\t\t\tWHERE \${d} = \${m.literal(c)} AND \${a} IS NOT NULL
\t\t\`)).get(0);
\t\tv.push({
\t\t\tlabel: \`(other \${r.toLocaleString()})\`,
\t\t\tcolor: i.otherColor,
\t\t\tpredicate: s.length > 0 ? m.sql\`\${a} IS NOT NULL AND \${o} NOT IN (\${s.map((e) => m.literal(e.value)).join(",")})\` : m.sql\`\${a} IS NOT NULL\`,
\t\t\tcount: h
\t\t});
\t}
\tlet y = g > 0 && h <= 0 ? c : l;
\treturn g > 0 && v.push({
\t\tlabel: "(null)",
\t\tcolor: i.nullColor,
\t\tpredicate: m.isNull(a),
\t\tcount: g
\t}), {
\t\tindexColumn: u(y),
\t\tlegend: v
\t};
}
async function G6(e, t, n, r) {
\tlet i = faReadOnlyCategoryExpr(n), a = await V6(e, t, i), o, s, c;
\tif (a?.quantitative) o = P6(a.quantitative, { desiredCount: 5 }), s = m.cast(i, "DOUBLE"), c = H6;
\telse if (a?.temporal) {
\t\to = B6(a.temporal, { desiredCount: 5 }), s = m.epoch_ms(i);
\t\tlet e = a.temporal.hasTimezone;
\t\tc = (t) => U6(t, e);
\t} else throw Error("invalid data type");
\tlet l = o.binIndexExpr(s), u = Array.from(await e.query(m.Query.from(t).select({
\t\tindex: l,
\t\tcount: m.cast(m.count(), "INT")
\t}).groupby(l).orderby(l))), d = null, f = null, p = /* @__PURE__ */ new Map();
\tfor (let { index: e, count: t } of u) e != null && ((d == null || e < d) && (d = e), (f == null || e > f) && (f = e)), p.set(e, t);
\tlet h = [];
\tif (d != null && f != null) {
\t\tlet e = xPe(r, f - d + 1), t = /* @__PURE__ */ new Set();
\t\tfor (let e = d; e <= f; e++) {
\t\t\tlet [n, r] = o.rangeForIndex(e);
\t\t\tt.add(n), t.add(r);
\t\t}
\t\tlet n = c(Array.from(t));
\t\tfor (let t = d; t <= f; t++) {
\t\t\tlet [r, i] = o.rangeForIndex(t);
\t\t\th.push({
\t\t\t\tlabel: \`[\${n(r)}, \${n(i)})\`,
\t\t\t\tcolor: e[t - d],
\t\t\t\tpredicate: m.eq(l, m.literal(t)),
\t\t\t\tcount: p.get(t) ?? 0
\t\t\t});
\t\t}
\t}
\tlet g = l;
\tif (p.has(null)) {
\t\tlet e = h.length;
\t\tg = m.sql\`CASE WHEN \${l} IS NULL THEN \${m.literal(e)} ELSE \${l} END\`, h.push({
\t\t\tlabel: "(null / nan / inf)",
\t\t\tcolor: r.nullColor,
\t\t\tpredicate: m.isNull(l),
\t\t\tcount: p.get(null) ?? 0
\t\t});
\t}
\treturn {
\t\tindexColumn: g,
\t\tlegend: h
\t};
}`;

const originalCategoryFunctionsStart = "async function yPe(e, t, n, r) {";
const originalCategoryFunctionsEnd = "function bPe(e, t) {";

function replaceCategoryFunctions(code) {
  const start = code.indexOf(originalCategoryFunctionsStart);
  const end = code.indexOf(originalCategoryFunctionsEnd);
  if (start < 0 || end < 0 || end <= start) fail("missing category helper block");
  if (code.indexOf(originalCategoryFunctionsStart, start + 1) >= 0) fail("ambiguous category helper block");
  return code.slice(0, start) + readonlyCategoryAdapter + adaptedCategoryFunctions + "\n" + code.slice(end);
}

export function transformEmbeddingAtlasChunk(code) {
  let transformed = replaceCategoryFunctions(code);
  // Full-text search indexes are otherwise cached solely by predicate. Retain
  // upstream's serialized rebuild queue, but also invalidate after corpus edits.
  transformed = replaceOnce(transformed,
    "}, n = this.predicateString(e);",
    "}, n = this.predicateString(e), r = this.coordinator.__atlasDataVersion ?? 0;",
    "search index data version");
  transformed = replaceOnce(transformed,
    "if (this.currentIndex.predicate != n) {",
    "if (this.currentIndex.predicate != n || this.currentIndex.dataVersion !== r) {",
    "search index invalidation");
  for (const indent of ["\t\t\t\t\t", "\t\t\t\t"]) {
    transformed = replaceOnce(transformed,
      `${indent}predicate: n,\n${indent}promise: e`,
      `${indent}predicate: n,\n${indent}dataVersion: r,\n${indent}promise: e`,
      "search index cache entry");
  }
  transformed = replaceOnce(
    transformed,
    "...i == null ? {} : { category: m.sql`${m.column(i)}::INT` },",
    "...i == null ? {} : { category: m.sql`${faReadOnlyCategoryExpr(i)}::INT` },",
    "point fetch category select"
  );
  transformed = replaceOnce(
    transformed,
    "...r.category == null ? {} : { c: m.sql`${m.column(r.category)}::UTINYINT` }",
    "...r.category == null ? {} : { c: m.sql`${faReadOnlyCategoryExpr(r.category)}::UTINYINT` }",
    "embedding point category select"
  );
  transformed = replaceOnce(
    transformed,
      "t.context.cache.value(`embedding/category/${L(_)}/${L(S)}`, () => yPe(t.context.coordinator, L(_), L(S), L(g))).then((e) => {\n\t\t\tI(C, e), (L(C)?.legend.length ?? 0) > s && t.onSpecChange((e) => {\n\t\t\t\te.mode = \"points\";\n\t\t\t});\n\t\t});",
    "let e = 0, n = (r = !1) => {\n\t\t\tlet i = ++e, a = faReadOnlyCategoryCacheKey(L(_), L(S)), o = () => yPe(t.context.coordinator, L(_), L(S), L(g)), c = r ? o() : t.context.cache.value(a, o);\n\t\t\tr && t.context.cache.set(a, c);\n\t\t\treturn c.then((n) => {\n\t\t\t\tif (i !== e) return;\n\t\t\t\tI(C, n), (L(C)?.legend.length ?? 0) > s && t.onSpecChange((e) => {\n\t\t\t\t\te.mode = \"points\";\n\t\t\t\t});\n\t\t\t});\n\t\t}, r = faRegisterReadOnlyCategoryRefresh(t.context.coordinator, () => n(!0));\n\t\treturn n(), () => {\n\t\t\te++, r();\n\t\t};",
    "embedding category cache refresh site"
  );
  transformed = replaceOnce(
    transformed,
    "m.eq(m.cast(m.column(i), \"INTEGER\"), m.literal(e.category))",
    "m.eq(m.cast(faReadOnlyCategoryExpr(i), \"INTEGER\"), m.literal(e.category))",
    "selection predicate category read"
  );
  transformed = replaceOnce(
    transformed,
    "...t.category == null ? {} : { maxCategory: m.sql`MAX(${m.column(t.category)}::UTINYINT)` }",
    "...t.category == null ? {} : { maxCategory: m.sql`MAX(${faReadOnlyCategoryExpr(t.category)}::UTINYINT)` }",
    "embedding stats max category read"
  );
  transformed = replaceOnce(
    transformed,
    "g = t.category == null ? null : m.column(t.category)",
    "g = t.category == null ? null : faReadOnlyCategoryExpr(t.category)",
    "embedding density group category read"
  );
  transformed = replaceOnce(
    transformed,
    "export { cs as n, SGe as r, axe as t };",
    "export { cs as n, SGe as r, axe as t };",
    "chunk export surface"
  );
  if (/async function (W6|G6)[\s\S]*?await e\.exec/.test(transformed)) {
    fail("adapted category helpers still call exec");
  }
  for (const needle of [
    "ALTER TABLE",
    "ADD COLUMN IF NOT EXISTS",
    "SET ${m.column(a)}",
    "SET ${m.column(c)}",
  ]) {
    const start = transformed.indexOf("async function W6");
    const end = transformed.indexOf("function bPe", start);
    if (transformed.slice(start, end).includes(needle)) fail(`adapted category block still contains ${needle}`);
  }
  return transformed;
}

export function transformEmbeddingAtlasIndex(code) {
  replaceOnce(
    code,
    'import { n as e, r as t, t as n } from "./chunk-BYfXPSbk.js";',
    'import { n as e, r as t, t as n } from "./chunk-BYfXPSbk.js";',
    "index chunk import"
  );
  replaceOnce(
    code,
    "export { t as EmbeddingAtlas, a as EmbeddingView, o as EmbeddingViewMosaic, u as createNNDescent, c as createUMAP, l as createUMAPFromKNN, i as defaultCategoryColors, e as defaultCharts, s as findClusters, r as maxDensityModeCategories, n as registerRenderer };",
    "export { t as EmbeddingAtlas, a as EmbeddingView, o as EmbeddingViewMosaic, u as createNNDescent, c as createUMAP, l as createUMAPFromKNN, i as defaultCategoryColors, e as defaultCharts, s as findClusters, r as maxDensityModeCategories, n as registerRenderer };",
    "index export surface"
  );
  return code;
}

export function readonlyCategoryTransformPlugin() {
  assertPackagePinned();
  return {
    name: "feedback-atlas-readonly-category-transform",
    enforce: "pre",
    transform(code, id) {
      if (!id.includes("/node_modules/embedding-atlas/")) return null;
      if (id.endsWith(`/dist/${CHUNK_ID}`)) {
        assertExpectedHash("dist/chunk-BYfXPSbk.js", code);
        return { code: transformEmbeddingAtlasChunk(code), map: null };
      }
      if (id.endsWith("/dist/index.js")) {
        assertExpectedHash("dist/index.js", code);
        return { code: transformEmbeddingAtlasIndex(code), map: null };
      }
      return null;
    },
  };
}
