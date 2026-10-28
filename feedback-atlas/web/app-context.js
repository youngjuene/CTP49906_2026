const scriptBase = new URL(".", import.meta.url);
const basePath = scriptBase.pathname.endsWith("/") ? scriptBase.pathname : `${scriptBase.pathname}/`;

function cleanPath(path) {
  return String(path || "").replace(/^\/+/, "");
}

export function apiUrl(path = "") {
  const url = new URL(cleanPath(path), scriptBase);
  return `${url.pathname}${url.search}${url.hash}`;
}

export function wsUrl(path = "") {
  const proto = location.protocol === "https:" ? "wss:" : "ws:";
  return `${proto}//${location.host}${apiUrl(path)}`;
}

export function storageKey(name) {
  const scope = basePath === "/" ? "classroom" : basePath.replace(/^\/|\/$/g, "").replace(/\//g, ":");
  return `${name}:${scope || "classroom"}`;
}
