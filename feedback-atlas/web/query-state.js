/* Membership is shared by the map, keyboard list, counts and CSV preview. */
export function queryState(patch = {}) {
  return Object.freeze({weeks:[1,2,3,4], targetIds:[], sources:[], search:'', selectedIds:[], emphasis:null, ...patch});
}
export function matchesQuery(point, query) {
  return query.weeks.includes(Number(point.week)) &&
    (!query.targetIds.length || query.targetIds.includes(point.target_id)) &&
    (!query.sources.length || query.sources.includes(point.source)) &&
    (!query.search || point.text.toLowerCase().includes(query.search.toLowerCase()));
}
export const visiblePoints = (points, query) => [...points.values()].filter(p => matchesQuery(p, query));
export function exportIds(scope, points, query) {
  const ids = scope === 'all' ? [...points.keys()] : scope === 'selected'
    ? query.selectedIds.filter(id => points.has(id)) : visiblePoints(points, query).map(p => p.id);
  return [...new Set(ids)].sort();
}
export const counts = points => ({units:points.length, submissions:new Set(points.map(p => p.submission_id || p.id)).size});
