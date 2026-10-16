/** A subset rerun is not a failed attempt at all paper datasets. */
export function runDatasetScope(run: {
  paper_spec: Record<string, unknown>
  result_json: Record<string, unknown>
}): string[] {
  const revision = run.result_json.revision as { datasets?: unknown } | undefined
  const rerun = run.result_json.rerun as { datasets?: unknown } | undefined
  const names = revision?.datasets ?? rerun?.datasets ?? run.paper_spec.datasets
  if (!Array.isArray(names)) return []
  const seen = new Set<string>()
  return names.filter((name): name is string => {
    if (typeof name !== 'string' || !name.trim() || seen.has(name.toLowerCase())) return false
    seen.add(name.toLowerCase())
    return true
  })
}
