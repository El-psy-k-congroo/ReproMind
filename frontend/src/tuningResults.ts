/** Presentation only: never overwrite baseline results or select using test scores. */
export function selectedTuningMetrics(result: Record<string, any>, dataset: string): Record<string, number> {
  const same = (value: unknown) => String(value || '').toLowerCase() === dataset.toLowerCase()
  const sessions = (result.tuning_sessions || []).filter((s: any) =>
    s.status === 'completed' && (s.plans || []).some((p: any) => same(p.dataset)))
  const session = sessions.at(-1)
  const selected = [...(result.tuning_attempts || [])].reverse().find((row: any) =>
    same(row.dataset) && row.selected && (session ? row.session_id === session.id : !row.session_id)
    && (row.return_code === undefined || row.return_code === 0))
  if (!selected) return {}
  const metrics = selected.test_metrics || (typeof selected.test_accuracy === 'number' ? { accuracy: selected.test_accuracy } : {})
  return Object.fromEntries(Object.entries(metrics).filter(([, value]) => typeof value === 'number' && Number.isFinite(value))) as Record<string, number>
}
