import type { DatasetResult } from './types'

// Result verification owns metric direction, units, tolerance and protocol scope.
// Recomputing it in the view can turn an error rate into a successful accuracy.
export function baselineComparison(item: DatasetResult): 'matched' | 'deviated' | null {
  if (item.return_code !== 0 || ['functional_check', 'modified_configuration'].includes(item.comparison_scope || '')) return null
  const verdict = item.baseline_verdict || (!item.tuning_selected ? item.verdict : '')
  return verdict === 'matched' || verdict === 'deviated' ? verdict : null
}
