import assert from 'node:assert/strict'
import { test } from 'node:test'
import fs from 'node:fs'
import ts from 'typescript'
const source = fs.readFileSync(new URL('../src/tuningResults.ts', import.meta.url), 'utf8')
const compiled = ts.transpileModule(source, { compilerOptions: { module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2022 } }).outputText
const { selectedTuningMetrics } = await import(`data:text/javascript;base64,${Buffer.from(compiled).toString('base64')}`)
test('show validation-selected test value, not highest test value; preserve baselines', () => {
  const result = { dataset_results: [{ dataset: 'alpha', metrics: { accuracy: .53 } }],
    tuning_sessions: [{ id: 's', status: 'completed', plans: [{ dataset: 'alpha' }] }],
    tuning_attempts: [{ session_id: 's', dataset: 'alpha', selected: false, test_metrics: { accuracy: .72 } },
      { session_id: 's', dataset: 'alpha', selected: true, test_metrics: { accuracy: .61 } }] }
  const snapshot = JSON.stringify(result)
  assert.deepEqual(selectedTuningMetrics(result, 'ALPHA'), { accuracy: .61 })
  assert.deepEqual(selectedTuningMetrics(result, 'other'), {})
  assert.equal(JSON.stringify(result), snapshot)
})
test('active or failed search cannot supply a completed tuned bar', () => {
  for (const status of ['running', 'failed', 'cancelled']) assert.deepEqual(selectedTuningMetrics({
    tuning_sessions: [{ id: 's', status, plans: [{ dataset: 'beta' }] }],
    tuning_attempts: [{ session_id: 's', dataset: 'beta', selected: true, test_metrics: { rmse: 1.2 } }]
  }, 'beta'), {})
})
test('latest completed search and finite zero values, without mixing metrics', () => {
  const result = { tuning_sessions: [{ id: 'old', status: 'completed', plans: [{ dataset: 'beta' }] },
    { id: 'new', status: 'completed', plans: [{ dataset: 'beta' }] }], tuning_attempts: [
    { session_id: 'old', dataset: 'beta', selected: true, test_metrics: { rmse: .1 } },
    { session_id: 'new', dataset: 'beta', selected: true, test_metrics: { rmse: 0, mae: NaN } }] }
  assert.deepEqual(selectedTuningMetrics(result, 'beta'), { rmse: 0 })
})
