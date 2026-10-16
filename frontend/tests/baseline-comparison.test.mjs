import assert from 'node:assert/strict'
import { test } from 'node:test'
import fs from 'node:fs'
import ts from 'typescript'
const source = fs.readFileSync(new URL('../src/baselineComparison.ts', import.meta.url), 'utf8')
const compiled = ts.transpileModule(source, { compilerOptions: { module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2022 } }).outputText
const { baselineComparison } = await import(`data:text/javascript;base64,${Buffer.from(compiled).toString('base64')}`)

test('honor recorded direction and uncertainty instead of inferring accuracy from numeric values', () => {
  assert.equal(baselineComparison({ return_code: 0, metric_key: 'top1error', metrics: { top1error: .8452 }, target: .1447, verdict: 'deviated' }), 'deviated')
  assert.equal(baselineComparison({ return_code: 0, metric_key: 'accuracy', metrics: { accuracy: .82 }, target: .823, verdict: 'matched' }), 'matched')
  assert.equal(baselineComparison({ return_code: 0, metrics: { accuracy: .81 }, target: .823, verdict: 'configuration_unverified' }), null)
})

test('tuned scores cannot replace baseline acceptance and functional checks have no paper verdict', () => {
  assert.equal(baselineComparison({ return_code: 0, verdict: 'tuned_result', tuning_selected: true }), null)
  assert.equal(baselineComparison({ return_code: 0, verdict: 'tuned_result', tuning_selected: true, baseline_verdict: 'deviated' }), 'deviated')
  assert.equal(baselineComparison({ return_code: 0, verdict: 'matched', comparison_scope: 'functional_check' }), null)
  assert.equal(baselineComparison({ return_code: 1, verdict: 'matched' }), null)
})
