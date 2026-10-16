import assert from 'node:assert/strict'
import { test } from 'node:test'
import fs from 'node:fs'
import ts from 'typescript'
const source = fs.readFileSync(new URL('../src/runScope.ts', import.meta.url), 'utf8')
const compiled = ts.transpileModule(source, { compilerOptions: { module: ts.ModuleKind.ESNext } }).outputText
const { runDatasetScope } = await import(`data:text/javascript;base64,${Buffer.from(compiled).toString('base64')}`)

test('subset revision shows one approved dataset, without mutating paper targets or history', () => {
  const run = {paper_spec: {datasets: ['Alpha', 'Beta']}, result_json: {revision: {datasets: ['Beta']}}}
  const snapshot = JSON.stringify(run)
  assert.deepEqual(runDatasetScope(run), ['Beta'])
  assert.equal(JSON.stringify(run), snapshot)
})
test('reruns use their scope, original experiments retain all paper datasets', () => {
  assert.deepEqual(runDatasetScope({paper_spec: {datasets: ['Alpha', 'Beta']}, result_json: {rerun: {datasets: ['Alpha']}}}), ['Alpha'])
  assert.deepEqual(runDatasetScope({paper_spec: {datasets: ['Alpha', 'Beta']}, result_json: {}}), ['Alpha', 'Beta'])
})
test('invalid and duplicate names do not inflate the approved denominator', () => {
  assert.deepEqual(runDatasetScope({paper_spec: {}, result_json: {revision: {datasets: ['custom', 'CUSTOM', '', null]}}}), ['custom'])
})
