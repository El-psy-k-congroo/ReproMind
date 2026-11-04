import assert from 'node:assert/strict'
import { test } from 'node:test'
import fs from 'node:fs'
import ts from 'typescript'

const source = fs.readFileSync(new URL('../src/demoStages.ts', import.meta.url), 'utf8')
const js = ts.transpileModule(source, { compilerOptions: { module: ts.ModuleKind.ES2022 } }).outputText
const { resolveRunId, demoLink, demoStages } = await import(`data:text/javascript;base64,${Buffer.from(js).toString('base64')}`)

test('each tab keeps its requested experiment instead of switching to the latest run', () => {
  const history = [{ id: 'latest' }, { id: 'baseline' }, { id: 'changed' }]
  assert.equal(resolveRunId(history, '', 'baseline'), 'baseline')
  assert.equal(resolveRunId(history, 'latest', 'changed'), 'changed')
  assert.equal(resolveRunId(history, 'baseline', ''), 'baseline')
  assert.throws(() => resolveRunId(history, '', 'deleted'), /不存在/)
})

test('five stage links name the same run and total five minutes', () => {
  assert.equal(demoStages.reduce((sum, stage) => sum + stage.seconds, 0), 300)
  for (const stage of demoStages) {
    const url = new URL(demoLink('project', 'baseline', stage.id), 'http://localhost')
    assert.equal(url.searchParams.get('run'), 'baseline')
    assert.equal(url.searchParams.get('stage'), stage.id)
  }
})
