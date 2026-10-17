import assert from 'node:assert/strict'
import { test } from 'node:test'
import fs from 'node:fs'
import ts from 'typescript'
const source = fs.readFileSync(new URL('../src/curveSamples.ts', import.meta.url), 'utf8')
const compiled = ts.transpileModule(source, { compilerOptions: { module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2022 } }).outputText
const { curveSamples, nearestSample } = await import(`data:text/javascript;base64,${Buffer.from(compiled).toString('base64')}`)

test('fractional and sparse epochs retain actual horizontal position and hover', () => {
  const points = [{ epoch: .025, loss: .4 }, { epoch: .05, loss: .3 }, { epoch: 1, accuracy: .8 }]
  const samples = curveSamples(points, 'loss', .2, .5)
  assert.equal(samples.length, 2)
  assert.equal(samples[0].x, 48)
  assert.ok(samples[1].x > 48 && samples[1].x < 60)
  assert.equal(nearestSample(samples, 59), 1)
  assert.equal(curveSamples(points, 'accuracy', .7, .9)[0].x, 436)
})

test('single logged value remains a visible sample without inventing a line', () => {
  const [sample] = curveSamples([{ epoch: 1, loss: .2 }], 'loss', .1, .3)
  assert.equal(sample.x, 242)
  assert.ok(Math.abs(sample.y - 80) < 1e-9)
  assert.equal(curveSamples([{ epoch: 1 }], 'loss', .1, .3).length, 0)
})
