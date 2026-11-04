import assert from 'node:assert/strict'
import { test } from 'node:test'
import fs from 'node:fs'
import ts from 'typescript'

const source = fs.readFileSync(new URL('../src/routeRecovery.ts', import.meta.url), 'utf8')
const compiled = ts.transpileModule(source, { compilerOptions: { module: ts.ModuleKind.ESNext } }).outputText
const { recoverRouteImport } = await import(`data:text/javascript;base64,${Buffer.from(compiled).toString('base64')}`)
function memoryStorage() {
  const values = new Map()
  return { getItem: key => values.get(key) ?? null, setItem: (key, value) => values.set(key, value) }
}

test('missing route chunk retries the intended URL, preserving project and run', () => {
  const targets = []
  const target = '/agents?project=project-a&run=run-a'
  assert.equal(recoverRouteImport(new Error('Failed to fetch dynamically imported module: /assets/old.js'), target, memoryStorage(), value => targets.push(value), 100_000), 'reloading')
  assert.deepEqual(targets, [target])
})

test('a persistent deployment error cannot trap a tab in a reload loop', () => {
  const storage = memoryStorage()
  const targets = []
  const error = new Error('Unable to preload CSS for /assets/old.css')
  const navigate = value => targets.push(value)
  assert.equal(recoverRouteImport(error, '/models', storage, navigate, 100_000), 'reloading')
  assert.equal(recoverRouteImport(error, '/models', storage, navigate, 101_000), 'failed')
  assert.deepEqual(targets, ['/models'])
  assert.equal(recoverRouteImport(error, '/models', storage, navigate, 161_000), 'reloading')
})

test('one tab recovering does not consume another tab’s retry', () => {
  const error = new Error('Importing a module script failed')
  assert.equal(recoverRouteImport(error, '/agents', memoryStorage(), () => {}, 100_000), 'reloading')
  assert.equal(recoverRouteImport(error, '/agents', memoryStorage(), () => {}, 100_000), 'reloading')
})

test('unrelated errors and denied storage do not cause automatic navigation', () => {
  const navigate = () => assert.fail('unexpected reload')
  assert.equal(recoverRouteImport(new Error('API unavailable'), '/models', memoryStorage(), navigate), 'unrelated')
  const denied = { getItem() { throw new Error('Access denied') }, setItem() {} }
  assert.equal(recoverRouteImport(new Error('error loading dynamically imported module'), '/agents', denied, navigate), 'failed')
})
