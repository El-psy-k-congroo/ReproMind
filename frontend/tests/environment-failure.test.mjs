import assert from 'node:assert/strict'
import { test } from 'node:test'
import fs from 'node:fs'
import ts from 'typescript'
const source = fs.readFileSync(new URL('../src/environmentFailure.ts', import.meta.url), 'utf8')
const compiled = ts.transpileModule(source, { compilerOptions: { module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2022 } }).outputText
const { installationTransportFailed } = await import(`data:text/javascript;base64,${Buffer.from(compiled).toString('base64')}`)

test('transport errors take precedence over pip no-matching-version summaries', () => {
  assert.equal(installationTransportFailed('ProxyError: Cannot connect to proxy; No matching distribution found', null), true)
  assert.equal(installationTransportFailed('No matching distribution found', { kind: 'network' }), true)
  assert.equal(installationTransportFailed('No matching distribution found', { kind: 'incompatible_package' }), false)
})
