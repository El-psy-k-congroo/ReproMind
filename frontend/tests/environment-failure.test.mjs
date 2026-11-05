import assert from 'node:assert/strict'
import { test } from 'node:test'
import fs from 'node:fs'
import ts from 'typescript'
const source = fs.readFileSync(new URL('../src/environmentFailure.ts', import.meta.url), 'utf8')
const compiled = ts.transpileModule(source, { compilerOptions: { module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2022 } }).outputText
const { installationTransportFailed, installationIndexUnavailable, tensorflowFailureHeading } = await import(`data:text/javascript;base64,${Buffer.from(compiled).toString('base64')}`)

test('missing channel metadata explains a legacy package-not-found diagnostic without calling it a timeout', () => {
  const error = 'RuntimeError: TensorFlow 环境迁移第 1 步失败：PackagesNotFoundError'
  const failure = {kind:'command_error'}
  const logs = 'WARNING conda: No repodata found for channel conda-forge. Solve will fail.'
  assert.equal(installationIndexUnavailable(error, failure, logs), true)
  assert.match(tensorflowFailureHeading(error, failure, logs), /索引未加载/)
  assert.doesNotMatch(tensorflowFailureHeading(error, failure, logs), /超时/)
})

test('a current network failure is not overwritten by historical missing-index output', () => {
  const error = 'TensorFlow 环境迁移第 1 步失败：SSLError'
  const failure = {kind:'network'}
  const logs = 'old attempt: No repodata found for channel conda-forge'
  assert.equal(installationIndexUnavailable(error, failure, logs), false)
  assert.match(tensorflowFailureHeading(error, failure, logs), /连接失败/)
  assert.doesNotMatch(tensorflowFailureHeading(error, failure, logs), /超时/)
})

test('transport errors take precedence over pip no-matching-version summaries', () => {
  assert.equal(installationTransportFailed('ProxyError: Cannot connect to proxy; No matching distribution found', null), true)
  assert.equal(installationTransportFailed('No matching distribution found', { kind: 'network' }), true)
  assert.equal(installationTransportFailed('No matching distribution found', { kind: 'incompatible_package' }), false)
})
