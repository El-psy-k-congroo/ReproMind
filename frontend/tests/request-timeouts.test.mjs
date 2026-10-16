import assert from 'node:assert/strict'
import { test } from 'node:test'
import fs from 'node:fs'
import ts from 'typescript'
const source = fs.readFileSync(new URL('../src/api.ts', import.meta.url), 'utf8').replace('import.meta.env.VITE_API_BASE_URL', 'undefined')
const compiled = ts.transpileModule(source, { compilerOptions: { module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2022 } }).outputText
const { api } = await import(`data:text/javascript;base64,${Buffer.from(compiled).toString('base64')}`)
test('first conversation creation is bounded, fails explicitly and never silently retries', async () => {
  const previousWindow = globalThis.window, previousFetch = globalThis.fetch
  let calls = 0, duration = 0, expire, cleared = false
  globalThis.window = { setTimeout(callback, ms) { expire = callback; duration = ms; return 1 }, clearTimeout() { cleared = true } }
  globalThis.fetch = (_, options) => { calls++; return new Promise((resolve, reject) => options.signal.addEventListener('abort', () => reject(options.signal.reason))) }
  try {
    const request = api.createConversation('offline')
    assert.equal(duration, 25000)
    expire()
    await assert.rejects(request, /请求超时/)
    assert.equal(calls, 1); assert.equal(cleared, true)
  } finally { globalThis.window = previousWindow; globalThis.fetch = previousFetch }
})
test('timeout also covers a stalled JSON response body', async () => {
  const previousWindow = globalThis.window, previousFetch = globalThis.fetch
  let expire, signal, bodyStarted = false
  globalThis.window = { setTimeout(callback) { expire = callback; return 1 }, clearTimeout() {} }
  globalThis.fetch = async (_, options) => { signal = options.signal; return { ok: true, status: 200,
    json: () => { bodyStarted = true; return new Promise((resolve, reject) => signal.addEventListener('abort', () => reject(signal.reason))) } } }
  try {
    const request = api.createConversation('offline')
    await Promise.resolve(); assert.equal(bodyStarted, true)
    expire(); await assert.rejects(request, /请求超时/)
  } finally { globalThis.window = previousWindow; globalThis.fetch = previousFetch }
})
