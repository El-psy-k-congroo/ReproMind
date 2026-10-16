import test from 'node:test'
import assert from 'node:assert/strict'
import fs from 'node:fs'
import ts from 'typescript'

const source = fs.readFileSync(new URL('../src/api.ts', import.meta.url), 'utf8')
const compiled = ts.transpileModule(source, {
  compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.ES2022 },
}).outputText.replace('import.meta.env.VITE_API_BASE_URL', "'/api'")
const { api } = await import('data:text/javascript;base64,' + Buffer.from(compiled).toString('base64'))
globalThis.window = { setTimeout, clearTimeout }

test('HTTP validation array is a readable field error', async () => {
  globalThis.fetch = async () => new Response(JSON.stringify({ detail: [
    { loc: ['body', 'name'], msg: 'String should have at least 2 characters', type: 'string_too_short' },
  ] }), { status: 422, headers: { 'Content-Type': 'application/json' } })
  await assert.rejects(api.createProject({ name: 'a', description: '', research_domain: '' }),
    /name: String should have at least 2 characters/)
})

test('string detail and non-JSON errors preserve meaningful messages', async () => {
  globalThis.fetch = async () => new Response(JSON.stringify({ detail: '项目不存在' }), { status: 404 })
  await assert.rejects(api.getProject('missing'), /项目不存在/)
  globalThis.fetch = async () => new Response('upstream unavailable', { status: 502 })
  await assert.rejects(api.getProject('missing'), /502/)
})
