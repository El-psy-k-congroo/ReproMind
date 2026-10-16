import assert from 'node:assert/strict'
import {test} from 'node:test'
import fs from 'node:fs'
import ts from 'typescript'

const source = fs.readFileSync(new URL('../src/components/ResearchCopilot.vue', import.meta.url), 'utf8')
const start = source.indexOf('async function approveEnvironmentMigration(')
const handler = source.slice(start, source.indexOf('\nasync function ', start + 1))
const javascript = ts.transpileModule(handler, {compilerOptions:{target:ts.ScriptTarget.ES2022}}).outputText
function fixture(framework='pytorch', accept=true) {
  const questions=[], readiness={framework, gpu_environment_plan:{needed:true,supported:true,
    packages:['torch==reviewed'], candidates:[{profile:'reviewed-cuda',packages:['torch==reviewed']}],warning:'project only'}}
  const ctx={props:{projectId:'p',run:{id:'r'}},api:{runReadiness:async()=>readiness},
    window:{confirm:text=>{questions.push(text);return accept}}}
  const run=()=>new Function(...Object.keys(ctx), `${javascript};return approveEnvironmentMigration(true)`)(...Object.values(ctx))
  return {run,readiness,questions}
}

test('PyTorch confirmation transmits explicit environment approval and candidate range', async()=>{
  const f=fixture(); assert.equal(await f.run(),true)
  assert.match(f.questions[0],/reviewed-cuda/)
})
test('declined environment preparation cannot submit a run', async()=>{
  const f=fixture('pytorch',false); assert.equal(await f.run(),null)
})
test('GPU-ready environment needs no dependency approval', async()=>{
  const f=fixture(); f.readiness.gpu_environment_plan.needed=false
  assert.equal(await f.run(),false); assert.equal(f.questions.length,0)
})
test('unsupported dependency plan cannot be approved', async()=>{
  const f=fixture(); f.readiness.gpu_environment_plan.supported=false
  await assert.rejects(f.run(),/不兼容/); assert.equal(f.questions.length,0)
})
test('rerun shows preparation before coding, no automatic remote coding request', ()=>{
  assert.ok(source.includes('<GpuPreparationPlan'))
  const auto=source.slice(source.indexOf('const lastAnswer = messages.value.at(-1)'),source.indexOf('function pauseResponse'))
  assert.ok(!auto.includes("['gpu_revision', 'code_revision']"))
  assert.ok(auto.includes("=== 'code_revision'"))
})
