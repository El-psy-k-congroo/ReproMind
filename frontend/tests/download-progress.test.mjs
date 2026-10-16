import assert from 'node:assert/strict'
import { test } from 'node:test'
import fs from 'node:fs'
import ts from 'typescript'

const view = fs.readFileSync(new URL('../src/views/ProjectView.vue', import.meta.url), 'utf8')
const source = view.slice(view.indexOf('let refreshingRun ='), view.indexOf('\nasync function selectRun('))
const javascript = ts.transpileModule(source, { compilerOptions: { target: ts.ScriptTarget.ES2022 } }).outputText
function fixture(api) {
  const ctx = { api, project: {value:{id:'project'}}, selectedRunId:{value:'run'},
    runHistory:{value:[]}, reproduction:{value:{id:'run', marker:'existing'}}, error:{value:''},
    stopReproductionPolling:()=>{ctx.stopped++}, stopped:0 }
  const refresh = new Function(...Object.keys(ctx), `${javascript};return refreshReproduction`)(...Object.values(ctx))
  return {ctx,refresh}
}

test('slow status requests cannot pile up and duplicate polling', async()=>{
  let resolve, calls=0
  const f=fixture({listReproductionRuns:()=>{calls++;return new Promise(r=>{resolve=r})},
    getReproduction:async()=>({id:'run',status:'executing'})})
  const first=f.refresh()
  await f.refresh(); await f.refresh()
  assert.equal(calls,1)
  resolve([{id:'run',status:'executing'}]); await first
  assert.equal(f.ctx.reproduction.value.status,'executing')
})

test('poll timeout preserves last record, keeps polling and is not labelled download failure',async()=>{
  const f=fixture({listReproductionRuns:async()=>{throw new Error('请求超时')}})
  await f.refresh()
  assert.equal(f.ctx.reproduction.value.marker,'existing')
  assert.equal(f.ctx.stopped,0)
  assert.match(f.ctx.error.value,/不是数据下载或实验失败/)
  assert.match(f.ctx.error.value,/自动重试/)
})

test('late response for previously selected run cannot overwrite current run',async()=>{
  let resolve
  const f=fixture({listReproductionRuns:async()=>[{id:'run',status:'executing'}],
    getReproduction:()=>new Promise(r=>{resolve=r})})
  const request=f.refresh()
  await Promise.resolve(); f.ctx.selectedRunId.value='other'
  resolve({id:'run',status:'executing'}); await request
  assert.equal(f.ctx.reproduction.value.marker,'existing')
})

test('download card exposes live transfer fields and paused stage preserves preparation',()=>{
  const progress=fs.readFileSync(new URL('../src/components/ReproductionProgress.vue', import.meta.url),'utf8')
  assert.match(progress,/downloaded_bytes/); assert.match(progress,/total_bytes/)
  assert.match(progress,/run\.current_stage\) \|\| activity\?\.operation/)
  assert.match(progress,/paused_from_stage/)
  assert.ok(!progress.includes("paused_from_status === 'executing' ? 'executing'"))
})
