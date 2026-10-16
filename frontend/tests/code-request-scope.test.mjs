import assert from 'node:assert/strict'
import {test} from 'node:test'
import fs from 'node:fs'
import ts from 'typescript'

const source = fs.readFileSync(new URL('../src/components/ResearchCopilot.vue', import.meta.url), 'utf8')
const start = source.indexOf('async function previewActionPlan(')
const handler = source.slice(start, source.indexOf('\nconst previewGpuPlan', start))
const javascript = ts.transpileModule(handler, {compilerOptions:{target:ts.ScriptTarget.ES2022}}).outputText

test('new natural-language requirement is not replaced by an older GPU proposal', async () => {
  const cell = value => ({value}), calls = []
  const ctx = {
    props:{projectId:'p',run:{id:'r'}}, branchBusy:cell(false), actionMessageId:cell(''),
    revisionDataset:cell(''), revisionDatasets:cell([]), isolatedCopy:cell(false),
    branchInstruction:cell(''), pendingRevision:cell({instruction:'old GPU request'}), revisionPlan:cell({action:'prepare_code'}),
    branchError:cell(''), branchPreview:cell(null), codingModel:cell(''),
    api:{previewBranch:async (...args) => {calls.push(args); return {proposal_id:'new'}}},
  }
  await new Function(...Object.keys(ctx), `${javascript};return previewActionPlan({id:'m',context:{action_plan:{kind:'code_revision',run_id:'r',instruction:'追加耗时清单',selected_datasets:[],dataset:''}}})`)(...Object.values(ctx))
  assert.equal(calls[0][3], '追加耗时清单')
  assert.equal(ctx.pendingRevision.value, null)
  assert.equal(ctx.revisionPlan.value, null)
  assert.equal(ctx.branchPreview.value.proposal_id, 'new')
  assert.equal(ctx.branchBusy.value, false)
})

test('no fixed GPU compatibility feature entry and navigation preserves reproduction', () => {
  assert.ok(!source.includes('async function prepareGpuCompatibility'))
  const app = fs.readFileSync(new URL('../src/App.vue', import.meta.url), 'utf8')
  const nav = app.slice(app.indexOf('<nav'), app.indexOf('</nav>'))
  assert.ok(nav.indexOf('项目总览') < nav.indexOf('证据知识库'))
  assert.ok(nav.indexOf('使用统计') < nav.indexOf('Agent 协作网络'))
  assert.ok(nav.indexOf('Agent 协作网络') < nav.indexOf('多模型调度'))
  assert.ok(!nav.includes('审计'))
  const project = fs.readFileSync(new URL('../src/views/ProjectView.vue', import.meta.url), 'utf8')
  assert.ok(project.includes('<ReproductionProgress'))
  assert.ok(project.includes('<ExperimentDashboard'))
  assert.ok(!project.includes('runAnalysis'))
})
