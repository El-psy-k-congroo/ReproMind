import assert from 'node:assert/strict'
import {test} from 'node:test'
import fs from 'node:fs'
import ts from 'typescript'

const source = fs.readFileSync(new URL('../src/components/ResearchCopilot.vue', import.meta.url), 'utf8')
const handler = ['previewActionPlan', 'previewBranch', 'acceptBranchResponse'].map(name => {
  const start = source.indexOf(`function ${name}(`)
  const prefix = source.slice(start - 6, start) === 'async ' ? 'async ' : ''
  return prefix + source.slice(start, source.indexOf('\n}', start) + 2)
}).join('\n')
const javascript = ts.transpileModule(handler, {compilerOptions:{target:ts.ScriptTarget.ES2022}}).outputText

test('new natural-language requirement is not replaced by an older GPU proposal', async () => {
  const cell = value => ({value}), calls = []
  const ctx = {
    props:{projectId:'p',run:{id:'r'}}, branchBusy:cell(false), actionMessageId:cell(''),
    revisionDataset:cell(''), revisionDatasets:cell([]), isolatedCopy:cell(false),
    branchInstruction:cell(''), pendingRevision:cell({instruction:'old GPU request'}), revisionPlan:cell({action:'prepare_code'}),
    branchError:cell(''), branchPreview:cell(null), branchRunPlan:cell(null), showBranch:cell(false), codingModel:cell(''),
    api:{previewBranch:async (...args) => {calls.push(args); return {proposal_id:'new'}}},
  }
  await new Function(...Object.keys(ctx), `${javascript};return previewActionPlan({id:'m',context:{action_plan:{kind:'code_revision',run_id:'r',instruction:'追加耗时清单',selected_datasets:[],dataset:''}}})`)(...Object.values(ctx))
  assert.equal(calls[0][3], '追加耗时清单')
  assert.equal(ctx.pendingRevision.value, null)
  assert.equal(ctx.revisionPlan.value, null)
  assert.equal(ctx.branchPreview.value.proposal_id, 'new')
  assert.equal(ctx.branchBusy.value, false)
})

for (const [instruction,selected] of [['请在gpu上跑完这四个数据集',['A','B','C','D']],['只跑NELL',['NELL']]]) {
test(`workbench prepares ${instruction} without applying or running`, async () => {
  const cell = value => ({value}), calls=[]
  const plan={kind:'rerun_confirm',run_id:'r',gpu_requested:true,selected_datasets:selected}
  const ctx={props:{projectId:'p',run:{id:'r'}},branchInstruction:cell(instruction),
    pendingRevision:cell(null),revisionPlan:cell(null),actionMessageId:cell(''),revisionDatasets:cell([]),revisionDataset:cell(''),
    branchBusy:cell(false),branchError:cell('old review error'),branchPreview:cell(null),branchRunPlan:cell(null),showBranch:cell(true),
    branchFile:cell(''),codingModel:cell(''),api:{previewBranch:async(...args)=>{calls.push(args);return {action_plan:plan}}}}
  await new Function(...Object.keys(ctx),`${javascript};return previewBranch()`)(...Object.values(ctx))
  assert.deepEqual(ctx.branchRunPlan.value,plan)
  assert.equal(ctx.branchPreview.value,null)
  assert.equal(ctx.branchError.value,'')
  assert.equal(calls.length,1)
})
}

test('a request response cannot switch the run plan after changing experiments', async () => {
  const cell = value => ({value})
  const ctx={props:{projectId:'p',run:{id:'r'}},branchInstruction:cell('请在gpu上跑完这四个数据集'),
    pendingRevision:cell(null),revisionPlan:cell(null),actionMessageId:cell(''),revisionDatasets:cell([]),revisionDataset:cell(''),
    branchBusy:cell(false),branchError:cell(''),branchPreview:cell(null),branchRunPlan:cell(null),showBranch:cell(true),
    branchFile:cell(''),codingModel:cell(''),api:{previewBranch:async()=>{ctx.props.run={id:'other'};return {action_plan:{kind:'rerun_confirm'}}}}}
  await new Function(...Object.keys(ctx),`${javascript};return previewBranch()`)(...Object.values(ctx))
  assert.equal(ctx.branchRunPlan.value,null)
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
