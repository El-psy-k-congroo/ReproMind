import assert from 'node:assert/strict'
import { test } from 'node:test'
import fs from 'node:fs'
import ts from 'typescript'

// Execute the real component handler with fake boundaries. No fetch/DOM/training.
const source = fs.readFileSync(new URL('../src/components/ResearchCopilot.vue', import.meta.url), 'utf8')
const start = source.indexOf('async function runRevision() {')
const handler = source.slice(start, source.indexOf('\nasync function ', start + 1))
const javascript = ts.transpileModule(handler, { compilerOptions: { target: ts.ScriptTarget.ES2022 } }).outputText
function fixture(action = 'ready') {
  const cell = value => ({value})
  const events = [], calls = []
  const context = {
    props: { projectId: 'project', run: {id: 'base'} }, pendingRevision: cell({proposal_id: 'proposal'}),
    branchBusy: cell(false), branchError: cell(''), branchStage: cell(''), revisionPlan: cell(null),
    actionMessageId: cell(''), revisionDatasets: cell([]), revisionDataset: cell('NELL'), isolatedCopy: cell(true), showBranch: cell(true),
    api: { revisionRunPlan: async () => ({action, proposal_id: 'proposal', gpu_requested: true, reason: 'requires preparation', selected_datasets:['NELL'], datasets:['Cora','NELL'], scope_explicit:true}),
      runRevision: async (...args) => { calls.push(args); return {run_id:'new'} } },
    window: {confirm: () => true}, approveEnvironmentMigration: async () => true,
    emit: (...args) => events.push(args),
  }
  const run = () => new Function(...Object.keys(context), `${javascript};return runRevision()`)(...Object.values(context))
  return {context, run, calls, events}
}

for (const action of ['prepare_code', 'blocked']) test(`${action}: show the reason, do not approve or submit`, async () => {
  const f = fixture(action)
  f.context.window.confirm = () => assert.fail('Must not ask to run a blocked experiment')
  await f.run()
  assert.equal(f.calls.length, 0); assert.equal(f.context.branchError.value, 'requires preparation')
  assert.equal(f.context.branchBusy.value, false)
  assert.equal(f.context.pendingRevision.value.proposal_id, 'proposal')
})
test('cancel approval retains the proposal and makes no mutation request', async () => {
  const f = fixture(); f.context.window.confirm = () => false
  await f.run(); assert.equal(f.calls.length, 0); assert.match(f.context.branchStage.value, /取消/)
})
test('cancel environment approval does not submit an experiment', async () => {
  const f = fixture('approve_environment'); f.context.approveEnvironmentMigration = async () => null
  await f.run(); assert.equal(f.calls.length, 0); assert.match(f.context.branchStage.value, /取消环境/)
})
test('approved ready flow submits once, then switches to the returned run', async () => {
  const f = fixture('approve_environment')
  await f.run(); await f.run()
  assert.equal(f.calls.length, 1); assert.equal(f.calls[0][2], 'proposal')
  assert.deepEqual(f.calls[0][4], ['NELL']); assert.equal(f.calls[0][5], true)
  assert.deepEqual(f.events, [['branchStarted', 'new']]); assert.equal(f.context.pendingRevision.value, null)
})
test('an in-progress submit prevents a duplicate click', async () => {
  const f = fixture(); f.context.branchBusy.value = true
  await f.run(); assert.equal(f.calls.length, 0)
})
test('changing the selected experiment during preflight does not approve it', async () => {
  const f = fixture(); f.context.api.revisionRunPlan = async () => {
    f.context.props.run.id = 'other'; return {action:'ready'}
  }
  await f.run(); assert.equal(f.calls.length, 0); assert.match(f.context.branchError.value, /已变化/)
})
test('network errors are visible and preserve the retryable proposal', async () => {
  const f = fixture(); f.context.api.revisionRunPlan = async () => {throw new Error('probe timeout')}
  await f.run(); assert.equal(f.calls.length, 0); assert.equal(f.context.branchError.value, 'probe timeout')
  assert.equal(f.context.branchBusy.value, false); assert.ok(f.context.pendingRevision.value)
})

test('reloaded/manual workbench uses authoritative proposal scope, not an empty chat selection', async () => {
  const f = fixture(); f.context.revisionDataset.value = '@intent'
  f.context.revisionDatasets.value = []; f.context.actionMessageId.value = ''
  let confirmation = ''
  f.context.window.confirm = message => {confirmation = message; return true}
  await f.run()
  assert.deepEqual(f.calls[0][4], ['NELL'])
  assert.match(confirmation, /仅运行：NELL/)
})

test('an explicit NELL instruction cannot expand to all datasets through a stale select', async () => {
  const f = fixture(); f.context.revisionDataset.value = ''
  await f.run(); assert.equal(f.calls.length, 0)
  assert.match(f.context.branchError.value, /超出原指令/)
})
