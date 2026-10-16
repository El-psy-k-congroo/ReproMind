import assert from 'node:assert/strict'
import {test} from 'node:test'
import fs from 'node:fs'
import ts from 'typescript'
import {reactive, watch, nextTick} from 'vue'

test('real copilot watcher does not probe GPU again for unchanged polling snapshots', async () => {
  const source = fs.readFileSync(new URL('../src/components/ResearchCopilot.vue', import.meta.url), 'utf8')
  const statement = source.split('\n').find(line => line.startsWith('watch([() => props.run?.id'))
  assert.ok(statement)
  const js = ts.transpileModule(statement, {}).outputText
  const props = reactive({run: {id:'run', execution_command:{pythonpath:'source'}, code_spec:{source_path:'source'}}})
  let probes = 0
  new Function('watch', 'props', 'loadBranchFiles', 'loadRunReadiness', js)(watch, props, async () => {}, async () => {probes++})
  for (let i=0;i<10;i++) {props.run = JSON.parse(JSON.stringify(props.run)); await nextTick()}
  assert.equal(probes, 0)
  props.run.execution_command.pythonpath = 'revised'
  await nextTick(); await nextTick()
  assert.equal(probes, 1)
})

test('runtime display refreshes on verified evidence changes, not on every progress poll', async () => {
  const source = fs.readFileSync(new URL('../src/components/ReproductionProgress.vue', import.meta.url), 'utf8')
  const statement = source.match(/watch\(\[.*gpu_runtime_verified.*\], \(\) => \{[\s\S]*?\n\}\)/)?.[0]
  assert.ok(statement)
  const props = reactive({run:{status:'executing', current_stage:'executing', environment_spec:{gpu_runtime_verified:{checked_at:'first'}}}})
  let probes = 0
  new Function('watch', 'props', 'refreshGpuReadiness', statement)(watch, props, () => {probes++})
  for (let i=0;i<10;i++) {props.run = JSON.parse(JSON.stringify(props.run)); await nextTick()}
  assert.equal(probes, 0)
  props.run.environment_spec.gpu_runtime_verified.checked_at = 'second'
  await nextTick()
  assert.equal(probes, 1)
})
