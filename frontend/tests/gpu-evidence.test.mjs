import assert from 'node:assert/strict'
import {test} from 'node:test'
import fs from 'node:fs'
import ts from 'typescript'

const source = fs.readFileSync(new URL('../src/gpuEvidence.ts', import.meta.url), 'utf8')
const module = {exports:{}}
new Function('exports', ts.transpileModule(source, {compilerOptions:{module:ts.ModuleKind.CommonJS}}).outputText)(module.exports)
const {trainingDeviceLabel} = module.exports

test('display distinguishes actual compute from CUDA enumeration and initialization', () => {
  assert.match(trainingDeviceLabel([{kind:'gpu_compute_placement'}]), /GPU 计算算子/)
  assert.match(trainingDeviceLabel([{kind:'training_device_cpu'}]), /CPU 训练设备/)
  assert.match(trainingDeviceLabel([{kind:'gpu_initialized_not_training_proof'}]), /尚无明确/)
  assert.match(trainingDeviceLabel(), /尚无明确/)
})
