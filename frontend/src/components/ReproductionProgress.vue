<script setup lang="ts">
import { computed, onMounted, onUnmounted, ref, watch } from 'vue'
import { api } from '../api'
import { runDatasetScope } from '../runScope'
import { trainingDeviceLabel } from '../gpuEvidence'
import { baselineComparison } from '../baselineComparison'
import { installationTransportFailed } from '../environmentFailure'
import type { ReproductionActivity, ReproductionRun, TuningPlan } from '../types'

const props = defineProps<{ run: ReproductionRun; approving?: boolean }>()
defineEmits<{ approve: [budget: number]; resume: []; uploadData: []; uploadCode: []; chooseRepository: [url: string]; continuePartial: [] }>()
const enableTuning = ref(false)
const tuningPlans = ref<TuningPlan[]>([])
watch([() => props.run.id, () => props.run.status], async () => {
  enableTuning.value = false
  tuningPlans.value = []
  if (props.run.status !== 'awaiting_approval') return
  const id = props.run.id
  try { const response = await api.tuningPlan(props.run.project_id, id); if (props.run.id === id) tuningPlans.value = response.plans }
  catch { /* No plan means no optional tuning approval. */ }
}, { immediate: true })
const repositoryInput = ref('')
watch(() => props.run.id, () => { repositoryInput.value = props.run.repository_url || '' }, { immediate: true })

const codingIssues = computed(() => {
  const plan = props.run.code_spec.coding_plan as { issue_ledger?: unknown } | undefined
  const rows = props.run.code_spec.issue_ledger || plan?.issue_ledger
  return Array.isArray(rows) ? rows.filter(row => row && typeof row === 'object').map(row => ({
    id: String(row.id || ''), file: String(row.file || ''), reason: String(row.reason || ''),
    status: String(row.status || 'open'), verification: String(row.verification || ''),
  })) : []
})
const codingIssueStatus = (status: string) => status === 'resolved' ? '已复核解决' : status === 'pending_verification' ? '已修改，待复核' : '待处理'
const codingPhase = (phase?: string) => ({ write: '文件生成/修复', review: '方法审查', review_protocol: '审查响应校验', plan: '实现规划' }[phase || ''] || '检查')

const stages = [
  ['理解论文', ['paper_parsing', 'repo_discovery']],
  ['准备复现', ['source_acquisition', 'experiment_planning', 'needs_input', 'code_inspection', 'environment_planning', 'data_planning', 'awaiting_data', 'awaiting_approval']],
  ['构建环境', ['environment_building', 'data_downloading']],
  ['执行实验', ['executing', 'repairing', 'tuning']],
  ['验证结果', ['result_verification', 'completed', 'failed']],
] as const

const currentIndex = computed(() => {
  const stage = props.run.status === 'paused'
    ? String(props.run.result_json.paused_from_stage ||
      [...props.run.events].reverse().find(event => event.stage !== 'paused')?.stage || 'paper_parsing')
    : props.run.status === 'failed'
    ? [...props.run.events].reverse().find(event => event.stage !== 'failed')?.stage || 'failed'
    : props.run.current_stage
  return Math.max(0, stages.findIndex(item => (item[1] as readonly string[]).includes(stage)))
})
const command = computed(() => String(props.run.execution_command.display || '等待 Environment Agent 生成'))
const missing = computed(() => (props.run.environment_spec.missing as string[] | undefined) || [])
const environmentName = computed(() => String(props.run.environment_spec.environment_name || '正在识别'))
const platformAlert = computed(() => props.run.environment_spec.platform_alert as {
  title: string; reason: string; issues: { requirement: string; reason: string; options: string[] }[]; recommended_python?: string
} | undefined)
const platformAlertOpen = ref(false)
watch(() => `${props.run.id}:${JSON.stringify(platformAlert.value || null)}`, () => {
  platformAlertOpen.value = Boolean(platformAlert.value?.issues?.length)
}, { immediate: true })
const datasetEvidence = computed(() => Object.entries(
  {...(props.run.dataset_spec.local_dataset_evidence || {}) as Record<string, { path?: string; ready?: boolean; file_count?: number }>,
    ...gpuReadiness.value?.current_local_dataset_evidence},
).map(([label, value]) => ({ label, ...value })))
const unplannedDatasets = computed(() => (props.run.execution_command.unplanned_datasets as string[] | undefined) || [])
const hardwarePlan = computed(() => props.run.environment_spec.hardware_plan as { requirement: string; available: boolean; gpu?: { name: string } } | undefined)
const gpuReadiness = ref<Awaited<ReturnType<typeof api.runReadiness>> | null>(null)
const runtimeLabel = computed(() => {
  if (gpuReadiness.value?.framework === 'mixed') return '源码同时使用 TensorFlow 与 PyTorch；框架迁移尚未完成'
  const evidence = gpuReadiness.value?.runtime
  if (!evidence?.checked && evidence?.error?.includes("No module named 'torch'")) return '项目环境尚未安装 PyTorch，GPU 未就绪'
  if (!evidence?.checked) return evidence?.error || 'GPU 运行时尚未验证'
  return `${gpuReadiness.value?.framework} ${evidence.version || ''}，CUDA ${evidence.cuda_available ? '可用' : '不可用'}`
})
const sourceGpuLabel = computed(() => !gpuReadiness.value ? '尚未完成' :
  gpuReadiness.value.framework === 'mixed' ? '框架混用，需协调迁移' :
  !['pytorch', 'tensorflow'].includes(gpuReadiness.value.framework) ? '此框架暂不支持源码级验证；不代表源码不支持 GPU' :
  gpuReadiness.value.gpu_code_ready ? '通过' : '发现潜在设备问题，需查看详情')
const trainingGpuLabel = computed(() => trainingDeviceLabel(gpuReadiness.value?.device_log_evidence))
const dependencySourceIsHistorical = computed(() => {
  const source = String(props.run.environment_spec.dependency_source || '')
  return (gpuReadiness.value?.framework === 'pytorch' && source.includes('TensorFlow')) ||
    (gpuReadiness.value?.framework === 'tensorflow' && String(gpuReadiness.value.runtime?.version).startsWith('2.') && source.includes('TensorFlow 1'))
})
let gpuReadinessRequest = 0
async function refreshGpuReadiness() {
  const runId = props.run.id
  const request = ++gpuReadinessRequest
  try {
    const result = await api.runReadiness(props.run.project_id, runId)
    if (props.run.id === runId && request === gpuReadinessRequest) gpuReadiness.value = result
  } catch { if (props.run.id === runId && request === gpuReadinessRequest) gpuReadiness.value = null }
}
const isActive = computed(() => ['queued', 'running', 'executing'].includes(props.run.status))
const draftFiles = computed(() => Array.isArray(props.run.code_spec.draft_files) ? props.run.code_spec.draft_files as string[] : [])
const codeValidation = computed(() => draftFiles.value.length ? undefined : props.run.code_spec.validation as { method_fidelity?: string; runtime?: string } | undefined)
const missingDatasets = computed(() => (props.run.dataset_spec.missing_datasets as string[] | undefined) || [])
const publicSources = computed(() => {
  const candidates = (props.run.dataset_spec.dataset_source_candidates || {}) as Record<string, Array<Record<string, unknown>>>
  const downloads = (props.run.dataset_spec.download_requests || []) as Array<Record<string, unknown>>
  return Object.entries(candidates).filter(([, rows]) => rows.length).map(([dataset, rows]) => {
    const download = downloads.find(row => row.dataset === dataset)
    return { dataset, url: String(rows[0]?.url || ''), source: String(rows[0]?.source || ''),
      status: gpuReadiness.value?.current_local_dataset_evidence?.[dataset]?.ready ? '当前文件已找到；论文处理版本仍需核对' :
        download?.download_status === 'ready' ? '文件已适配，论文处理版本待核对' :
        download?.download_status === 'failed' ? '下载或适配失败；等待重试或补充，不影响其他就绪数据集' :
        currentOperation.value?.dataset === dataset && activity.value?.active ? '正在下载；进度见实时操作日志' :
        props.run.status === 'paused' && props.run.result_json.paused_from_stage === 'data_downloading' ? '下载任务已暂停；文件尚未验证完成' :
        download ? '已找到下载来源，批准执行后下载' : '已找到公开候选，身份或加载格式待核对' }
  })
})
const priorPreparation = computed(() => {
  const rows = props.run.result_json.preparation_history as Array<Record<string, unknown>> | undefined
  return (rows || []).map(row => ({ time: String(row.time || ''),
    reason: String((row.action_required as Record<string, unknown> | undefined)?.reason || row.user_action || ''),
    missing: ((row.dataset_spec as Record<string, unknown> | undefined)?.missing_datasets || []) as string[],
  })).filter(row => row.reason || row.missing.length)
})
const reviewResponseText = (value?: Record<string, unknown>) => value ? JSON.stringify(value, null, 2) : ''
const reviewBlocked = computed(() => props.run.result_json.action_required?.failure_kind === 'review_response_invalid'
  || props.run.error_message.includes('方法审查未完成')
  || (props.run.result_json.code_generation_attempts || []).some(row => row.phase === 'review_protocol' && row.status !== 'resolved'))
const verified = computed(() => Boolean(props.run.result_json.finished_at && props.run.events.some(item => item.stage === 'result_verification')))
const reportReady = computed(() => verified.value && Boolean(props.run.result_json.report_path))
const reportUrl = computed(() => api.reproductionReportUrl(props.run.project_id, props.run.id))
const datasetResults = computed(() => props.run.result_json.dataset_results || [])
const claimResults = computed(() => verified.value ? props.run.result_json.claim_results || [] : [])
const metricClaims = computed(() => claimResults.value.filter(item => item.dataset))
const contextualClaims = computed(() => claimResults.value.filter(item => !item.dataset))
const baselineClaims = computed(() => metricClaims.value.filter(item => item.scope !== 'tuned'))
const supportedCount = computed(() => baselineClaims.value.filter(item => item.supported).length)
const baselineComparisons = computed(() => datasetResults.value.filter(item => baselineComparison(item) !== null))
const baselineWithinTolerance = computed(() => baselineComparisons.value.filter(item => baselineComparison(item) === 'matched').length)
const claimLabel = (claim: { supported: boolean; status: string }) => claim.supported ? '达到指标容差' :
  claim.status === 'functional_completed' ? '功能运行完成' :
  claim.status === 'deviated' ? '指标有偏差' : claim.status === 'failed' ? '运行失败' : '尚未验证'
const completeCount = computed(() => datasetResults.value.filter(item => item.return_code === 0 && Object.keys(item.metrics).length).length)
const targetCount = computed(() => runDatasetScope(props.run).length)
const reproductionScope = computed(() => {
  const scope = props.run.paper_spec.reproduction_scope as { required?: Array<{ name: string }>; optional?: Array<{ name: string }> } | undefined
  return scope?.required?.length ? scope : null
})
const needsData = computed(() => props.run.status === 'awaiting_data' || datasetResults.value.some(item => item.diagnosis.type === 'missing_dataset'))
const activity = ref<ReproductionActivity | null>(null)
const activityUnavailable = ref(false)
const now = ref(Date.now())
const codeWaitSeconds = computed(() => {
  const started = Date.parse(props.run.result_json.code_agent_progress?.updated_at || '')
  return Number.isFinite(started) ? Math.max(0, Math.floor((now.value - started) / 1000)) : 0
})
let timer: number | undefined
const operationMinutes = computed(() => {
  const started = activity.value?.operation?.started_at || props.run.result_json.active_operation?.started_at
  const timestamp = Date.parse(started || '')
  return Number.isFinite(timestamp) ? Math.max(0, Math.floor((now.value - timestamp) / 60000)) : null
})
const currentOperation = computed(() => activity.value ? activity.value.operation : props.run.result_json.active_operation)
const currentMessage = computed(() => props.run.status === 'paused' && activity.value?.worker_active === false
  ? '任务已暂停，后台步骤已停止；已有数据和实验记录保留，不会自动恢复' : props.run.message)
const operationLabel = computed(() => currentOperation.value?.kind === 'dataset_download' || props.run.current_stage === 'data_downloading'
  ? '数据下载' : props.run.current_stage === 'environment_building' ? '环境构建命令' : '运行操作')
const downloadProgress = computed(() => {
  const op = currentOperation.value
  if (op?.downloaded_bytes === undefined) return ''
  const bytes = (value: number) => `${(value / 1024 / 1024).toFixed(2)} MB`
  return `${op.dataset || ''} · ${op.state || '下载中'} · ${bytes(op.downloaded_bytes)} / ${op.total_bytes ? bytes(op.total_bytes) : '总量未知'}${op.attempt ? ` · 尝试 ${op.attempt}/2` : ''}`
})
let refreshingActivity = false
async function refreshActivity() {
  now.value = Date.now()
  if (refreshingActivity) return
  if (!['queued', 'running', 'executing', 'paused'].includes(props.run.status)) { activity.value = null; return }
  const runId = props.run.id
  refreshingActivity = true
  try {
    const value = await api.activity(props.run.project_id, runId)
    if (props.run.id === runId) { activity.value = value; activityUnavailable.value = false }
  }
  catch {
    if (props.run.id === runId) { activity.value = null; activityUnavailable.value = true }
  } finally { refreshingActivity = false }
}
onMounted(() => { refreshActivity(); refreshGpuReadiness(); timer = window.setInterval(refreshActivity, 2500) })
onUnmounted(() => { if (timer) window.clearInterval(timer) })
watch(() => props.run.id, () => { activity.value = null; activityUnavailable.value = false; gpuReadiness.value = null; refreshActivity(); refreshGpuReadiness() })
watch(() => props.run.status, refreshActivity)
watch([() => props.run.status, () => props.run.current_stage, () => JSON.stringify(props.run.environment_spec.gpu_runtime_verified || null)], () => {
  if (['executing', 'completed', 'failed'].includes(props.run.current_stage)) refreshGpuReadiness()
})
const percent = (value: number | null | undefined) => value === null || value === undefined ? '—' : `${(value * 100).toFixed(1)}%`
const metricLabel = (key: string) => key === 'accuracy' ? '准确率' : key.toUpperCase()
const metricValue = (value: number | null | undefined, key: string) => value === null || value === undefined ? '—' : ['accuracy', 'auc', 'auroc', 'aupr', 'f1', 'precision', 'recall'].includes(key) ? percent(value) : value.toFixed(4)
const installFailureDetail = computed(() => {
  if (props.run.status === 'failed' && installationTransportFailed(props.run.error_message, props.run.environment_spec.last_failure)) {
    return '依赖来源的网络、代理或 TLS 连接失败，未证明该版本与本机不兼容。请先核对连接后重试准备；不会据此更换 Python、框架或执行主机。实验尚未启动。'
  }
  if (props.run.status === 'failed' && /No matching distribution|Could not find a version/.test(props.run.error_message)) {
    const plan = gpuReadiness.value?.gpu_environment_plan
    return `安装包与当前解释器／平台不匹配，不是模型训练错误。${plan?.supported && plan.packages?.length ? `可预检的兼容方案：${plan.packages.join('、')}。` : plan?.reason || '需重新核对 Python 与依赖版本。'}重试先做依赖预检，不会无条件重装或更换 Python。`
  }
  if (props.run.status !== 'failed' || !/返回码 124|命令超过 \d+ 秒|安装命令超过 \d+ 秒/.test(props.run.error_message)) return ''
  const failedStage = [...props.run.events].reverse().find(event => event.stage !== 'failed')?.stage
  if (failedStage !== 'environment_building' && !/TensorFlow 环境迁移|Conda 环境构建失败|当前环境补装依赖失败/.test(props.run.error_message)) return ''
  const downloads = [...(props.run.logs || '').matchAll(/Downloading\s+([^\r\n]+)/g)]
  const last = downloads.at(-1)?.[1] || ''
  return last ? `安装超时前正在下载：${last}。这是环境安装失败，不是模型训练错误。原依赖快照保留；重试前须核验环境和安装计划。` : '环境安装超时，未进入训练。重试前须检查下载状态、原依赖快照和安装变更计划；不会通过修改模型代码解决下载超时。'
})
const displayedError = computed(() => {
  if (installFailureDetail.value && props.run.error_message.includes('TensorFlow 环境迁移')) {
    const seconds = props.run.error_message.match(/(?:安装)?命令超过 (\d+) 秒/)?.[1]
    return `TensorFlow GPU 环境安装超时${seconds ? `（${Math.round(Number(seconds) / 60)} 分钟）` : ''}，已停止安装，未启动训练。`
  }
  return props.run.error_message.replace(/\u001b\[[0-9;]*[A-Za-z]/g, '')
})
</script>

<template>
  <section class="reproduction-console" :class="`run-${run.status}`">
    <div class="reproduction-head">
      <div>
        <span class="kicker">AUTONOMOUS REPRODUCTION</span>
        <h2>自主复现流水线</h2>
        <p>{{ currentMessage }}</p>
      </div>
      <div class="run-head-metrics">
        <div class="progress-number"><strong>{{ run.progress }}</strong><span>%</span><small>流程完成度</small></div>
        <div class="experiment-count"><strong>{{ targetCount ? `${completeCount}/${targetCount}` : '—' }}</strong><small>{{ targetCount ? '实验产出指标' : '实验目标待核验' }}</small></div>
      </div>
    </div>

    <div class="progress-track reproduction-track"><span :style="{ width: `${run.progress}%` }"></span></div>
    <div class="stage-grid">
      <div v-for="(stage, index) in stages" :key="stage[0]" class="stage-item" :class="{ done: index < currentIndex || run.status === 'completed', active: index === currentIndex && run.status !== 'failed', failed: index === currentIndex && run.status === 'failed' }">
        <span class="stage-dot">{{ index < currentIndex || run.status === 'completed' ? '✓' : index + 1 }}</span>
        <small>{{ stage[0] }}</small>
      </div>
    </div>

    <div class="run-detail-grid">
      <div class="run-detail-card">
        <span>源码策略</span>
        <strong>{{ run.source_mode === 'paper_repository' ? '论文仓库' : run.source_mode === 'uploaded_code' ? '上传代码' : run.source_mode === 'generated_implementation' || run.source_mode === 'generated_scaffold' ? 'Agent 根据论文实现' : run.source_mode === 'copilot_revision' ? '助手修改后的项目代码' : run.source_mode === 'copilot_branch' ? '独立实验分支' : run.source_mode === 'generated_fallback' ? '仓库获取失败后生成' : '正在识别' }}</strong>
        <a v-if="run.repository_url" :href="run.repository_url" target="_blank" rel="noreferrer">{{ run.repository_url }}</a>
        <small v-else>尚未确定仓库地址</small>
      </div>
      <div class="run-detail-card">
        <span>待执行命令</span>
        <pre>{{ command }}</pre>
        <small>当前模式：{{ run.execution_mode === 'full' ? unplannedDatasets.length ? '完整实验目标 · 部分命令待补' : '已批准范围完整运行：' + ((run.execution_command.commands as Array<{ dataset: string }> | undefined) || []).map(item => item.dataset).join('、') : '仅语法试运行' }}</small>
      </div>
      <div class="run-detail-card">
        <span>独立 Conda 环境</span>
        <strong>{{ environmentName }}</strong>
        <small>{{ missing.length ? `Agent 推断 ${missing.length} 个依赖` : `${dependencySourceIsHistorical ? '原基线依赖记录：' : ''}${String(run.environment_spec.dependency_source || '正在分析依赖')}` }}</small>
      </div>
    </div>
    <div v-if="run.workspace_path" class="workspace-path">
      <span>论文项目工作区</span><code>{{ run.workspace_path }}</code>
    </div>
    <div v-if="datasetEvidence.length" class="dataset-evidence"><strong>作者数据核验</strong><span v-for="item in datasetEvidence" :key="item.label" :class="item.ready ? 'dataset-found' : 'dataset-incomplete'">{{ item.ready ? '✓' : '△' }} {{ item.label }} · {{ item.path }} · {{ item.file_count || 0 }} 文件</span></div>
    <div v-if="reproductionScope" class="run-notice"><strong>本次复现范围</strong><span>必须实现：{{ reproductionScope.required?.map(item => item.name).join('、') }}。</span><span v-if="reproductionScope.optional?.length">未选择的扩展：{{ reproductionScope.optional.map(item => item.name).join('、') }}。这些扩展不计入本次通过标准，不代表已复现全部论文实验。</span></div>
    <details v-if="publicSources.length" class="run-events" open>
      <summary>公开数据来源与适配状态 · {{ publicSources.length }} 组</summary>
      <ol><li v-for="item in publicSources" :key="item.dataset"><strong>{{ item.dataset }}</strong><span>{{ item.status }} · {{ item.source }}<br><a v-if="item.url.startsWith('https://')" :href="item.url" target="_blank" rel="noreferrer">查看来源</a></span></li></ol>
      <small>公开原始数据不代表论文所用的编码、缺失值处理和划分已完全一致；未核实版本不会标记为复现成功。</small>
    </details>
    <div v-if="unplannedDatasets.length" class="compatibility-note"><strong>实验命令待补</strong><span>{{ unplannedDatasets.join('、') }} 尚未得到可安全执行的独立命令；不会计入完整复现。</span></div>
    <div v-if="run.environment_spec.compatibility_note" class="compatibility-note">
      <strong>依赖兼容信息</strong><span>{{ gpuReadiness?.framework === 'pytorch' && String(run.environment_spec.compatibility_note).includes('TensorFlow 1.x') ? '这是原 TensorFlow 基线的环境记录；当前源码已改用 PyTorch，需要按实际 Python 版本预检 GPU 包，不能沿用旧框架标签判断可用。' : String(run.environment_spec.compatibility_note) }}</span>
    </div>
    <div v-if="run.source_mode === 'generated_implementation'" class="compatibility-note"><strong>无作者源码</strong><span>代码由 Agent 根据论文生成；请重点核对模型结构、数据预处理与论文实验设置。语法通过和训练运行不等于论文方法已经复现。</span><span v-if="!draftFiles.length && codeValidation?.method_fidelity === 'reviewed' && Array.isArray(run.code_spec.limitations) && run.code_spec.limitations.length">限制：{{ run.code_spec.limitations.join('；') }}</span></div>
    <div v-if="run.result_json.code_agent_progress || draftFiles.length || codeValidation" class="operation-live">
      <strong>{{ isActive ? '论文编码进度' : '论文编码检查记录' }}</strong>
      <span v-if="run.result_json.code_agent_progress">{{ run.result_json.code_agent_progress.message }}</span>
      <small v-if="isActive && run.result_json.code_agent_progress">距本步骤开始 {{ codeWaitSeconds }} 秒。正在等待模型返回或检查源码；此处不是训练进度，也不表示实验已启动。</small>
      <small v-if="draftFiles.length">已保留草稿：{{ draftFiles.join('、') }}。草稿不等于通过全部检查；重试时会保留有效文件，继续修正未完成部分。</small>
      <small v-if="codeValidation">编译与本地导入检查已通过；{{ codeValidation.method_fidelity === 'reviewed' ? '已完成模型辅助方法审查，仍需人工核对论文' : '方法一致性尚待审查' }}；{{ codeValidation.runtime === 'pending_environment' ? '运行验证待环境与数据准备后进行' : '运行状态请以实验记录为准' }}。</small>
      <details v-if="codingIssues.length" :open="run.status === 'needs_input'">
        <summary>代码问题记录 · {{ codingIssues.filter(item => item.status !== 'resolved').length }} 项待处理 / {{ codingIssues.length }} 项累计</summary>
        <div v-for="item in codingIssues" :key="item.id">
          <strong>{{ codingIssueStatus(item.status) }} · {{ item.file }}</strong>
          <small>{{ item.reason }}</small>
          <small v-if="item.verification">复核依据：{{ item.verification }}</small>
        </div>
      </details>
    </div>
    <div v-if="hardwarePlan" class="hardware-note"><strong>计算设备</strong><span>主机：{{ hardwarePlan.available ? String(hardwarePlan.gpu?.name || '已检测到 GPU') : '未检测到 GPU' }}；当前环境：{{ runtimeLabel }}；源码 GPU 静态检查：{{ sourceGpuLabel }}；{{ trainingGpuLabel }}</span></div>
    <div v-if="['environment_building', 'data_downloading'].includes(run.current_stage) || activity?.operation" class="operation-live">
      <div><strong>{{ run.status === 'paused' ? `最近${operationLabel}（任务已暂停）` : `当前${operationLabel}` }}</strong><span>{{ operationMinutes === null ? '耗时未知' : `${operationMinutes} 分钟` }} · {{ currentOperation?.step || '—' }}/{{ currentOperation?.total_steps || '—' }} 步</span></div>
      <small v-if="downloadProgress">{{ downloadProgress }}</small>
      <small v-if="currentOperation?.error">最近异常：{{ currentOperation.error }}</small>
      <code>{{ currentOperation?.command || '当前命令未回传' }}</code>
      <pre v-if="activity?.log_tail">{{ activity.log_tail }}</pre>
      <small v-if="activityUnavailable">日志接口暂时不可用，无法确认实时输出；页面任务状态不代表当前命令的实际进度。</small>
      <small v-else-if="!currentOperation">后端未提供当前操作和实时日志，无法确认下载或安装进度。这不代表已经完成，也不能据此认定没有错误。</small>
      <small v-else-if="!activity?.log_tail">当前命令尚无日志输出；依赖解析或下载可能暂时没有输出，请以实时日志为准。</small>
    </div>
    <div v-if="run.status === 'paused'" class="operation-live"><strong>任务已暂停</strong><small>已完成的数据集结果会保留。若作者代码不支持断点续训，恢复后当前未完成的数据集将从头训练，而不是从暂停时的 Epoch 继续。</small></div>

    <section v-if="datasetResults.length" class="experiment-section">
      <div class="section-title-row compact"><div><span class="kicker">EXPERIMENT RESULTS</span><h3>逐数据集结果</h3></div><small>{{ completeCount }} / {{ datasetResults.length }} 组产生指标</small></div>
      <div class="experiment-grid">
        <article v-for="item in datasetResults" :key="item.dataset" class="experiment-card" :class="`experiment-${item.verdict}`">
          <div class="experiment-card-title"><strong>{{ item.dataset }}</strong><span>{{ item.verdict === 'matched' ? '✓ 指标在容差内' : item.verdict === 'result_produced' ? '△ 已产生指标，目标待核' : item.verdict === 'metric_missing' ? '△ 缺少指标' : item.verdict === 'deviated' ? '△ 指标超出容差' : item.verdict === 'tuned_result' ? '△ 调参结果，非原配置' : item.verdict === 'configuration_unverified' ? '△ 配置待核' : item.verdict === 'dataset_version_unverified' ? '△ 版本待核' : item.return_code === 0 ? '△ 待核验' : '✗ 未完成' }}</span></div>
          <div class="experiment-value">{{ metricValue(item.metrics[item.metric_key || 'accuracy'], item.metric_key || 'accuracy') }}</div>
          <small>{{ metricLabel(item.metric_key || 'accuracy') }} · 论文 {{ metricValue(item.target, item.metric_key || 'accuracy') }} · 差值 {{ item.delta === null ? '—' : item.delta.toFixed(4) }}</small>
          <small v-if="item.comparison_scope === 'functional_check'">{{ item.comparison_basis }}</small>
          <small v-if="item.process_exit_code === 0 && item.return_code !== 0">进程退出码为 0，但日志报告了明确运行异常，按运行失败记录；原始退出码和日志均保留。</small>
          <p v-if="item.diagnosis.suggestion">{{ item.diagnosis.suggestion }}</p>
        </article>
      </div>
    </section>

    <section v-if="claimResults.length" class="claim-section">
      <div class="section-title-row compact"><div><span class="kicker">CLAIM VERIFICATION</span><h3>实验指标核验</h3></div><small>{{ baselineWithinTolerance }} / {{ baselineComparisons.length }} 组基线数值在容差内</small></div>
      <small>{{ supportedCount }} 项具有配置核验证据支持；数值比较不替代数据版本、配置和复现协议核验。</small>
      <small>指标接近不等于论文全部结论已复现；尚未验证不代表结论错误。作者配置基线和调参结果分别展示。</small>
      <div v-for="claim in metricClaims" :key="claim.claim" class="claim-row"><span :class="claim.supported ? 'claim-yes' : ['deviated', 'failed'].includes(claim.status) ? 'claim-no' : ''">{{ claim.supported ? '✓' : ['deviated', 'failed'].includes(claim.status) ? '!' : '—' }}</span><div><strong>{{ claim.claim }} · {{ claimLabel(claim) }}</strong><small>{{ claim.evidence }}</small></div></div>
      <details v-if="contextualClaims.length"><summary>其他论文主张：{{ contextualClaims.length }} 项（本轮未检验）</summary><div v-for="claim in contextualClaims" :key="claim.claim" class="claim-row"><span>—</span><div><strong>{{ claim.claim }}</strong><small>{{ claim.evidence }}</small></div></div></details>
    </section>

    <div v-if="run.status === 'needs_input' && run.result_json.action_required?.kind === 'hardware_unavailable'" class="run-error data-required"><strong>执行主机不满足 GPU 要求</strong><span>{{ run.result_json.action_required.reason }}</span><button class="button secondary" @click="$emit('resume')">重新检测设备</button></div>
    <div v-if="run.status === 'needs_input' && run.result_json.action_required?.kind === 'paper_spec_failed'" class="run-error data-required"><strong>实验信息待核验（不是作者代码运行失败）</strong><span>{{ run.result_json.action_required.reason }}</span><span>{{ run.current_stage === 'paper_parsing' ? '旧任务被论文规范校验阻断；继续后先获取作者源码，再核验数据集与指标。' : run.result_json.action_required.detail }}</span><button class="button primary" @click="$emit('resume')">继续作者源码复现准备（不运行实验）</button></div>
    <div v-if="(run.status === 'needs_input' && !['hardware_unavailable', 'paper_spec_failed'].includes(run.result_json.action_required?.kind || '')) || (run.status === 'failed' && !run.approved_at)" class="run-error data-required">
      <strong>{{ run.result_json.action_required?.kind === 'code_generation_failed' ? '未开源自动编码已暂缓，旧草稿保留' : '需要获取作者源码' }}</strong>
      <span>{{ run.result_json.action_required?.detail || run.result_json.action_required?.reason || run.message }}</span>
      <details v-if="run.result_json.code_generation_attempts?.length"><summary>检查与请求记录（不代表完整生成失败次数）</summary>
        <div v-for="attempt in run.result_json.code_generation_attempts || []" :key="attempt.attempt"><small>{{ attempt.file ? `${attempt.file} · ` : '' }}{{ codingPhase(attempt.phase) }}：{{ attempt.reason }}</small><details v-if="attempt.review_response"><summary>原始审查意见（未通过证据校验，不作为已确认缺陷）</summary><pre>{{ reviewResponseText(attempt.review_response) }}</pre></details></div>
      </details>
      <small>目前仅复现作者仓库或上传的源码，不会生成替代实现。对已有源码的修改和纠错仍可使用。</small>
      <label>作者仓库地址<input v-model="repositoryInput" aria-label="作者仓库地址" placeholder="https://github.com/作者/仓库" /></label>
      <div class="decision-actions"><button class="button primary" :disabled="!repositoryInput.trim()" @click="$emit('chooseRepository', repositoryInput.trim())">获取此作者仓库</button><button class="button secondary" @click="$emit('uploadCode')">上传作者代码</button><button v-if="run.repository_url" class="button secondary" @click="$emit('resume')">重试作者仓库</button></div>
    </div>

    <div v-if="run.status === 'awaiting_data'" class="run-error data-required">
      <strong>数据准备未完成（不代表数据未公开）</strong>
      <span>{{ String(run.result_json.user_action || `缺少 ${missingDatasets.join('、')} 的作者数据。`) }}</span>
      <div class="decision-actions"><button class="button secondary" @click="$emit('resume')">重试数据准备</button><button class="button secondary" @click="$emit('uploadData')">上传数据并继续</button><button class="button secondary" @click="$emit('continuePartial')">跳过缺失数据，继续可用实验</button></div>
      <small>跳过的数据集会在报告中标记为未验证，不会视为完整复现。</small>
    </div>

    <div v-if="run.status === 'awaiting_approval'" class="approval-panel">
      <div><strong>执行前需要人工确认</strong><p>批准后先检查项目 Conda 环境：首次使用时创建，后续复用现有环境并按需安装依赖，再运行计划中的实验。</p></div>
      <details v-if="tuningPlans.some(plan => plan.trials.length)"><summary>查看备用调参方案（本轮先完成作者配置基线）</summary><div v-for="plan in tuningPlans" :key="plan.dataset"><strong>{{ plan.dataset }}</strong><p v-for="item in plan.space" :key="item.parameter">{{ item.parameter }}：基线 {{ item.baseline }}，候选 {{ item.values.join('、') }}；来源 {{ item.source }}</p><small>{{ plan.trials.length }} 次局部/联合组合，最多额外 30 分钟；{{ plan.selection_rule }}。{{ plan.limitation }}</small></div><label><input v-model="enableTuning" type="checkbox" />保留备用搜索计划；基线未复现后需复核原因并另行批准，达到指标的配置不搜索</label></details>
      <button class="button primary" :disabled="approving" @click="$emit('approve', enableTuning ? 6 : 0)">{{ approving ? '正在启动…' : '批准执行全部实验' }}</button>
    </div>

    <div v-if="run.error_message && !['awaiting_data', 'needs_input'].includes(run.status)" class="run-error"><strong>任务诊断</strong><span>{{ displayedError }}<br v-if="installFailureDetail">{{ installFailureDetail }}</span><button v-if="run.status === 'failed'" class="button secondary" @click="$emit('resume')">{{ installFailureDetail ? '检查环境后重试准备' : run.approved_at ? '继续未完成实验' : '继续作者源码复现准备' }}</button></div>

    <div v-if="reportReady" class="report-download">
      <div><strong>{{ run.result_json.reproduction_success ? '复现验证报告已生成' : '复现诊断报告已生成' }}</strong><span>包含结论核验、逐数据集指标、修复与调参记录，以及未完成原因。</span></div>
      <a class="button secondary" :href="reportUrl" download>下载复现报告</a>
    </div>

    <details v-if="run.events.length" class="run-events" :open="isActive">
      <summary>执行记录 · {{ run.events.length }} 条</summary>
      <ol>
        <li v-for="(event, index) in [...run.events].reverse()" :key="`${event.time}-${index}`"><time>{{ new Date(event.time).toLocaleTimeString('zh-CN', { hour12: false }) }}</time><span>{{ event.message }}</span></li>
      </ol>
    </details>
    <details v-if="priorPreparation.length" class="run-events"><summary>此前准备诊断 · 历史记录，不代表本轮状态</summary><ol><li v-for="(item, index) in priorPreparation" :key="index"><time>{{ new Date(item.time).toLocaleString('zh-CN') }}</time><span>{{ item.reason || `此前缺少：${item.missing.join('、')}` }}</span></li></ol></details>
    <details v-if="run.logs" class="run-logs"><summary>查看终端日志</summary><pre>{{ run.logs }}</pre></details>
    <div v-if="platformAlertOpen && platformAlert" class="platform-alert-backdrop">
      <section role="alertdialog" aria-modal="true" aria-labelledby="platform-alert-title" class="platform-alert-dialog">
        <h3 id="platform-alert-title">{{ platformAlert.title }}</h3>
        <p>{{ platformAlert.reason }}</p>
        <div v-for="issue in platformAlert.issues" :key="issue.requirement"><strong>{{ issue.requirement }}</strong><p>{{ issue.reason }}</p><ul><li v-for="option in issue.options" :key="option">{{ option }}</li></ul></div>
        <p v-if="platformAlert.recommended_python">这些固定包的共同候选 Python：{{ platformAlert.recommended_python }}；仍需完整依赖求解和设备验证。</p>
        <button class="button primary" autofocus @click="platformAlertOpen = false">了解兼容性问题</button>
      </section>
    </div>
  </section>
</template>

<style scoped>
.platform-alert-backdrop { position: fixed; inset: 0; background: #0007; display: grid; place-items: center; z-index: 1000; padding: 24px; }
.platform-alert-dialog { background: var(--surface, #fff); color: var(--text, #17202a); padding: 24px; border-radius: 10px; max-width: 680px; max-height: 80vh; overflow: auto; box-shadow: 0 12px 40px #0004; }
</style>
