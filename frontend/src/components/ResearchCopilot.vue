<script setup lang="ts">
import { computed, nextTick, onMounted, onUnmounted, ref, watch } from 'vue'
import { api } from '../api'
import { renderResearchText } from '../renderResearchText'
import RevisionRunStatus from './RevisionRunStatus.vue'
import GpuPreparationPlan from './GpuPreparationPlan.vue'
import type { CopilotConversation, CopilotMessage, ModelCatalog, ReproductionRun, RevisionRunPlan, TuningPlan } from '../types'
import 'katex/dist/katex.min.css'

const props = defineProps<{ projectId: string; modelId: string; run: ReproductionRun | null }>()
const emit = defineEmits<{ close: []; uploadCode: []; uploadData: []; continuePartial: []; approve: []; branchStarted: [runId: string]; runUpdated: [run: ReproductionRun] }>()
const conversations = ref<CopilotConversation[]>([])
const selectedId = ref('')
const messages = ref<CopilotMessage[]>([])
const catalog = ref<ModelCatalog | null>(null)
const chatModel = ref('')
const codingModel = ref('')
const question = ref('')
const busy = ref(false)
const conversationBusy = ref(false)
const streamStatus = ref('正在核对项目证据…')
const error = ref('')
const scrollArea = ref<HTMLElement | null>(null)
const showBranch = ref(false)
const branchFiles = ref<string[]>([])
const branchFile = ref('')
const branchInstruction = ref('')
const branchPreview = ref<{ proposal_id: string; base_run_id: string; files: string[]; diffs: Record<string, string>; rationale: string; expected_observation: string; coding_trace?: Array<{ agent: string; model: string; files?: string[]; status: string }> } | null>(null)
const pendingRevision = ref<{ proposal_id: string; files: string[]; instruction: string; applied_at: string; runnable?: boolean; validation_error?: string } | null>(null)
const isolatedCopy = ref(true)
const revisionDataset = ref('@intent')
const revisionDatasets = ref<string[]>([])
const branchBusy = ref(false)
const branchError = ref('')
const branchStage = ref('')
const revisionPlan = ref<RevisionRunPlan | null>(null)
const runReadiness = ref<Awaited<ReturnType<typeof api.runReadiness>> | null>(null)
const rerunTuningPlans = ref<TuningPlan[]>([])
const rerunTuningEnabled = ref(false)
const rerunBusy = ref(false)
const rerunError = ref('')
const actionMessageId = ref('')
const controlBusy = ref(false)
const controlError = ref('')
const completedControlId = ref('')
const generationBusy = ref(false)
const generationError = ref('')
async function retryPaperSpecification() {
  if (!props.run || generationBusy.value) return
  generationBusy.value = true; generationError.value = ''
  try { emit('runUpdated', await api.resumeReproduction(props.projectId, props.run.id)) }
  catch (err) { generationError.value = (err as Error).message }
  finally { generationBusy.value = false }
}
const legacyGeneratedFailure = computed(() => props.run?.status === 'failed'
  && props.run.source_mode === 'generated_implementation'
  && !Array.isArray(props.run.code_spec?.files))
const latestControl = computed(() => {
  const answer = [...messages.value].reverse().find(item => item.role === 'assistant')
  const kind = answer?.context?.action_plan?.kind
  return kind === 'pause_confirm' || kind === 'resume_confirm' ? answer : null
})
const latestDatasetChoice = computed(() => {
  const answer = [...messages.value].reverse().find(item => item.role === 'assistant')
  return answer?.context?.action_plan?.kind === 'needs_dataset' ? answer : null
})
let aborter: AbortController | null = null
let pendingAsk = false
let historyGeneration = 0
let stoppedByUser = false
const suggestions = ['论文、代码和本次实验的参数有哪些差异？', '这次失败的证据和最小修复方案是什么？', '下一步应做哪项控制变量实验？']
async function confirmExperimentControl() {
  const item = latestControl.value
  const plan = item?.context?.action_plan
  if (!item || !plan || !props.run || plan.run_id !== props.run.id || controlBusy.value) return
  const pause = plan.kind === 'pause_confirm'
  if (pause ? !['queued', 'running', 'executing'].includes(props.run.status) : props.run.status !== 'paused') return
  const warning = pause
    ? '暂停会停止当前训练进程。未完成的数据集或轮次可能需要从头运行，已有结果会保留。'
    : '恢复后，未完成的数据集可能从头运行；请确认当前代码和环境仍符合预期。'
  if (!window.confirm(`${pause ? '确认暂停' : '确认恢复'}实验 ${props.run.id.slice(0, 8)}？\n${warning}`)) return
  controlBusy.value = true; controlError.value = ''
  try {
    const updated = pause ? await api.pauseReproduction(props.projectId, props.run.id)
      : await api.resumePaused(props.projectId, props.run.id)
    completedControlId.value = item.id
    emit('runUpdated', updated)
  } catch (err) { controlError.value = (err as Error).message }
  finally { controlBusy.value = false }
}
async function loadRunReadiness() {
  runReadiness.value = null
  rerunTuningPlans.value = []; rerunTuningEnabled.value = false
  if (!props.run) return
  try {
    runReadiness.value = await api.runReadiness(props.projectId, props.run.id)
    rerunTuningPlans.value = (await api.tuningPlan(props.projectId, props.run.id)).plans
  } catch (err) { rerunError.value = (err as Error).message }
}
async function confirmRerun(item: CopilotMessage) {
  const plan = item.context?.action_plan
  if (!props.run || !plan || !['rerun_confirm', 'initial_run_confirm'].includes(plan.kind) || plan.run_id !== props.run.id || rerunBusy.value) return
  const runId = props.run.id
  if (plan.kind === 'initial_run_confirm') {
    if (!window.confirm(`批准首次运行 ${plan.dataset}？\n请核对所选作者入口、参数和环境方案。批准后将准备依赖、数据并执行实验。${plan.gpu_requested ? '\n要求 GPU；无法使用时停止。' : ''}`)) return
    rerunBusy.value = true; rerunError.value = ''
    try {
      if (props.run?.id !== runId) throw new Error('所选实验已变化，请重新检查')
      await api.approveReproduction(props.projectId, runId)
      emit('branchStarted', runId)
    } catch (err) { rerunError.value = (err as Error).message }
    finally { rerunBusy.value = false }
    return
  }
  if (!window.confirm(`确认运行 ${plan.dataset}？${plan.gpu_requested ? '\n要求 GPU；CUDA 不可用时停止，不退回 CPU。' : ''}\n新实验记录与原结果分开，复用当前 Conda 环境。${rerunTuningEnabled.value ? '\n批准恢复搜索：仅限上一轮基线未复现的数据集，先完成全部基线，再按验证集搜索。已达到指标的配置不调参；每个数据集最多 6 次、额外 30 分钟。请先核对数据版本、划分和作者参数。' : '\n不运行额外调参。'}`)) return
  rerunBusy.value = true; rerunError.value = ''
  try {
    const migration = await approveEnvironmentMigration(!!plan.gpu_requested)
    if (migration === null) return
    if (props.run?.id !== runId) throw new Error('所选实验已变化，请重新检查，未提交运行')
    const created = await api.rerunDataset(props.projectId, plan.run_id, plan.selected_datasets, !!plan.gpu_requested, rerunTuningEnabled.value ? 6 : 0, migration)
    emit('branchStarted', created.run_id)
  } catch (err) { rerunError.value = (err as Error).message }
  finally { rerunBusy.value = false }
}
async function approveEnvironmentMigration(gpu: boolean, checked?: RevisionRunPlan['readiness']): Promise<boolean | null> {
  if (!gpu || !props.run) return false
  const readiness = checked || await api.runReadiness(props.projectId, props.run.id)
  const migration = readiness.gpu_environment_plan
  if (!migration?.needed) return false
  if (readiness.framework === 'pytorch') {
    if (!migration.supported) throw new Error(migration.reason || '当前 Python 与 GPU 包不兼容，需审查版本方案')
    const alternatives = migration.candidates?.map(row => `${row.profile}：${row.packages.join('、')}`).join('\n')
    return window.confirm(`单独批准项目 PyTorch GPU 环境配置？\n${alternatives || migration.packages?.join('、')}\n${migration.warning}\n仅允许所示兼容方案；先预检已有依赖是否会被替换，再验证 GPU 运算；失败停止。`) ? true : null
  }
  if (readiness.framework !== 'tensorflow') return false
  if (!migration.supported || migration.source_changes_needed) throw new Error(migration.reason || '请先完成 TensorFlow 兼容入口审查')
  return window.confirm(`单独批准 GPU 环境迁移？\n\n${migration.packages?.join('、')}\n${migration.warning}\n\n将先记录原依赖版本，再安装和检查真实 GPU 运算；失败停止，不回退 CPU。`) ? true : null
}
async function previewActionPlan(item: CopilotMessage) {
  const plan = item.context?.action_plan
  if (!props.run || !plan || !['gpu_revision', 'code_revision'].includes(plan.kind) || plan.run_id !== props.run.id || branchBusy.value) return
  actionMessageId.value = item.id
  revisionDataset.value = plan.dataset
  revisionDatasets.value = plan.selected_datasets
  isolatedCopy.value = true
  branchInstruction.value = String(plan.instruction || '')
  // A historical applied proposal must not be presented as this new request.
  if (pendingRevision.value?.instruction !== branchInstruction.value) {
    pendingRevision.value = null; revisionPlan.value = null
  }
  branchBusy.value = true; branchError.value = ''; branchPreview.value = null
  try { branchPreview.value = await api.previewBranch(props.projectId, plan.run_id, '', branchInstruction.value, codingModel.value, plan.kind === 'gpu_revision') }
  catch (err) { branchError.value = (err as Error).message }
  finally { branchBusy.value = false }
}
const previewGpuPlan = previewActionPlan

async function loadBranchFiles() {
  branchFiles.value = []; branchFile.value = ''; branchPreview.value = null
  pendingRevision.value = null; actionMessageId.value = ''; revisionDatasets.value = []
  revisionPlan.value = null; branchStage.value = ''
  revisionDataset.value = '@intent'
  if (!props.run) return
  const id = props.run.id
  try {
    const loaded = await api.branchFiles(props.projectId, id)
    if (props.run?.id !== id) return
    branchFiles.value = loaded.files
    branchFile.value = ''
    pendingRevision.value = await api.pendingRevision(props.projectId, id)
    if (props.run?.id !== id) { pendingRevision.value = null; return }
    branchError.value = pendingRevision.value?.validation_error || ''
    syncPendingAction()
    if (pendingRevision.value) {
      const proposalId = pendingRevision.value.proposal_id
      const plan = await api.revisionRunPlan(props.projectId, id, proposalId)
      if (props.run?.id === id && pendingRevision.value?.proposal_id === proposalId) revisionPlan.value = plan
    }
  } catch (err) { branchError.value = (err as Error).message }
}
function syncPendingAction() {
  if (!pendingRevision.value || !props.run) return
  const matching = [...messages.value].reverse().find(item => {
    const plan = item.context?.action_plan
    return item.role === 'assistant' && ['gpu_revision', 'code_revision'].includes(plan?.kind || '') && plan?.run_id === props.run?.id
      && pendingRevision.value!.instruction.includes(plan?.instruction || plan?.dataset || '\u0000')
  })
  const plan = matching?.context?.action_plan
  if (matching && plan) {
    actionMessageId.value = matching.id
    revisionDatasets.value = plan.selected_datasets
    revisionDataset.value = plan.dataset
    isolatedCopy.value = true
  }
}
function toggleBranch() {
  showBranch.value = !showBranch.value
  if (showBranch.value) { actionMessageId.value = ''; revisionDatasets.value = []; revisionDataset.value = '@intent' }
  if (showBranch.value && !branchInstruction.value) {
    branchInstruction.value = [...messages.value].reverse().find(item => item.role === 'user')?.content || ''
  }
}
async function previewBranch() {
  if (!props.run || branchInstruction.value.trim().length < 8) return
  if (pendingRevision.value?.instruction !== branchInstruction.value.trim()) {
    pendingRevision.value = null; revisionPlan.value = null
  }
  actionMessageId.value = ''; revisionDatasets.value = []
  revisionDataset.value = '@intent'
  branchBusy.value = true; branchError.value = ''; branchPreview.value = null
  try { branchPreview.value = await api.previewBranch(props.projectId, props.run.id, branchFile.value, branchInstruction.value.trim(), codingModel.value) }
  catch (err) { branchError.value = (err as Error).message }
  finally { branchBusy.value = false }
}
async function applyRevision() {
  if (!props.run || !branchPreview.value || branchBusy.value) return
  if (!window.confirm(`确认修改 ${branchPreview.value.files.join('、')}？\n\n将备份这些文件的原版本，不启动实验，也不创建新 Conda 环境。`)) return
  branchBusy.value = true; branchError.value = ''
  try {
    const applied = await api.applyRevision(props.projectId, props.run.id, branchPreview.value.proposal_id)
    pendingRevision.value = { proposal_id: applied.proposal_id, files: applied.files,
      instruction: branchInstruction.value, applied_at: new Date().toISOString() }
    branchPreview.value = null
    revisionPlan.value = await api.revisionRunPlan(props.projectId, props.run.id, applied.proposal_id)
    revisionDataset.value = '@intent'
  } catch (err) { branchError.value = (err as Error).message }
  finally { branchBusy.value = false }
}
async function restoreBranchPreview() {
  if (!props.run || branchBusy.value) return
  const id = props.run.id
  branchBusy.value = true; branchError.value = ''
  try {
    const saved = await api.savedBranchPreview(props.projectId, id)
    if (props.run?.id !== id) return
    if (!saved) throw new Error('没有可恢复的已通过预览；源码变化或已应用的提案不能恢复。')
    branchInstruction.value = saved.instruction
    branchFile.value = ''; actionMessageId.value = ''
    await nextTick()
    branchPreview.value = saved
  } catch (err) { branchError.value = (err as Error).message }
  finally { branchBusy.value = false }
}
async function runRevision() {
  if (!props.run || !pendingRevision.value || branchBusy.value) return
  const runId = props.run.id, proposalId = pendingRevision.value.proposal_id
  branchBusy.value = true; branchError.value = ''
  branchStage.value = '正在核验源码和 GPU 环境；尚未启动实验…'
  try {
    const plan = await api.revisionRunPlan(props.projectId, runId, proposalId)
    if (props.run?.id !== runId || pendingRevision.value?.proposal_id !== proposalId) throw new Error('所选实验已变化，请重新检查')
    revisionPlan.value = plan
    if (plan.action === 'already_created' && plan.run_id) { emit('branchStarted', plan.run_id); return }
    if (plan.action === 'blocked' || plan.action === 'prepare_code') { branchStage.value = ''; branchError.value = plan.reason; return }
    const selected = actionMessageId.value || revisionDataset.value === '@intent' ? (plan.selected_datasets || [])
      : revisionDataset.value ? [revisionDataset.value] : (plan.datasets || [])
    if (!selected.length) throw new Error('未确定运行范围；请重新检查提案，不会默认运行全部数据集')
    if (plan.scope_explicit && selected.some(name => !plan.selected_datasets?.includes(name))) throw new Error('选择的范围超出原指令，请重新生成提案')
    if (!window.confirm(`批准运行修改后的实验？\n仅运行：${selected.join('、')}\n${plan.gpu_requested ? '要求 GPU，GPU 运算验证失败时停止，不回退 CPU。\n' : ''}将检查项目环境，不创建新 Conda 环境。${isolatedCopy.value ? '另存源码和数据副本。' : '直接使用项目源码，固定输出文件可能被覆盖。'}指标、日志和报告另存记录。`)) {
      branchStage.value = '已取消运行批准；没有启动实验'; return
    }
    const migration = await approveEnvironmentMigration(plan.gpu_requested, plan.readiness)
    if (migration === null) { branchStage.value = '已取消环境迁移批准；没有安装依赖或启动实验'; return }
    if (props.run?.id !== runId || pendingRevision.value?.proposal_id !== proposalId) throw new Error('所选实验或提案已变化，请重新检查，未提交运行')
    branchStage.value = '正在提交已批准的实验；请勿重复点击…'
    const created = await api.runRevision(props.projectId, runId, proposalId, isolatedCopy.value, selected, migration)
    pendingRevision.value = null; showBranch.value = false; actionMessageId.value = ''
    revisionPlan.value = null; branchStage.value = ''
    emit('branchStarted', created.run_id)
  } catch (err) { branchStage.value = ''; branchError.value = (err as Error).message }
  finally { branchBusy.value = false }
}

async function loadConversations(preserveError = false) {
  const generation = historyGeneration
  if (!preserveError) error.value = ''
  try {
    const loaded = await api.listConversations(props.projectId)
    if (pendingAsk || conversationBusy.value || generation !== historyGeneration) return
    conversations.value = loaded
    if (!conversations.value.some(item => item.id === selectedId.value)) selectedId.value = conversations.value[0]?.id || ''
    await loadMessages()
  } catch (err) { error.value = (err as Error).message }
}
async function loadMessages() {
  if (pendingAsk) return
  if (!selectedId.value) { messages.value = []; return }
  const requestedId = selectedId.value
  try {
    const loaded = await api.conversationMessages(props.projectId, requestedId)
    if (selectedId.value === requestedId && !pendingAsk) { messages.value = loaded; syncPendingAction() }
  }
  catch (err) { error.value = (err as Error).message }
  if (selectedId.value !== requestedId) return
  await nextTick()
  scrollArea.value?.scrollTo({ top: scrollArea.value.scrollHeight })
}
async function newConversation() {
  if (busy.value || conversationBusy.value) return
  conversationBusy.value = true
  historyGeneration += 1
  try {
    const item = await api.createConversation(props.projectId)
    conversations.value.unshift(item)
    selectedId.value = item.id
    messages.value = []
    error.value = ''
  } catch (err) { error.value = (err as Error).message }
  finally { conversationBusy.value = false }
}
async function deleteConversation(item: CopilotConversation) {
  if (busy.value || !window.confirm(`删除“${item.title}”及其全部历史消息？此操作不可恢复。`)) return
  try {
    await api.deleteConversation(props.projectId, item.id)
    conversations.value = conversations.value.filter(row => row.id !== item.id)
    if (selectedId.value === item.id) { selectedId.value = conversations.value[0]?.id || ''; await loadMessages() }
    error.value = ''
  } catch (err) { error.value = (err as Error).message }
}
async function ask(text = question.value) {
  const value = text.trim()
  if (!value || busy.value || pendingAsk || conversationBusy.value) return
  pendingAsk = true
  stoppedByUser = false
  let conversationId = selectedId.value
  question.value = ''
  error.value = ''
  busy.value = true
  streamStatus.value = conversationId ? '正在核对项目证据…' : '正在创建对话…'
  aborter = new AbortController()
  const startedAt = Date.now()
  const timeout = window.setTimeout(() => aborter?.abort('对话等待超过 120 秒，已停止；请重试或切换模型。'), 120000)
  const progressTimer = window.setInterval(() => {
    if (busy.value && !messages.value.at(-1)?.content) {
      const seconds = Math.floor((Date.now() - startedAt) / 1000)
      if (seconds >= 12) streamStatus.value = `仍在等待模型响应（${seconds} 秒）；可以暂停并重试。`
    }
  }, 5000)
  const requestId = crypto.randomUUID()
  const user: CopilotMessage = { id: `pending-user-${requestId}`, role: 'user', content: value, context: { status: 'pending' }, token_usage: 0, created_at: new Date().toISOString(), conversation_id: conversationId }
  const assistant: CopilotMessage = { id: `pending-assistant-${Date.now()}`, role: 'assistant', content: '', context: {}, token_usage: 0, created_at: new Date().toISOString(), conversation_id: conversationId }
  const last = messages.value.at(-1)
  if (last?.role === 'user' && last.content === value) messages.value[messages.value.length - 1] = user
  else messages.value.push(user)
  messages.value.push(assistant)
  const assistantIndex = messages.value.length - 1
  try {
    if (!conversationId) {
      const item = await api.createConversation(props.projectId, aborter.signal)
      conversationId = item.id
      if (!conversations.value.some(row => row.id === item.id)) conversations.value.unshift(item)
      selectedId.value = item.id
      user.conversation_id = item.id; assistant.conversation_id = item.id
      streamStatus.value = '正在核对项目证据…'
    }
    const response = await api.streamCopilot(props.projectId, conversationId, value, chatModel.value, requestId, props.run?.id || '', aborter.signal)
    if (!response.body) throw new Error('浏览器未提供流式响应')
    const reader = response.body.getReader()
    const decoder = new TextDecoder()
    let buffer = ''
    let completed = false
    while (true) {
      const { done, value: chunk } = await reader.read()
      if (done) break
      buffer += decoder.decode(chunk, { stream: true }).replace(/\r\n/g, '\n')
      let boundary = buffer.indexOf('\n\n')
      while (boundary >= 0) {
        const frame = buffer.slice(0, boundary)
        buffer = buffer.slice(boundary + 2)
        const event = frame.split('\n').find(line => line.startsWith('event:'))?.slice(6).trim()
        const raw = frame.split('\n').find(line => line.startsWith('data:'))?.slice(5).trim() || '{}'
        let payload: Record<string, unknown> = {}
        try { payload = JSON.parse(raw) } catch { /* Ignore malformed frames. */ }
        if (event === 'user' && typeof payload.id === 'string') messages.value[assistantIndex - 1] = payload as unknown as CopilotMessage
        if (event === 'status') streamStatus.value = String(payload.message || '正在生成…')
        if (event === 'delta') messages.value[assistantIndex].content += String(payload.text || '')
        if (event === 'replace') messages.value[assistantIndex].content = String(payload.text || '')
        if (event === 'done') { messages.value[assistantIndex] = payload as unknown as CopilotMessage; completed = true }
        if (event === 'error') error.value = String(payload.message || '对话生成失败')
        boundary = buffer.indexOf('\n\n')
      }
      await nextTick()
      scrollArea.value?.scrollTo({ top: scrollArea.value.scrollHeight })
    }
    if (!completed && !error.value) error.value = '流式连接提前结束，已生成的文字可能不完整。'
  } catch (err) { error.value = aborter?.signal.aborted ? String(aborter.signal.reason || '对话已中断') : (err as Error).message }
  finally {
    window.clearTimeout(timeout)
    window.clearInterval(progressTimer)
    if (error.value && !stoppedByUser) question.value = value
    busy.value = false
    aborter = null
    pendingAsk = false
    if (stoppedByUser) await new Promise(resolve => window.setTimeout(resolve, 350))
    await loadConversations(true)
    await nextTick()
    scrollArea.value?.scrollTo({ top: scrollArea.value.scrollHeight })
    const lastAnswer = messages.value.at(-1)
    if (lastAnswer?.context?.action_plan) showBranch.value = false
    if (lastAnswer?.context?.action_plan?.kind === 'code_revision') showBranch.value = true
    // A rerun first presents preparation facts. Only an explicit code request
    // auto-generates a draft; GPU reruns do not become slow coding API calls.
    if (lastAnswer?.role === 'assistant' && lastAnswer.context?.action_plan?.kind === 'code_revision'
        && !error.value && !stoppedByUser && props.run && !['queued', 'running', 'executing'].includes(props.run.status)) await previewActionPlan(lastAnswer)
  }
}
function pauseResponse() {
  if (!busy.value || !aborter) return
  stoppedByUser = true
  streamStatus.value = '已停止生成'
  const last = messages.value.at(-1)
  if (last?.role === 'assistant') last.context = { ...last.context, status: 'partial' }
  aborter.abort('已停止生成。未完成的回答不会作为最终结论；可以重新提问。')
}
function answerScope(item: CopilotMessage) {
  if (item.id.startsWith('pending-')) return '当前回答'
  const recorded = item.context?.run_id || item.context?.action_plan?.run_id
  if (!recorded) return '历史回答（未记录实验编号）'
  return `实验 ${recorded.slice(0, 8)}${recorded !== props.run?.id ? ' · 当前查看的实验已变化' : ' · 已保存回答'}`
}
onMounted(async () => {
  // Remote model discovery and GPU probing must not delay local conversation initialization.
  void api.listModels().then(value => { catalog.value = value }).catch(() => {})
  await Promise.all([loadConversations(), loadBranchFiles(), loadRunReadiness()])
})
watch(() => props.projectId, async () => { aborter?.abort(); selectedId.value = ''; messages.value = []; await loadConversations(); await loadBranchFiles(); await loadRunReadiness() })
watch([() => props.run?.id, () => props.run?.execution_command.pythonpath, () => props.run?.code_spec.source_path], async () => { await loadBranchFiles(); await loadRunReadiness() })
watch([branchFile, branchInstruction], () => { branchPreview.value = null })
watch(selectedId, () => { if (!pendingAsk) loadMessages() })
onUnmounted(() => aborter?.abort())
</script>

<template>
  <aside class="research-chat" aria-label="项目研究协作窗口">
    <header class="chat-head"><div><span class="kicker">PROJECT COPILOT</span><h2>复现协作</h2><small>可基于项目证据提出多文件修改；代码改动和运行分别确认</small></div><button class="icon-button" aria-label="收起对话" @click="$emit('close')">×</button></header>
    <div class="chat-toolbar"><button class="button secondary" :disabled="busy || conversationBusy" @click="newConversation">＋ 新建对话</button><label>对话模型<select v-model="chatModel" aria-label="选择对话模型"><option value="">按理解与报告组调度</option><option v-for="model in catalog?.models || []" :key="model.id" :value="model.id">{{ model.label }}</option></select></label></div>
    <nav class="chat-conversations" aria-label="本项目历史对话"><div v-for="item in conversations" :key="item.id" class="chat-conversation" :class="{ active: item.id === selectedId }"><button :title="item.title" :disabled="busy || conversationBusy" @click="selectedId = item.id">{{ item.title }}</button><button class="conversation-delete" :aria-label="`删除对话 ${item.title}`" @click="deleteConversation(item)">×</button></div><small v-if="!conversations.length">本项目还没有对话</small></nav>
    <div ref="scrollArea" class="copilot-messages" @wheel.stop>
      <div v-if="!messages.length" class="copilot-intro"><strong>基于这个项目的证据提问</strong><p>可追问当前卡点、作者数据、训练指标、调参尝试及结论核验。对话历史仅保存在本项目，并按会话隔离。</p><button v-for="item in suggestions" :key="item" @click="ask(item)">{{ item }}</button></div>
<article v-for="(item, index) in messages" v-show="!(item.role === 'user' && messages[index + 1]?.role === 'user' && messages[index + 1]?.content === item.content)" :key="item.id" class="copilot-message" :class="item.role"><small>{{ item.role === 'user' ? '你' : '科研助手' }}<span v-if="item.role === 'assistant'"> · {{ answerScope(item) }}</span></small><div class="research-markdown" v-html="renderResearchText(item.content || (busy ? streamStatus : ''))"></div><GpuPreparationPlan v-if="item.role === 'assistant' && item.context?.action_plan?.gpu_requested && item.context.action_plan.run_id === run?.id && item.context.action_plan.preparation" :plan="item.context.action_plan" /><div v-if="item.role === 'assistant' && item.context?.action_plan && ['rerun_confirm', 'initial_run_confirm'].includes(item.context.action_plan.kind) && item.context.action_plan.run_id === run?.id" class="chat-inline-action"><strong>待确认实验</strong><p>数据集：{{ item.context.action_plan.dataset }}；设备：{{ item.context.action_plan.gpu_requested ? '仅 GPU' : '沿用当前环境' }}。{{ item.context.action_plan.kind === 'initial_run_confirm' ? '将在当前已准备的记录中首次执行。' : '将另存实验记录，不覆盖原结果。' }}</p><details v-if="rerunTuningPlans.some(plan => plan.trials.length)"><summary>可选：审查有界调参方案</summary><div v-for="plan in rerunTuningPlans.filter(plan => item.context.action_plan?.selected_datasets.includes(plan.dataset))" :key="plan.dataset"><strong>{{ plan.dataset }}</strong><p v-for="parameter in plan.space" :key="parameter.parameter">{{ parameter.parameter }}：{{ parameter.baseline }} → {{ parameter.values.join('、') }}</p><small>{{ plan.selection_rule }}；最多 6 次、额外 30 分钟</small></div><label><input v-model="rerunTuningEnabled" type="checkbox" />同时批准所示搜索；保留原配置基线</label></details><button class="button primary" :disabled="rerunBusy || !!busy || ['queued', 'running', 'executing'].includes(run?.status || '')" @click="confirmRerun(item)">确认并开始实验</button><small v-if="rerunError" class="chat-error">{{ rerunError }}</small></div><div v-if="item.role === 'assistant' && item.context?.action_plan?.kind === 'gpu_revision' && item.context.action_plan.run_id === run?.id" class="chat-inline-action"><strong>下一步：源码兼容修改</strong><p>目标数据集：{{ item.context.action_plan.dataset }}。先审查修改，批准代码后仍需单独批准运行。</p><button v-if="actionMessageId !== item.id || (!branchPreview && !pendingRevision && !branchBusy)" class="button secondary" :disabled="branchBusy || ['queued', 'running', 'executing'].includes(run?.status || '')" @click="previewGpuPlan(item)">生成修改预览</button><small v-if="actionMessageId === item.id && branchBusy">正在检查源码并生成变更…</small><div v-if="actionMessageId === item.id && branchPreview" class="branch-preview"><strong>待审查：{{ branchPreview.files.join('、') }}</strong><p>{{ branchPreview.rationale }}</p><details v-if="branchPreview.coding_trace?.length"><summary>需求规划与编码记录</summary><p v-for="(step, i) in branchPreview.coding_trace" :key="i">{{ step.agent }} · {{ step.model }} · {{ step.status }}<small v-if="step.files?.length">读取：{{ step.files.join('、') }}</small></p></details><div v-for="file in branchPreview.files" :key="file"><strong>{{ file }}</strong><pre>{{ branchPreview.diffs[file] }}</pre></div><button class="button primary" :disabled="branchBusy || ['queued', 'running', 'executing'].includes(run?.status || '')" @click="applyRevision">确认修改代码（不运行）</button><button class="button secondary" @click="branchPreview = null">拒绝修改</button></div><div v-if="actionMessageId === item.id && pendingRevision" class="branch-preview"><strong>代码已修改，尚未运行</strong><p>修改：{{ pendingRevision.files.join('、') }}；计划只运行 {{ revisionDataset }}，实验记录独立保存。</p><RevisionRunStatus :plan="revisionPlan" :busy="branchBusy" :stage="branchStage" :error="branchError" /><button class="button primary" :disabled="branchBusy || ['queued', 'running', 'executing'].includes(run?.status || '')" @click="runRevision">{{ revisionPlan?.action === 'prepare_code' || revisionPlan?.action === 'blocked' ? '重新检查运行条件' : '批准运行实验' }}</button><button class="button secondary" @click="actionMessageId = ''">暂不运行</button></div><small v-if="actionMessageId === item.id && branchError" class="chat-error">{{ branchError }}</small></div><div v-if="item.role === 'assistant' && item.context?.status === 'partial'" class="chat-unanswered"><span>回答已中断，以上不是完整结论。</span><button @click="ask(messages[index - 1]?.content || '')">重新生成</button></div><small v-if="item.role === 'assistant' && item.context?.status === 'degraded'">模型未返回完整内容；这是基于项目状态的降级提示。</small><div v-if="item.role === 'user' && index === messages.length - 1 && !busy" class="chat-unanswered"><span>这条提问尚未得到回复{{ item.context?.error ? `：${item.context.error.slice(0, 100)}` : '。' }}</span><button @click="ask(item.content)">重试</button></div></article>
    </div>
    <div v-if="latestControl && latestControl.id !== completedControlId && latestControl.context?.action_plan?.run_id === run?.id" class="chat-action-strip">
      <span>{{ latestControl.context?.action_plan?.kind === 'pause_confirm' ? '待确认暂停实验' : '待确认恢复实验' }}</span>
      <button class="button secondary" :disabled="controlBusy || busy || (latestControl.context?.action_plan?.kind === 'pause_confirm' ? !['queued', 'running', 'executing'].includes(run?.status || '') : run?.status !== 'paused')" @click="confirmExperimentControl">
        {{ controlBusy ? '正在处理…' : latestControl.context?.action_plan?.kind === 'pause_confirm' ? '确认暂停实验' : '确认恢复实验' }}
      </button>
      <small v-if="controlError" class="chat-error">{{ controlError }}</small>
    </div>
    <div v-if="latestDatasetChoice && latestDatasetChoice.context?.action_plan?.run_id === run?.id" class="chat-action-strip">
      <span>请选择本次要运行的数据集；选择后仍需单独确认启动实验</span>
      <button v-for="dataset in latestDatasetChoice.context.action_plan?.datasets || []" :key="dataset" class="button secondary" :disabled="busy" @click="ask(`请重新运行 ${dataset} 数据集`)">仅 {{ dataset }}</button>
      <button class="button secondary" :disabled="busy" @click="ask('请重新运行全部数据集')">全部数据集</button>
    </div>
    <div v-if="run?.status === 'needs_input' || run?.status === 'awaiting_data' || run?.status === 'awaiting_approval'" class="chat-action-strip"><span>当前任务需要操作</span><button v-if="run.status === 'needs_input' && ['paper_spec_failed', 'code_generation_failed'].includes(run.result_json.action_required?.kind || '')" :disabled="generationBusy" @click="retryPaperSpecification">继续作者源码复现准备</button><button v-if="run.status === 'needs_input'" @click="$emit('uploadCode')">上传作者代码</button><button v-if="run.status === 'awaiting_data'" @click="$emit('uploadData')">上传数据</button><button v-if="run.status === 'awaiting_data'" @click="$emit('continuePartial')">继续可用实验</button><button v-if="run.status === 'awaiting_approval'" @click="$emit('approve')">批准执行</button><small v-if="generationError" class="chat-error">{{ generationError }}</small></div>
    <div v-if="legacyGeneratedFailure" class="chat-action-strip"><span>未开源自动编码已暂缓，旧草稿保留</span><button class="button secondary" @click="$emit('uploadCode')">上传作者代码</button></div>
    <div v-if="run" class="branch-workbench">
      <button v-if="messages.some(m => m.role === 'user')" class="branch-toggle" :aria-expanded="showBranch" @click="toggleBranch">{{ showBranch ? '收起代码工作台' : '＋ 根据对话需求修改实验代码' }}</button>
        <small v-if="!showBranch && branchError && !pendingRevision" class="chat-error">{{ branchError }}</small>
      <div v-if="showBranch" class="branch-form">
        <small>基于实验 {{ run.id.slice(0, 8) }}。Agent 可修改多个相关文件；先审查差异，再决定是否应用。后续运行复用当前项目环境，结果另存一条实验记录。</small>
        <label>优先关注的源码<select v-model="branchFile"><option value="">由 Agent 扫描并选择相关文件</option><option v-for="file in branchFiles" :key="file" :value="file">{{ file }}</option></select></label>
        <label>编码模型<select v-model="codingModel" aria-label="选择本次编码模型"><option value="">按编码与纠错组调度</option><option v-for="model in catalog?.models || []" :key="model.id" :value="model.id">{{ model.label }}</option></select></label><small>对话模型只用于回答；编码模型默认按能力组选择。<RouterLink to="/models">配置能力组模型</RouterLink></small>
        <label>实验需求<textarea v-model="branchInstruction" rows="3" maxlength="1600" placeholder="例如：只调整隐藏维度为 128，保持其余超参数不变，并重新运行全部数据集。"></textarea></label>
        <button class="button secondary" :disabled="branchBusy || !!pendingRevision" @click="restoreBranchPreview">恢复上次通过检查的预览</button>
        <button class="button secondary" :disabled="branchBusy || !branchFiles.length || branchInstruction.trim().length < 8" @click="previewBranch">{{ branchBusy ? '正在处理…' : '生成并检查多文件变更' }}</button>
        <div v-if="branchPreview" class="branch-preview"><strong>变更预览 · {{ branchPreview.files.length }} 个文件</strong><p>{{ branchPreview.rationale }}</p><details v-if="branchPreview.coding_trace?.length"><summary>需求规划与编码记录</summary><p v-for="(step, i) in branchPreview.coding_trace" :key="i">{{ step.agent }} · {{ step.model }} · {{ step.status }}<small v-if="step.files?.length">读取：{{ step.files.join('、') }}</small></p></details><div v-for="file in branchPreview.files" :key="file"><strong>{{ file }}</strong><pre>{{ branchPreview.diffs[file] }}</pre></div><small>预期观察：{{ branchPreview.expected_observation || '请根据实验结果核验' }}</small><button class="button primary" :disabled="branchBusy || ['queued', 'running', 'executing'].includes(run.status) || !!run.result_json.ablation_active" @click="applyRevision">确认修改代码（暂不运行）</button><button class="button secondary" @click="branchPreview = null">拒绝这次修改</button><small v-if="['queued', 'running', 'executing'].includes(run.status)">当前实验仍在运行；请先暂停并等待停止，或待结果生成后再修改。</small></div>
        <div v-if="pendingRevision && !branchPreview" class="branch-preview">
          <strong>已修改，尚未运行</strong><p>修改文件：{{ pendingRevision.files.join('、') }}。原文件已备份；代码批准不等于环境与实验批准。</p>
          <RevisionRunStatus :plan="revisionPlan" :busy="branchBusy" :stage="branchStage" :error="branchError" />
          <label>运行范围<select v-model="revisionDataset"><option value="@intent">按本次需求：{{ revisionPlan?.selected_datasets?.join('、') || '等待核验' }}</option><option v-if="!revisionPlan?.scope_explicit" value="">全部已规划数据集</option><option v-for="item in (revisionPlan?.scope_explicit ? revisionPlan.selected_datasets : runReadiness?.datasets) || []" :key="item" :value="item">仅 {{ item }}</option></select></label>
          <label class="revision-isolation"><input v-model="isolatedCopy" type="checkbox" /> 另存源码与数据副本，避免固定输出路径覆盖旧文件（不新建 Conda 环境）</label>
          <small v-if="!isolatedCopy">固定目录中的模型文件可能被覆盖；旧指标和日志仍保留。</small>
          <button class="button primary" :disabled="branchBusy || !!branchPreview || ['queued', 'running', 'executing'].includes(run.status)" @click="runRevision">{{ branchBusy ? '正在检查或提交…' : revisionPlan?.action === 'prepare_code' || revisionPlan?.action === 'blocked' ? '重新检查运行条件' : '批准运行实验' }}</button>
          <button class="button secondary" :disabled="branchBusy" @click="showBranch = false">暂不运行</button>
        </div>
          <div v-if="branchError && !pendingRevision" class="chat-error">{{ branchError }}</div>
      </div>
    </div>
    <div v-if="error" class="chat-error">{{ error }}</div>
    <form class="copilot-compose" @submit.prevent="ask()"><textarea v-model="question" rows="3" maxlength="1600" placeholder="询问当前复现任务、失败原因或结论证据…" @keydown.enter.exact.prevent="ask()"></textarea><div><small>Enter 发送 · Shift+Enter 换行</small><button v-if="busy" type="button" class="button secondary" aria-label="暂停当前回答" @click="pauseResponse">■ 暂停生成</button><button v-else class="button primary" :disabled="!question.trim() || conversationBusy">发送</button></div></form>
  </aside>
</template>
