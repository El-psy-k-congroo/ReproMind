<script setup lang="ts">
import { computed, nextTick, onMounted, onUnmounted, ref, watch } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { api } from '../api'
import UploadPanel from '../components/UploadPanel.vue'
import ReproductionProgress from '../components/ReproductionProgress.vue'
import ExperimentDashboard from '../components/ExperimentDashboard.vue'
import ResearchCopilot from '../components/ResearchCopilot.vue'
import type { ArtifactKind, ModelCatalog, Project, ReproductionRun } from '../types'

const route = useRoute()
const router = useRouter()
const project = ref<Project | null>(null)
const loading = ref(true)
const demoMode = ref(false)
const uploading = ref(false)
const error = ref('')
const showUpload = ref(false)
const uploadKind = ref<ArtifactKind>('paper')
const reproduction = ref<ReproductionRun | null>(null)
const runHistory = ref<Array<{ id: string; status: string; auxiliary_active?: boolean; created_at: string; branch?: { base_run_id: string; title: string; file: string }; revision?: { base_run_id: string; title: string; files: string[] }; rerun?: { base_run_id: string; datasets: string[]; gpu_requested: boolean }; message: string }>>([])
const selectedRunId = ref('')
const startingReproduction = ref(false)
const reproductionGoal = ref('复现论文核心实验')
const authorRepository = ref('')
const approvingReproduction = ref(false)
const deletingProject = ref(false)
const modelCatalog = ref<ModelCatalog | null>(null)
const selectedModel = ref('deepseek-v4-flash-0731')
const showCopilot = ref(true)
const showModelMenu = ref(false)
const workbench = ref<HTMLElement | null>(null)
const chatWidth = ref(Number(window.localStorage.getItem('repromind-chat-width')) || 430)
let reproductionTimer: number | undefined

const kindMeta: Record<ArtifactKind, { label: string; symbol: string }> = {
  paper: { label: '论文', symbol: 'P' },
  code: { label: '代码', symbol: '</>' },
  config: { label: '配置', symbol: '⚙' },
  result: { label: '结果', symbol: 'Σ' },
  note: { label: '说明', symbol: 'N' },
  dataset: { label: '数据集', symbol: 'D' },
}

async function load() {
  loading.value = true
  error.value = ''
  try {
    project.value = await api.getProject(route.params.id as string)
  } catch (err) {
    error.value = (err as Error).message
  } finally {
    loading.value = false
  }
}

async function upload(kind: ArtifactKind, file: File) {
  if (!project.value) return
  uploading.value = true
  error.value = ''
  try {
    await api.uploadArtifact(project.value.id, kind, file)
    showUpload.value = false
    await load()
    if (((kind === 'dataset' && ['awaiting_data', 'failed'].includes(reproduction.value?.status || ''))
      || (kind === 'code' && reproduction.value?.status === 'needs_input')) && reproduction.value) {
      reproduction.value = await api.resumeReproduction(project.value.id, reproduction.value.id)
      startReproductionPolling()
    }
  } catch (err) {
    error.value = (err as Error).message
  } finally {
    uploading.value = false
  }
}

async function remove(artifactId: string) {
  if (!project.value || !window.confirm('确认移除该证据文件？')) return
  try {
    await api.deleteArtifact(project.value.id, artifactId)
    await load()
  } catch (err) {
    error.value = (err as Error).message
  }
}

async function pauseReproduction() {
  if (!project.value || !reproduction.value) return
  try { reproduction.value = await api.pauseReproduction(project.value.id, reproduction.value.id); startReproductionPolling() }
  catch (err) { error.value = (err as Error).message }
}

async function resumePaused() {
  if (!project.value || !reproduction.value) return
  try {
    reproduction.value = await api.resumePaused(project.value.id, reproduction.value.id)
    if (['queued', 'running', 'executing'].includes(reproduction.value.status)) startReproductionPolling()
  } catch (err) { error.value = (err as Error).message }
}

function startResize(event: PointerEvent) {
  ;(event.currentTarget as HTMLElement).setPointerCapture(event.pointerId)
  document.body.classList.add('resizing-workbench')
}
function resize(event: PointerEvent) {
  if (!(event.currentTarget as HTMLElement).hasPointerCapture(event.pointerId) || !workbench.value) return
  const bounds = workbench.value.getBoundingClientRect()
  chatWidth.value = Math.round(Math.max(320, Math.min(bounds.right - event.clientX, Math.min(760, bounds.width - 470))))
}
function stopResize(event: PointerEvent) {
  const handle = event.currentTarget as HTMLElement
  if (handle.hasPointerCapture(event.pointerId)) handle.releasePointerCapture(event.pointerId)
  document.body.classList.remove('resizing-workbench')
  window.localStorage.setItem('repromind-chat-width', String(chatWidth.value))
}

function stopReproductionPolling() {
  if (reproductionTimer !== undefined) window.clearInterval(reproductionTimer)
  reproductionTimer = undefined
}

let refreshingRun = false
let nextPollAt = 0
let pollFailures = 0
async function refreshReproduction() {
  if (!project.value || refreshingRun || Date.now() < nextPollAt) return
  const projectId = project.value.id
  refreshingRun = true
  try {
    const history = await api.listReproductionRuns(projectId)
    if (project.value?.id !== projectId) return
    runHistory.value = history
    if (!runHistory.value.some(item => item.id === selectedRunId.value)) selectedRunId.value = runHistory.value[0]?.id || ''
    const runId = selectedRunId.value
    const run = runId ? await api.getReproduction(projectId, runId) : null
    if (project.value?.id !== projectId || selectedRunId.value !== runId) return
    reproduction.value = run
    pollFailures = 0
    if (error.value.startsWith('状态查询暂时失败')) error.value = ''
    if (!runHistory.value.some(item => ['queued', 'running', 'executing'].includes(item.status) || item.auxiliary_active)) {
      stopReproductionPolling()
    }
  } catch (err) {
    if (project.value?.id !== projectId) return
    error.value = `状态查询暂时失败：${(err as Error).message}。这不是数据下载或实验失败，正在自动重试。`
    nextPollAt = Date.now() + Math.min(15000, 1500 * 2 ** Math.min(++pollFailures, 4))
  } finally { refreshingRun = false }
}

async function selectRun(id: string) {
  selectedRunId.value = id
  if (!project.value) return
  try { reproduction.value = await api.getReproduction(project.value.id, id) }
  catch (err) { error.value = (err as Error).message }
}

async function branchStarted(runId: string) {
  selectedRunId.value = runId
  await refreshReproduction()
  startReproductionPolling()
}

function copilotRunUpdated(run: ReproductionRun) {
  reproduction.value = run
  if (['queued', 'running', 'executing'].includes(run.status)) startReproductionPolling()
  else stopReproductionPolling()
}

function startReproductionPolling() {
  stopReproductionPolling()
  nextPollAt = 0
  reproductionTimer = window.setInterval(refreshReproduction, 1500)
}

async function startReproduction() {
  if (!project.value) return
  startingReproduction.value = true
  error.value = ''
  try {
    reproduction.value = await api.startReproduction(project.value.id, selectedModel.value, 'full', reproductionGoal.value.trim() || '复现论文核心实验', authorRepository.value.trim())
    selectedRunId.value = reproduction.value.id
    startReproductionPolling()
  } catch (err) {
    error.value = (err as Error).message
  } finally {
    startingReproduction.value = false
  }
}

async function resumeReproduction() {
  if (!project.value || !reproduction.value) return
  try {
    reproduction.value = await api.resumeReproduction(project.value.id, reproduction.value.id)
    startReproductionPolling()
  } catch (err) { error.value = (err as Error).message }
}

async function chooseRepository(repositoryUrl: string) {
  if (!project.value || !reproduction.value) return
  try { reproduction.value = await api.chooseRepositorySource(project.value.id, reproduction.value.id, repositoryUrl); startReproductionPolling() }
  catch (err) { error.value = (err as Error).message }
}

async function continuePartial() {
  if (!project.value || !reproduction.value || !window.confirm('确认跳过当前缺失的数据集，仅对可用数据执行实验？报告会明确标记未验证项。')) return
  try { reproduction.value = await api.continuePartial(project.value.id, reproduction.value.id); startReproductionPolling() }
  catch (err) { error.value = (err as Error).message }
}

async function deleteCurrentProject() {
  if (!project.value || deletingProject.value) return
  const name = project.value.name
  if (!window.confirm(`永久删除“${name}”？论文附件、源码、数据、报告及该项目创建的 Conda 环境都会删除。`)) return
  deletingProject.value = true
  try {
    await api.deleteProject(project.value.id)
    await router.push('/')
  } catch (err) { error.value = (err as Error).message }
  finally { deletingProject.value = false }
}

async function scrollToSection() {
  const section = String(route.query.section || '')
  await nextTick()
  if (section === 'evidence' || section === 'experiment') {
    document.getElementById(section)?.scrollIntoView({ behavior: 'smooth', block: 'start' })
  }
}

function openDatasetUpload() {
  uploadKind.value = 'dataset'
  showUpload.value = true
  document.getElementById('evidence')?.scrollIntoView({ behavior: 'smooth' })
}

function openCodeUpload() {
  uploadKind.value = 'code'
  showUpload.value = true
  document.getElementById('evidence')?.scrollIntoView({ behavior: 'smooth' })
}

async function approveReproduction(tuningBudget = 0) {
  if (!project.value || !reproduction.value) return
  const display = String(reproduction.value.execution_command.display || '待执行命令')
  if (!window.confirm(`确认检查项目 Conda 环境、准备作者数据并执行全部作者配置基线：\n\n${display}${tuningBudget ? `\n\n保留备用调参计划（最多 ${tuningBudget} 次）；本轮不启动搜索，基线未复现后需复核原因并另行批准。` : '\n\n不运行额外调参。'}`)) return
  approvingReproduction.value = true
  error.value = ''
  try {
    reproduction.value = await api.approveReproduction(project.value.id, reproduction.value.id, tuningBudget)
    startReproductionPolling()
  } catch (err) {
    error.value = (err as Error).message
  } finally {
    approvingReproduction.value = false
  }
}

function formatBytes(value: number) {
  if (value < 1024) return `${value} B`
  if (value < 1024 * 1024) return `${(value / 1024).toFixed(1)} KB`
  return `${(value / 1024 / 1024).toFixed(1)} MB`
}

onMounted(async () => {
  void api.deployment().then(info => { demoMode.value = info.mode === 'demo' }).catch(() => {})
  const initialModel = selectedModel.value
  void (async () => { try {
    modelCatalog.value = await api.listModels()
    if (selectedModel.value === initialModel) selectedModel.value = modelCatalog.value.default
  } catch { /* Project evidence remains available while the model catalog is slow. */ } })()
  await load()
  await refreshReproduction()
  if (runHistory.value.some(item => ['queued', 'running', 'executing'].includes(item.status) || item.auxiliary_active)) {
    startReproductionPolling()
  }
  await scrollToSection()
})
watch(() => route.params.id, async (id, oldId) => {
  if (id === oldId) return
  stopReproductionPolling()
  reproduction.value = null
  selectedRunId.value = ''
  await load()
  await refreshReproduction()
  if (runHistory.value.some(item => ['queued', 'running', 'executing'].includes(item.status) || item.auxiliary_active)) startReproductionPolling()
  await scrollToSection()
})
watch(() => route.query.section, scrollToSection)
onUnmounted(stopReproductionPolling)
</script>

<template>
  <div v-if="loading" class="full-loading"><span class="loader"></span><p>正在构建证据视图…</p></div>
  <div v-else-if="project" ref="workbench" class="project-workbench" :class="{ 'chat-open': showCopilot }" :style="{ '--chat-width': `${chatWidth}px` }">
  <div class="project-view page-enter workbench-main">
    <header class="project-header">
      <div>
        <RouterLink class="breadcrumb" to="/">项目总览</RouterLink><span class="breadcrumb-sep">/</span><span>{{ project.research_domain }}</span>
        <h1>{{ project.name }}</h1>
        <p>{{ project.description }}</p>
      </div>
      <div class="project-actions">
        <input v-model="reproductionGoal" aria-label="本次复现目标" maxlength="500" placeholder="例如：仅运行 Cora 的核心实验" />
        <input v-model="authorRepository" aria-label="本次作者源码地址" maxlength="500" placeholder="作者 GitHub / Zenodo 地址（可选）" />
        <div class="model-menu-wrap"><button class="model-menu-trigger" :aria-expanded="showModelMenu" @click="showModelMenu = !showModelMenu"><span>模型</span><strong>{{ selectedModel }}</strong><b>⌄</b></button><div v-if="showModelMenu" class="model-menu"><small>新复现任务使用此模型</small><button v-for="model in modelCatalog?.models || [{ id: selectedModel, label: selectedModel }]" :key="model.id" :class="{ selected: selectedModel === model.id }" @click="selectedModel = model.id; showModelMenu = false">{{ model.label }}<span v-if="selectedModel === model.id">✓</span></button></div></div>
        <button class="button secondary" @click="uploadKind = 'paper'; showUpload = !showUpload">＋ 添加证据</button>
        <button class="button secondary" @click="showCopilot = true">复现协作</button>
        <button v-if="reproduction && (['queued', 'running', 'executing'].includes(reproduction.status) || (!!reproduction.result_json.ablation_active || !!reproduction.result_json.tuning_active) && reproduction.status === 'completed')" class="button secondary" @click="pauseReproduction">暂停任务</button>
        <button v-if="reproduction?.status === 'paused'" class="button secondary" @click="resumePaused">恢复任务</button>
        <button class="button ghost danger-text" :disabled="deletingProject" @click="deleteCurrentProject">{{ deletingProject ? '正在删除项目与环境…' : '删除项目' }}</button>
        <button class="button primary" :title="demoMode ? '请在本地部署后运行完整复现流程' : ''" :disabled="demoMode || startingReproduction || !project.artifacts.some(item => item.kind === 'paper') || runHistory.some(item => ['queued', 'running', 'executing'].includes(item.status))" @click="startReproduction">
          <span v-if="startingReproduction" class="loader small"></span>{{ startingReproduction ? '正在创建…' : '启动自主复现' }}
        </button>
      </div>
    </header>

    <div v-if="error" class="alert error">{{ error }}</div>
    <UploadPanel v-if="showUpload" :busy="uploading" :initial-kind="uploadKind" @upload="upload" />

    <section id="evidence" class="evidence-section">
      <div class="section-title-row compact">
        <div><span class="kicker">EVIDENCE SOURCES</span><h2>证据知识库</h2></div>
        <span class="section-note">{{ project.artifacts.length }} 个文件已索引</span>
      </div>
      <div class="artifact-grid">
        <article v-for="artifact in project.artifacts" :key="artifact.id" class="artifact-card">
          <div class="artifact-icon" :class="`artifact-${artifact.kind}`">{{ kindMeta[artifact.kind].symbol }}</div>
          <div class="artifact-copy"><span>{{ kindMeta[artifact.kind].label }}</span><strong>{{ artifact.filename }}</strong><small>{{ formatBytes(artifact.size_bytes) }}</small></div>
          <button class="remove-button" title="移除文件" @click="remove(artifact.id)">×</button>
        </article>
        <button class="artifact-card artifact-add" @click="showUpload = true"><span>＋</span><strong>补充证据</strong></button>
      </div>
    </section>

    <div v-if="runHistory.length" class="run-history-strip"><label>实验记录 <select :value="selectedRunId" @change="selectRun(($event.target as HTMLSelectElement).value)"><option v-for="(item, index) in runHistory" :key="item.id" :value="item.id">{{ item.rerun ? `重跑 ${item.rerun.datasets.join('、')}${item.rerun.gpu_requested ? ' · GPU' : ''}` : item.revision ? `代码修改实验 · ${item.revision.title}` : item.branch ? `分支 · ${item.branch.title}` : `原实验 ${runHistory.length - index}` }} · {{ item.status }} · {{ item.id.slice(0, 8) }}</option></select></label><span v-if="reproduction?.result_json.rerun">当前查看独立重跑记录：{{ reproduction.result_json.rerun.datasets.join('、') }}；{{ reproduction.result_json.rerun.gpu_requested ? '要求 GPU，CUDA 不可用即停止' : '按现有环境运行' }}。基于 {{ reproduction.result_json.rerun.base_run_id.slice(0, 8) }}，可切回原实验。以下进度、图表及结果仅属于当前所选记录。</span><span v-else-if="reproduction?.result_json.revision">当前查看代码修改后的独立实验记录；复用同一 Conda 环境，可从列表切回原结果。以下进度和图表仅属于所选记录。</span><span v-else-if="reproduction?.result_json.branch">当前查看独立分支；基于 {{ String(reproduction.result_json.branch.base_run_id).slice(0, 8) }}。原实验可从列表切回。</span></div>
    <ReproductionProgress v-if="reproduction" :run="reproduction" :approving="approvingReproduction" @approve="approveReproduction" @resume="resumeReproduction" @upload-data="openDatasetUpload" @upload-code="openCodeUpload" @choose-repository="chooseRepository" @continue-partial="continuePartial" />
    <button v-if="reproduction?.result_json.rerun && runHistory.some(item => item.id === reproduction?.result_json.rerun?.base_run_id)" class="button secondary" @click="selectRun(reproduction.result_json.rerun.base_run_id)">查看原实验完整结果（不重跑）</button>
    <ExperimentDashboard v-if="reproduction && (reproduction.approved_at || ['executing', 'completed'].includes(reproduction.status))" :run="reproduction" @ablation-started="startReproductionPolling" @recovery-started="branchStarted" />

  </div>
    <div v-if="showCopilot" class="workbench-resizer" role="separator" aria-orientation="vertical" aria-label="调整功能区与对话区宽度" @pointerdown="startResize" @pointermove="resize" @pointerup="stopResize" @pointercancel="stopResize"></div>
    <ResearchCopilot v-if="showCopilot" :project-id="project.id" :model-id="selectedModel" :run="reproduction" @close="showCopilot = false" @upload-code="openCodeUpload" @upload-data="openDatasetUpload" @continue-partial="continuePartial" @approve="approveReproduction" @branch-started="branchStarted" @run-updated="copilotRunUpdated" />
  </div>
  <div v-else class="full-loading"><p>{{ error || '项目不存在' }}</p><RouterLink class="button secondary" to="/">返回项目总览</RouterLink></div>
</template>
