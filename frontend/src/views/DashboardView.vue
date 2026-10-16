<script setup lang="ts">
import { onMounted, ref } from 'vue'
import { useRouter } from 'vue-router'
import { api } from '../api'
import type { AnalyticsOverview, ProjectListItem } from '../types'

const router = useRouter()
const projects = ref<ProjectListItem[]>([])
const overview = ref<AnalyticsOverview | null>(null)
const loading = ref(true)
const actionBusy = ref(false)
const error = ref('')
const showCreate = ref(false)
const openMenu = ref<string | null>(null)
const deletingId = ref<string | null>(null)
const form = ref({ name: '', description: '', research_domain: '计算机科学' })

async function load() {
  loading.value = true
  try {
    projects.value = await api.listProjects()
    try { overview.value = await api.overview() } catch { /* Project list remains available. */ }
  } catch (err) {
    error.value = (err as Error).message
  } finally {
    loading.value = false
  }
}

async function deleteProject(project: ProjectListItem) {
  openMenu.value = null
  if (!window.confirm(`永久删除“${project.name}”？将清除该项目的数据库记录、附件、工作区和其独立 Conda 环境。`)) return
  deletingId.value = project.id
  try { await api.deleteProject(project.id); projects.value = projects.value.filter(item => item.id !== project.id); window.dispatchEvent(new Event('repromind:projects-changed')); overview.value = await api.overview() }
  catch (err) { error.value = (err as Error).message }
  finally { deletingId.value = null }
}

function runLabel(status: string | null) {
  return ({ queued: '排队中', running: '准备中', needs_input: '等待选择源码', awaiting_data: '等待数据', awaiting_approval: '待批准执行', executing: '执行中', completed: '已完成', failed: '执行失败' } as Record<string, string>)[status || ''] || '尚未启动'
}

async function createProject() {
  if (!form.value.name.trim()) return
  actionBusy.value = true
  try {
    const project = await api.createProject(form.value)
    await router.push(`/projects/${project.id}`)
  } catch (err) {
    error.value = (err as Error).message
  } finally {
    actionBusy.value = false
  }
}

function formatDate(value: string) {
  return new Intl.DateTimeFormat('zh-CN', { month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit' }).format(new Date(value))
}

onMounted(load)
</script>

<template>
  <div class="dashboard page-enter">
    <header class="topbar">
      <div><span class="kicker">RESEARCH OPERATIONS</span><h1>科研复现控制台</h1></div>
      <div class="top-actions">
        <button class="button primary" @click="showCreate = true">新建复现项目</button>
      </div>
    </header>

    <div v-if="error" class="alert error">{{ error }}</div>

    <section class="hero-panel">
      <div class="hero-copy">
        <span class="hero-label">EVIDENCE-DRIVEN RESEARCH</span>
        <h2>从论文到可执行实验<br /><em>逐步复现与核验</em></h2>
        <p>获取作者代码与数据，构建独立环境，运行实验并记录修复、调参和结论核验；遇到无法自动解决的问题会明确请你选择下一步。</p>
        <div class="hero-actions">
          <button class="button primary light" @click="showCreate = true">开始一个复现任务</button>
        </div>
      </div>
      <div class="evidence-map" aria-label="证据工作流示意">
        <div class="map-orbit orbit-one"></div><div class="map-orbit orbit-two"></div>
        <div class="map-center"><span>RM</span><small>交叉核验</small></div>
        <div class="map-node node-paper"><b>01</b><span>论文</span></div>
        <div class="map-node node-code"><b>02</b><span>代码</span></div>
        <div class="map-node node-config"><b>03</b><span>配置</span></div>
        <div class="map-node node-result"><b>04</b><span>结果</span></div>
      </div>
    </section>

    <section class="metric-strip">
      <div><strong>{{ projects.length }}</strong><span>复现项目</span></div>
      <div><strong>{{ projects.reduce((sum, item) => sum + item.artifact_count, 0) }}</strong><span>证据文件</span></div>
      <div><strong>{{ overview?.reproduction_runs ?? '—' }}</strong><span>近 30 天实验记录</span></div>
      <div><strong>{{ overview?.reproduced_projects ?? '—' }}</strong><span>近 30 天完整复现的项目</span></div>
    </section>

    <section class="projects-section">
      <div class="section-title-row">
        <div><span class="kicker">ACTIVE WORK</span><h2>最近项目</h2></div>
        <span class="section-note">按最近更新排序</span>
      </div>
      <div v-if="loading" class="empty-state"><span class="loader"></span><p>正在读取项目…</p></div>
      <div v-else-if="projects.length" class="project-grid">
        <article v-for="project in projects" :key="project.id" class="project-card" :class="{ 'card-deleting': deletingId === project.id }">
          <div class="project-card-top">
            <span class="domain-pill">{{ project.research_domain }}</span>
            <div class="card-menu-wrap"><button class="card-menu-button" aria-label="项目操作" :aria-expanded="openMenu === project.id" @click="openMenu = openMenu === project.id ? null : project.id">•••</button><div v-if="openMenu === project.id" class="card-menu"><button @click="router.push(`/projects/${project.id}`)">打开项目</button><button class="danger-text" @click="deleteProject(project)">删除项目与环境</button></div></div>
          </div>
          <button class="project-card-link" @click="router.push(`/projects/${project.id}`)"><h3>{{ project.name }}</h3><p>{{ project.description || '尚未填写项目说明。' }}</p></button>
          <div class="project-progress">
            <div class="progress-label"><span>复现进度</span><strong>{{ project.latest_reproduction_progress ?? 0 }}%</strong></div>
            <div class="progress-track"><span :style="{ width: `${project.latest_reproduction_progress ?? 0}%` }"></span></div>
          </div>
          <div class="project-card-foot">
            <span>{{ project.artifact_count }} 个证据 · {{ formatDate(project.updated_at) }}</span>
            <span class="project-run-status" :class="`status-${project.latest_reproduction_status}`">{{ runLabel(project.latest_reproduction_status) }}</span>
          </div>
        </article>
      </div>
      <button v-else class="empty-project" @click="showCreate = true">
        <span>＋</span><strong>创建第一个复现项目</strong><small>上传论文后，Agent 将建立独立工作区并规划复现实验</small>
      </button>
    </section>

    <div v-if="showCreate" class="modal-backdrop" @click.self="showCreate = false">
      <form class="modal" @submit.prevent="createProject">
        <div class="modal-head"><div><span class="kicker">NEW PROJECT</span><h2>新建复现项目</h2></div><button type="button" class="icon-button" @click="showCreate = false">×</button></div>
        <label>项目名称<input v-model="form.name" autofocus placeholder="输入论文或模型名称" minlength="2" required /></label>
        <label>研究领域<input v-model="form.research_domain" placeholder="计算机科学" /></label>
        <label>项目说明<textarea v-model="form.description" rows="4" placeholder="说明复现目标、论文版本和当前进度"></textarea></label>
        <div class="modal-actions"><button type="button" class="button ghost" @click="showCreate = false">取消</button><button class="button primary" :disabled="actionBusy">创建项目</button></div>
      </form>
    </div>
  </div>
</template>
