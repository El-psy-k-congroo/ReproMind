<script setup lang="ts">
import { onMounted, onUnmounted, ref, watch } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { api } from './api'
import type { ProjectListItem } from './types'
import { session, logout, startup } from './session'
function retryStartup() { window.location.reload() }

const route = useRoute()
const router = useRouter()
const projects = ref<ProjectListItem[]>([])
const openProjectMenu = ref('')
const sidebarError = ref('')
async function refreshProjects() {
  if (!session.loaded || session.required && !session.user) { projects.value=[]; return }
  try { projects.value = await api.listProjects() } catch { /* Preserve the last successful project list. */ }
}
onMounted(() => { refreshProjects(); window.addEventListener('repromind:projects-changed', refreshProjects) })
onUnmounted(() => window.removeEventListener('repromind:projects-changed', refreshProjects))
watch(() => route.fullPath, refreshProjects)
watch(() => [session.loaded, session.user?.id], () => { projects.value=[]; void refreshProjects() })
async function signOut() { try { await logout() } catch (err) { sidebarError.value=(err as Error).message } }
async function removeSidebarProject(item: ProjectListItem) {
  openProjectMenu.value = ''
  if (!window.confirm(`删除“${item.name}”及其本地项目文件、独立 Conda 环境和对话历史？此操作不可恢复。`)) return
  try {
    await api.deleteProject(item.id)
    if (route.params.id === item.id) await router.push('/')
    await refreshProjects()
    sidebarError.value = ''
  } catch (err) { sidebarError.value = (err as Error).message }
}
const projectTarget = (section: string) => {
  const id = String(route.params.id || route.query.project || projects.value[0]?.id || '')
  return id ? `/projects/${id}?section=${section}` : '/'
}
</script>

<template>
  <div v-if="startup.error" class="full-loading"><strong>ReproMind</strong><p>{{ startup.error }}</p><button class="button primary" @click="retryStartup">重新连接</button></div>
  <div v-else-if="!session.loaded" class="full-loading"><span class="loader"></span><p>正在连接研究工作台…</p></div>
  <RouterView v-else-if="route.meta.public" />
  <div v-else class="app-shell">
    <aside class="sidebar">
      <RouterLink class="brand" to="/">
        <span class="brand-mark">R</span>
        <span>
          <strong>ReproMind</strong>
          <small>科研复现工作台</small>
        </span>
      </RouterLink>

      <nav class="nav-list" aria-label="主导航">
        <RouterLink class="nav-item" :class="{ active: route.name === 'dashboard' }" to="/">
          <span class="nav-symbol">⌂</span> 项目总览
        </RouterLink>
        <RouterLink class="nav-item" :class="{ active: route.query.section === 'evidence' }" :to="projectTarget('evidence')">
          <span class="nav-symbol">◇</span> 证据知识库
        </RouterLink>
        <RouterLink class="nav-item" :class="{ active: route.query.section === 'experiment' }" :to="projectTarget('experiment')">
          <span class="nav-symbol">⌁</span> 实验过程
        </RouterLink>
        <RouterLink class="nav-item" :class="{ active: route.name === 'insights' }" to="/insights">
          <span class="nav-symbol">▥</span> 使用统计
        </RouterLink>
        <RouterLink class="nav-item" :class="{ active: route.name === 'agents' }" :to="route.params.id || route.query.project ? `/agents?project=${route.params.id || route.query.project}` : '/agents'"><span class="nav-symbol">◈</span> Agent 协作网络</RouterLink>
        <RouterLink class="nav-item" :class="{ active: route.name === 'models' }" to="/models"><span class="nav-symbol">⇄</span> 多模型调度</RouterLink>
      </nav>

      <div class="sidebar-projects">
        <div class="sidebar-section-title"><span>最近项目</span><RouterLink to="/">查看全部</RouterLink></div>
        <div v-for="item in projects.slice(0, 8)" :key="item.id" class="sidebar-project-row" :class="{ selected: route.params.id === item.id }">
          <RouterLink class="sidebar-project" :to="`/projects/${item.id}`" :title="item.name"><span>▣</span><span>{{ item.name }}</span></RouterLink>
          <button class="sidebar-project-more" :aria-label="`管理项目 ${item.name}`" @click="openProjectMenu = openProjectMenu === item.id ? '' : item.id">···</button>
          <div v-if="openProjectMenu === item.id" class="sidebar-project-menu"><button @click="removeSidebarProject(item)">删除项目与独立环境</button></div>
        </div>
        <small v-if="sidebarError" class="sidebar-error">{{ sidebarError }}</small>
        <small v-if="!projects.length" class="sidebar-empty">还没有项目</small>
      </div>

      <div class="sidebar-context">
        <p class="eyebrow">SYSTEM STATUS</p>
        <div class="status-line"><span class="status-dot"></span> 复现工作台</div>
        <p>作者源码优先；代码、环境与实验操作分别确认。</p>
      </div>

      <div class="sidebar-foot">
        <div class="avatar">{{ session.user?.username.slice(0,1).toUpperCase() || 'R' }}</div>
        <RouterLink v-if="session.required" to="/account" class="sidebar-account"><strong>{{ session.user?.username || '我的账号' }}</strong><small>个人工作空间</small></RouterLink>
        <div v-else class="sidebar-account"><strong>本地研究空间</strong><small>完整复现工作流</small></div>
        <button v-if="session.required" class="logout-button" aria-label="退出登录" title="退出登录" @click="signOut">↪</button>
      </div>
    </aside>
    <main class="main-content">
      <RouterView />
    </main>
  </div>
</template>
