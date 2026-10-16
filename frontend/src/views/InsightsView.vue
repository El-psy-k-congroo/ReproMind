<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'
import { api } from '../api'
import type { AnalyticsOverview } from '../types'

const overview = ref<AnalyticsOverview | null>(null)
const error = ref('')
const tokenHover = ref<number | null>(null)
const activityHover = ref<number | null>(null)
const days = computed(() => overview.value?.daily_tokens || [])
const maxTokens = computed(() => Math.max(1, ...days.value.map(item => item.count)))
const tokenLine = computed(() => days.value.map((item, index) => `${index ? 'L' : 'M'}${34 + index * 690 / Math.max(days.value.length - 1, 1)},${192 - item.count / maxTokens.value * 152}`).join(' '))
const roles = computed(() => Object.entries(overview.value?.tokens_by_role || {}))
const roleTotal = computed(() => Math.max(1, roles.value.reduce((sum, [, count]) => sum + count, 0)))
const statuses = computed(() => Object.entries(overview.value?.status_breakdown || {}).filter(([, count]) => count > 0))
const maxRuns = computed(() => Math.max(1, ...(overview.value?.daily_runs.map(item => item.count) || [1])))
const statusNames: Record<string, string> = { queued: '排队', running: '准备中', needs_input: '待选源码', awaiting_data: '待补数据', awaiting_approval: '待批准', executing: '运行中', completed: '已完成', failed: '失败' }
function hoverToken(event: MouseEvent) {
  if (!days.value.length) return
  const rect = (event.currentTarget as SVGSVGElement).getBoundingClientRect()
  const x = (event.clientX - rect.left) / rect.width * 760
  tokenHover.value = Math.max(0, Math.min(days.value.length - 1, Math.round((x - 34) / 690 * (days.value.length - 1))))
}
onMounted(async () => { try { overview.value = await api.overview() } catch (err) { error.value = (err as Error).message } })
</script>

<template>
  <div class="dashboard page-enter insights-page">
    <header class="topbar"><div><span class="kicker">WORKSPACE INSIGHTS</span><h1>工作区观测</h1><p>最近 30 天的模型调用、任务状态与复现活动；仅统计本系统记录的数据。</p></div></header>
    <div v-if="error" class="alert error">{{ error }}</div>
    <template v-if="overview">
      <section class="insight-metrics">
        <article><small>模型 Token</small><strong>{{ overview.tokens.toLocaleString() }}</strong><span>复现与项目对话</span></article>
        <article><small>完整复现项目</small><strong>{{ overview.reproduced_projects }}</strong><span>计划指标均得到支持</span></article>
        <article><small>待用户处理</small><strong>{{ overview.waiting_projects }}</strong><span>源码选择或数据补充</span></article>
        <article><small>复现尝试</small><strong>{{ overview.reproduction_runs }}</strong><span>其中 {{ overview.failed_runs }} 次失败</span></article>
      </section>
      <div class="insight-grid">
        <section class="insight-chart token-trend chart-interactive"><div class="section-title-row compact"><div><span class="kicker">MODEL USAGE</span><h2>每日 Token 消耗</h2></div><small>近 30 天</small></div><svg viewBox="0 0 760 230" role="img" aria-label="每日 Token 消耗曲线" @mousemove="hoverToken" @mouseleave="tokenHover = null"><path class="chart-gridline" d="M34 40 H724 M34 116 H724 M34 192 H724"/><path v-if="tokenLine" class="token-area" :d="`${tokenLine} L724,192 L34,192 Z`"/><path v-if="tokenLine" class="chart-line token-stroke" :d="tokenLine"/><template v-if="tokenHover !== null && days[tokenHover]"><path class="chart-cursor" :d="`M${34 + tokenHover * 690 / Math.max(days.length - 1, 1)} 40 V192`"/><circle class="chart-focus" :cx="34 + tokenHover * 690 / Math.max(days.length - 1, 1)" :cy="192 - days[tokenHover].count / maxTokens * 152" r="5"/></template><text x="34" y="219">{{ days[0]?.date.slice(5) }}</text><text x="677" y="219">{{ days.at(-1)?.date.slice(5) }}</text></svg><div v-if="tokenHover !== null && days[tokenHover]" class="chart-tooltip" :style="{ left: `${Math.max(15, Math.min(80, (34 + tokenHover * 690 / Math.max(days.length - 1, 1)) / 760 * 100))}%` }">{{ days[tokenHover].date }} · {{ days[tokenHover].count.toLocaleString() }} Token</div><div class="chart-foot"><span>峰值 {{ maxTokens.toLocaleString() }} Token / 日</span><span>总量 {{ overview.tokens.toLocaleString() }}</span></div></section>
        <section class="insight-chart usage-breakdown"><div class="section-title-row compact"><div><span class="kicker">MODEL ROUTING</span><h2>调用环节</h2></div></div><div v-for="([role, count], index) in roles" :key="role" class="usage-row"><div><strong>{{ role }}</strong><span>{{ count.toLocaleString() }} Token</span></div><div class="usage-track" :title="`${role} · ${count.toLocaleString()} Token`"><span :class="`usage-color-${index}`" :style="{ width: `${count / roleTotal * 100}%` }"></span></div></div><p>由网关返回的 usage 累加；供应商计费口径可能不同。</p></section>
      </div>
      <div class="insight-grid secondary-grid">
        <section class="insight-chart chart-interactive"><div class="section-title-row compact"><div><span class="kicker">REPRODUCTION ACTIVITY</span><h2>每日任务</h2></div><small>按创建时间</small></div><div class="activity-bars" role="img" aria-label="每日复现任务数"><div v-for="(day, index) in overview.daily_runs" :key="day.date" :title="`${day.date} · ${day.count} 次`" @mouseenter="activityHover = index" @mouseleave="activityHover = null"><span v-if="day.count" :style="{ height: `${Math.max(10, day.count / maxRuns * 100)}%` }"></span></div></div><div v-if="activityHover !== null && overview.daily_runs[activityHover]" class="chart-tooltip" :style="{ left: `${Math.max(15, Math.min(80, activityHover / Math.max(overview.daily_runs.length - 1, 1) * 100))}%` }">{{ overview.daily_runs[activityHover].date }} · {{ overview.daily_runs[activityHover].count }} 次</div><div class="chart-foot"><span>{{ overview.daily_runs[0]?.date }}</span><span>{{ overview.daily_runs.at(-1)?.date }}</span></div></section>
        <section class="insight-chart"><div class="section-title-row compact"><div><span class="kicker">WORKFLOW HEALTH</span><h2>任务状态</h2></div></div><div v-if="statuses.length" class="status-breakdown"><div v-for="([status, count]) in statuses" :key="status"><span>{{ statusNames[status] || status }}</span><strong>{{ count }}</strong></div></div><p v-else>最近 30 天没有复现任务。</p></section>
      </div>
    </template>
    <div v-if="!overview && !error" class="full-loading"><span class="loader"></span><p>正在读取统计…</p></div>
  </div>
</template>
