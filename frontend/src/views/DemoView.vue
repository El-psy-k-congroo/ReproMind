<script setup lang="ts">
import { computed, ref, watch } from 'vue'
import { useRoute } from 'vue-router'
import { api } from '../api'
import { demoLink, demoStages } from '../demoStages'
import ExperimentDashboard from '../components/ExperimentDashboard.vue'
import type { Project, ReproductionRun } from '../types'

const route = useRoute()
const project = ref<Project | null>(null)
const run = ref<ReproductionRun | null>(null)
const preview = ref<Awaited<ReturnType<typeof api.savedBranchPreview>>>(null)
const error = ref('')
const previewError = ref('')
const loading = ref(false)
const active = computed(() => demoStages.find(item => item.id === route.query.stage))
const completed = computed(() => run.value?.status === 'completed' && !run.value.result_json.tuning_active && !run.value.result_json.ablation_active)
const pretty = (value: unknown) => JSON.stringify(value, null, 2)
const materialNames: Record<string, string> = { paper: '论文', code: '源码', config: '配置', result: '结果', dataset: '数据', note: '说明' }
let generation = 0

async function load() {
  const token = ++generation
  loading.value = true; error.value = ''; previewError.value = ''; run.value = null; preview.value = null
  try {
    const id = String(route.params.id)
    const runId = String(route.query.run || '')
    if (!runId) throw new Error('请从项目的“五分钟演示”入口选择一个真实实验记录。')
    const [p, r] = await Promise.all([api.getProject(id), api.getReproduction(id, runId)])
    if (token !== generation) return
    project.value = p; run.value = r
    if (route.query.stage === 'coding') {
      try { const saved = await api.savedBranchPreview(id, runId, true); if (token === generation) preview.value = saved }
      catch (err) { if (token === generation) previewError.value = (err as Error).message }
    }
  } catch (err) { if (token === generation) error.value = (err as Error).message }
  finally { if (token === generation) loading.value = false }
}

function openStages() {
  if (!run.value || !project.value) return
  let blocked = 0
  for (const item of demoStages) {
    const tab = window.open(demoLink(project.value.id, run.value.id, item.id), '_blank')
    if (tab) tab.opener = null
    else blocked++
  }
  if (blocked) error.value = '浏览器阻止了部分标签页，请使用下面的“新标签页打开”分别打开。'
}
watch(() => [route.params.id, route.query.run, route.query.stage], load, { immediate: true })
</script>

<template>
  <main class="demo-view page-enter">
    <p v-if="loading" role="status">正在读取实验记录…</p>
    <p v-if="error" class="alert error" role="alert">{{ error }}</p>
    <template v-if="project && run">
      <header class="demo-header">
        <div><RouterLink :to="`/projects/${project.id}?run=${run.id}`">返回实验工作台</RouterLink><h1>{{ project.name }} · 五分钟演示</h1><p>展示已保存的真实实验材料。打开页面不会创建环境、下载数据或启动训练。</p></div>
        <button v-if="!active" class="button primary" @click="openStages">打开五个阶段页面</button>
      </header>
      <div class="demo-record"><strong>展示记录 {{ run.id.slice(0, 8) }}</strong><span>状态：{{ run.status }}</span><span>保存时间：{{ new Date(run.updated_at).toLocaleString('zh-CN') }}</span></div>
      <p v-if="!completed" class="alert warning">此记录尚未全部结束。当前页面只展示已经保存的内容，演示前请优先选择已完成记录。</p>
      <nav class="demo-nav" aria-label="演示阶段"><RouterLink v-for="item in demoStages" :key="item.id" :to="demoLink(project.id, run.id, item.id)" :aria-current="active?.id === item.id ? 'page' : undefined">{{ item.title }}</RouterLink></nav>
      <section v-if="!active" class="demo-outline">
        <h2>演示前分别打开，现场按顺序切换</h2>
        <p>环境创建、数据下载和训练提前完成。五个页面固定在当前实验记录，不随其他标签页的实验选择变化。</p>
        <ol><li v-for="item in demoStages" :key="item.id"><RouterLink :to="demoLink(project.id, run.id, item.id)">{{ item.title }}</RouterLink><span>建议 {{ item.seconds }} 秒</span><a :href="demoLink(project.id, run.id, item.id)" target="_blank" rel="noopener">新标签页打开</a></li></ol>
        <p>代码修改页显示保存的修改差异。需要现场新操作时，返回实验工作台按正常流程确认。</p>
      </section>
      <section v-else-if="active.id === 'evidence'" class="demo-content">
        <h2>论文与实验目标</h2><h3>本次目标</h3><p>{{ run.goal }}</p>
        <h3>项目材料</h3><ul><li v-for="artifact in project.artifacts" :key="artifact.id">{{ materialNames[artifact.kind] }}：{{ artifact.filename }}</li></ul>
        <h3>提取的实验规范</h3><pre>{{ pretty({ model: run.paper_spec.model_name, datasets: run.paper_spec.datasets, metrics: run.paper_spec.metrics, hyperparameters: run.paper_spec.hyperparameters, targets: run.paper_spec.target_results }) }}</pre>
      </section>
      <section v-else-if="active.id === 'preparation'" class="demo-content">
        <h2>源码、环境与数据准备记录</h2><h3>作者源码</h3><a v-if="run.repository_url" :href="run.repository_url" target="_blank" rel="noopener">{{ run.repository_url }}</a>
        <h3>实验环境</h3><p>{{ run.environment_spec.environment_name }}，Python {{ run.environment_spec.python }}</p><pre>{{ pretty(run.environment_spec.display_commands) }}</pre>
        <h3>数据准备</h3><pre>{{ pretty(run.dataset_spec) }}</pre>
        <details><summary>查看准备阶段的真实时间记录</summary><p v-for="(event, index) in run.events.filter(item => !['executing', 'completed', 'result_verification'].includes(item.stage))" :key="index">{{ new Date(event.time).toLocaleString('zh-CN') }}　{{ event.message }}</p></details>
      </section>
      <section v-else-if="active.id === 'execution'" class="demo-content">
        <h2>执行命令与训练日志</h2><pre>{{ run.execution_command.display }}</pre><p>以下为保存日志的末尾，便于直接查看训练与测试结果。</p><pre class="demo-log">{{ run.logs.slice(-14000) || '该记录尚未产生运行日志。' }}</pre><details><summary>完整运行日志</summary><pre>{{ run.logs }}</pre></details>
      </section>
      <section v-else-if="active.id === 'coding'" class="demo-content">
        <h2>代码修改与校验</h2><p v-if="previewError" role="alert">{{ previewError }}</p>
        <template v-if="preview"><h3>修改要求</h3><p>{{ preview.instruction }}</p><details v-for="(diff, file) in preview.diffs" :key="file" open><summary>{{ file }}</summary><pre>{{ diff }}</pre></details><p v-if="preview.run_id"><RouterLink class="button secondary" :to="`/projects/${project.id}?run=${preview.run_id}`" target="_blank">查看修改后的独立实验</RouterLink></p><h3>Agent 协作记录</h3><ul class="demo-agents"><li v-for="(item, index) in preview.coding_trace || []" :key="index"><strong>{{ item.agent }}</strong><span>{{ item.status === 'completed' ? '已通过' : item.status }}</span><small>{{ item.model }}</small></li></ul><details><summary>修改依据与验收条件</summary><p>{{ preview.rationale }}</p><p>{{ preview.expected_observation }}</p></details><details><summary>完整协作记录</summary><pre>{{ pretty(preview.coding_trace) }}</pre></details></template>
        <p v-else-if="!previewError">此记录没有保存代码修改方案。请先在工作台完成一次真实修改，再准备本阶段页面。</p>
        <details><summary>已保存的修改记录</summary><pre>{{ pretty(run.result_json.code_revisions || run.result_json.revision || run.result_json.branch || []) }}</pre></details>
      </section>
      <section v-else class="demo-content"><h2>实验结果、曲线与报告</h2><a v-if="run.result_json.report_path" class="button secondary" :href="api.reproductionReportUrl(project.id, run.id)" download>下载复现报告</a><ExperimentDashboard :key="run.id" :run="run" read-only /></section>
    </template>
  </main>
</template>

<style scoped>
.demo-view{max-width:1280px;margin:auto;padding:32px}.demo-header{display:flex;justify-content:space-between;gap:24px;align-items:center}.demo-header h1{margin:16px 0}.demo-header p,.demo-record{color:var(--text-muted,#607d8b)}.demo-record{display:flex;gap:24px;padding:14px 0;flex-wrap:wrap}.demo-nav{display:flex;gap:12px;border-block:1px solid #dce5eb;padding:16px 0;margin:12px 0 24px;flex-wrap:wrap}.demo-nav a{padding:8px 12px}.demo-nav a[aria-current=page]{background:#e4f1fb;color:#0053a3;font-weight:700;border-radius:6px}.demo-content,.demo-outline{background:var(--surface,#fff);padding:28px;border-radius:12px}.demo-outline li{display:flex;gap:28px;padding:20px 0;border-bottom:1px solid #e5ebef}.demo-outline li>a:first-child{flex:1;font-weight:600}.demo-content h3{margin-top:26px}.demo-content pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f4f7fa;padding:20px;border-radius:8px;line-height:1.65;max-height:60vh;overflow:auto;font-size:13px}.demo-content p{line-height:1.8}.demo-log{font-family:Consolas,monospace}.demo-content details{margin:16px 0}.demo-content :deep(.experiment-dashboard.read-only button){display:none}@media(max-width:760px){.demo-view{padding:16px}.demo-header{display:block}.demo-outline li{gap:12px;flex-wrap:wrap}.demo-content{padding:16px}}
.demo-agents{list-style:none;padding:0}.demo-agents li{display:flex;align-items:baseline;gap:20px;padding:10px 0;border-bottom:1px solid #e5ebef;flex-wrap:wrap}.demo-agents small{color:#607d8b}
</style>
