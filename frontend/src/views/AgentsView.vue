<script setup lang="ts">
import { computed, onMounted, onUnmounted, ref } from 'vue'
import { useRoute } from 'vue-router'
import { api } from '../api'
import type { ProjectListItem, ReproductionRun } from '../types'

const route = useRoute()
const projects = ref<ProjectListItem[]>([])
const projectId = ref(String(route.query.project || ''))
const run = ref<ReproductionRun | null>(null)
const error = ref('')
const selected = ref('paper')
const moving = ref(true)
let generation = 0
let timer: number | undefined
// Logical roles correspond to the engine/API workflows, not independently running model processes.
const agents = [
  { id: 'paper', name: 'Paper Agent', title: '论文理解', mark: '01', color: '#82caba', input: '论文与已索引证据', output: '数据集、指标、目标值与实验规范', rule: '缺少证据时标注未知，不编造目标指标。', module: 'reproduction_engine · paper_parsing', x: 145, y: 90 },
  { id: 'source', name: 'Source Agent', title: '作者源码定位', mark: '02', color: '#82caba', input: '论文链接、上传源码', output: '作者仓库与可追溯源码快照', rule: '作者仓库和上传源码优先；未开源自动实现暂缓。', module: 'reproduction_engine · repo_discovery', x: 365, y: 90 },
  { id: 'code', name: 'Code Agent', title: '代码检查与迁移', mark: '03', color: '#82caba', input: '源码、模块接口与修改要求', output: '入口、依赖及多文件变更预览', rule: '统一迁移后集中校验；批准保存代码不等于批准实验。', module: 'project_coding · Code Planning Agent → Code Agent', x: 585, y: 90 },
  { id: 'environment', name: 'Environment Agent', title: '环境准备', mark: '04', color: '#82caba', input: '框架、Python ABI、已有依赖', output: '兼容方案、安装日志与运行时探测', rule: '复用项目环境需检查；涉及安装与实验须获批准。', module: 'environment_recovery · runtime_evidence', x: 805, y: 90 },
  { id: 'data', name: 'Data Agent', title: '数据来源核对', mark: '05', color: '#8ca5df', input: '作者说明、数据路径与划分要求', output: '本地文件检查、来源和版本差异', rule: '数据文件存在不代表与论文版本及划分完全一致。', module: 'dataset_discovery · official_datasets', x: 805, y: 280 },
  { id: 'execution', name: 'Execution Agent', title: '作者配置基线', mark: '06', color: '#8ca5df', input: '已批准命令、源码、环境和数据', output: '基线指标、训练日志与退出状态', rule: '先完成作者配置基线；请求GPU时不可静默回退CPU。', module: 'reproduction_engine · execute_reproduction', x: 585, y: 280 },
  { id: 'repair', name: 'Repair Agent', title: '异常诊断与纠错', mark: '07', color: '#dbac78', input: '真实异常与相关源码接口', output: '隔离副本中的修复与重试记录', rule: '仅在失败后介入；不修改其他实验正在使用的源码。', module: 'repair_isolation · _agent_repair_code', x: 365, y: 280 },
  { id: 'tuning', name: 'Tuning Agent', title: '有界恢复搜索', mark: '08', color: '#dbac78', input: '未复现基线、可调参数与审批预算', output: '搜索空间、验证选参及测试报告', rule: '基线完成后单独审批；只按验证集选参，保留基线。', module: 'tuning · tuning_sessions', x: 145, y: 280 },
  { id: 'result', name: 'Result Agent', title: '结果与主张核验', mark: '09', color: '#b7a2da', input: '论文目标、实际指标与执行证据', output: '偏差与已检验/未检验结论', rule: '未检验不等于错误；调参结果不等于作者原配置复现。', module: 'reproduction_engine · result_verification', x: 145, y: 470 },
  { id: 'report', name: 'Report Agent', title: '可追溯交付', mark: '10', color: '#b7a2da', input: '实验记录、修复及搜索证据', output: '复现报告与结果展示', rule: '保留历史记录，不把离线测试写成论文复现成功。', module: 'reproduction_engine · _write_reproduction_report', x: 365, y: 470 },
  { id: 'copilot', name: 'Project Copilot', title: '项目证据协作', mark: '11', color: '#b7a2da', input: '当前项目证据与用户问题', output: '只读回答、变更提案与确认操作', rule: '状态查询走本地证据；实验和代码改动分别确认。', module: 'api/copilot · ResearchCopilot', x: 805, y: 470 },
]
const edges = [
  ['paper', 'source', 'main'], ['source', 'code', 'main'], ['code', 'environment', 'main'],
  ['environment', 'data', 'main'], ['data', 'execution', 'main'],
  ['execution', 'repair', 'conditional'], ['repair', 'execution', 'return'],
  ['execution', 'result', 'main'], ['result', 'tuning', 'conditional'], ['tuning', 'result', 'return'],
  ['result', 'report', 'main'], ['copilot', 'code', 'conditional'],
]
const detail = computed(() => agents.find(a => a.id === selected.value)!)
const currentAgent = computed(() => {
  if (run.value?.result_json.tuning_active) return 'tuning'
  if (!run.value || !['queued', 'running', 'executing'].includes(run.value.status)) return ''
  const stage = run.value.current_stage
  return ({ paper_parsing: 'paper', repo_discovery: 'source', code_inspection: 'code', environment_planning: 'environment', environment_building: 'environment', data_planning: 'data', data_downloading: 'data', executing: 'execution', result_verification: 'result' } as Record<string, string>)[stage] || ''
})
function edgePath(edge: string[]) {
  const from = agents.find(a => a.id === edge[0])!, to = agents.find(a => a.id === edge[1])!
  const direction = to.x > from.x ? 1 : -1
  if (from.y === to.y) {
    const x1 = from.x + direction * 88, x2 = to.x - direction * 88
    if (edge[2] === 'return') return `M ${x1} ${from.y + 22} Q ${(x1 + x2) / 2} ${from.y + 92} ${x2} ${to.y + 22}`
    return `M ${x1} ${from.y} L ${x2} ${to.y}`
  }
  const y1 = from.y + (to.y > from.y ? 46 : -46), y2 = to.y + (to.y > from.y ? -46 : 46)
  return `M ${from.x} ${y1} C ${from.x} ${(y1 + y2) / 2} ${to.x} ${(y1 + y2) / 2} ${to.x} ${y2}`
}
async function refresh() {
  const request = ++generation, id = projectId.value
  if (!id) { run.value = null; error.value = ''; return }
  try {
    const value = await api.getLatestReproduction(id)
    if (request === generation) { run.value = value; error.value = '' }
  } catch (err) { if (request === generation) { run.value = null; error.value = (err as Error).message } }
}
onMounted(async () => {
  try { projects.value = await api.listProjects() } catch (err) { error.value = (err as Error).message }
  await refresh()
  timer = window.setInterval(refresh, 12000)
})
onUnmounted(() => { generation++; window.clearInterval(timer) })
</script>

<template>
  <section class="agents-page">
    <header class="agents-heading"><div><span class="eyebrow">MULTI-AGENT SYSTEM</span><h1>Agent 协作网络</h1><p>从论文证据到可追溯实验，观察角色之间的交接与反馈。</p></div><div class="agents-controls"><label>项目状态<select v-model="projectId" aria-label="选择可视化项目" @change="refresh"><option value="">仅查看系统架构</option><option v-for="p in projects" :key="p.id" :value="p.id">{{ p.name }}</option></select></label><button class="button secondary" @click="moving = !moving">{{ moving ? '暂停动效' : '播放动效' }}</button></div></header>
    <p v-if="error" role="alert">{{ error }}</p>
    <div class="agents-orchestrator"><div><span class="agent-indicator"></span><strong>流程编排与权限门控</strong><small>项目隔离 · 源码快照 · 审批 · 证据记录</small></div><span v-if="run">任务 {{ run.id.slice(0, 8) }} · {{ run.status }} · {{ run.current_stage }}{{ run.result_json.tuning_active ? ' · 额外调参中' : '' }}</span><span v-else>架构视图 · 未读取活动实验</span></div>
    <div class="agents-layout">
      <div class="agents-network" :class="{ paused: !moving }">
        <div class="network-top"><span>REPRODUCTION GRAPH</span><small>悬停或选择节点查看职责</small></div>
        <div class="network-scroll"><div class="network-canvas">
          <svg viewBox="0 0 950 555" class="network-links" aria-hidden="true"><defs><marker id="agent-arrow" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto"><path d="M0 0 L6 3 L0 6" fill="currentColor" /></marker></defs><path v-for="(edge, index) in edges" :key="index" :d="edgePath(edge)" :class="['agent-edge', edge[2], { related: edge.includes(selected) }]" marker-end="url(#agent-arrow)" /></svg>
          <button v-for="a in agents" :key="a.id" class="agent-node" :class="{ chosen: a.id === selected, current: a.id === currentAgent }" :style="{ left: `${a.x / 950 * 100}%`, top: `${a.y / 555 * 100}%`, '--agent-color': a.color }" :aria-label="`${a.name}：${a.title}`" :aria-pressed="a.id === selected" @mouseenter="selected = a.id" @focus="selected = a.id" @click="selected = a.id"><span class="agent-node-top"><span>{{ a.mark }}</span><span class="node-dot"></span></span><strong>{{ a.title }}</strong><small>{{ a.name }}</small><span v-if="a.id === currentAgent" class="node-state">当前任务阶段</span></button>
        </div></div>
        <footer class="network-legend"><span><i></i>顺序交接</span><span><i class="conditional"></i>条件触发 / 用户确认</span><small>连线动效是流程演示，不代表实时 Agent 调用。</small></footer>
      </div>
      <aside class="agent-detail" aria-live="polite"><span class="eyebrow">ROLE INSPECTOR</span><span class="detail-mark" :style="{ color: detail.color }">{{ detail.mark }}</span><h2>{{ detail.title }}</h2><p>{{ detail.name }}</p><dl><dt>接收</dt><dd>{{ detail.input }}</dd><dt>产出</dt><dd>{{ detail.output }}</dd><dt>执行边界</dt><dd>{{ detail.rule }}</dd></dl><div class="detail-module"><small>实现映射</small><code>{{ detail.module }}</code></div><RouterLink v-if="projectId" class="button secondary" :to="`/projects/${projectId}`">查看项目证据与日志 →</RouterLink></aside>
    </div>
    <div class="agents-notes"><article><strong>先基线，后恢复搜索</strong><p>作者参数执行结束后，核对偏差和数据版本，再审批调参空间与预算。</p></article><article><strong>修改与运行分离</strong><p>编码角色提交可审查的变更；环境安装、实验执行仍需独立确认。</p></article><article><strong>角色不等于并行进程</strong><p>这是系统逻辑角色图。模型调用与规则检查共同实现工作流，不声称所有 Agent 同时运行。</p></article></div>
  </section>
</template>

<style scoped>
.agents-page { max-width: 1600px; margin: 0 auto; padding: 36px; }.agents-heading { display:flex; justify-content:space-between; gap:24px; align-items:center; margin-bottom:24px; }.agents-heading h1 { margin:10px 0; font-size:29px; }.agents-heading p { color:#74877f; font-size:13px; }.agents-controls { display:flex; align-items:end; gap:12px; }.agents-controls label { display:grid; gap:7px; color:#74877f; font-size:11px; }.agents-controls select { min-width:175px; padding:10px; border:1px solid #d2ded6; border-radius:6px; background:white; }.agents-orchestrator { display:flex; justify-content:space-between; gap:14px; align-items:center; padding:20px 24px; background:#e8f0ec; border:1px solid #d3e2d9; border-radius:10px; margin-bottom:16px; font-size:12px; color:#4a675c; }.agents-orchestrator > div { display:flex; align-items:center; gap:13px; }.agents-orchestrator small { color:#81948b; }.agent-indicator { width:9px; height:9px; border-radius:50%; background:#328d72; box-shadow:0 0 0 5px #328d7214; }.agents-layout { display:grid; grid-template-columns:minmax(0,1fr) 280px; gap:16px; }.agents-network { overflow:hidden; border:1px solid #233b40; border-radius:12px; background:radial-gradient(ellipse at 50% 30%,#15343a 0,#0d212a 65%); box-shadow:0 14px 30px #16353014; }.network-top { display:flex; justify-content:space-between; padding:20px 24px; color:#90b0ad; font:11px monospace; letter-spacing:1px; }.network-top small { letter-spacing:0; color:#6c8c90; }.network-scroll { overflow-x:auto; }.network-canvas { position:relative; aspect-ratio:950/555; min-width:690px; background-image:radial-gradient(#88aba81c 1px,transparent 1px); background-size:22px 22px; }.network-links { width:100%; height:100%; position:absolute; }.agent-edge { fill:none; stroke:#496d70; stroke-width:1.6; color:#496d70; }.agent-edge.related { stroke:#7ec4b4; color:#7ec4b4; stroke-width:2; }.agent-edge.main { stroke-dasharray:7 6; animation:agent-flow 2s linear infinite; }.agent-edge.conditional,.agent-edge.return { stroke-dasharray:3 6; color:#a4977d; stroke:#a4977d; }.agent-node { width:18.5%; position:absolute; transform:translate(-50%,-50%); padding:12px 14px; min-height:88px; text-align:left; border:1px solid #395457; border-radius:9px; color:#d9e9e5; background:linear-gradient(140deg,#1b343c,#132a32); cursor:pointer; box-shadow:0 5px 18px #0002; transition:border-color .2s,box-shadow .2s,background .2s; }.agent-node.chosen,.agent-node:focus-visible { border-color:var(--agent-color); background:linear-gradient(140deg,#23464b,#1b343b); box-shadow:0 0 0 3px #88bea915,0 8px 18px #0003; outline:none; }.agent-node-top { display:flex; justify-content:space-between; margin-bottom:7px; font:10px monospace; color:var(--agent-color); }.node-dot { width:5px; height:5px; border-radius:50%; background:var(--agent-color); }.agent-node strong { display:block; font-size:12px; font-weight:600; }.agent-node small { display:block; font:9px monospace; color:#86a3a6; margin-top:5px; }.agent-node.current { box-shadow:0 0 0 3px #8fd2ba33; }.node-state { font-size:8px; color:#9acfb9; }.network-legend { display:flex; flex-wrap:wrap; gap:14px; align-items:center; padding:17px 24px; border-top:1px solid #254147; color:#84a09e; font-size:10px; }.network-legend span { display:flex; align-items:center; gap:6px; }.network-legend i { display:block; width:20px; border-top:2px solid #75ad9e; }.network-legend i.conditional { border-top-style:dashed; border-color:#a4977d; }.network-legend small { margin-left:auto; color:#72908f; }.agent-detail { padding:25px; border:1px solid #d8e4dc; border-radius:12px; background:#fff; display:flex; flex-direction:column; }.detail-mark { margin:26px 0 10px; font:42px monospace; opacity:.9; }.agent-detail h2 { font-size:19px; margin:0 0 8px; }.agent-detail > p { font:11px monospace; color:#7d948b; }.agent-detail dl { margin:26px 0 14px; }.agent-detail dt { font-size:10px; color:#85988e; margin-bottom:7px; }.agent-detail dd { margin:0 0 22px; font-size:12px; color:#36554a; line-height:1.8; }.detail-module { margin-top:auto; padding:13px; background:#f1f6f3; border-radius:7px; display:grid; gap:8px; }.detail-module small { color:#80978a; }.detail-module code { font-size:10px; line-height:1.6; overflow-wrap:anywhere; color:#4d6e60; }.agent-detail a { margin-top:20px; text-align:center; font-size:11px; }.agents-notes { display:grid; grid-template-columns:repeat(3,1fr); gap:16px; margin-top:24px; }.agents-notes article { border-left:2px solid #b7d3c5; padding:4px 16px; }.agents-notes strong { font-size:12px; }.agents-notes p { font-size:11px; color:#7d9085; line-height:1.8; }.paused .agent-edge { animation:none; }@keyframes agent-flow { to { stroke-dashoffset:-26; } }@media(prefers-reduced-motion:reduce) { .agent-edge { animation:none!important; }.agent-node { transition:none; } }@media(max-width:1150px) { .agents-layout { grid-template-columns:1fr; }.agent-detail { min-height:280px; }.agents-heading { align-items:start; flex-direction:column; }.agents-orchestrator,.agents-orchestrator > div { flex-wrap:wrap; } }@media(max-width:700px) { .agents-page { padding:16px; }.agents-notes { grid-template-columns:1fr; }.agents-controls { flex-wrap:wrap; } }
.network-canvas { min-width:600px; }
@media (min-width:1151px) and (max-width:1400px) { .agent-node { padding:10px; }.agent-node strong { font-size:11px; }.agent-node small { font-size:8px; } }
</style>
