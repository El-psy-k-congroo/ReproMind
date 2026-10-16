<script setup lang="ts">
import { computed, onMounted, onUnmounted, ref, watch } from 'vue'
import { api } from '../api'
import { selectedTuningMetrics } from '../tuningResults'
import { runDatasetScope } from '../runScope'
import type { DatasetResult, ReproductionActivity, ReproductionRun, TelemetryPoint, TelemetrySeries, TuningPlan } from '../types'

const props = defineProps<{ run: ReproductionRun }>()
const emit = defineEmits<{ ablationStarted: []; recoveryStarted: [runId: string] }>()
const recoveryPlans = ref<TuningPlan[]>([])
const recoveryOpen = ref(false)
const recoveryBusy = ref(false)
const recoveryError = ref('')
const recoveryReviewed = ref(false)
const approvedPlans = computed(() => recoveryPlans.value.filter(plan => plan.ready))
const parameterText = (value: number) => Number(value.toPrecision(6)).toString()
const recoveryDatasets = computed(() => results.value.filter(row => row.return_code === 0 && row.verdict === 'deviated' && !row.tuning_selected).map(row => row.dataset))
const canReviewRecovery = computed(() => props.run.status === 'completed' && !props.run.result_json.tuning_active && recoveryDatasets.value.length > 0)
watch(() => props.run.id, () => { recoveryOpen.value = false; recoveryPlans.value = []; recoveryReviewed.value = false; recoveryError.value = '' })
async function reviewRecovery() {
  if (recoveryBusy.value || !canReviewRecovery.value) return
  recoveryBusy.value = true; recoveryError.value = ''; recoveryReviewed.value = false
  const id = props.run.id
  try {
    const response = await api.tuningPlan(props.run.project_id, id)
    if (id !== props.run.id) return
    recoveryPlans.value = response.plans.filter(plan => recoveryDatasets.value.includes(plan.dataset) && plan.trials.length)
    recoveryOpen.value = true
    if (!recoveryPlans.value.length) recoveryError.value = '当前源码没有可审查的参数空间；未启动训练或调参。'
  } catch (err) { recoveryError.value = (err as Error).message }
  finally { recoveryBusy.value = false }
}
async function startRecovery() {
  if (recoveryBusy.value || !canReviewRecovery.value || !recoveryReviewed.value || !approvedPlans.value.length) return
  const datasets = approvedPlans.value.map(plan => plan.dataset)
  if (!window.confirm(`确认批准恢复搜索：${datasets.join('、')}？\n搜索记录保存在当前实验的“自动调参与修复”，不重跑基线、不新建实验、不覆盖其他结果。\n每个数据集最多 6 组、额外 30 分钟；只用验证集选参，不按测试集选参。\n复用当前环境，不安装依赖。`)) return
  recoveryBusy.value = true; recoveryError.value = ''
  try {
    const created = await api.startTuning(props.run.project_id, props.run.id, approvedPlans.value)
    emit('recoveryStarted', created.run_id)
  } catch (err) { recoveryError.value = (err as Error).message }
  finally { recoveryBusy.value = false }
}
const series = ref<TelemetrySeries[]>([])
const ablationOptions = ref<Array<{ dataset: string; flag: string; label: string }>>([])
const ablationReason = ref('')
const ablationActivity = ref<ReproductionActivity | null>(null)
const selectedAblation = ref('')
const startingAblation = ref(false)
const selectedDataset = ref('')
const selectedAttempt = ref('')
const loadError = ref('')
const paperTargetOverride = ref<Record<string, Record<string, number>> | null>(null)
const paperTargetReason = ref('')
const loadingPaperTargets = ref(false)
const curveHover = ref<{ key: string; index: number } | null>(null)
const barHover = ref<{ dataset: string; metric: string; series: 'paper' | 'actual' | 'tuned' } | null>(null)
let timer: number | undefined

const datasets = computed(() => [...new Set(series.value.map(item => item.dataset))])
const attempts = computed(() => series.value.filter(item => item.dataset === selectedDataset.value))
const current = computed(() => attempts.value.find(item => item.attempt === selectedAttempt.value && item.points.length) || attempts.value.find(item => item.points.length))
const trainingSamples = computed(() => current.value?.points.length || 0)
const results = computed(() => props.run.result_json.dataset_results || [])
const approvedDatasets = computed(() => runDatasetScope(props.run))
const comparisonRows = computed<DatasetResult[]>(() => {
  const names = [...new Set([...approvedDatasets.value, ...results.value.map(item => item.dataset)])]
  return names.map(name => results.value.find(item => item.dataset.toLowerCase() === name.toLowerCase()) || {
    dataset: name, return_code: -1, metrics: {}, target: null, delta: null, verdict: 'not_run', diagnosis: {},
  })
})
const elapsed = computed(() => results.value.reduce((sum, item) => sum + (item.wall_time_seconds || 0), 0))
const repairs = computed(() => props.run.result_json.repair_attempts?.length || 0)
const tunings = computed(() => props.run.result_json.tuning_attempts || [])
const metricNames: Record<string, string> = { acc: 'Accuracy', accuracy: 'Accuracy', auc: 'AUC', auroc: 'AUC', aupr: 'AUPR', f1: 'F1', loss: 'Loss', rmse: 'RMSE', mae: 'MAE', mse: 'MSE', prd: 'PRD', cr: 'CR', fid: 'FID', perplexity: 'Perplexity', top1error: 'Top-1 Error', top5error: 'Top-5 Error', bleu: 'BLEU', rouge: 'ROUGE', precision: 'Precision', recall: 'Recall' }
const preferredMetrics = computed(() => (props.run.paper_spec.metrics as string[] | undefined || []).map(item => item.toLowerCase()))
const metricKeys = computed(() => {
  const available = [...new Set((current.value?.points || []).flatMap(point => Object.keys(point).filter(key => key !== 'epoch' && typeof point[key] === 'number')))]
  const canonical = (key: string) => key.replace(/^(train|val|test)_/, '').replace('accuracy', 'acc')
  const paperMatches = (key: string) => preferredMetrics.value.some(label => {
    const target = canonical(key)
    return label.includes(target) || target === 'acc' && label.includes('accuracy') || target === 'aupr' && label.includes('precision-recall')
  })
  const score = (key: string) => (paperMatches(key) ? 100 : 0)
    + (key.startsWith('val_') ? 25 : key.startsWith('test_') ? 20 : key.startsWith('train_') ? 5 : 10)
    + (canonical(key) === 'loss' ? 8 : 0)
  const ranked = available.sort((a, b) => score(b) - score(a))
  const chosen: string[] = []
  for (const key of ranked) {
    if (!chosen.some(existing => canonical(existing) === canonical(key))) chosen.push(key)
    if (chosen.length === 3) break
  }
  return chosen
})
function metricTitle(key: string): string { return `${key.startsWith('val_') ? '验证集' : key.startsWith('test_') ? '测试集' : key.startsWith('train_') ? '训练集' : ''}${metricNames[key.replace(/^(train|val|test)_/, '')] || key}` }
function bounds(key: string) {
  const values = (current.value?.points || []).map(point => point[key]).filter((value): value is number => value !== undefined)
  const min = Math.min(...values)
  const max = Math.max(...values)
  const padding = Math.max((max - min) * .08, .005)
  return { min: min - padding, max: max + padding, values }
}
function line(key: string): string {
  const points = (current.value?.points || []).filter(point => point[key] !== undefined)
  if (points.length < 2) return ''
  const { min, max } = bounds(key)
  return points.map((point, index) => `${index ? 'L' : 'M'}${48 + index * 388 / (points.length - 1)},${130 - (point[key]! - min) * 100 / (max - min)}`).join(' ')
}
function number(value: number | undefined, key: string): string {
  if (value === undefined || !Number.isFinite(value)) return '—'
  const isRatio = /(?:acc|auc|aupr|f1|precision|recall|rouge)/.test(key) && value >= 0 && value <= 1.2
  return isRatio ? `${(value * 100).toFixed(1)}%` : value.toFixed(3)
}
function latest(point: TelemetryPoint[] | undefined, key: string): number | undefined { return [...(point || [])].reverse().find(item => item[key] !== undefined)?.[key] }
function canonicalMetric(key: string): string {
  const normalized = key.toLowerCase().replace(/[^a-z0-9]/g, '')
  return ({ acc: 'accuracy', auroc: 'auc', aucroc: 'auc', rocauc: 'auc', aucpr: 'aupr', prauc: 'aupr', f1score: 'f1' } as Record<string, string>)[normalized] || normalized
}
function paperTargets(item: DatasetResult): Record<string, number> {
  const source = paperTargetOverride.value && Object.keys(paperTargetOverride.value).length
    ? paperTargetOverride.value : props.run.paper_spec.target_results
  if (!source || typeof source !== 'object') return {}
  const rows = Array.isArray(source) ? source : Object.entries(source as Record<string, unknown>).map(([name, value]) => ({ dataset: name, metrics: value }))
  const found = rows.find(row => {
    if (!row || typeof row !== 'object') return false
    const record = row as Record<string, unknown>
    return String(record.dataset || record.name || '').toLowerCase() === item.dataset.toLowerCase()
  }) as Record<string, unknown> | undefined
  const raw = found?.metrics && typeof found.metrics === 'object' ? found.metrics : found
  const output: Record<string, number> = {}
  const scalarKey = item.metric_key || (preferredMetrics.value.length === 1 ? canonicalMetric(preferredMetrics.value[0]!) : '')
  if (typeof found?.metrics === 'number' && scalarKey && metricNames[scalarKey]) output[canonicalMetric(scalarKey)] = found.metrics > 1 && !/rmse|mae|mse|loss/.test(scalarKey) ? found.metrics / 100 : found.metrics
  if (raw && typeof raw === 'object') for (const [key, value] of Object.entries(raw)) {
    if (key === 'metrics' || key === 'dataset' || key === 'name') continue
    if (typeof value === 'number' && Number.isFinite(value)) output[canonicalMetric(key)] = value > 1 && /accuracy|auc|aupr|f1|precision|recall/.test(key.toLowerCase()) ? value / 100 : value
  }
  return output
}
function comparisonMetrics(item: DatasetResult) {
  const targets = paperTargets(item)
  if (item.metric_key && item.target !== null) targets[canonicalMetric(item.metric_key)] = item.target
  const observed = Object.fromEntries(Object.entries(item.baseline_metrics || item.metrics).filter(([, value]) => typeof value === 'number' && Number.isFinite(value)).map(([key, value]) => [canonicalMetric(key), value]))
  const tuned = Object.fromEntries(Object.entries(selectedTuningMetrics(props.run.result_json, item.dataset)).map(([key, value]) => [canonicalMetric(key), value]))
  if (item.tuning_selected && !Object.keys(tuned).length) for (const [key, value] of Object.entries(item.metrics)) tuned[canonicalMetric(key)] = value
  const declared = (props.run.paper_spec.metrics as string[] | undefined || []).flatMap(label =>
    Object.keys(metricNames).filter(key => key !== 'acc' && new RegExp(`\\b${key}\\b`, 'i').test(label)).map(canonicalMetric))
  const paperKeys = Object.keys(targets)
  return [...new Set(paperKeys.length ? paperKeys : declared)].map(key => ({ key, paper: targets[key], actual: observed[key], tuned: tuned[key], maximum: chartMax(targets[key], Math.max(observed[key] || 0, tuned[key] || 0), key) }))
}
function chartMax(paper: number | undefined, actual: number | undefined, key: string): number {
  const observed = Math.max(0, paper || 0, actual || 0)
  return /^(accuracy|auc|aupr|f1|precision|recall)$/.test(key) && observed <= 1 ? 1 : Math.max(0.01, observed) * 1.15
}
function barHeight(value: number | undefined, maximum: number): number { return value === undefined ? 0 : Math.max(0, Math.min(122, value / maximum * 122)) }
const datasetPalette = ['#258e77', '#6979bd', '#bd8050', '#9d6ba7', '#397fa1', '#a36a5e']
function datasetColor(dataset: string): string {
  const index = comparisonRows.value.findIndex(item => item.dataset.toLowerCase() === dataset.toLowerCase())
  return datasetPalette[Math.max(0, index) % datasetPalette.length]
}
function curvePoint(key: string, index: number) {
  const points = (current.value?.points || []).filter(point => point[key] !== undefined)
  const point = points[index]
  if (!point || points.length < 2) return null
  const range = bounds(key)
  return { x: 48 + index * 388 / (points.length - 1), y: 130 - (point[key]! - range.min) * 100 / (range.max - range.min), point }
}
function hoverCurve(event: MouseEvent, key: string) {
  const points = (current.value?.points || []).filter(point => point[key] !== undefined)
  if (!points.length) return
  const rect = (event.currentTarget as SVGSVGElement).getBoundingClientRect()
  const x = (event.clientX - rect.left) / rect.width * 470
  curveHover.value = { key, index: Math.max(0, Math.min(points.length - 1, Math.round((x - 48) / 388 * (points.length - 1)))) }
}
async function refreshPaperTargets() {
  if (loadingPaperTargets.value) return
  loadingPaperTargets.value = true; paperTargetReason.value = ''
  try {
    const response = await api.refreshPaperTargets(props.run.project_id, props.run.id)
    paperTargetOverride.value = response.target_results || {}
    paperTargetReason.value = response.reason || response.source || ''
  } catch (err) { paperTargetReason.value = (err as Error).message }
  finally { loadingPaperTargets.value = false }
}
function chooseDefaultAttempt() {
  const picked = tunings.value.find(item => item.dataset?.toString().toLowerCase() === selectedDataset.value.toLowerCase() && item.selected)
  const selectedTuning = picked ? attempts.value.find(item => item.attempt === (picked.session_id ? `${String(picked.session_id).slice(0, 8)}:tuning-${picked.trial}` : picked.trial ? `tuning-${picked.trial}` : `lr-${picked.value}`) && item.points.length) : undefined
  const mostEvidence = [...attempts.value].sort((a, b) => b.points.length - a.points.length)[0]
  selectedAttempt.value = selectedTuning?.attempt || mostEvidence?.attempt || attempts.value[0]?.attempt || ''
}
async function refresh() {
  try {
    series.value = (await api.telemetry(props.run.project_id, props.run.id)).series
    if (props.run.result_json.ablation_active || props.run.result_json.tuning_active) ablationActivity.value = await api.activity(props.run.project_id, props.run.id)
    else ablationActivity.value = null
    if (!datasets.value.includes(selectedDataset.value)) {
      const mostEvidence = [...series.value].sort((a, b) => b.points.length - a.points.length)[0]
      selectedDataset.value = mostEvidence?.dataset || datasets.value[0] || ''
    }
    if (!attempts.value.some(item => item.attempt === selectedAttempt.value)) chooseDefaultAttempt()
    loadError.value = ''
  } catch (err) { loadError.value = (err as Error).message }
}
async function loadAblations() {
  try {
    const response = await api.ablationOptions(props.run.project_id, props.run.id)
    ablationOptions.value = response.options
    ablationReason.value = response.reason
    if (!ablationOptions.value.some(item => `${item.dataset}|${item.flag}` === selectedAblation.value))
      selectedAblation.value = ablationOptions.value.length ? `${ablationOptions.value[0].dataset}|${ablationOptions.value[0].flag}` : ''
  } catch (err) { ablationReason.value = (err as Error).message }
}
async function startAblation() {
  const item = ablationOptions.value.find(option => `${option.dataset}|${option.flag}` === selectedAblation.value)
  if (!item || startingAblation.value || !window.confirm(`运行 ${item.dataset} 的 ${item.flag} 消融实验？将使用当前项目独立环境和作者原生开关。`)) return
  startingAblation.value = true
  try {
    await api.startAblation(props.run.project_id, props.run.id, item.dataset, item.flag)
    emit('ablationStarted')
    loadError.value = ''
  } catch (err) { loadError.value = (err as Error).message }
  finally { startingAblation.value = false }
}
onMounted(() => { refresh(); loadAblations(); if (!Object.keys(props.run.paper_spec.target_results || {}).length && !props.run.paper_spec.target_results_checked_at) refreshPaperTargets(); timer = window.setInterval(refresh, 4000) })
onUnmounted(() => { if (timer) window.clearInterval(timer) })
watch(() => props.run.id, () => { selectedDataset.value = ''; selectedAttempt.value = ''; paperTargetOverride.value = null; paperTargetReason.value = ''; refresh(); loadAblations(); if (!Object.keys(props.run.paper_spec.target_results || {}).length && !props.run.paper_spec.target_results_checked_at) refreshPaperTargets() })
watch(selectedDataset, chooseDefaultAttempt)
</script>

<template>
  <section id="experiment" class="experiment-dashboard">
    <div class="section-title-row compact"><div><span class="kicker">实验观测</span><h2>实验过程与结果</h2></div><small>{{ run.status === 'executing' ? '每 4 秒读取实际日志' : '来自实际运行记录' }}</small></div>
    <div class="experiment-kpis"><article><small>本次批准的数据集</small><strong>{{ results.length }} / {{ approvedDatasets.length || '—' }}</strong></article><article><small>已取得最终结果</small><strong>{{ results.filter(item => Object.keys(item.metrics).length).length }}</strong><small>当前曲线 {{ trainingSamples }} 个真实采样点</small></article><article><small>修复 / 调参</small><strong>{{ repairs }} / {{ tunings.length }}</strong></article><article><small>累计执行时间</small><strong>{{ elapsed ? `${Math.round(elapsed)} 秒` : '—' }}</strong></article></div>
    <div v-if="loadError" class="alert error">{{ loadError }}</div>
    <div class="chart-selection"><label>数据集<select v-model="selectedDataset"><option v-for="dataset in datasets" :key="dataset">{{ dataset }}</option></select></label><label>实验轮次<select v-model="selectedAttempt"><option v-for="attempt in attempts" :key="attempt.attempt" :value="attempt.attempt">{{ attempt.attempt }}</option></select></label><small>{{ metricKeys.length === 1 && metricKeys[0].includes('loss') ? '作者日志当前仅逐轮输出损失；其他指标须在评估产生后展示。' : '仅显示与论文指标最相关的最多 3 条曲线；未伪造缺失指标。' }}</small></div>
    <div v-if="metricKeys.length" class="experiment-charts" :class="{ 'three-charts': metricKeys.length === 3 }">
      <article v-for="key in metricKeys" :key="key" class="experiment-chart-card chart-interactive"><div class="chart-heading"><div><small>训练过程</small><h3>{{ metricTitle(key) }}</h3></div><strong>{{ number(latest(current?.points, key), key) }}</strong></div><svg viewBox="0 0 470 170" role="img" :aria-label="`${selectedDataset} ${metricTitle(key)} 曲线`" @mousemove="hoverCurve($event, key)" @mouseleave="curveHover = null"><path class="chart-gridline" d="M48 30 H436 M48 80 H436 M48 130 H436"/><text x="4" y="34">{{ number(bounds(key).max, key) }}</text><text x="4" y="84">{{ number((bounds(key).max + bounds(key).min) / 2, key) }}</text><text x="4" y="134">{{ number(bounds(key).min, key) }}</text><text x="48" y="160">{{ current?.points[0]?.epoch ?? 0 }}</text><text x="407" y="160">{{ current?.points.at(-1)?.epoch ?? 0 }}</text><path v-if="line(key)" class="chart-line" :class="key.includes('loss') ? 'loss-line' : 'accuracy-line'" :d="line(key)"/><template v-if="curveHover?.key === key && curvePoint(key, curveHover.index)"><path class="chart-cursor" :d="`M${curvePoint(key, curveHover.index)!.x} 30 V130`"/><circle class="chart-focus" :cx="curvePoint(key, curveHover.index)!.x" :cy="curvePoint(key, curveHover.index)!.y" r="5" /></template></svg><div v-if="curveHover?.key === key && curvePoint(key, curveHover.index)" class="chart-tooltip" :style="{ left: `${Math.max(15, Math.min(75, curvePoint(key, curveHover.index)!.x / 470 * 100))}%` }">第 {{ curvePoint(key, curveHover.index)!.point.epoch }} 轮 · {{ number(curvePoint(key, curveHover.index)!.point[key], key) }}</div><p>训练轮次 → · {{ current?.points.length || 0 }} 个日志采样点 · 鼠标移动可查看真实数值</p></article>
    </div>
    <div v-else class="experiment-chart-card chart-empty">等待可解析的训练指标。该仓库若不输出逐轮指标，原始日志仍可查看。</div>
    <div class="experiment-chart-card result-comparison">
      <div class="chart-heading"><div><small>论文与实测</small><h3>论文指标与本次实验</h3></div><button class="button secondary" :disabled="loadingPaperTargets" @click="refreshPaperTargets">{{ loadingPaperTargets ? '正在核对论文表格…' : '重新提取论文数值' }}</button></div>
      <p v-if="paperTargetReason">{{ paperTargetReason }}</p><p v-else>仅展示论文指标，沿用论文中的英文名称；代码额外输出不在此展示。悬停柱形查看数值。</p>
      <div v-if="comparisonRows.length" class="comparison-datasets">
        <section v-for="result in comparisonRows" :key="result.dataset" class="comparison-dataset" :style="{ '--dataset-color': datasetColor(result.dataset) }">
          <h4>{{ result.dataset }} <small>{{ result.return_code === 0 ? '实验已结束' : '未完成' }}{{ result.tuning_selected ? ' · 调参结果，非原配置复现' : '' }}</small></h4>
          <p v-if="result.baseline_metrics && Object.keys(result.baseline_metrics).length">原配置基线：<span v-for="(value, key) in result.baseline_metrics" :key="key">{{ metricNames[String(key)] || key }} {{ number(value, String(key)) }}； </span></p>
          <p v-if="Object.keys(selectedTuningMetrics(run.result_json, result.dataset)).length">额外调参已完成；原配置基线保留，新增调参后的测试结果。</p>
          <p v-else-if="result.tuning_note">{{ result.tuning_note.includes('调参未获批准') ? '仅未批准额外搜索，不影响已完成的基线；可在下方审查恢复调参计划。' : result.tuning_note }}</p>
          <div class="comparison-metrics">
            <article v-for="metric in comparisonMetrics(result)" :key="metric.key" class="comparison-metric">
              <strong>{{ metricNames[metric.key] || metric.key.toUpperCase() }}</strong>
              <svg viewBox="0 0 220 185" role="img" :aria-label="`${result.dataset} ${metric.key}：论文 ${number(metric.paper, metric.key)}；原配置 ${number(metric.actual, metric.key)}；调参后 ${number(metric.tuned, metric.key)}`" @mouseleave="barHover = null">
                <path class="comparison-axis" d="M43 18 V146 H204 M43 82 H204 M43 18 H204" />
                <text x="2" y="22">{{ number(metric.maximum, metric.key) }}</text>
                <text x="2" y="86">{{ number(metric.maximum / 2, metric.key) }}</text>
                <text x="2" y="150">0</text>
                <rect v-if="metric.paper !== undefined" x="66" :y="146 - barHeight(metric.paper, metric.maximum)" width="28" :height="barHeight(metric.paper, metric.maximum)" class="comparison-paper-column" @mouseenter="barHover = { dataset: result.dataset, metric: metric.key, series: 'paper' }"><title>论文：{{ number(metric.paper, metric.key) }}</title></rect>
                <rect v-if="metric.actual !== undefined" x="116" :y="146 - barHeight(metric.actual, metric.maximum)" width="28" :height="barHeight(metric.actual, metric.maximum)" class="comparison-run-column" @mouseenter="barHover = { dataset: result.dataset, metric: metric.key, series: 'actual' }"><title>原配置：{{ number(metric.actual, metric.key) }}</title></rect>
                <rect v-if="metric.tuned !== undefined" x="166" :y="146 - barHeight(metric.tuned, metric.maximum)" width="28" :height="barHeight(metric.tuned, metric.maximum)" class="comparison-tuned-column" @mouseenter="barHover = { dataset: result.dataset, metric: metric.key, series: 'tuned' }"><title>按验证集选择的调参后测试结果：{{ number(metric.tuned, metric.key) }}</title></rect>
                <text x="80" y="165" text-anchor="middle">论文</text><text x="130" y="165" text-anchor="middle">原配置</text><text v-if="metric.tuned !== undefined" x="180" y="165" text-anchor="middle">调参后</text>
              </svg>
              <div v-if="barHover?.dataset === result.dataset && barHover?.metric === metric.key" class="comparison-hover">{{ barHover.series === 'paper' ? '论文' : barHover.series === 'tuned' ? '调参后' : '原配置' }} · {{ number(barHover.series === 'paper' ? metric.paper : barHover.series === 'tuned' ? metric.tuned : metric.actual, metric.key) }}</div>
              <div class="comparison-values"><span>{{ metric.paper === undefined ? '论文未给数值' : number(metric.paper, metric.key) }}</span><span>{{ metric.actual === undefined ? '尚未取得' : number(metric.actual, metric.key) }}</span><span v-if="metric.tuned !== undefined">{{ number(metric.tuned, metric.key) }}</span></div>
              <small v-if="metric.tuned !== undefined">按验证集选参；展示对应测试值，不等于原配置复现。</small>
            </article>
          </div>
          <p v-if="!comparisonMetrics(result).length">尚未提取到该数据集的论文评估指标；不会用代码输出代替论文指标。</p>
        </section>
      </div>
      <p v-else>执行后将显示逐数据集的论文与实测指标。</p>
    </div>
<div class="experiment-chart-card tuning-panel">
      <div class="chart-heading"><div><small>修复与调参</small><h3>自动调参与修复</h3></div><small>{{ tunings.length }} 次参数尝试</small></div>
      <div v-if="tunings.length" class="tuning-table"><div class="tuning-row tuning-head"><span>数据集</span><span>参数</span><span>尝试值/组合</span><span>验证指标</span><span>测试指标</span><span>最终采用</span></div><div v-for="(item, index) in tunings" :key="index" class="tuning-row"><span>{{ item.dataset }}</span><span>{{ item.parameter || '参数组合' }}</span><span>{{ item.parameters ? JSON.stringify(item.parameters) : item.value }}</span><span>{{ item.validation_score ?? item.validation_accuracy ?? '—' }}</span><span>{{ item.test_metrics ? JSON.stringify(item.test_metrics) : item.test_accuracy ?? '—' }}</span><span>{{ item.selected ? '✓' : '—' }}</span></div></div>
      <p v-else>没有执行额外搜索，不影响已完成的作者配置基线。额外搜索会增加运行时间，需审查计划后单独批准。</p>
      <p v-if="run.result_json.tuning_active" role="status">正在当前实验内搜索 {{ run.result_json.tuning_active.dataset || '已批准数据集' }} · 第 {{ run.result_json.tuning_active.trial || '—' }} 组；全部基线结果保留。可用“暂停任务”中止额外搜索。</p>
      <pre v-if="run.result_json.tuning_active && ablationActivity?.log_tail">{{ ablationActivity.log_tail }}</pre>
      <div v-for="session in run.result_json.tuning_sessions || []" :key="session.id">
        <strong>搜索 {{ session.id.slice(0, 8) }} · {{ session.status }}</strong>
        <div v-for="plan in session.plans" :key="plan.dataset"><p>{{ plan.dataset }} · 最多 {{ plan.max_trials }} 组 / {{ plan.max_seconds / 60 }} 分钟</p><p v-for="item in plan.space" :key="item.parameter">{{ item.parameter }}：基线 {{ parameterText(item.baseline) }}；搜索候选 {{ item.values.map(parameterText).join('、') }}（{{ item.source }}）</p></div>
        <p>{{ session.note }}</p><p v-for="(note, dataset) in session.outcomes || {}" :key="dataset">{{ dataset }}：{{ note }}</p>
      </div>
      <button v-if="canReviewRecovery" class="button secondary" :disabled="recoveryBusy" @click="reviewRecovery">{{ recoveryBusy ? '正在处理…' : '审查未复现数据集的恢复调参计划（不启动实验）' }}</button>
      <div v-if="recoveryOpen">
        <div v-for="plan in recoveryPlans" :key="plan.dataset"><strong>{{ plan.dataset }}</strong><p v-for="item in plan.space" :key="item.parameter">{{ item.parameter }}：基线 {{ parameterText(item.baseline) }}；候选 {{ item.values.map(parameterText).join('、') }}；源码 {{ item.source }}</p><p>{{ plan.trials.length }} 组局部/联合组合，最多额外 30 分钟。{{ plan.selection_rule }}；{{ plan.limitation }}</p><details><summary>查看将执行的参数组合</summary><pre>{{ JSON.stringify(plan.trials, null, 2) }}</pre></details><p v-if="plan.blocked_reason" role="alert">不能启动：{{ plan.blocked_reason }}</p><p v-else-if="plan.baseline_identity !== 'fingerprint'">历史基线未保存源码指纹，仅完成日志与源码时间检查，仍需核对基线来源。</p></div>
        <label v-if="approvedPlans.length"><input v-model="recoveryReviewed" type="checkbox" />已核对数据版本、划分和作者参数；同意仅对未达标基线恢复搜索</label>
        <button v-if="approvedPlans.length" class="button primary" :disabled="recoveryBusy || !recoveryReviewed" @click="startRecovery">批准当前实验内调参（保留全部基线）</button>
      </div>
      <p v-if="recoveryError" role="alert">{{ recoveryError }}</p>
      <p v-if="tunings.length && !run.result_json.tuning_policy">以上为旧流程历史记录，未按新搜索策略重跑；旧记录未保存的基线不会补造。</p><p>只用验证集选参；不依据测试结果启动搜索。缺少验证指标时停止。基线单独保存，调参结果不等于按论文原配置复现。</p>
    </div>
    <div class="experiment-chart-card ablation-panel"><div class="chart-heading"><div><small>组件消融</small><h3>组件消融实验</h3></div><small>仅运行作者代码明确支持的关闭开关</small></div><p>消融对照必须在同一数据集、同一独立环境中关闭一个模型组件。普通学习率调整不算消融。</p><div v-if="run.result_json.ablation_active" class="operation-live">正在运行 {{ run.result_json.ablation_active.dataset }} · {{ run.result_json.ablation_active.flag }}；可用项目的暂停按钮中止。<pre v-if="ablationActivity?.log_tail">{{ ablationActivity.log_tail }}</pre></div><div v-if="ablationOptions.length" class="ablation-controls"><select v-model="selectedAblation" aria-label="选择组件消融"><option v-for="item in ablationOptions" :key="`${item.dataset}|${item.flag}`" :value="`${item.dataset}|${item.flag}`">{{ item.dataset }} · {{ item.flag }}</option></select><button class="button secondary" :disabled="startingAblation || !!run.result_json.ablation_active" @click="startAblation">{{ startingAblation ? '启动中…' : '运行消融' }}</button></div><p v-else>{{ ablationReason || '正在检查作者代码的消融开关…' }}</p><div v-if="run.result_json.ablation_results?.length" class="tuning-table"><div class="tuning-row ablation-head"><span>数据集</span><span>消融项</span><span>基线</span><span>消融值</span><span>差值</span><span>状态</span></div><div v-for="(item, index) in run.result_json.ablation_results" :key="index" class="tuning-row ablation-head"><span>{{ item.dataset }}</span><span>{{ item.label }}</span><span>{{ item.baseline ?? '—' }}</span><span>{{ item.value ?? '—' }}</span><span>{{ item.delta ?? '—' }}</span><span>{{ item.status }}</span></div></div></div>
  </section>
</template>
