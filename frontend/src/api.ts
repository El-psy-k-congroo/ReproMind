import type { AnalyticsOverview, ArtifactKind, CopilotConversation, CopilotMessage, GpuEnvironmentPlan, ModelCatalog, ModelRouting, RevisionRunPlan, Project, ProjectListItem, ReproductionActivity, ReproductionRun, TelemetrySeries, TuningPlan } from './types'

const API_BASE = import.meta.env.VITE_API_BASE_URL || '/api'

function errorDetail(detail: unknown, fallback: string): string {
  if (typeof detail === 'string' && detail.trim()) return detail
  if (Array.isArray(detail)) {
    const messages = detail.flatMap(item => {
      if (!item || typeof item.msg !== 'string') return []
      const field = Array.isArray(item.loc) ? item.loc.filter((part: unknown) =>
        !['body', 'query', 'path'].includes(String(part))).join('.') : ''
      return [field ? `${field}: ${item.msg}` : item.msg]
    })
    if (messages.length) return messages.join('；')
  }
  return fallback
}

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const controller = new AbortController()
  const timeoutMs = path.endsWith('/branch-preview') ? 260000
    : path.includes('/revision-run-plan?') || path.endsWith('/run-readiness') ? 45000
    : path.endsWith('/revision-runs') ? 90000 : 25000
  const timeout = window.setTimeout(() => controller.abort(new Error('请求超时，请重试')), timeoutMs)
  const signal = options?.signal ? AbortSignal.any([options.signal, controller.signal]) : controller.signal
  try {
  const response = await fetch(`${API_BASE}${path}`, { ...options, signal })
  if (!response.ok) {
    let message = `请求失败（${response.status}）`
    try {
      const body = await response.json()
      message = errorDetail(body?.detail, message)
    } catch {
      // Keep the status-based error.
    }
    throw new Error(message)
  }
  if (response.status === 204) return undefined as T
  return await response.json() as T
  } finally { window.clearTimeout(timeout) }
}

export const api = {
  listProjects: () => request<ProjectListItem[]>('/projects'),
  getProject: (id: string) => request<Project>(`/projects/${id}`),
  deleteProject: (id: string) => request<void>(`/projects/${id}`, { method: 'DELETE' }),
  listModels: () => request<ModelCatalog>('/models'),
  modelRouting: () => request<ModelRouting>('/model-routing'),
  saveModelRouting: (models: Record<string, string>, revision: string) => request<ModelRouting>('/model-routing', {
    method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ models, revision }),
  }),
  overview: () => request<AnalyticsOverview>('/analytics/overview'),
  createProject: (payload: { name: string; description: string; research_domain: string }) =>
    request<Project>('/projects', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    }),
  uploadArtifact: (projectId: string, kind: ArtifactKind, file: File) => {
    const form = new FormData()
    form.append('kind', kind)
    form.append('file', file)
    return request(`/projects/${projectId}/artifacts`, { method: 'POST', body: form })
  },
  deleteArtifact: (projectId: string, artifactId: string) =>
    request<void>(`/projects/${projectId}/artifacts/${artifactId}`, { method: 'DELETE' }),
  startReproduction: (projectId: string, modelId: string, executionMode: 'smoke' | 'full' = 'full', goal = '复现论文核心实验', repositoryUrl = '') =>
    request<ReproductionRun>(`/projects/${projectId}/reproductions`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ goal, repository_url: repositoryUrl, execution_mode: executionMode, model_id: modelId }),
    }),
  getLatestReproduction: (projectId: string) =>
    request<ReproductionRun | null>(`/projects/${projectId}/reproductions/latest`),
  listReproductionRuns: (projectId: string) =>
    request<Array<{ id: string; status: string; auxiliary_active?: boolean; created_at: string; branch?: { base_run_id: string; title: string; file: string }; revision?: { base_run_id: string; title: string; files: string[] }; rerun?: { base_run_id: string; datasets: string[]; gpu_requested: boolean }; message: string }>>(`/projects/${projectId}/reproductions`),
  getReproduction: (projectId: string, runId: string) =>
    request<ReproductionRun>(`/projects/${projectId}/reproductions/${runId}`),
  refreshPaperTargets: (projectId: string, runId: string) =>
    request<{ target_results: Record<string, Record<string, number>>; source?: string; reason?: string }>(`/projects/${projectId}/reproductions/${runId}/paper-targets`, { method: 'POST' }),
  branchFiles: (projectId: string, runId: string) =>
    request<{ files: string[] }>(`/projects/${projectId}/reproductions/${runId}/branch-files`),
  previewBranch: (projectId: string, runId: string, file: string, instruction: string, modelId: string, preparationOnly = false) =>
    request<{ proposal_id: string; base_run_id: string; files: string[]; diffs: Record<string, string>; rationale: string; expected_observation: string; coding_trace?: Array<{ agent: string; model: string; files?: string[]; status: string }> }>(`/projects/${projectId}/reproductions/${runId}/branch-preview`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ file, instruction, model_id: modelId, preparation_only: preparationOnly }),
    }),
  applyRevision: (projectId: string, runId: string, proposalId: string) =>
    request<{ proposal_id: string; status: string; files: string[]; message: string }>(`/projects/${projectId}/reproductions/${runId}/revisions`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ proposal_id: proposalId }),
    }),
  savedBranchPreview: (projectId: string, runId: string) =>
    request<{ proposal_id: string; base_run_id: string; instruction: string; files: string[]; diffs: Record<string, string>; rationale: string; expected_observation: string; coding_trace?: Array<{ agent: string; model: string; files?: string[]; status: string }> } | null>(`/projects/${projectId}/reproductions/${runId}/branch-preview`),
  pendingRevision: (projectId: string, runId: string) =>
    request<{ proposal_id: string; files: string[]; instruction: string; applied_at: string } | null>(`/projects/${projectId}/reproductions/${runId}/pending-revision`),
  revisionRunPlan: (projectId: string, runId: string, proposalId: string) =>
    request<RevisionRunPlan>(`/projects/${projectId}/reproductions/${runId}/revision-run-plan?proposal_id=${encodeURIComponent(proposalId)}`),
  runReadiness: (projectId: string, runId: string) =>
request<{ datasets: string[]; current_local_dataset_evidence?: Record<string, { path?: string; ready?: boolean; file_count?: number; dataset_variant?: string }>; framework: string; gpu_code_ready: boolean; gpu_code_issues: string[]; device_log_evidence?: Array<{kind: string; log: string; line: string}>; environment_name: string; gpu_environment_plan?: GpuEnvironmentPlan; runtime?: { checked: boolean; version?: string; cuda_build?: string | boolean | null; cuda_available?: boolean; device_count?: number; error?: string }; torch_runtime: { checked: boolean; torch?: string; cuda_build?: string | null; cuda_available: boolean; device_count?: number; error?: string }; note: string }>(`/projects/${projectId}/reproductions/${runId}/run-readiness`),
  tuningPlan: (projectId: string, runId: string, purpose: 'recovery' | 'validation_optimization' = 'recovery') => request<{ plans: TuningPlan[]; executed: boolean }>(`/projects/${projectId}/reproductions/${runId}/tuning-plan?purpose=${purpose}`),
  startTuning: (projectId: string, runId: string, plans: TuningPlan[], purpose: 'recovery' | 'validation_optimization' = 'recovery') =>
    request<{ run_id: string; session_id: string }>(`/projects/${projectId}/reproductions/${runId}/tuning-sessions`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ datasets: plans.map(plan => plan.dataset), budget: 6, purpose,
        review_tokens: Object.fromEntries(plans.map(plan => [plan.dataset, plan.review_token])) }),
    }),
  rerunDataset: (projectId: string, runId: string, datasets: string[], gpuRequested: boolean, tuningBudget = 0, gpuEnvironmentMigration = false) =>
    request<{ run_id: string; status: string }>(`/projects/${projectId}/reproductions/${runId}/reruns`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ datasets, gpu_requested: gpuRequested, isolated_copy: true, tuning_budget: tuningBudget, gpu_environment_migration: gpuEnvironmentMigration }),
    }),
  runRevision: (projectId: string, runId: string, proposalId: string, isolatedCopy: boolean, datasets: string[] = [], gpuEnvironmentMigration = false) =>
    request<{ run_id: string; status: string }>(`/projects/${projectId}/reproductions/${runId}/revision-runs`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ proposal_id: proposalId, isolated_copy: isolatedCopy, datasets, gpu_environment_migration: gpuEnvironmentMigration }),
    }),
  reproductionReportUrl: (projectId: string, runId: string) =>
    `${API_BASE}/projects/${projectId}/reproductions/${runId}/report`,
  approveReproduction: (projectId: string, runId: string, tuningBudget = 0) =>
    request<ReproductionRun>(`/projects/${projectId}/reproductions/${runId}/approve`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ tuning_budget: tuningBudget }) }),
  resumeReproduction: (projectId: string, runId: string) =>
    request<ReproductionRun>(`/projects/${projectId}/reproductions/${runId}/resume`, { method: 'POST' }),
  chooseRepositorySource: (projectId: string, runId: string, repositoryUrl: string) =>
    request<ReproductionRun>(`/projects/${projectId}/reproductions/${runId}/source-choice`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ choice: 'repository', repository_url: repositoryUrl }),
    }),
  continuePartial: (projectId: string, runId: string) =>
    request<ReproductionRun>(`/projects/${projectId}/reproductions/${runId}/continue-partial`, { method: 'POST' }),
  telemetry: (projectId: string, runId: string) =>
    request<{ series: TelemetrySeries[] }>(`/projects/${projectId}/reproductions/${runId}/telemetry`),
  ablationOptions: (projectId: string, runId: string) =>
    request<{ options: Array<{ dataset: string; flag: string; label: string }>; reason: string }>(`/projects/${projectId}/reproductions/${runId}/ablations/options`),
  startAblation: (projectId: string, runId: string, dataset: string, flag: string) =>
    request<{ status: string }>(`/projects/${projectId}/reproductions/${runId}/ablations`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ dataset, flag }),
    }),
  activity: (projectId: string, runId: string) =>
    request<ReproductionActivity>(`/projects/${projectId}/reproductions/${runId}/activity`),
  pauseReproduction: (projectId: string, runId: string) =>
    request<ReproductionRun>(`/projects/${projectId}/reproductions/${runId}/pause`, { method: 'POST' }),
  resumePaused: (projectId: string, runId: string) =>
    request<ReproductionRun>(`/projects/${projectId}/reproductions/${runId}/resume-paused`, { method: 'POST' }),
  listConversations: (projectId: string) => request<CopilotConversation[]>(`/projects/${projectId}/copilot/conversations`),
  createConversation: (projectId: string, signal?: AbortSignal) => request<CopilotConversation>(`/projects/${projectId}/copilot/conversations`, { method: 'POST', signal }),
  deleteConversation: (projectId: string, conversationId: string) =>
    request<void>(`/projects/${projectId}/copilot/conversations/${conversationId}`, { method: 'DELETE' }),
  conversationMessages: (projectId: string, conversationId: string) =>
    request<CopilotMessage[]>(`/projects/${projectId}/copilot/conversations/${conversationId}/messages`),
  streamCopilot: async (projectId: string, conversationId: string, question: string, modelId: string, requestId: string, runId = '', signal?: AbortSignal): Promise<Response> => {
    const response = await fetch(`${API_BASE}/projects/${projectId}/copilot/conversations/${conversationId}/stream`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question, model_id: modelId, request_id: requestId, run_id: runId }),
      signal,
    })
    if (!response.ok) {
      const body = await response.json().catch(() => ({}))
      throw new Error(errorDetail(body?.detail, `流式请求失败（${response.status}）`))
    }
    return response
  },
}
