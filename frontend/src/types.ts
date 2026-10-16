export type ArtifactKind = 'paper' | 'code' | 'config' | 'result' | 'note' | 'dataset'

export interface Artifact {
  id: string
  kind: ArtifactKind
  filename: string
  media_type: string
  size_bytes: number
  metadata_json: Record<string, unknown>
  created_at: string
}

export interface Project {
  id: string
  name: string
  description: string
  research_domain: string
  created_at: string
  updated_at: string
  artifacts: Artifact[]
}

export interface ProjectListItem {
  id: string
  name: string
  description: string
  research_domain: string
  artifact_count: number
  latest_reproduction_status: ReproductionStatus | null
  latest_reproduction_progress: number | null
  updated_at: string
}

export type ReproductionStatus = 'queued' | 'running' | 'needs_input' | 'awaiting_data' | 'awaiting_approval' | 'executing' | 'completed' | 'failed' | 'paused'

export interface DatasetResult {
  dataset: string
  metric_key?: string
  return_code: number
  metrics: Record<string, number>
  baseline_metrics?: Record<string, number>
  tuning_selected?: boolean
  tuning_note?: string
  target: number | null
  delta: number | null
  verdict: string
  wall_time_seconds?: number
  diagnosis: Record<string, string>
}

export interface ClaimResult {
  scope?: 'baseline' | 'tuned'
  claim: string
  dataset: string
  supported: boolean
  status: string
  evidence: string
}

export interface ReproductionEvent {
  stage: string
  progress: number
  message: string
  time: string
}

export interface TuningPlan {
  ready?: boolean
  blocked_reason?: string
  review_token?: string
  baseline_identity?: string
  dataset: string
  strategy: string
  space: Array<{ parameter: string; baseline: number; values: number[]; source: string; reason: string }>
  trials: Array<Record<string, number>>
  max_trials: number
  max_seconds: number
  selection_rule: string
  limitation: string
}

export interface ReproductionRun {
  id: string
  project_id: string
  status: ReproductionStatus
  current_stage: string
  progress: number
  message: string
  goal: string
  model_id: string
  source_mode: string
  repository_url: string
  workspace_path: string
  execution_mode: string
  execution_command: Record<string, unknown>
  paper_spec: Record<string, unknown>
  code_spec: Record<string, unknown>
  environment_spec: Record<string, unknown>
  dataset_spec: Record<string, unknown>
  result_json: Record<string, unknown> & {
    report_path?: string
    reproduction_success?: boolean
    dataset_results?: DatasetResult[]
    claim_results?: ClaimResult[]
    user_action?: string
    tuning_attempts?: Array<Record<string, unknown>>
    repair_attempts?: Array<Record<string, unknown>>
    code_generation_attempts?: Array<{ attempt: number; reason: string; phase?: string; file?: string; status?: string; model?: string; provider?: string; returned_files?: number; review_response?: Record<string, unknown> }>
    code_generation_issues?: Array<{ id: string; file: string; reason: string; status: string }>
    code_agent_progress?: { step: number; message: string; updated_at: string }
    finished_at?: string
    action_required?: { kind: string; reason?: string; detail?: string; options?: string[]; failure_kind?: string } | null
    skipped_datasets?: string[]
    active_operation?: RunOperation
    paused_from_status?: string
    branch?: { base_run_id: string; proposal_id: string; title: string; file: string; rationale?: string; expected_observation?: string }
    revision?: { base_run_id: string; proposal_id: string; title: string; files: string[]; rationale?: string; expected_observation?: string }
    rerun?: { base_run_id: string; datasets: string[]; gpu_requested: boolean; isolated_copy: boolean }
    ablation_results?: Array<Record<string, unknown>>
    ablation_active?: { dataset: string; flag: string; started_at: string }
    tuning_active?: { id: string; dataset?: string; trial?: number; parameters?: Record<string, number> }
    tuning_sessions?: Array<{ id: string; status: string; plans: TuningPlan[]; note?: string; outcomes?: Record<string, string> }>
  }
  events: ReproductionEvent[]
  logs: string
  error_message: string
  created_at: string
  updated_at: string
  approved_at?: string | null
  completed_at: string | null
}

export interface ModelCatalog {
  provider: string
  default: string
  models: Array<{ id: string; label: string }>
}

export interface AnalyticsOverview {
  period_days: number
  active_projects: number
  reproduced_projects: number
  reproduction_runs: number
  tokens: number
  daily_runs: Array<{ date: string; count: number }>
  daily_tokens: Array<{ date: string; count: number }>
  status_breakdown: Record<string, number>
  tokens_by_role: Record<string, number>
  waiting_projects: number
  failed_runs: number
}

export interface TelemetryPoint { epoch: number; [metric: string]: number | undefined }
export interface TelemetrySeries { dataset: string; attempt: string; points: TelemetryPoint[]; updated_at: number; note?: string }
export interface GpuEnvironmentPlan {
  needed?: boolean
  supported?: boolean
  source_changes_needed?: boolean
  source_files?: string[]
  packages?: string[]
  warning?: string
  reason?: string
  candidates?: Array<{ profile: string; packages: string[] }>
}
export interface GpuPreparation {
  framework?: string
  preparation?: { action: string; reason: string }
  runtime?: { checked?: boolean; version?: string; cuda_available?: boolean; error?: string }
  gpu_environment_plan?: GpuEnvironmentPlan
}
export interface CopilotMessage {
  id: string
  role: 'user' | 'assistant'
  content: string
    context: { run_id?: string; suggested_action?: string; stage?: string; status?: 'pending' | 'failed' | 'partial' | 'degraded' | 'completed'; error?: string; action_plan?: GpuPreparation & { kind: string; run_id: string; instruction?: string; dataset: string; datasets?: string[]; selected_datasets: string[]; gpu_requested?: boolean; reason?: string; gpu_code_issues?: string[] } }
  token_usage: number
  created_at: string
  conversation_id?: string | null
}

export interface CopilotConversation {
  id: string
  project_id: string
  title: string
  created_at: string
  updated_at: string
}

export interface RunOperation {
  kind: string; step: number; total_steps: number; started_at: string; command: string
  dataset?: string; state?: string; attempt?: number; downloaded_bytes?: number
  total_bytes?: number | null; updated_at?: string; error?: string
}
export interface ReproductionActivity {
  active: boolean
  worker_active?: boolean
  operation?: RunOperation
  log_tail: string
}
export interface ModelRouting {
  provider: string
  revision: string
  note: string
  groups: Array<{ id: string; label: string; roles: string[]; agents: string[]; model: string; effective_model: string }>
}

export interface RevisionRunPlan {
  proposal_id: string
  action: 'blocked' | 'prepare_code' | 'approve_environment' | 'ready' | 'already_created'
  reason: string
  gpu_requested: boolean
  selected_datasets?: string[]
  datasets?: string[]
  scope_explicit?: boolean
  scope_reason?: string
  run_id?: string
  readiness?: {
    framework: string
    gpu_environment_plan?: GpuEnvironmentPlan
  }
}
