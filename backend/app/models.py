import enum
import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    JSON, Boolean, CheckConstraint, DateTime, Enum, Float, ForeignKey, Index,
    Integer, String, Text, UniqueConstraint, Uuid,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .database import Base


JSON_TYPE = JSON().with_variant(JSONB(), "postgresql")


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def new_uuid() -> str:
    return str(uuid.uuid4())


class ArtifactKind(str, enum.Enum):
    paper = "paper"
    code = "code"
    config = "config"
    result = "result"
    note = "note"
    dataset = "dataset"


class AnalysisStatus(str, enum.Enum):
    pending = "pending"
    running = "running"
    completed = "completed"
    failed = "failed"


class ReproductionStatus(str, enum.Enum):
    queued = "queued"
    running = "running"
    awaiting_data = "awaiting_data"
    needs_input = "needs_input"
    awaiting_approval = "awaiting_approval"
    executing = "executing"
    completed = "completed"
    failed = "failed"
    paused = "paused"


class User(Base):
    """与 EduAgent users 表保持相同的租户、账号和角色设计。"""

    __tablename__ = "users"
    __table_args__ = (
        UniqueConstraint("tenant_id", "email", name="uq_users_tenant_email"),
        CheckConstraint("role IN ('researcher', 'reviewer', 'admin')", name="ck_users_role"),
        Index("idx_users_tenant_id", "tenant_id"),
        Index("idx_users_role", "role"),
    )

    id: Mapped[str] = mapped_column(Uuid(as_uuid=False), primary_key=True, default=new_uuid)
    tenant_id: Mapped[str] = mapped_column(String(64), default="tenant_default")
    username: Mapped[str] = mapped_column(String(64))
    email: Mapped[str] = mapped_column(String(128))
    password_hash: Mapped[str] = mapped_column(String(256))
    role: Mapped[str] = mapped_column(String(16), default="researcher")
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)


class Project(Base):
    __tablename__ = "research_projects"
    __table_args__ = (
        Index("idx_research_projects_tenant", "tenant_id"),
        Index("idx_research_projects_owner", "owner_id"),
        Index("idx_research_projects_status", "status"),
    )

    id: Mapped[str] = mapped_column(Uuid(as_uuid=False), primary_key=True, default=new_uuid)
    tenant_id: Mapped[str] = mapped_column(String(64), default="tenant_default")
    owner_id: Mapped[str | None] = mapped_column(Uuid(as_uuid=False), ForeignKey("users.id"), nullable=True)
    name: Mapped[str] = mapped_column(String(160))
    description: Mapped[str] = mapped_column(Text, default="")
    research_domain: Mapped[str] = mapped_column(String(80), default="通用科研")
    status: Mapped[str] = mapped_column(String(20), default="active")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)

    artifacts: Mapped[list["Artifact"]] = relationship(
        back_populates="project", cascade="all, delete-orphan", lazy="selectin"
    )
    analyses: Mapped[list["Analysis"]] = relationship(
        back_populates="project", cascade="all, delete-orphan", lazy="selectin"
    )
    reproduction_jobs: Mapped[list["ReproductionJob"]] = relationship(
        back_populates="project", cascade="all, delete-orphan", lazy="selectin"
    )
    copilot_messages: Mapped[list["CopilotMessage"]] = relationship(
        back_populates="project", cascade="all, delete-orphan", lazy="selectin"
    )
    copilot_conversations: Mapped[list["CopilotConversation"]] = relationship(
        back_populates="project", cascade="all, delete-orphan", lazy="selectin"
    )


class Artifact(Base):
    __tablename__ = "research_artifacts"
    __table_args__ = (
        Index("idx_research_artifacts_project", "project_id"),
        Index("idx_research_artifacts_kind", "kind"),
        Index("idx_research_artifacts_sha256", "sha256"),
    )

    id: Mapped[str] = mapped_column(Uuid(as_uuid=False), primary_key=True, default=new_uuid)
    tenant_id: Mapped[str] = mapped_column(String(64), default="tenant_default")
    project_id: Mapped[str] = mapped_column(
        Uuid(as_uuid=False), ForeignKey("research_projects.id", ondelete="CASCADE")
    )
    kind: Mapped[ArtifactKind] = mapped_column(Enum(ArtifactKind, native_enum=False, length=16))
    filename: Mapped[str] = mapped_column(String(255))
    media_type: Mapped[str] = mapped_column(String(120), default="application/octet-stream")
    size_bytes: Mapped[int] = mapped_column(Integer, default=0)
    storage_provider: Mapped[str] = mapped_column(String(16), default="local")
    storage_path: Mapped[str] = mapped_column("object_key", String(512))
    sha256: Mapped[str] = mapped_column(String(64), default="")
    parse_status: Mapped[str] = mapped_column(String(16), default="completed")
    content_text: Mapped[str] = mapped_column(Text, default="")
    metadata_json: Mapped[dict] = mapped_column("metadata", JSON_TYPE, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)

    project: Mapped[Project] = relationship(back_populates="artifacts")


class Analysis(Base):
    __tablename__ = "agent_runs"
    __table_args__ = (
        UniqueConstraint("thread_id", name="uq_agent_runs_thread_id"),
        Index("idx_agent_runs_project_created", "project_id", "created_at"),
        Index("idx_agent_runs_status", "status"),
    )

    id: Mapped[str] = mapped_column(Uuid(as_uuid=False), primary_key=True, default=new_uuid)
    tenant_id: Mapped[str] = mapped_column(String(64), default="tenant_default")
    project_id: Mapped[str] = mapped_column(
        Uuid(as_uuid=False), ForeignKey("research_projects.id", ondelete="CASCADE")
    )
    user_id: Mapped[str | None] = mapped_column(Uuid(as_uuid=False), ForeignKey("users.id"), nullable=True)
    thread_id: Mapped[str] = mapped_column(String(160), default=lambda: f"run_{new_uuid()}")
    goal: Mapped[str] = mapped_column(Text, default="")
    route: Mapped[str] = mapped_column(String(32), default="full_audit")
    status: Mapped[AnalysisStatus] = mapped_column(
        Enum(AnalysisStatus, native_enum=False, length=16), default=AnalysisStatus.pending
    )
    score: Mapped[int] = mapped_column(Integer, default=0)
    summary: Mapped[str] = mapped_column(Text, default="")
    report_json: Mapped[dict] = mapped_column("report", JSON_TYPE, default=dict)
    error_message: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    project: Mapped[Project] = relationship(back_populates="analyses")
    steps: Mapped[list["AgentStep"]] = relationship(cascade="all, delete-orphan")
    model_calls: Mapped[list["ModelCall"]] = relationship(cascade="all, delete-orphan")
    findings: Mapped[list["AuditFinding"]] = relationship(cascade="all, delete-orphan")
    reproduction_tasks: Mapped[list["ReproductionTask"]] = relationship(cascade="all, delete-orphan")


class AgentStep(Base):
    __tablename__ = "agent_steps"
    __table_args__ = (
        UniqueConstraint("run_id", "sequence_no", name="uq_agent_steps_run_sequence"),
        Index("idx_agent_steps_run", "run_id"),
    )

    id: Mapped[str] = mapped_column(Uuid(as_uuid=False), primary_key=True, default=new_uuid)
    run_id: Mapped[str] = mapped_column(Uuid(as_uuid=False), ForeignKey("agent_runs.id", ondelete="CASCADE"))
    sequence_no: Mapped[int] = mapped_column(Integer)
    agent_type: Mapped[str] = mapped_column(String(64))
    model_role: Mapped[str] = mapped_column(String(32), default="")
    status: Mapped[str] = mapped_column(String(16), default="completed")
    input_summary: Mapped[dict] = mapped_column(JSON_TYPE, default=dict)
    output: Mapped[dict] = mapped_column(JSON_TYPE, default=dict)
    fallback_used: Mapped[bool] = mapped_column(Boolean, default=False)
    duration_ms: Mapped[float] = mapped_column(Float, default=0)
    error_message: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class ModelCall(Base):
    __tablename__ = "model_calls"
    __table_args__ = (Index("idx_model_calls_run", "run_id"),)

    id: Mapped[str] = mapped_column(Uuid(as_uuid=False), primary_key=True, default=new_uuid)
    run_id: Mapped[str] = mapped_column(Uuid(as_uuid=False), ForeignKey("agent_runs.id", ondelete="CASCADE"))
    step_id: Mapped[str | None] = mapped_column(Uuid(as_uuid=False), ForeignKey("agent_steps.id", ondelete="SET NULL"), nullable=True)
    provider: Mapped[str] = mapped_column(String(32))
    model_id: Mapped[str] = mapped_column(String(128))
    model_role: Mapped[str] = mapped_column(String(32))
    prompt_tokens: Mapped[int] = mapped_column(Integer, default=0)
    completion_tokens: Mapped[int] = mapped_column(Integer, default=0)
    total_tokens: Mapped[int] = mapped_column(Integer, default=0)
    latency_ms: Mapped[float] = mapped_column(Float, default=0)
    success: Mapped[bool] = mapped_column(Boolean, default=True)
    fallback_used: Mapped[bool] = mapped_column(Boolean, default=False)
    error_message: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class AuditFinding(Base):
    __tablename__ = "audit_findings"
    __table_args__ = (
        Index("idx_audit_findings_run", "run_id"),
        Index("idx_audit_findings_severity", "severity"),
    )

    id: Mapped[str] = mapped_column(Uuid(as_uuid=False), primary_key=True, default=new_uuid)
    run_id: Mapped[str] = mapped_column(Uuid(as_uuid=False), ForeignKey("agent_runs.id", ondelete="CASCADE"))
    severity: Mapped[str] = mapped_column(String(16))
    category: Mapped[str] = mapped_column(String(64))
    title: Mapped[str] = mapped_column(String(256))
    description: Mapped[str] = mapped_column(Text, default="")
    recommendation: Mapped[str] = mapped_column(Text, default="")
    status: Mapped[str] = mapped_column(String(16), default="open")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    evidence_refs: Mapped[list["EvidenceReference"]] = relationship(cascade="all, delete-orphan")


class EvidenceReference(Base):
    __tablename__ = "evidence_refs"
    __table_args__ = (Index("idx_evidence_refs_finding", "finding_id"),)

    id: Mapped[str] = mapped_column(Uuid(as_uuid=False), primary_key=True, default=new_uuid)
    finding_id: Mapped[str] = mapped_column(Uuid(as_uuid=False), ForeignKey("audit_findings.id", ondelete="CASCADE"))
    artifact_id: Mapped[str | None] = mapped_column(Uuid(as_uuid=False), ForeignKey("research_artifacts.id", ondelete="SET NULL"), nullable=True)
    source: Mapped[str] = mapped_column(String(255))
    locator: Mapped[str] = mapped_column(String(255), default="")
    excerpt: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class ReproductionTask(Base):
    __tablename__ = "reproduction_tasks"
    __table_args__ = (
        UniqueConstraint("run_id", "step_no", name="uq_reproduction_tasks_run_step"),
        Index("idx_reproduction_tasks_run", "run_id"),
    )

    id: Mapped[str] = mapped_column(Uuid(as_uuid=False), primary_key=True, default=new_uuid)
    run_id: Mapped[str] = mapped_column(Uuid(as_uuid=False), ForeignKey("agent_runs.id", ondelete="CASCADE"))
    step_no: Mapped[int] = mapped_column(Integer)
    title: Mapped[str] = mapped_column(String(256))
    description: Mapped[str] = mapped_column(Text, default="")
    status: Mapped[str] = mapped_column(String(20), default="needs_review")
    inputs: Mapped[dict] = mapped_column(JSON_TYPE, default=dict)
    expected_outputs: Mapped[dict] = mapped_column(JSON_TYPE, default=dict)
    verification: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class ReproductionJob(Base):
    """自主复现任务；与审计报告分离，保存实际获取和执行代码的状态。"""

    __tablename__ = "reproduction_jobs"
    __table_args__ = (
        Index("idx_reproduction_jobs_project_created", "project_id", "created_at"),
        Index("idx_reproduction_jobs_status", "status"),
    )

    id: Mapped[str] = mapped_column(Uuid(as_uuid=False), primary_key=True, default=new_uuid)
    tenant_id: Mapped[str] = mapped_column(String(64), default="tenant_default")
    project_id: Mapped[str] = mapped_column(
        Uuid(as_uuid=False), ForeignKey("research_projects.id", ondelete="CASCADE")
    )
    status: Mapped[ReproductionStatus] = mapped_column(
        Enum(ReproductionStatus, native_enum=False, length=24),
        default=ReproductionStatus.queued,
    )
    current_stage: Mapped[str] = mapped_column(String(48), default="queued")
    progress: Mapped[int] = mapped_column(Integer, default=0)
    message: Mapped[str] = mapped_column(String(500), default="任务已创建")
    goal: Mapped[str] = mapped_column(Text, default="复现论文核心实验")
    model_id: Mapped[str] = mapped_column(String(128), default="deepseek-v4-flash-0731")
    source_mode: Mapped[str] = mapped_column(String(24), default="undetermined")
    repository_url: Mapped[str] = mapped_column(String(500), default="")
    workspace_path: Mapped[str] = mapped_column(String(600), default="")
    execution_mode: Mapped[str] = mapped_column(String(16), default="full")
    execution_command: Mapped[dict] = mapped_column(JSON_TYPE, default=dict)
    paper_spec: Mapped[dict] = mapped_column(JSON_TYPE, default=dict)
    code_spec: Mapped[dict] = mapped_column(JSON_TYPE, default=dict)
    environment_spec: Mapped[dict] = mapped_column(JSON_TYPE, default=dict)
    dataset_spec: Mapped[dict] = mapped_column(JSON_TYPE, default=dict)
    result_json: Mapped[dict] = mapped_column(JSON_TYPE, default=dict)
    events: Mapped[list] = mapped_column(JSON_TYPE, default=list)
    logs: Mapped[str] = mapped_column(Text, default="")
    error_message: Mapped[str] = mapped_column(Text, default="")
    approved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    project: Mapped[Project] = relationship(back_populates="reproduction_jobs")


class CopilotConversation(Base):
    __tablename__ = "project_copilot_conversations"
    __table_args__ = (Index("idx_copilot_conversation_project", "project_id", "updated_at"),)

    id: Mapped[str] = mapped_column(Uuid(as_uuid=False), primary_key=True, default=new_uuid)
    project_id: Mapped[str] = mapped_column(Uuid(as_uuid=False), ForeignKey("research_projects.id", ondelete="CASCADE"))
    title: Mapped[str] = mapped_column(String(160), default="新对话")
    summary: Mapped[str] = mapped_column(Text, default="")
    summary_count: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)

    project: Mapped[Project] = relationship(back_populates="copilot_conversations")
    messages: Mapped[list["CopilotMessage"]] = relationship(back_populates="conversation", cascade="all, delete-orphan")


class CopilotMessage(Base):
    __tablename__ = "project_copilot_messages"
    __table_args__ = (Index("idx_copilot_project_created", "project_id", "created_at"),)

    id: Mapped[str] = mapped_column(Uuid(as_uuid=False), primary_key=True, default=new_uuid)
    project_id: Mapped[str] = mapped_column(
        Uuid(as_uuid=False), ForeignKey("research_projects.id", ondelete="CASCADE")
    )
    conversation_id: Mapped[str | None] = mapped_column(
        Uuid(as_uuid=False), ForeignKey("project_copilot_conversations.id", ondelete="CASCADE"), nullable=True
    )
    role: Mapped[str] = mapped_column(String(16))
    content: Mapped[str] = mapped_column(Text)
    context_json: Mapped[dict] = mapped_column(JSON_TYPE, default=dict)
    token_usage: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    project: Mapped[Project] = relationship(back_populates="copilot_messages")
    conversation: Mapped[CopilotConversation | None] = relationship(back_populates="messages")
