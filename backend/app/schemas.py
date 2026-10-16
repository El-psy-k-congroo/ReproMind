from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from .models import ArtifactKind, ReproductionStatus


class ProjectCreate(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    name: str = Field(min_length=2, max_length=160)
    description: str = Field(default="", max_length=2000)
    research_domain: str = Field(default="通用科研", max_length=80)


class ReproductionCreate(BaseModel):
    goal: str = Field(default="复现论文核心实验", min_length=2, max_length=500)
    repository_url: str = Field(default="", max_length=500)
    execution_mode: Literal["smoke", "full"] = "full"
    model_id: str = Field(default="deepseek-v4-flash-0731", min_length=1, max_length=128)


class ReproductionRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: str
    project_id: str
    status: ReproductionStatus
    current_stage: str
    progress: int
    message: str
    goal: str
    model_id: str
    source_mode: str
    repository_url: str
    workspace_path: str
    execution_mode: str
    execution_command: dict[str, Any]
    paper_spec: dict[str, Any]
    code_spec: dict[str, Any]
    environment_spec: dict[str, Any]
    dataset_spec: dict[str, Any]
    result_json: dict[str, Any]
    events: list[dict[str, Any]]
    logs: str
    error_message: str
    created_at: datetime
    updated_at: datetime
    approved_at: datetime | None
    completed_at: datetime | None


class ArtifactRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: str
    kind: ArtifactKind
    filename: str
    media_type: str
    size_bytes: int
    metadata_json: dict[str, Any]
    created_at: datetime


class ProjectRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: str
    name: str
    description: str
    research_domain: str
    created_at: datetime
    updated_at: datetime
    artifacts: list[ArtifactRead] = []


class ProjectListItem(BaseModel):
    id: str
    name: str
    description: str
    research_domain: str
    artifact_count: int
    latest_reproduction_status: str | None
    latest_reproduction_progress: int | None
    updated_at: datetime
