from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "ReproMind API"
    app_env: str = "development"
    app_host: str = "127.0.0.1"
    app_port: int = 8010
    deployment_mode: Literal["local", "server"] = "local"
    public_origin: str = ""
    session_secure_cookie: bool = True
    execution_sandbox: bool = False
    default_tenant_id: str = "tenant_default"

    # 与 EduAgent 一致：支持拆分的 PostgreSQL 配置，也允许 DATABASE_URL 覆盖。
    database_url: str = "sqlite+aiosqlite:///./data/repromind.db"
    db_host: str = "127.0.0.1"
    db_port: int = 5432
    db_name: str = "repromind"
    db_user: str = "root"
    db_password: str = ""
    langgraph_checkpoint_url: str = ""

    redis_url: str = ""
    minio_endpoint: str = ""
    minio_access_key: str = ""
    minio_secret_key: str = ""
    minio_bucket: str = "repromind"
    milvus_uri: str = ""
    milvus_db_name: str = "repromind"
    milvus_collection_name: str = "research_evidence"
    upload_dir: Path = Path("./data/uploads")
    paper_projects_dir: Path = Path("./data/paper-projects")
    max_upload_mb: int = 40
    git_clone_timeout_seconds: int = 45
    reproduction_timeout_seconds: int = 900
    dataset_download_timeout_seconds: int = Field(default=1800, ge=1, le=86400)
    dataset_download_read_timeout_seconds: int = Field(default=180, ge=1, le=86400)
    dataset_download_connect_timeout_seconds: int = Field(default=60, ge=1, le=86400)
    dataset_max_download_bytes: int = Field(default=12_000_000_000, ge=1)
    dataset_max_extract_bytes: int = Field(default=50_000_000_000, ge=1)
    dataset_max_archive_files: int = Field(default=250_000, ge=1)
    reproduction_use_system_proxy: bool = True
    experiment_timeout_seconds: int = 43200  # Full multi-fold training may take hours.
    reproduction_log_chars: int = 60_000
    hf_endpoint: str = "https://hf-mirror.com"
    model_provider: str = "openai_compatible"
    llm_api_key: str = Field(
        default="",
        validation_alias=AliasChoices("LLM_API_KEY", "DEEPSEEK_API_KEY"),
    )
    llm_base_url: str = "https://api.chatanywhere.tech/v1"
    llm_model_chat: str = "gpt-4o-mini"
    llm_model_coder: str = "gpt-4o-mini"
    llm_model_reasoning: str = "gpt-4o-mini"
    llm_timeout_seconds: int = 60
    # Includes reasoning for models that cannot turn thinking off. Individual
    # lightweight requests still select much smaller visible answer budgets.
    llm_max_tokens_per_call: int = 8192
    moma_api_key: str = ""
    moma_base_url: str = "https://zhenze-huhehaote.cmecloud.cn/v1"
    moma_model_router: str = ""
    moma_model_chat: str = ""
    moma_model_coder: str = ""
    moma_model_reasoning: str = ""
    model_routing_path: Path = Path('./data/model-routing.json')

    model_config = SettingsConfigDict(
        env_file=("../.env", ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @property
    def llm_enabled(self) -> bool:
        if self.model_provider.lower() == "moma":
            return bool(self.moma_api_key.strip() and self.moma_model_chat.strip())
        return bool(self.llm_api_key.strip())

    @property
    def async_database_url(self) -> str:
        url = self.database_url.strip()
        if url:
            if url.startswith("postgresql+psycopg://"):
                return url.replace("postgresql+psycopg://", "postgresql+asyncpg://", 1)
            if url.startswith("postgresql://"):
                return url.replace("postgresql://", "postgresql+asyncpg://", 1)
            if url.startswith("sqlite:///"):
                return url.replace("sqlite:///", "sqlite+aiosqlite:///", 1)
            return url
        return (
            f"postgresql+asyncpg://{self.db_user}:{self.db_password}"
            f"@{self.db_host}:{self.db_port}/{self.db_name}"
        )


@lru_cache
def get_settings() -> Settings:
    return Settings()
