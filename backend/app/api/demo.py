import hashlib

from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from ..config import get_settings
from ..database import get_db
from ..models import Artifact, ArtifactKind, Project
from ..schemas import ProjectRead
from ..services.file_parser import parse_bytes


router = APIRouter(prefix="/demo", tags=["demo"])
DEMO_PROJECT_NAME = "Transformer Base 复现实验"

PAPER = """Attention Is All You Need — Transformer base configuration (paraphrased demo evidence)
The base model uses 6 encoder and decoder layers, d_model = 512 and 8 attention heads.
Dropout = 0.1 is applied to residual connections. Adam training uses 4000 warmup steps.
The token batch size is 25000. Translation quality is evaluated with BLEU.
"""

CONFIG = """model:
  layers: 6
  d_model: 512
  num_heads: 8
  dropout: 0.2
training:
  warmup_steps: 2000
  token_batch_size: 25000
  seed: 42
"""

RESULTS = """method,BLEU,BLEU_std
Transformer-base-reference,27.3,0.4
Local-reproduction,26.7,0.0001
"""

CODE = """import random
import numpy as np

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)

def train(config):
    set_seed(config['training']['seed'])
    return train_translation_model(config)
"""


@router.post("", response_model=ProjectRead)
async def create_demo(db: AsyncSession = Depends(get_db)) -> Project:
    existing = await db.scalar(
        select(Project)
        .where(
            Project.tenant_id == get_settings().default_tenant_id,
            Project.name == DEMO_PROJECT_NAME,
        )
        .order_by(Project.created_at.desc())
        .options(selectinload(Project.artifacts), selectinload(Project.analyses))
        .limit(1)
    )
    if existing:
        return existing

    project = Project(
        tenant_id=get_settings().default_tenant_id,
        name=DEMO_PROJECT_NAME,
        description="以经典 Transformer 论文为背景，演示论文、代码、配置与实验结果证据管理。",
        research_domain="人工智能 · 自然语言处理",
        status="demo",
    )
    db.add(project)
    await db.flush()
    samples = [
        (ArtifactKind.paper, "paper.txt", PAPER),
        (ArtifactKind.code, "source.py", CODE),
        (ArtifactKind.config, "config.yaml", CONFIG),
        (ArtifactKind.result, "results.csv", RESULTS),
    ]
    for kind, filename, text in samples:
        raw = text.encode("utf-8")
        content, metadata = parse_bytes(filename, raw)
        db.add(
            Artifact(
                tenant_id=project.tenant_id,
                project_id=project.id,
                kind=kind,
                filename=filename,
                media_type="text/plain",
                size_bytes=len(raw),
                storage_provider="demo",
                storage_path=f"demo://{filename}",
                sha256=hashlib.sha256(raw).hexdigest(),
                parse_status="completed",
                content_text=content,
                metadata_json=metadata,
            )
        )
    await db.flush()
    return await db.scalar(
        select(Project)
        .where(Project.id == project.id)
        .options(selectinload(Project.artifacts), selectinload(Project.analyses))
        .execution_options(populate_existing=True)
    )
