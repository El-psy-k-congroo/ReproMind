import hashlib
from collections import defaultdict
from typing import Any

from ..models import Artifact


# 保留足够的证据片段，同时避免每个 Agent 重复发送过长上下文。
PER_SOURCE_LIMIT = 12_000
TOTAL_CONTEXT_LIMIT = 36_000


def serialize_artifacts(artifacts: list[Artifact]) -> list[dict[str, Any]]:
    """将 ORM 实体转为可持久化的 Agent 上下文，避免图节点持有数据库会话。"""
    serialized: list[dict[str, Any]] = []
    consumed = 0
    for artifact in artifacts:
        remaining = max(0, TOTAL_CONTEXT_LIMIT - consumed)
        text = artifact.content_text[: min(PER_SOURCE_LIMIT, remaining)]
        consumed += len(text)
        serialized.append(
            {
                "id": artifact.id,
                "kind": artifact.kind.value,
                "filename": artifact.filename,
                "content": text,
                "metadata": artifact.metadata_json,
                "sha256": hashlib.sha256(artifact.content_text.encode("utf-8")).hexdigest(),
            }
        )
    return serialized


def build_evidence_context(artifacts: list[dict[str, Any]]) -> dict[str, Any]:
    by_kind: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in artifacts:
        by_kind[item["kind"]].append(item)
    return {
        "artifact_count": len(artifacts),
        "available_kinds": sorted(by_kind),
        "files_by_kind": {
            kind: [item["filename"] for item in items] for kind, items in by_kind.items()
        },
        "evidence_manifest": [
            {
                "id": item["id"],
                "kind": item["kind"],
                "filename": item["filename"],
                "sha256": item["sha256"],
            }
            for item in artifacts
        ],
    }


def source_bundle(artifacts: list[dict[str, Any]], kinds: set[str]) -> str:
    return "\n\n".join(
        f"SOURCE={item['filename']}; KIND={item['kind']}\n{item['content']}"
        for item in artifacts
        if item["kind"] in kinds and item["content"]
    )
