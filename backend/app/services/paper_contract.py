"""Fail-closed validation of the experiment target and selected method routes."""

import re


def vague_spec(spec: dict) -> bool:
    if not isinstance(spec, dict):
        return True
    for key in ("datasets", "metrics"):
        values = spec.get(key)
        if key == 'metrics' and values == [] and spec.get('execution_phase') == 'sampling' and (spec.get('spec_validation') or {}).get('scope') == 'author_source_workflow':
            continue  # Sampling artifacts are checked separately from paper metrics.
        if not isinstance(values, list) or not values or any(not isinstance(value, str) or not value.strip() for value in values):
            return True
        if any(re.search(r"未.{0,6}(?:明确|指定|提供)|多个数据集|多个数据|various datasets|unspecified", value, re.I) for value in values):
            return True
    return False


def validate_scope(spec: dict, paper: str, goal: str) -> dict:
    scope = spec.get("reproduction_scope")
    if not isinstance(scope, dict) or not isinstance(scope.get("required"), list) or not scope["required"]:
        raise ValueError("尚未明确本次必须实现的论文方法或路线")
    if scope.get("goal") and scope["goal"] != goal:
        raise ValueError("旧复现范围不适用于当前用户目标，需要重新核验")
    names = set()
    result = {"goal": goal, "required": [], "optional": []}
    for kind in ("required", "optional"):
        rows = scope.get(kind, [])
        if not isinstance(rows, list) or len(rows) > 12:
            raise ValueError("复现范围必须是有界的组件列表")
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError("复现范围组件格式错误")
            name, quote = row.get("name"), row.get("paper_evidence")
            if (not isinstance(name, str) or not name.strip() or len(name) > 120
                    or not isinstance(quote, str) or not quote.strip() or len(quote) > 1200 or quote not in paper):
                raise ValueError("复现范围组件必须附带可回查论文的短原文证据")
            if name.strip().lower() in names:
                raise ValueError("必需路线与可选路线重复或冲突")
            names.add(name.strip().lower())
            result[kind].append({"name": name.strip(), "paper_evidence": quote})
    return result


def spec_ready(spec: dict, paper: str, goal: str) -> bool:
    if vague_spec(spec):
        return False
    try:
        validate_scope(spec, paper, goal)
    except ValueError:
        return False
    return True
