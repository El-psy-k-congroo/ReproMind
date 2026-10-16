"""Validate review findings independently; one bad quote cannot erase good findings."""

from .code_validation import grounded_issue
from .paper_evidence import core_implementation_concerns


def inspect_review(data: dict, files: dict[str, str], paper: str, pending: list[dict], required_names=None) -> dict:
    if not isinstance(data, dict) or not isinstance(data.get("issues"), list):
        raise ValueError("方法审查响应格式不完整")
    if len(data["issues"]) > 6:
        raise ValueError("一次审查最多处理六项问题，请优先报告阻塞项")
    confirmed, rejected, checked = [], [], {}
    for item in data["issues"]:
        try:
            if not isinstance(item, dict):
                raise ValueError("方法审查 issues 必须是对象列表")
            confirmed.append(grounded_issue(item, files, paper))
        except ValueError as exc:
            rejected.append({"kind": "unverified_finding", "reason": str(exc), "finding": item})
    checks = data.get("issue_checks", [])
    if not isinstance(checks, list):
        rejected.append({"kind": "invalid_checks", "reason": "issue_checks 必须是列表"})
        checks = []
    # Duplicate or fabricated IDs must never resolve an existing issue.
    for row in pending:
        matches = [item for item in checks if isinstance(item, dict) and item.get("id") == row["id"]]
        try:
            if len(matches) != 1 or matches[0].get("status") not in {"resolved", "open"} or not isinstance(
                    matches[0].get("reason"), str) or not matches[0]["reason"].strip():
                raise ValueError("未逐项复核上轮问题；不能把遗漏问题视为已解决")
            check = dict(matches[0])
            if check["status"] == "open":
                current = grounded_issue({**row, "code_evidence": check.get("code_evidence", row["code_evidence"])}, files, paper)
                check["code_evidence"] = current["code_evidence"]
            checked[row["id"]] = check
        except ValueError as exc:
            rejected.append({"kind": "unverified_check", "id": row["id"], "reason": str(exc)})
    limits = data.get("limitations", [])
    if isinstance(limits, str):
        limits = [limits]
    if not isinstance(limits, list) or any(not isinstance(value, str) for value in limits):
        rejected.append({"kind": "invalid_limitations", "reason": "limitations 必须是文本或文本列表"})
        limits = []
    concerns = core_implementation_concerns(limits, required_names)
    rejected.extend({"kind": "unlocated_core_concern", "reason": "核心实现缺失不能作为限制放行；请提供可定位证据或澄清：" + value,
                     "finding": value} for value in concerns)
    resources = data.get("required_resources", [])
    if not isinstance(resources, list) or any(not isinstance(value, dict) for value in resources):
        rejected.append({"kind": "invalid_resources", "reason": "required_resources 必须是对象列表"})
        resources = []
    return {"confirmed": confirmed, "rejected": rejected, "checked": checked,
            "limitations": limits, "required_resources": resources}
