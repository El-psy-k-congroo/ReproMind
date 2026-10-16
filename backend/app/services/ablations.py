"""Conservative, opt-in ablations through author-provided component switches.

No flag is invented and hyperparameter tuning is never labelled an ablation.
"""

import re
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession

from ..config import get_settings
from ..database import AsyncSessionLocal
from ..models import ReproductionJob, ReproductionStatus
from .reproduction_engine import (
    CURRENT_JOB_ID, ReproductionPaused, _begin_work, _end_work, _extract_metrics, _run_process, _slug,
    _write_reproduction_report,
)


_SWITCH = re.compile(
    r"add_argument\(\s*['\"](--(?:no|disable|without)[-_][a-zA-Z][\w-]*)['\"]"
    r"[^)]{0,220}?action\s*=\s*['\"]store_true['\"]",
    re.DOTALL,
)
_NON_COMPONENT = {"--no-cuda", "--no_cuda", "--no-gpu", "--no_gpu", "--no-shuffle", "--no_shuffle"}


def _workspace(job: ReproductionJob) -> Path:
    workspace = Path(job.workspace_path).resolve()
    root = get_settings().paper_projects_dir.resolve()
    if workspace.parent != root or not workspace.name.endswith(f"-{job.project_id[:8]}"):
        raise ValueError("项目工作区路径不合法")
    return workspace


def discover_ablations(job: ReproductionJob) -> list[dict]:
    """Return only switches explicitly implemented by the author's entry script."""
    workspace = _workspace(job)
    result: list[dict] = []
    for command in (job.execution_command or {}).get("commands", [])[:12]:
        argv = [str(value) for value in command.get("argv", [])]
        cwd = Path(str(command.get("cwd", ""))).resolve()
        if not argv or (cwd != workspace and workspace not in cwd.parents):
            continue
        script = next((value for value in argv[1:] if value.endswith(".py")), "")
        source = (cwd / script).resolve() if script else cwd / "__missing__"
        if workspace not in source.parents or not source.is_file() or source.stat().st_size > 256_000:
            continue
        text = source.read_text(encoding="utf-8", errors="replace")
        for flag in dict.fromkeys(_SWITCH.findall(text)):
            if flag in _NON_COMPONENT or flag in argv:
                continue
            result.append({"dataset": str(command.get("dataset", "default")), "flag": flag,
                           "label": flag.removeprefix("--").replace("_", "-")})
    return result[:24]


async def execute_ablation(job_id: str, dataset: str, flag: str) -> None:
    _begin_work(job_id)
    token = CURRENT_JOB_ID.set(job_id)
    try:
        async with AsyncSessionLocal() as session:
            job = await session.get(ReproductionJob, job_id)
            if job is None or job.status != ReproductionStatus.completed:
                return
            workspace = _workspace(job)
            if not any(item["dataset"] == dataset and item["flag"] == flag for item in discover_ablations(job)):
                raise ValueError("作者代码未提供此消融开关")
            command = next(item for item in (job.execution_command or {}).get("commands", [])
                           if str(item.get("dataset", "default")) == dataset)
            argv = [str(value) for value in command["argv"]]
            cwd = Path(str(command["cwd"])).resolve()
            prefix = Path(str((job.environment_spec or {}).get("prefix", ""))).resolve()
            python = prefix / ("python.exe" if __import__("os").name == "nt" else "bin/python")
            if not python.is_file() or Path(argv[0]).resolve() != python.resolve():
                raise ValueError("消融实验只能使用本项目已创建的 Conda 环境")
            if cwd != workspace and workspace not in cwd.parents:
                raise ValueError("消融实验不能离开当前项目工作区")
            result = dict(job.result_json or {})
            result["ablation_active"] = {"dataset": dataset, "flag": flag,
                                         "started_at": datetime.now(timezone.utc).isoformat()}
            job.result_json = result
            await session.commit()
            log = workspace / "runs" / job.id / "telemetry" / f"ablation-{_slug(dataset)}-{flag.removeprefix('--')}.log"
            code, output = await _run_process(
                [*argv, flag], cwd, get_settings().reproduction_timeout_seconds,
                {"REPROMIND_DATA_DIR": str(workspace / "data")}, output_path=log,
            )
            measured = _extract_metrics(output).get("accuracy")
            baseline = next(((item.get("metrics") or {}).get("accuracy")
                             for item in result.get("dataset_results", []) if item.get("dataset") == dataset), None)
            record = {"dataset": dataset, "label": flag, "baseline": baseline,
                      "value": measured, "delta": round(measured - baseline, 6)
                      if measured is not None and baseline is not None else None,
                      "status": "completed" if code == 0 and measured is not None else "failed",
                      "return_code": code, "log_path": str(log),
                      "explanation": "作者仓库原生组件关闭开关；仅对比同一数据集的实测指标。"}
            result = dict(job.result_json or {})
            result["ablation_results"] = [item for item in result.get("ablation_results", [])
                                          if not (item.get("dataset") == dataset and item.get("label") == flag)] + [record]
            result.pop("ablation_active", None)
            job.result_json = result
            job.events = [*(job.events or []), {"stage": "ablation", "progress": job.progress,
                          "message": f"消融 {dataset} {flag}：{record['status']}",
                          "time": datetime.now(timezone.utc).isoformat()}]
            _write_reproduction_report(job, workspace)
            await session.commit()
    except ReproductionPaused:
        return
    except Exception as exc:
        async with AsyncSessionLocal() as session:
            job = await session.get(ReproductionJob, job_id)
            if job:
                result = dict(job.result_json or {})
                result.pop("ablation_active", None)
                result["ablation_results"] = [*result.get("ablation_results", []), {
                    "dataset": dataset, "label": flag, "status": "failed", "explanation": str(exc)[:500],
                }]
                job.result_json = result
                await session.commit()
    finally:
        CURRENT_JOB_ID.reset(token)
        _end_work(job_id)
