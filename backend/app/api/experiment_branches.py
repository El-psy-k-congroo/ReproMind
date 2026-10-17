"""Project-scoped, reviewed multi-file code revisions and opt-in reruns."""

import asyncio
import ast
import difflib
import hashlib
import json
import re
import shutil
import subprocess
from collections import Counter
from itertools import islice
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID, uuid4

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..config import get_settings
from ..core.model_router import ModelRole, get_model_router
from ..database import get_db
from ..models import ReproductionJob, ReproductionStatus
from ..services.reproduction_engine import (
    _environment_python, _named_environment_prefix, _environment_owned_by_job, _align_dataset_commands,
    _hardware_plan, _infer_python_dependencies, _execution_command,
    _native_path, _repository_manifest,
    execute_reproduction, has_active_work,
)
from .models_catalog import validate_model
from ..services.tuning import build_tuning_plans, recovery_datasets
from ..services.runtime_evidence import probe_framework, probe_interpreter, execution_logs, device_log_evidence
from ..services.environment_recovery import pytorch_cuda_plan
from ..services.dataset_discovery import inspect_local_datasets
from ..services.experiment_scope import revision_scope
from ..services.run_parameters import inherit_dataset_parameters, changed_cli_parameters, historical_parameter_context
from ..services.gpu_preparation import source_bodies, source_framework, tensorflow_gpu_plan, tensorflow_compat_changes, requested_framework, gpu_preparation_state
from ..services.editable_source import editable_file, requirements_file, validate_requirements

router = APIRouter(prefix="/projects/{project_id}/reproductions", tags=["experiment-branches"])
ACTIVE = {ReproductionStatus.queued, ReproductionStatus.running, ReproductionStatus.executing}
_REVISION_RUN_LOCKS: dict[tuple[str, str, str], asyncio.Lock] = {}


class BranchPreviewRequest(BaseModel):
    file: str = Field(default="", max_length=240)
    instruction: str = Field(min_length=8, max_length=1600)
    model_id: str = Field(default="", max_length=128)
    preparation_only: bool = False


class BranchConfirmRequest(BaseModel):
    proposal_id: str


class RevisionRunRequest(BaseModel):
    proposal_id: str
    isolated_copy: bool = False
    datasets: list[str] = Field(default_factory=list, max_length=20)
    gpu_environment_migration: bool = False


class RerunRequest(BaseModel):
    datasets: list[str] = Field(min_length=1, max_length=20)
    gpu_requested: bool = False
    isolated_copy: bool = True
    tuning_budget: int = Field(default=0, ge=0, le=12)
    gpu_environment_migration: bool = False


async def _job(db: AsyncSession, project_id: str, job_id: str) -> ReproductionJob:
    job = await db.scalar(select(ReproductionJob).where(
        ReproductionJob.id == job_id, ReproductionJob.project_id == project_id,
    ))
    if not job:
        raise HTTPException(status_code=404, detail="原实验不存在")
    return job


def _scope(job: ReproductionJob) -> tuple[Path, Path]:
    workspace = Path(job.workspace_path).resolve()
    root = get_settings().paper_projects_dir.resolve()
    if workspace.parent != root or not workspace.name.endswith(f"-{job.project_id[:8]}"):
        raise HTTPException(status_code=409, detail="项目工作区未通过路径校验")
    source = Path(str((job.execution_command or {}).get("pythonpath")
                      or (job.code_spec or {}).get("source_path") or "")).resolve()
    if not source.is_dir() or workspace not in source.parents:
        raise HTTPException(status_code=409, detail="原实验没有可隔离的源码目录")
    return workspace, source


def _code_file(source: Path, relative: str) -> Path:
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts or not editable_file(relative):
        raise HTTPException(status_code=400, detail="只能修改当前实验中的 Python、YAML、TOML 或 JSON 文件")
    target = (source / path).resolve()
    if source not in target.parents or not target.is_file() or target.is_symlink():
        raise HTTPException(status_code=400, detail="源码文件不存在或越过项目边界")
    if target.stat().st_size > 60_000:
        raise HTTPException(status_code=409, detail="文件超过 60 KB；请先缩小需要修改的模块")
    return target


def _select_code_file(source: Path, instruction: str) -> Path:
    terms = [item.lower() for item in re.findall(r"[A-Za-z_][A-Za-z0-9_]{2,}", instruction)][:16]
    ranked: list[tuple[int, Path]] = []
    for path in islice(source.rglob("*.py"), 250):
        if (not path.is_file() or path.is_symlink() or path.stat().st_size > 60_000
                or any(part in {".git", ".conda", ".venv", "__pycache__", "data", "data_set"}
                       for part in path.relative_to(source).parts)):
            continue
        body = path.read_text(encoding="utf-8", errors="replace").lower()
        name = path.name.lower()
        score = sum(8 for term in terms if term in name) + sum(4 * min(3, body.count(term)) for term in terms)
        score += 2 if name in {"train.py", "run.py", "main.py"} else 0
        ranked.append((score, path))
    if not ranked:
        raise HTTPException(status_code=409, detail="没有可供模型修改的项目 Python 文件")
    return max(ranked, key=lambda row: row[0])[1]


def _candidate_code_files(source: Path, instruction: str, selected: str = "") -> list[Path]:
    """Bound token use while allowing one request to change several related modules."""
    terms = [item.lower() for item in re.findall(r"[A-Za-z_][A-Za-z0-9_]{2,}", instruction)][:20]
    migration = bool(requested_framework(instruction))
    if migration:
        terms += ['tensorflow', 'torch', 'placeholder', 'session']
    if any(term in instruction.lower() for term in ("gpu", "cuda", "显卡", "设备")):
        terms += ["cuda", "device", "torch", "cpu", "to("]
    ranked: list[tuple[int, Path]] = []
    for path in islice(source.rglob("*.py"), 250):
        if (not path.is_file() or path.is_symlink() or path.stat().st_size > 45_000
                or any(part in {".git", ".conda", ".venv", "__pycache__", "data", "data_set"}
                       for part in path.relative_to(source).parts)):
            continue
        body = path.read_text(encoding="utf-8", errors="replace").lower()
        score = sum(12 if term in path.name.lower() else min(5, body.count(term)) * 2 for term in terms)
        score += 4 if path.name.lower() in {"run.py", "train.py", "main.py"} else 0
        ranked.append((score, path))
    ranked.sort(key=lambda item: (-item[0], item[1].as_posix()))
    limit = 12 if migration else 6
    chosen = [path for _, path in ranked[:limit]]
    if selected:
        explicit = _code_file(source, selected)
        chosen = [explicit, *[path for path in chosen if path != explicit]][:limit]
    return chosen


def _gpu_preview_issues(source: Path, revisions: dict[str, str], entrypoints=None) -> list[str]:
    """Reject an obviously CPU-bound GPU proposal; this is not a runtime proof."""
    from ..services.source_interfaces import interface_issues
    issues: list[str] = interface_issues(source, revisions)
    try:
        bodies = source_bodies(source, revisions,entrypoints)
    except ValueError as exc:
        return [str(exc)]
    framework = source_framework(bodies)
    if framework == 'mixed':
        return ['源码混用 TensorFlow 和 PyTorch；需先完成模型、层、损失与训练入口的一致性迁移，不能视为 GPU 适配已通过']
    if framework == 'tensorflow':
        # TensorFlow places graph operations itself; torch.cuda/.to are unrelated.
        for name, body in bodies.items():
            tree = ast.parse(body)
            # Profiling must observe an existing step, not silently add training.
            if 'RunMetadata' in body:
                def optimizer_counts(statements):
                    counts = Counter()
                    for statement in statements:
                        if isinstance(statement, ast.If):
                            left = optimizer_counts(statement.body)
                            right = optimizer_counts(statement.orelse)
                            counts.update({key: max(left[key], right[key]) for key in left.keys() | right.keys()})
                        else:
                            for call in ast.walk(statement):
                                if isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute) and call.func.attr == 'run' and call.args:
                                    counts.update(ast.dump(child) for child in ast.walk(call.args[0])
                                                  if isinstance(child, ast.Attribute) and child.attr in {'opt_op', 'train_op', 'training_op'})
                    return counts
                for loop in ast.walk(tree):
                    if isinstance(loop, (ast.For, ast.While)) and any(count > 1 for count in optimizer_counts(loop.body).values()):
                        issues.append(f'{name}: 设备采样重复执行同一优化器；应把 options/run_metadata 附加到原有 Session.run，并用 if/else 保证每轮只执行原训练步骤')
                    if isinstance(loop, (ast.For, ast.While)):
                        for statement in loop.body:
                            if not isinstance(statement, (ast.Assign, ast.Expr)):
                                continue
                            for call in ast.walk(statement):
                                if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute) and call.func.attr == 'run'):
                                    continue
                                options = next((kw.value for kw in call.keywords if kw.arg == 'options'), None)
                                if (options is not None and any(isinstance(node, ast.Attribute) and node.attr == 'FULL_TRACE' for node in ast.walk(options))
                                        and not any(isinstance(node, ast.IfExp) for node in ast.walk(options))):
                                    issues.append(f'{name}: 每轮都执行 FULL_TRACE 会增加训练开销；在首个原有训练步骤条件采样，其他轮次保持原 Session.run，不额外训练')
                for statement in tree.body:
                    if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.For, ast.While)):
                        continue
                    for call in ast.walk(statement):
                        if (isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
                                and call.func.attr == 'run' and call.args
                                and any(isinstance(child, ast.Attribute) and child.attr in {'opt_op', 'train_op', 'training_op'}
                                        for child in ast.walk(call.args[0]))):
                            issues.append(f'{name}: 设备采样在原训练循环之外额外执行了优化器，改变实验协议；将 RunMetadata 附加到首个原有训练步骤，不得额外训练或重复初始化')
            for node in ast.walk(tree):
                if not isinstance(node, ast.With):
                    continue
                device_context = any(isinstance(item.context_expr, ast.Call)
                                     and isinstance(item.context_expr.func, ast.Attribute)
                                     and item.context_expr.func.attr == 'device' for item in node.items)
                if device_context and any(isinstance(child, ast.Call) and isinstance(child.func, ast.Attribute)
                                          and child.func.attr == 'run' for child in ast.walk(node)):
                    issues.append(f'{name}: tf.device 包裹 Session.run 不会改变已建图算子的设备；请在建图阶段设置，或读取真实 RunMetadata 的设备')
            if (re.search(r'print\([^\n]*(?:training|train|model)[^\n]*(?:gpu|cuda)', body, re.I)
                    and not re.search(r'RunMetadata|log_device_placement\s*=\s*True', body)):
                issues.append(f'{name}: 固定字符串 GPU 声明不是训练设备证据；使用 RunOptions(trace_level=FULL_TRACE) 与 RunMetadata.step_stats.dev_stats 或实际设备放置日志，不得打印臆测设备')
            if re.search(r'(?:tf|tensorflow)\.device\s*\(\s*[\'\"]/?(?:device:)?cpu', body, re.I):
                issues.append(f'{name} 显式指定 CPU 设备；需审查是否限制训练运算')
            if re.search(r'(?:device_count\s*=\s*\{[^}]*[\'\"]GPU[\'\"]\s*:\s*0|CUDA_VISIBLE_DEVICES[\'\"]\s*\]\s*=\s*[\'\"]-1)', body):
                issues.append(f'{name} 显式禁用了 GPU')
        return issues
    if framework == 'unknown' and not any(re.search(r'torch\.|\.to\(|nn\.', body) for body in bodies.values()):
        return ['未识别可静态检查的 GPU 框架；不会套用 PyTorch 规则或直接放行']
    contents = {Path(name).name.lower(): body for name, body in bodies.items()
                if Path(name).name.lower() in {'run.py', 'train.py', 'main.py', 'model.py', 'models.py'}}
    entry = (bodies.get(entrypoints[0],'') if entrypoints else
             next((contents[name] for name in ("run.py", "train.py", "main.py") if name in contents), ""))
    model = contents.get("model.py", contents.get('models.py', ''))
    device_source = '\n'.join(bodies.values())
    if entry and not re.search(r"torch\.cuda\.is_available|torch\.device\s*\(\s*['\"]cuda|\bdevice\s*=\s*['\"]cuda|\.cuda\s*\(", device_source):
        issues.append("训练入口没有显式检查或指定 CUDA 设备")
    if entry and not re.search(r"\.to\s*\(\s*(?:(?:self\.)?device|['\"]cuda)|\.cuda\s*\(|\bdevice_map\s*=", device_source):
        issues.append("训练入口没有将模型或批次迁移到 CUDA")
    if model and re.search(r"\.to\s*\(\s*['\"]cpu['\"]\s*\)", model):
        issues.append("模型仍将张量固定迁移到 CPU")
    if model and re.search(r"self\.adj_(?:entity|relation)\[entities\[h\]\]", model):
        issues.append("邻接表仍直接用可能位于 GPU 的张量索引")
    local_layer = re.search(r"^[ \t]*(?!self\.)\w+[ \t]*=[ \t]*nn\.Linear\([^\n]+\)[ \t]*$", model, re.M)
    if local_layer:
        issues.append("模型中仍临时创建未注册的线性层：" + local_layer.group(0).strip()[:90])
    return issues


def _complete_known_gpu_patches(body: str, filename: str) -> tuple[str, list[dict]]:
    """Propose narrowly recognizable device fixes; never writes author files."""
    changes: list[dict] = []
    updated = body
    for method, next_method, device_name in (
        ("get_neighbors_cell", "get_neighbors_drug", "cells"),
        ("get_neighbors_drug", "sum_aggregator", "drugs"),
    ):
        match = re.search(rf"    def {method}\(.*?(?=    def {next_method}\()", updated, re.S)
        if not match:
            continue
        old = match.group(0)
        new = old.replace("self.adj_entity[entities[h]]",
                          "self.adj_entity[entities[h].detach().cpu().numpy()]")
        new = new.replace("self.adj_relation[entities[h]]",
                          "self.adj_relation[entities[h].detach().cpu().numpy()]")
        new = re.sub(r"\.to\(\s*['\"]cpu['\"]\s*\)", f".to({device_name}.device)", new)
        if new != old:
            updated = _apply_text(updated, old, new, filename)
            changes.append({"file": filename, "old_text": old, "new_text": new,
                            "rationale": "邻接表在 CPU 索引，所得邻居张量迁回当前输入设备"})
    local = re.search(
        r"(?m)^([ \t]*)([A-Za-z_]\w*)[ \t]*=[ \t]*nn\.Linear\(self\.dim,[ \t]*self\.dim\)"
        r"(?:\.to\([^\n)]+\))?[ \t]*\n"
        r"[ \t]*emb_i[ \t]*=[ \t]*\2\(emb_i\.sum\(dim=1\)\)", updated,
    )
    if local:
        anchor = re.search(r"(?m)^[ \t]*self\.dim = [^\n]+$", updated)
        if anchor and "self.message_projection" not in updated:
            old = anchor.group(0)
            new = old + "\n        self.message_projection = nn.Linear(self.dim, self.dim)"
            updated = _apply_text(updated, old, new, filename)
            changes.append({"file": filename, "old_text": old, "new_text": new,
                            "rationale": "将投影层注册为模型参数，随模型迁移到 CUDA 并参与优化"})
        if "self.message_projection" in updated:
            old = local.group(0)
            new = local.group(1) + "emb_i = self.message_projection(emb_i.sum(dim=1))"
            updated = _apply_text(updated, old, new, filename)
            changes.append({"file": filename, "old_text": old, "new_text": new,
                            "rationale": "前向传播复用已注册的投影层"})
    return updated, changes


def _apply_text(original: str, old: str, new: str, filename: str) -> str:
    whole_file = old == original and len(old) <= 60_000 and len(new) <= 60_000
    if not old.strip() or (not whole_file and (len(old) > 6000 or len(new) > 10000)) or original.count(old) != 1:
        raise HTTPException(status_code=422, detail="模型变更片段为空、过大或不唯一，未修改代码")
    updated = original.replace(old, new, 1)
    try:
        import tomllib
        import yaml
        suffix = Path(filename).suffix.lower()
        if suffix == '.py': compile(updated, filename, 'exec')
        elif suffix in {'.yaml', '.yml'}: yaml.safe_load(updated)
        elif suffix == '.toml': tomllib.loads(updated)
        elif suffix == '.json': json.loads(updated)
        elif requirements_file(filename): validate_requirements(updated)
    except (SyntaxError, ValueError, yaml.YAMLError) as exc:
        raise HTTPException(status_code=422, detail=f"候选文件语法错误：{exc}") from exc
    return updated


async def _ensure_idle(db: AsyncSession, project_id: str) -> None:
    jobs = (await db.scalars(select(ReproductionJob).where(ReproductionJob.project_id == project_id))).all()
    if any(item.status in ACTIVE or has_active_work(item.id) or
           bool((item.result_json or {}).get("ablation_active") or (item.result_json or {}).get('tuning_active')) for item in jobs):
        raise HTTPException(status_code=409, detail="项目仍有实验正在运行或停止中；请先暂停并等待进程退出")


def _copy_bounded(source: Path, target: Path, excluded_roots=()) -> None:
    source,target=_native_path(source),_native_path(target)
    if target.exists():
        raise HTTPException(status_code=409, detail="实验分支目录已存在")
    total = 0
    count = 0
    for path in source.rglob("*"):
        if path.relative_to(source).parts[0] in excluded_roots:
            continue
        if any(part in {".git", ".conda", ".venv", "__pycache__"} for part in path.relative_to(source).parts):
            continue
        if path.is_symlink():
            raise HTTPException(status_code=409, detail="源码含符号链接，不能安全复制为独立实验")
        if path.is_file():
            count += 1
            total += path.stat().st_size
        if count > 50000 or total > 2_000_000_000:
            raise HTTPException(status_code=409, detail="源码或数据超过分支复制上限；请先精简实验输入")
    try:
        ordinary_ignore = shutil.ignore_patterns(".git", ".conda", ".venv", "__pycache__")
        def ignore(directory, names):
            return set(ordinary_ignore(directory, names)) | (set(names) & set(excluded_roots)
                if Path(directory) == source else set())
        shutil.copytree(source, target, ignore=ignore)
    except (OSError,shutil.Error) as exc:
        raise HTTPException(status_code=422,detail='独立源码复制失败，原始源码未修改：'+str(exc)[:400]) from exc


def _materialized_cache_exclusions(base, source, datasets):
    """Omit only the platform's download staging cache after checking its outputs.

    Actual dataset files are copied in the independent source tree. Unknown or
    partially materialized data retains the conservative full-copy behavior.
    """
    source = _native_path(source)
    requests = (base.dataset_spec or {}).get('download_requests') or []
    for dataset in datasets:
        request = next((row for row in requests if row.get('dataset') == dataset), None)
        if not request or request.get('download_status') != 'ready' or not request.get('installed_files'):
            return ()
        for name in request['installed_files']:
            relative = Path(str(name))
            path = (source / relative).resolve()
            if (relative.is_absolute() or '..' in relative.parts or source not in path.parents
                    or path.is_symlink() or not path.is_file() or not path.stat().st_size):
                return ()
    # Never omit data explicitly referenced by an existing command.
    cache = str(Path(base.workspace_path).resolve() / 'data' / 'author').replace('\\', '/').lower()
    if any(cache in str(value).replace('\\', '/').lower()
           for row in (base.execution_command or {}).get('commands', []) for value in row.get('argv', [])):
        return ()
    return ('author',) if datasets else ()


def _remap_command(command: dict, source: Path, branch_source: Path, branch_data: Path,
                   workspace: Path | None = None) -> dict:
    result = json.loads(json.dumps(command))
    workspace = workspace or source.parent.parent
    def remap(value: str) -> str:
        path = Path(value).resolve()
        if path != source and source not in path.parents:
            raise HTTPException(status_code=409, detail=f"原运行目录不在可隔离源码内：{value}")
        return str(branch_source / path.relative_to(source))
    result["cwd"] = remap(str(result.get("cwd") or ""))
    result["pythonpath"] = str(branch_source)
    result["data_dir"] = str(branch_data)
    def remap_arg(value: str) -> str:
        remaining = value.replace(str(source), "").replace(str(workspace / "data"), "")
        if str(workspace) in remaining:
            raise HTTPException(status_code=409, detail="命令仍指向原项目中的其他可写路径，不能保证实验隔离")
        return value.replace(str(source), str(branch_source)).replace(str(workspace / "data"), str(branch_data))
    for item in result.get("commands", []):
        item["cwd"] = remap(str(item.get("cwd") or ""))
        if item.get("pythonpath"):
            item["pythonpath"] = remap(str(item["pythonpath"]))
        item["argv"] = [remap_arg(str(value)) for value in item.get("argv", [])]
    result["argv"] = result["commands"][0]["argv"] if result.get("commands") else [remap_arg(str(value)) for value in result.get("argv", [])]
    result["display"] = "\n".join(
        f"[{item.get('dataset', 'default')}] {' '.join(item['argv'])}" for item in result.get("commands", [])
    )[:4000]
    return result


def _branch_environment(base: ReproductionJob, branch_id: str, gpu_preparation=False) -> dict:
    original = base.environment_spec or {}
    conda = str(original.get("conda_executable") or "")
    name = str(original.get("environment_name") or "")
    prefix = Path(str(original.get("prefix") or "")).resolve()
    gpu_recovery=(gpu_preparation and (original.get('dependencies_installed') is True
                  or str(getattr(base,'error_message','') or '').startswith('RuntimeError: 源码要求 GPU，但独立环境中 torch.cuda.is_available() 未通过')))
    if (not conda or not Path(conda).is_file() or not name.endswith(f"-{base.project_id[:8]}")
            or prefix != _named_environment_prefix(conda, name).resolve()
            or not _environment_python(prefix).is_file() or not ((prefix / ".repromind-ready").is_file() or gpu_recovery)):
        raise HTTPException(status_code=409, detail="原实验环境尚未构建完成，不能安全创建独立分支环境")
    if not _environment_owned_by_job(base, prefix):
        raise HTTPException(status_code=409, detail="当前环境的项目归属无法验证，不会修改该环境")
    return {**original, "ready": True, "create_commands": [], "install_commands": [],
            'gpu_recovery_pending':bool(gpu_recovery),
            "recheck_on_run": True, "isolation": "后续实验复用本项目原有 Conda 环境；运行前检查依赖与 GPU"}


def _fresh_run_command(base: ReproductionJob, source: Path, run_workspace: Path,
                       environment: dict, datasets: list[str], changed_parameters=()) -> dict:
    """Rebuild static dataset variants from current source, never reuse old variants."""
    original = base.execution_command or {}
    entrypoints = original.get("entrypoint_candidates") or []
    if not entrypoints:
        raise HTTPException(status_code=409, detail="原实验没有可验证的训练入口")
    # A fresh run still needs the author's documented flags and output roles.
    # Keep the selected entries; rescanning usage must not switch experiments.
    manifest = {**_repository_manifest(source), "entrypoints": entrypoints}
    planned = _execution_command(
        source, manifest, "full", environment,
        base.paper_spec or {}, base.dataset_spec or {}, run_workspace,
    )
    requested = {label.casefold() for label in datasets if label.strip()}
    if requested:
        available = {str(item.get("dataset") or "").casefold() for item in planned.get("commands", [])}
        unknown = requested - available
        if unknown:
            raise HTTPException(status_code=422, detail=f"没有已验证的可执行数据集：{', '.join(sorted(unknown))}")
        planned["commands"] = [item for item in planned["commands"]
                               if str(item.get("dataset") or "").casefold() in requested]
    if not planned.get("commands"):
        raise HTTPException(status_code=409, detail="没有可执行的数据集命令")
    # Show the real author data variant in the approval preview as well as
    # checking it again before training. Never show an alias known to be absent.
    holder = SimpleNamespace(execution_command=planned)
    _align_dataset_commands(holder)
    planned = holder.execution_command
    try:
        planned = inherit_dataset_parameters(base, planned, source, changed_parameters,
            fresh_output_roots=(run_workspace, run_workspace.parents[1] / 'outputs' / run_workspace.name[:8]))
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    planned["argv"] = planned["commands"][0]["argv"]
    planned["cwd"] = planned["commands"][0]["cwd"]
    planned["display"] = "\n".join(
        f"[{item.get('dataset', 'default')}] {' '.join(item['argv'])}"
        for item in planned["commands"]
    )[:4000]
    planned["adaptations"] = [item for item in planned.get("adaptations", [])
                              if not requested or str(item.get("dataset") or "").casefold() in requested]
    return planned


@router.get("")
async def list_reproduction_runs(project_id: str, db: AsyncSession = Depends(get_db)) -> list[dict]:
    jobs = (await db.scalars(select(ReproductionJob).where(
        ReproductionJob.project_id == project_id,
    ).order_by(ReproductionJob.created_at.desc()).limit(30))).all()
    return [{"id": item.id, "status": item.status.value, "created_at": item.created_at,
             "branch": (item.result_json or {}).get("branch"),
             "revision": (item.result_json or {}).get("revision"),
             "rerun": (item.result_json or {}).get("rerun"),
             'auxiliary_active': bool((item.result_json or {}).get('ablation_active') or (item.result_json or {}).get('tuning_active')),
             "message": item.message} for item in jobs]


@router.get("/{job_id}/branch-files")
async def branch_files(project_id: str, job_id: str, db: AsyncSession = Depends(get_db)) -> dict:
    _, source = _scope(await _job(db, project_id, job_id))
    source = _native_path(source)
    files = [path.relative_to(source).as_posix() for path in islice(source.rglob("*.py"), 500)
             if path.is_file() and not path.is_symlink() and path.stat().st_size <= 60_000
             and not any(part in {".git", ".conda", ".venv", "__pycache__", "data", "data_set"}
                         for part in path.relative_to(source).parts)]
    return {"files": sorted(files)[:200]}


@router.post("/{job_id}/branch-preview")
async def preview_branch(project_id: str, job_id: str, payload: BranchPreviewRequest,
                         db: AsyncSession = Depends(get_db)) -> dict:
    """One general coding workflow; keyword recognition only sets safety checks."""
    from ..services.project_coding import propose_revision
    job = await _job(db, project_id, job_id)
    workspace, source = _scope(job)
    if payload.file:
        _code_file(source, payload.file)
    if payload.model_id:
        await validate_model(payload.model_id)
    gpu = any(term in payload.instruction.lower() for term in ("gpu", "cuda", "显卡"))
    readiness = await inspect_run_readiness(job) if gpu else {}
    # Structured preparation action, not a GCN/command template. Only exact,
    # reviewed TF import transformations bypass generation; complex requests
    # still use the same general planner/read/edit/validate workflow below.
    migration = readiness.get('gpu_environment_plan') or {}
    if (payload.preparation_only and gpu and not requested_framework(payload.instruction)
            and readiness.get('framework') == 'tensorflow' and migration.get('supported')
            and migration.get('source_changes_needed') and not readiness.get('gpu_code_issues')):
        from ..services.framework_migration import validate_bundle
        bodies = source_bodies(source)
        try:
            changes = tensorflow_compat_changes(bodies)
            originals = {row['file']: bodies[row['file']] for row in changes}
            revisions = dict(originals)
            for row in changes:
                revisions[row['file']] = _apply_text(revisions[row['file']], row['old_text'], row['new_text'], row['file'])
            if not revisions:
                raise ValueError('源码准备状态已变化，请重新检查')
            issues = validate_bundle(source, revisions, None, lambda path, updated: _gpu_preview_issues(path, updated,(getattr(job,'code_spec',None) or {}).get('entrypoints')))
            if issues or tensorflow_compat_changes(source_bodies(source, revisions)):
                raise ValueError('规则兼容修改未通过完整源码检查，需重新生成通用修改提案：' + '；'.join(issues))
        except (ValueError, SyntaxError) as exc:
            raise HTTPException(status_code=422, detail=str(exc) + '；未修改源码') from exc
        proposal = _store_preview(workspace, job, payload.instruction, originals, revisions, changes,
                                 '根据已核验的项目环境方案替换 TensorFlow 根导入为 compat.v1 图模式；保留算法、CLI、划分和超参数。不调用模型重写源码。',
                                 '仅通过静态兼容检查；应用代码、环境安装和实验执行仍需批准，GPU 运算仍需运行时验证。',
                                 '', {'total_tokens': 0}, [], migration)
        proposal_path, saved = _proposal(workspace, job.id, proposal['proposal_id'])
        trace = [{'agent': 'Compatibility Rules', 'model': '未调用模型', 'files': sorted(revisions), 'status': 'static_checked'}]
        saved['coding_trace'] = trace
        proposal_path.write_text(json.dumps(saved, ensure_ascii=False), encoding='utf-8')
        return {**proposal, 'coding_trace': trace}
    from ..services.task_context import task_contract
    context = {"paper_spec": job.paper_spec or {}, 'task_contract': task_contract(job),
               "environment": {"name": (job.environment_spec or {}).get("environment_name"),
                               "hardware_plan": (job.environment_spec or {}).get("hardware_plan")},
               "execution": {"datasets": [row.get("dataset") for row in (job.execution_command or {}).get("commands", [])],
                             "recorded_parameter_evidence": historical_parameter_context(job)},
               "runtime_readiness": readiness,
               'selected_run_execution_logs':execution_logs(workspace,job.id,limit=3),
               "recent_errors": (job.result_json or {}).get("repair_attempts", [])[-2:]}
    def validate_revision(source, revisions):
        issues = _gpu_preview_issues(source, revisions,(getattr(job,'code_spec',None) or {}).get('entrypoints')) if gpu else []
        from ..services.host_environment import spawn_safety_issues
        issues.extend(spawn_safety_issues(source,revisions,(job.code_spec or {}).get('entrypoints')))
        migration = readiness.get("gpu_environment_plan") or {}
        if gpu and migration.get("source_changes_needed"):
            missing = tensorflow_compat_changes(source_bodies(source, revisions))
            if missing:
                issues.append("当前环境证据要求完整图模式兼容入口，仍未修改：" +
                              "、".join(row["file"] for row in missing))
        return issues
    try:
        preparation = (tensorflow_compat_changes(source_bodies(source))
                       if gpu and migration.get('supported') and migration.get('source_changes_needed')
                       and not requested_framework(payload.instruction) else [])
        originals, revisions, changes, response, usage, trace = await propose_revision(
            source, payload.instruction, payload.model_id, preferred=payload.file,
            context=context, target=requested_framework(payload.instruction),
            validator=validate_revision, preparation_changes=preparation)
    except (ValueError, TimeoutError, OSError) as exc:
        job.result_json = {**(job.result_json or {}), 'code_request_failures': [
            *(job.result_json or {}).get('code_request_failures', [])[-19:],
            {'time': datetime.now(timezone.utc).isoformat(), 'instruction': payload.instruction[:1600],
             'kind': type(exc).__name__, 'error': str(exc)[:1000], 'source_changed': False}]}
        await db.commit()
        raise HTTPException(status_code=422, detail=str(exc) or "编码提案超时；未修改源码") from exc
    proposal = _store_preview(workspace, job, payload.instruction, originals, revisions, changes,
                             str(response.data.get("rationale") or "通用多文件修改预览")[:800],
                             str(response.data.get("expected_observation") or "静态校验不等于实验成功；应用与运行仍需分别批准")[:500],
                             response.model, {"total_tokens": usage}, [],
                             readiness.get("gpu_environment_plan"))
    # Trace is evidence of real model calls, not an animated architecture claim.
    proposal_path, saved = _proposal(workspace, job.id, proposal["proposal_id"])
    saved["coding_trace"] = trace
    proposal_path.write_text(json.dumps(saved, ensure_ascii=False), encoding="utf-8")
    proposal["coding_trace"] = trace
    return proposal


def _store_preview(workspace, job, instruction, originals, revisions, changes, rationale, observation, model, usage, diagnostics, migration=None):
    diffs = {relative: "".join(difflib.unified_diff(
        originals[relative].splitlines(keepends=True), updated.splitlines(keepends=True),
        fromfile=f"修改前/{relative}", tofile=f"修改后/{relative}"))[:20_000]
        for relative, updated in revisions.items()}
    proposal_id = str(uuid4())
    proposal = {"base_run_id": job.id, "files": sorted(revisions), "changes": changes,
                "instruction": instruction,
                "source_sha256": {name: hashlib.sha256(body.encode()).hexdigest() for name, body in originals.items()},
                "updated_sha256": {name: hashlib.sha256(body.encode()).hexdigest() for name, body in revisions.items()},
                "rationale": rationale, "expected_observation": observation,
                "model": model, "usage": usage, "diagnostics": diagnostics,
                'gpu_environment_plan': migration or {}}
    proposal.update(revision_scope(job, proposal))
    location = workspace / "runs" / job.id / "proposals"
    location.mkdir(parents=True, exist_ok=True)
    (location / f"{proposal_id}.json").write_text(json.dumps(proposal, ensure_ascii=False), encoding="utf-8")
    return {"proposal_id": proposal_id, "files": sorted(revisions), "diffs": diffs,
            "rationale": proposal["rationale"], "expected_observation": proposal["expected_observation"],
            "base_run_id": job.id, 'gpu_environment_plan': migration or {},
            **revision_scope(job, proposal)}


def _proposal(workspace: Path, job_id: str, proposal_id: str) -> tuple[Path, dict]:
    try:
        checked = str(UUID(proposal_id))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="变更提案 ID 无效") from exc
    path = workspace / "runs" / job_id / "proposals" / f"{checked}.json"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="变更提案不存在")
    return path, json.loads(path.read_text(encoding="utf-8"))


def _recover_preview(workspace, base, source):
    """Recover a passed, unapplied proposal only while its source hashes match."""
    directory = workspace / 'runs' / base.id / 'proposals'
    if not directory.is_dir():
        return None
    for path in sorted(directory.glob('*.json'), key=lambda p:p.stat().st_mtime, reverse=True)[:30]:
        try:
            _, saved = _proposal(workspace, base.id, path.stem)
            if saved.get('base_run_id') != base.id or saved.get('applied_at') or saved.get('run_id'):
                continue
            files=saved.get('files') or []
            if not 1<=len(files)<=12:
                continue
            originals={name:_code_file(source,name).read_text(encoding='utf-8',errors='replace') for name in files}
            if any(hashlib.sha256(body.encode()).hexdigest()!=saved.get('source_sha256',{}).get(name)
                   for name,body in originals.items()):
                continue
            updated=dict(originals)
            for change in saved['changes']:
                name=change['file']
                updated[name]=_apply_text(updated[name],change['old_text'],change['new_text'],name)
            if any(hashlib.sha256(body.encode()).hexdigest()!=saved.get('updated_sha256',{}).get(name)
                   for name,body in updated.items()):
                continue
            return {'proposal_id':path.stem,'base_run_id':base.id,'files':files,
                    'instruction':saved.get('instruction',''), 'rationale':saved.get('rationale',''),
                    'expected_observation':saved.get('expected_observation',''),
                    'coding_trace':saved.get('coding_trace',[]),
                    'diffs':{name:''.join(difflib.unified_diff(originals[name].splitlines(True),body.splitlines(True),
                             fromfile=f'修改前/{name}',tofile=f'修改后/{name}'))[:20000] for name,body in updated.items()}}
        except (HTTPException,ValueError,KeyError,TypeError,OSError):
            continue  # Malformed/obsolete previews are never applied or repaired here.
    return None


@router.get('/{job_id}/branch-preview')
async def saved_branch_preview(project_id: str, job_id: str, db: AsyncSession = Depends(get_db)) -> dict | None:
    base=await _job(db,project_id,job_id)
    workspace,source=_scope(base)
    return await asyncio.to_thread(_recover_preview,workspace,base,source)


@router.post("/{job_id}/revisions")
async def apply_revision(project_id: str, job_id: str, payload: BranchConfirmRequest,
                         db: AsyncSession = Depends(get_db)) -> dict:
    await _ensure_idle(db, project_id)
    base = await _job(db, project_id, job_id)
    if base.status not in {ReproductionStatus.paused, ReproductionStatus.completed,
                           ReproductionStatus.failed, ReproductionStatus.awaiting_data,
                           ReproductionStatus.awaiting_approval, ReproductionStatus.needs_input}:
        raise HTTPException(status_code=409, detail="实验尚未暂停或结束，不能修改代码")
    workspace, source = _scope(base)
    proposal_file, proposal = _proposal(workspace, base.id, payload.proposal_id)
    if proposal.get("base_run_id") != base.id:
        raise HTTPException(status_code=409, detail="提案与当前实验不匹配")
    if proposal.get("applied_at"):
        raise HTTPException(status_code=409, detail="该变更已经应用，不能重复修改")
    originals: dict[str, str] = {}
    updated: dict[str, str] = {}
    for name in proposal.get("files", []):
        original = _code_file(source, str(name)).read_text(encoding="utf-8", errors="replace")
        if hashlib.sha256(original.encode()).hexdigest() != (proposal.get("source_sha256") or {}).get(name):
            raise HTTPException(status_code=409, detail=f"{name} 在预览后已变化，请重新生成变更")
        originals[name] = original
        updated[name] = original
    for change in proposal.get("changes", []):
        name = str(change["file"])
        if name not in updated:
            raise HTTPException(status_code=422, detail="提案文件列表不一致")
        updated[name] = _apply_text(updated[name], str(change["old_text"]), str(change["new_text"]), name)
    target_framework = requested_framework(str(proposal.get('instruction') or ''))
    if target_framework and source_framework(source_bodies(source, updated)) != target_framework:
        raise HTTPException(status_code=422, detail=f'提案未完成 {target_framework} 框架迁移；未写入任何源码。请重新生成多文件提案。')
    if any(term in str(proposal.get('instruction') or '').lower() for term in ('gpu', 'cuda', '显卡')):
        issues = _gpu_preview_issues(source, updated,(base.code_spec or {}).get('entrypoints'))
        if issues:
            raise HTTPException(status_code=422, detail='GPU 提案未通过当前源码校验，未写入任何源码：' + '；'.join(issues))
    backup_root = workspace / "runs" / base.id / "revisions" / payload.proposal_id / "original"
    if backup_root.exists():
        raise HTTPException(status_code=409, detail="变更备份已存在，不会覆盖")
    for name, body in originals.items():
        backup = backup_root / name
        backup.parent.mkdir(parents=True, exist_ok=True)
        backup.write_text(body, encoding="utf-8")
    try:
        for name, body in updated.items():
            _code_file(source, name).write_text(body, encoding="utf-8")
        from ..services.editable_source import dependency_file
        dependency_hashes = {name: proposal['updated_sha256'][name] for name in updated if dependency_file(name)}
        if dependency_hashes:
            base.environment_spec = {**(base.environment_spec or {}), 'dependency_revision': {
                'proposal_id':payload.proposal_id, 'sha256':{
                    **((base.environment_spec or {}).get('dependency_revision') or {}).get('sha256', {}),
                    **dependency_hashes}}}
        if (base.status == ReproductionStatus.awaiting_approval and not getattr(base,'approved_at',None)
                and (getattr(base,'execution_command',None) or {}).get('entrypoint_candidates')):
            datasets = [str(item.get('dataset')) for item in (base.execution_command or {}).get('commands', []) if item.get('dataset')]
            base.execution_command = _fresh_run_command(base, source, workspace/'runs'/base.id,
                base.environment_spec or {}, datasets, changed_cli_parameters(proposal))
        proposal["applied_at"] = datetime.now(timezone.utc).isoformat()
        proposal_file.write_text(json.dumps(proposal, ensure_ascii=False), encoding="utf-8")
        base.events = [*(base.events or []), {"stage": "code_revision", "progress": base.progress,
                       "message": f"助手修改了 {', '.join(sorted(updated))}；尚未运行实验",
                       "time": proposal["applied_at"]}]
        base.result_json = {**(base.result_json or {}), "code_revisions": [
            *((base.result_json or {}).get("code_revisions") or []),
            {"proposal_id": payload.proposal_id, "files": sorted(updated),
             "backup_path": str(backup_root), "applied_at": proposal["applied_at"]},
        ]}
        await db.commit()
    except Exception:
        for name, body in originals.items():
            _code_file(source, name).write_text(body, encoding="utf-8")
        raise
    return {"proposal_id": payload.proposal_id, "base_run_id": base.id,
            "status": "applied_not_run", "files": sorted(updated),
            "message": "代码已修改并备份原文件。未启动实验；你可以稍后明确批准运行。"}


def _included_in_initial_approval(base, proposal: dict) -> bool:
    """A pre-approval edit is part of the initial run, not an unrun revision."""
    approved_at = getattr(base, 'approved_at', None)
    if not approved_at or not proposal.get('applied_at'):
        return False
    try:
        approved = datetime.fromisoformat(str(approved_at))
        applied = datetime.fromisoformat(str(proposal['applied_at']))
        if approved.tzinfo is None:
            approved = approved.replace(tzinfo=timezone.utc)
        if applied.tzinfo is None:
            applied = applied.replace(tzinfo=timezone.utc)
        return applied <= approved
    except (TypeError, ValueError):
        # Uncertain legacy timestamps must not silently hide an unrun edit.
        return False


@router.get("/{job_id}/pending-revision")
async def pending_revision(project_id: str, job_id: str,
                           db: AsyncSession = Depends(get_db)) -> dict | None:
    base = await _job(db, project_id, job_id)
    workspace, source = _scope(base)
    directory = workspace / "runs" / base.id / "proposals"
    if not directory.is_dir():
        return None
    for path in sorted(directory.glob("*.json"), key=lambda item: item.stat().st_mtime, reverse=True)[:30]:
        proposal = json.loads(path.read_text(encoding="utf-8"))
        if (proposal.get("applied_at") and not proposal.get("run_id")
                and not _included_in_initial_approval(base, proposal)):
            target = requested_framework(str(proposal.get('instruction') or ''))
            valid = not target or source_framework(source_bodies(source)) == target
            return {"proposal_id": path.stem, "files": proposal.get("files", []),
                    "runnable": valid,
                    "validation_error": '' if valid else f'旧提案仅修改了部分文件，尚未完成 {target} 框架迁移。不会批准运行；请重新生成完整多文件迁移，原文件备份和历史结果仍保留。',
                    "instruction": proposal.get("instruction", ""), "applied_at": proposal["applied_at"],
                    **revision_scope(base, proposal)}
    return None


@router.post("/{job_id}/revision-runs", status_code=202)
async def run_revision(project_id: str, job_id: str, payload: RevisionRunRequest,
                       background_tasks: BackgroundTasks, db: AsyncSession = Depends(get_db)) -> dict:
    key = (project_id, job_id, payload.proposal_id)
    lock = _REVISION_RUN_LOCKS.setdefault(key, asyncio.Lock())
    async with lock:
        return await _run_revision_once(project_id, job_id, payload, background_tasks, db)


@router.get("/{job_id}/revision-run-plan")
async def revision_run_plan(project_id: str, job_id: str, proposal_id: str,
                            db: AsyncSession = Depends(get_db)) -> dict:
    """Read-only transition check. Never approves, installs, copies or runs."""
    base = await _job(db, project_id, job_id)
    workspace, source = _scope(base)
    _, proposal = _proposal(workspace, base.id, proposal_id)
    return await inspect_revision_plan(base, source, proposal, proposal_id)


async def inspect_revision_plan(base, source, proposal, proposal_id):
    result = {'proposal_id': proposal_id, 'action': 'blocked', 'reason': '',
              'gpu_requested': any(term in str(proposal.get('instruction') or '').lower()
                                   for term in ('gpu', 'cuda', '显卡'))
                                   or bool((getattr(base, 'environment_spec', {}) or {}).get('gpu_requested')),
              **revision_scope(base, proposal)}
    if proposal.get('base_run_id') != base.id or not proposal.get('applied_at'):
        return {**result, 'reason': '请先审查并应用代码修改；没有启动实验'}
    if proposal.get('run_id'):
        return {**result, 'action': 'already_created', 'run_id': proposal['run_id'],
                'reason': '该提案已创建实验，不会重复启动'}
    if result['scope_reason']:
        return {**result, 'reason': result['scope_reason']}
    for name, digest in (proposal.get('updated_sha256') or {}).items():
        if hashlib.sha256(_code_file(source, name).read_text(encoding='utf-8', errors='replace').encode()).hexdigest() != digest:
            return {**result, 'reason': f'{name} 在修改后发生变化；请重新生成并审查提案'}
    target = requested_framework(str(proposal.get('instruction') or ''))
    if target and source_framework(source_bodies(source)) != target:
        return {**result, 'reason': f'尚未完成 {target} 多文件迁移，请审查完整迁移提案'}
    if not result['gpu_requested']:
        return {**result, 'action': 'ready', 'reason': '可以批准运行；执行时仍检查项目环境'}
    readiness = await inspect_run_readiness(base)
    result['readiness'] = readiness
    return {**result, **gpu_preparation_state(readiness)}


async def _run_revision_once(project_id, job_id, payload, background_tasks, db):
    base = await _job(db, project_id, job_id)
    workspace, source = _scope(base)
    _, proposal = _proposal(workspace, base.id, payload.proposal_id)
    if proposal.get("base_run_id") != base.id or not proposal.get("applied_at"):
        raise HTTPException(status_code=409, detail="请先确认并应用代码修改")
    if proposal.get("run_id"):
        existing = await _job(db, project_id, proposal['run_id'])
        return {"run_id": existing.id, "base_run_id": base.id,
                "status": existing.status.value, "revision": (existing.result_json or {}).get('revision', {}),
                "already_created": True}
    await _ensure_idle(db, project_id)
    target_framework = requested_framework(str(proposal.get('instruction') or ''))
    if target_framework and source_framework(source_bodies(source)) != target_framework:
        raise HTTPException(status_code=409, detail=f'当前源码未完成 {target_framework} 多文件迁移；不会创建实验记录、安装依赖或启动训练。请重新生成完整迁移提案。')
    for name, digest in (proposal.get("updated_sha256") or {}).items():
        body = _code_file(source, name).read_text(encoding="utf-8", errors="replace")
        if hashlib.sha256(body.encode()).hexdigest() != digest:
            raise HTTPException(status_code=409, detail=f"{name} 在修改后再次变化，请重新生成实验提案")
    scope = revision_scope(base, proposal, payload.datasets)
    if scope['scope_reason']:
        raise HTTPException(status_code=422, detail=scope['scope_reason'])
    run_id = str(uuid4())
    environment = _branch_environment(base, run_id,bool(payload.gpu_environment_migration))
    selected_entries=(base.execution_command or {}).get('entrypoint_candidates',[])[:1]
    environment["hardware_plan"] = _hardware_plan(source,selected_entries)
    environment["inferred_dependencies"] = _infer_python_dependencies(source,selected_entries)
    environment["gpu_requested"] = any(term in str(proposal.get("instruction") or "").lower()
                                       for term in ("gpu", "cuda", "显卡")) or bool(environment.get('gpu_requested'))
    if environment['gpu_requested']:
        await _approve_gpu_environment(base, environment, payload.gpu_environment_migration)
    run_source = source
    isolated_root = workspace / "branches" / run_id
    if payload.isolated_copy:
        isolated_source = isolated_root / "source"
        isolated_data = isolated_root / "data"
        await asyncio.to_thread(_copy_bounded, source, isolated_source)
        run_source = isolated_source
        try:
            if (workspace / "data").is_dir():
                await asyncio.to_thread(_copy_bounded, workspace / "data", isolated_data,
                    _materialized_cache_exclusions(base, source, scope['selected_datasets']))
            else:
                isolated_data.mkdir(parents=True)
        except Exception:
            if isolated_root.is_dir() and isolated_root.parent == workspace / "branches":
                shutil.rmtree(isolated_root)
            raise
    try:
        command = _fresh_run_command(base, run_source, workspace / "runs" / run_id,
                                     environment, scope['selected_datasets'], changed_cli_parameters(proposal))
        if payload.isolated_copy:
            command['data_dir'] = str(isolated_root / 'data')
    except Exception:
        if payload.isolated_copy and isolated_root.is_dir() and isolated_root.parent == workspace / "branches":
            shutil.rmtree(isolated_root)
        raise
    revision = {"base_run_id": base.id, "proposal_id": payload.proposal_id,
                "title": str(proposal.get("instruction") or "")[:70],
                "files": proposal.get("files", []), "rationale": proposal.get("rationale", ""),
                "expected_observation": proposal.get("expected_observation", ""),
                "isolated_copy": payload.isolated_copy,
                "datasets": [item.get("dataset") for item in command["commands"]]}
    now = datetime.now(timezone.utc)
    job = ReproductionJob(
        id=run_id, tenant_id=base.tenant_id, project_id=project_id,
        status=ReproductionStatus.executing, current_stage="environment_building", progress=82,
        message="检查当前项目环境，然后执行已批准的实验", goal=f"代码修改后的实验：{revision['title']}",
        model_id=base.model_id, source_mode="copilot_revision", repository_url=base.repository_url,
        workspace_path=str(workspace), execution_mode="full",
        execution_command=command,
        paper_spec=json.loads(json.dumps(base.paper_spec or {})),
        code_spec={**(base.code_spec or {}), "revision_files": revision["files"]},
        environment_spec=environment, dataset_spec=json.loads(json.dumps(base.dataset_spec or {})),
        result_json={"revision": revision}, approved_at=now,
        events=[{"stage": "revision_run", "progress": 82,
                 "message": "用户批准运行；复用并检查本项目现有 Conda 环境"
                            + ("；使用独立代码与数据副本" if payload.isolated_copy else "；直接使用当前项目源码"),
                 "time": now.isoformat()}],
    )
    db.add(job)
    original_proposal = dict(proposal)
    proposal["run_id"] = run_id
    proposal_file = workspace / "runs" / base.id / "proposals" / f"{payload.proposal_id}.json"
    try:
        proposal_file.write_text(json.dumps(proposal, ensure_ascii=False), encoding="utf-8")
        await db.commit()
    except Exception:
        proposal_file.write_text(json.dumps(original_proposal, ensure_ascii=False), encoding="utf-8")
        if payload.isolated_copy and isolated_root.is_dir() and isolated_root.parent == workspace / "branches":
            shutil.rmtree(isolated_root)
        raise
    background_tasks.add_task(execute_reproduction, run_id)
    return {"run_id": run_id, "base_run_id": base.id, "status": "executing", "revision": revision}


@router.get("/{job_id}/run-readiness")
async def run_readiness(project_id: str, job_id: str,
                        db: AsyncSession = Depends(get_db)) -> dict:
    base = await _job(db, project_id, job_id)
    try:
        return await inspect_run_readiness(base)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


async def inspect_run_readiness(base: ReproductionJob) -> dict:
    """Fresh, read-only device evidence for both UI and research chat."""
    workspace, source = _scope(base)
    framework = str(((base.environment_spec or {}).get("hardware_plan") or {}).get("framework") or "unknown")
    selected_entries=(base.code_spec or {}).get('entrypoints')
    actual_framework = source_framework(source_bodies(source,entrypoints=selected_entries))
    if actual_framework != 'unknown':
        framework = actual_framework
    issues = _gpu_preview_issues(source, {},selected_entries)
    commands = (base.execution_command or {}).get("commands") or []
    datasets = [str(item.get("dataset")) for item in commands if item.get("dataset")]
    runtime: dict = {"checked": False, "cuda_available": False,
                     "error": "当前框架不是 PyTorch，不能以 torch 的检测结果判断 GPU 可用" if framework != "pytorch" else ""}
    prefix = Path(str((base.environment_spec or {}).get("prefix") or ""))
    python = _environment_python(prefix)
    if python.is_file() and _environment_owned_by_job(base, prefix):
        runtime = await probe_framework(python, framework)
        interpreter = await probe_interpreter(python)
        runtime = {**runtime, **interpreter, 'interpreter_checked': bool(interpreter.get('python_version'))}
    elif python.is_file():
        runtime = {'checked': False, 'error': '项目解释器归属未核验，未执行探测'}
    logs = execution_logs(workspace, base.id)
    return {"datasets": list(dict.fromkeys(datasets)), "framework": framework,
            "current_local_dataset_evidence": inspect_local_datasets(source, datasets),
            "gpu_code_ready": not issues,
            "gpu_code_issues": issues, "environment_name": (base.environment_spec or {}).get("environment_name"),
            "runtime": runtime, "torch_runtime": runtime if framework == 'pytorch' else {},
            "tensorflow_runtime": runtime if framework == 'tensorflow' else {},
            "gpu_environment_plan": (tensorflow_gpu_plan(source, runtime, base.environment_spec or {}) if framework == 'tensorflow'
                                     else pytorch_cuda_plan(runtime, declared=(base.environment_spec or {}).get('inferred_dependencies')) if framework == 'pytorch' else {}),
            "device_log_evidence": device_log_evidence(logs),
            "note": "静态检查不是 GPU 训练证明；执行时仍会复检，主机有显卡不代表环境可用。"}


async def _approve_gpu_environment(base, environment, approved):
    # Recompute from current source/runtime; stale chat proposals cannot authorize upgrades.
    readiness = await inspect_run_readiness(base)
    state = gpu_preparation_state(readiness)
    if state['action'] in {'blocked', 'prepare_code'}:
        raise HTTPException(status_code=409, detail=state['reason'])
    migration = readiness.get('gpu_environment_plan') or {}
    if state['action'] == 'approve_environment':
        if not approved:
            raise HTTPException(status_code=409, detail=f"需单独批准 {readiness['framework']} GPU 环境配置；尚未安装依赖或启动实验")
        environment['gpu_environment_migration'] = {**migration, 'approved': True,
                                                    'framework': readiness['framework'],
                                                    'approved_at': datetime.now(timezone.utc).isoformat()}


@router.post("/{job_id}/reruns", status_code=202)
async def rerun_dataset(project_id: str, job_id: str, payload: RerunRequest,
                        background_tasks: BackgroundTasks, db: AsyncSession = Depends(get_db)) -> dict:
    await _ensure_idle(db, project_id)
    base = await _job(db, project_id, job_id)
    if base.status not in {ReproductionStatus.paused, ReproductionStatus.completed,
                           ReproductionStatus.failed, ReproductionStatus.awaiting_data}:
        raise HTTPException(status_code=409, detail="原实验尚未暂停或结束，不能开始重跑")
    workspace, source = _scope(base)
    if payload.gpu_requested:
        issues = _gpu_preview_issues(source, {},(base.code_spec or {}).get('entrypoints'))
        if issues:
            raise HTTPException(status_code=409, detail="当前代码尚未完成 GPU 适配：" + "；".join(issues))
    run_id = str(uuid4())
    environment = _branch_environment(base, run_id,bool(payload.gpu_requested and payload.gpu_environment_migration))
    environment["gpu_requested"] = payload.gpu_requested
    selected_entries=(base.execution_command or {}).get('entrypoint_candidates',[])[:1]
    environment["hardware_plan"] = _hardware_plan(source,selected_entries)
    environment["inferred_dependencies"] = _infer_python_dependencies(source,selected_entries)
    if payload.gpu_requested:
        await _approve_gpu_environment(base, environment, payload.gpu_environment_migration)
    isolated_root = workspace / "branches" / run_id
    run_source = source
    if payload.isolated_copy:
        run_source = isolated_root / "source"
        await asyncio.to_thread(_copy_bounded, source, run_source)
        try:
            if (workspace / "data").is_dir():
                await asyncio.to_thread(_copy_bounded, workspace / "data", isolated_root / "data",
                    _materialized_cache_exclusions(base, source, payload.datasets))
        except Exception:
            if isolated_root.is_dir() and isolated_root.parent == workspace / "branches":
                shutil.rmtree(isolated_root)
            raise
    try:
        command = _fresh_run_command(base, run_source, workspace / "runs" / run_id,
                                     environment, payload.datasets)
        if payload.isolated_copy:
            command['data_dir'] = str(isolated_root / 'data')
    except Exception:
        if payload.isolated_copy and isolated_root.is_dir() and isolated_root.parent == workspace / "branches":
            shutil.rmtree(isolated_root)
        raise
    selected = [item.get("dataset") for item in command["commands"]]
    rerun = {"base_run_id": base.id, "datasets": selected,
             "gpu_requested": payload.gpu_requested, "isolated_copy": payload.isolated_copy}
    now = datetime.now(timezone.utc)
    job = ReproductionJob(
        id=run_id, tenant_id=base.tenant_id, project_id=project_id,
        status=ReproductionStatus.executing, current_stage="environment_building", progress=82,
        message="检查现有项目环境后，仅重跑指定数据集", goal="重新运行：" + "、".join(selected),
        model_id=base.model_id, source_mode="copilot_rerun", repository_url=base.repository_url,
        workspace_path=str(workspace), execution_mode="full", execution_command=command,
        paper_spec=json.loads(json.dumps(base.paper_spec or {})), code_spec=json.loads(json.dumps(base.code_spec or {})),
        environment_spec=environment, dataset_spec=json.loads(json.dumps(base.dataset_spec or {})),
        result_json={"rerun": rerun}, approved_at=now,
        events=[{"stage": "rerun", "progress": 82,
                 "message": "用户批准仅重跑 " + "、".join(selected)
                            + ("；要求 GPU，无可用 CUDA 时不得退回 CPU" if payload.gpu_requested else ""),
                 "time": now.isoformat()}],
    )
    db.add(job)
    try:
        if payload.tuning_budget:
            job.result_json = {**job.result_json, 'tuning_policy': {
                'enabled': True, 'budget_per_dataset': payload.tuning_budget,
                'approved_at': now.isoformat(), 'test_feedback_allowed': False,
                'phase': 'after_all_baselines',
                'recovery_datasets': recovery_datasets((base.result_json or {}).get('dataset_results', []), selected),
                'requires_failure_review': True},
                'tuning_plans': await asyncio.to_thread(build_tuning_plans, job, payload.tuning_budget)}
        await db.commit()
    except Exception:
        if payload.isolated_copy and isolated_root.is_dir() and isolated_root.parent == workspace / "branches":
            shutil.rmtree(isolated_root)
        raise
    background_tasks.add_task(execute_reproduction, run_id)
    return {"run_id": run_id, "base_run_id": base.id, "status": "executing", "rerun": rerun}
