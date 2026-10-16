"""Opt-in searches attached to an existing run; never replace its baseline or claims."""
import asyncio
import hashlib
import json
import os
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from ..config import get_settings
from ..database import AsyncSessionLocal
from ..models import ReproductionJob, ReproductionStatus
from .tuning import build_tuning_plans, command_fingerprint, validation_score

APPROVAL_LOCK = asyncio.Lock()  # The application uses one backend worker.


def reviewed_plans(job, budget=6):
    workspace = Path(job.workspace_path).resolve()
    root = get_settings().paper_projects_dir.resolve()
    if workspace.parent != root or not workspace.name.endswith('-' + job.project_id[:8]):
        raise ValueError('工作区归属不合法')
    plans = build_tuning_plans(job, budget)
    execution = job.execution_command or {}
    for plan in plans:
        row = next((r for r in (job.result_json or {}).get('dataset_results', [])
                    if r.get('dataset') == plan['dataset']), {})
        command = next(c for c in execution['commands'] if c.get('dataset') == plan['dataset'])
        source = Path(command.get('pythonpath') or execution.get('pythonpath') or
                      (job.code_spec or {}).get('source_path') or '').resolve()
        reason = ''
        log = (workspace / 'runs' / job.id / str(row.get('baseline_log') or '__missing__')).resolve()
        run_root = (workspace / 'runs' / job.id).resolve()
        argv = row.get('baseline_argv')
        if row.get('return_code') != 0 or row.get('baseline_verdict', row.get('verdict')) != 'deviated':
            reason = '仅对已成功执行、经用户复核未复现原因的基线提出恢复搜索'
        elif not argv or argv != command.get('argv'):
            reason = '当前命令与原基线不同，需先独立验证修改后的基线，不能用调参替代纠错'
        elif run_root not in log.parents or not log.is_file() or log.is_symlink() or log.stat().st_size > 20_000_000:
            reason = '缺少可核验的基线日志'
        elif row.get('baseline_fingerprint') and row['baseline_fingerprint'] != plan['fingerprint']:
            reason = '源码已改变，不能沿用原基线开始搜索'
        elif not row.get('baseline_fingerprint') and any(p.stat().st_mtime > log.stat().st_mtime for p in source.rglob('*.py')):
            reason = '历史基线未保存源码指纹，且源码晚于基线日志；请先验证新源码基线'
        elif validation_score(log.read_text(encoding='utf-8', errors='replace'), row.get('metric_key', 'accuracy')) is None:
            reason = '基线日志没有验证集指标，不能按测试集选参'
        elif not plan['trials']:
            reason = '没有可核验的参数空间'
        plan['ready'] = not reason
        plan['blocked_reason'] = reason
        plan['baseline_identity'] = 'fingerprint' if row.get('baseline_fingerprint') else 'legacy_source_timestamps_requires_review'
        # Approval binds source, exact baseline command, measured log and budget together.
        plan['review_token'] = hashlib.sha256(json.dumps({
            'fingerprint': plan['fingerprint'], 'argv': argv, 'trials': plan['trials'],
            'log': hashlib.sha256(log.read_bytes()).hexdigest() if not reason else '',
        }, sort_keys=True).encode()).hexdigest()
    return plans


def new_session(plans):
    return {'id': uuid4().hex, 'status': 'approved', 'plans': plans,
            'approved_at': datetime.now(timezone.utc).isoformat(), 'attempts': []}


async def execute_tuning_session(job_id, session_id):
    from .reproduction_engine import (
        CURRENT_JOB_ID, ReproductionPaused, _begin_work, _end_work, _run_tuning, _write_reproduction_report,
    )
    _begin_work(job_id)
    token = CURRENT_JOB_ID.set(job_id)
    async with AsyncSessionLocal() as db:
        job = await db.get(ReproductionJob, job_id)
        try:
            active = (job.result_json or {}).get('tuning_active') or {}
            if active.get('id') == session_id and job.status == ReproductionStatus.paused:
                raise ReproductionPaused()
            if active.get('id') != session_id or job.status != ReproductionStatus.completed:
                return
            approved = next(s for s in job.result_json['tuning_sessions'] if s['id'] == session_id)
            workspace = Path(job.workspace_path).resolve()
            for plan in approved['plans']:
                await db.refresh(job)
                checked = next(p for p in reviewed_plans(job, plan['max_trials']) if p['dataset'] == plan['dataset'])
                if not checked['ready'] or checked['review_token'] != plan['review_token']:
                    raise ValueError(checked['blocked_reason'] or '批准后源码、命令或基线发生变化，停止搜索')
                command = next(c for c in job.execution_command['commands'] if c.get('dataset') == plan['dataset'])
                row = next(r for r in job.result_json['dataset_results'] if r['dataset'] == plan['dataset'])
                source = command.get('pythonpath') or job.execution_command.get('pythonpath') or job.code_spec['source_path']
                prefix = Path(str((job.environment_spec or {}).get('prefix') or '')).resolve()
                python = prefix / ('python.exe' if os.name == 'nt' else 'bin/python')
                if not python.is_file() or Path(command['argv'][0]).resolve() != python.resolve():
                    raise ValueError('调参仅允许使用本项目已经批准的环境，不创建或安装依赖')
                cwd = Path(command['cwd']).resolve()
                if workspace not in cwd.parents:
                    raise ValueError('调参命令不能离开项目工作区')
                log_root = workspace / 'runs' / job.id / 'telemetry' / 'tuning' / session_id
                log_root.mkdir(parents=True, exist_ok=True)
                baseline = workspace / 'runs' / job.id / row['baseline_log']
                async def progress(trial, parameters, path):
                    result = deepcopy(job.result_json)
                    result['tuning_active'] = {'id': session_id, 'dataset': plan['dataset'], 'trial': trial,
                                               'parameters': parameters, 'log_path': str(path)}
                    job.result_json = result
                    job.events = [*(job.events or []), {'stage': 'tuning', 'progress': job.progress,
                        'time': datetime.now(timezone.utc).isoformat(),
                        'message': f"Tuning Agent 搜索 {plan['dataset']}（{trial}/{len(plan['trials'])}）；原实验基线保留"}]
                    await db.commit()
                async def record(record):
                    result = deepcopy(job.result_json)
                    record = {**record, 'session_id': session_id,
                              'log': f"tuning/{session_id}/{record['log']}"}
                    result['tuning_attempts'] = [*result.get('tuning_attempts', []), record]
                    sessions = [dict(s) for s in result['tuning_sessions']]
                    current = next(s for s in sessions if s['id'] == session_id)
                    current['attempts'] = [*current['attempts'], record]
                    result['tuning_sessions'] = sessions
                    job.result_json = result
                    await db.commit()
                records, output, note = await _run_tuning(plan, row['baseline_argv'], cwd,
                    {'REPROMIND_DATA_DIR': str(workspace / 'data'), 'PYTHONPATH': str(source),
                     **((job.execution_command.get('runtime_environment') or {}).get('variables') or {})},
                    get_settings().experiment_timeout_seconds, log_root,
                    baseline.read_text(encoding='utf-8', errors='replace'), row.get('metric_key', 'accuracy'), progress, record)
                result = deepcopy(job.result_json)
                for item in result.get('tuning_attempts', []):
                    if item.get('session_id') == session_id and item.get('dataset') == plan['dataset']:
                        item['selected'] = any(r['selected'] and r['trial'] == item['trial'] for r in records)
                sessions = [dict(s) for s in result['tuning_sessions']]
                current = next(s for s in sessions if s['id'] == session_id)
                current['attempts'] = [r for r in result.get('tuning_attempts', []) if r.get('session_id') == session_id]
                current.setdefault('outcomes', {})[plan['dataset']] = note or (
                    '仅按验证集选择；基线结果不变' if output else '验证集没有改善，保留基线')
                if any(r['return_code'] != 0 or r['validation_score'] is None for r in records):
                    raise ValueError('搜索试验运行失败或缺少验证指标；停止本轮搜索，原基线保留')
                result['tuning_sessions'] = sessions
                job.result_json = result
                await db.commit()
            await _finish(db, job, session_id, 'completed', '搜索结束；原数据集指标和论文结论均未覆盖')
            _write_reproduction_report(job, workspace)
            await db.commit()
        except ReproductionPaused:
            await db.refresh(job)
            await _finish(db, job, session_id, 'cancelled', '用户中止额外搜索；基线与已完成尝试保留')
        except Exception as exc:
            if job:
                await _finish(db, job, session_id, 'failed', str(exc)[:600])
        finally:
            CURRENT_JOB_ID.reset(token)
            _end_work(job_id)


async def _finish(db, job, session_id, status, note):
    result = deepcopy(job.result_json or {})
    result.pop('tuning_active', None)
    sessions = [dict(s) for s in result.get('tuning_sessions', [])]
    for item in sessions:
        if item['id'] == session_id:
            item.update(status=status, note=note, finished_at=datetime.now(timezone.utc).isoformat())
    result['tuning_sessions'] = sessions
    job.result_json = result
    if job.status == ReproductionStatus.paused and result.get('paused_from_status') == 'completed':
        job.status = ReproductionStatus.completed
        job.current_stage = 'completed'
        job.message = note
    job.events = [*(job.events or []), {'stage': 'tuning', 'progress': job.progress,
        'time': datetime.now(timezone.utc).isoformat(), 'message': note}]
    await db.commit()
