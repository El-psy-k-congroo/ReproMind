"""Retain bounded failed candidates so a retry can use actual rejection evidence."""
import json
import re
import time
from pathlib import Path


def candidate_root(workspace, job_id):
    if Path(job_id).name != job_id or job_id in {'','.','..'}:
        raise ValueError('无效的修复任务标识')
    root=(Path(workspace)/'runs'/job_id/'repair-candidates').resolve()
    if Path(workspace).resolve() not in root.parents:
        raise ValueError('修复候选目录越界')
    return root


def save_candidate(workspace, job_id, dataset, attempt, payload):
    root=candidate_root(workspace,job_id)
    content=json.dumps(payload,ensure_ascii=False)
    if len(content.encode('utf-8'))>100000:
        raise ValueError('修复候选证据超过大小限制')
    root.mkdir(parents=True,exist_ok=True)
    label=re.sub(r'[^a-z0-9]+','-',str(dataset).casefold()).strip('-')[:40] or 'dataset'
    path=root/f'{label}-{attempt}-{time.time_ns()}.json'
    path.write_text(content,encoding='utf-8')
    return str(path)


def failed_candidates(job, workspace, dataset, source):
    root=candidate_root(workspace,job.id)
    rows=[]
    for record in reversed((getattr(job,'result_json',None) or {}).get('repair_attempts') or []):
        if record.get('dataset')!=dataset or record.get('file')!=str(source) or record.get('applied'):
            continue
        path=Path(str(record.get('candidate_evidence') or '')).resolve()
        if path.parent!=root or not path.is_file() or path.is_symlink() or path.stat().st_size>100000:
            continue
        try:payload=json.loads(path.read_text(encoding='utf-8'))
        except (ValueError,OSError):continue
        rows.append({'attempt':record.get('attempt'),'reason':record.get('reason'),'return_code':record.get('return_code'),
            'old_text':str(payload.get('old_text') or '')[:2600],
            'new_text':str(payload.get('new_text') or '')[:5200],
            'retry_failure_tail':str(record.get('retry_failure_tail') or '')[-2400:]})
        if len(rows)>=2:break
    return rows
