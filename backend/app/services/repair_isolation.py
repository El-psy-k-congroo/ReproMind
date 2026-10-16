"""Copy only a failed dataset's execution tree before automatic source repair."""
import json
import shutil
import time
from pathlib import Path


def isolate_failed_command(job, workspace, dataset, argv, cwd, env, output):
    workspace = workspace.resolve()
    execution = job.execution_command or {}
    commands = execution.get('commands', [])
    selected = [c for c in commands if c.get('dataset') == dataset]
    if len(selected) != 1:
        raise ValueError('不能唯一定位当前数据集命令')
    command = selected[0]
    root = Path(command.get('pythonpath') or execution.get('pythonpath') or cwd).resolve()
    if workspace not in root.parents or (cwd != root and root not in cwd.parents):
        raise ValueError('修复源码根目录不属于当前项目')
    shared = any(c is not command and (
        Path(c.get('pythonpath') or execution.get('pythonpath') or c.get('cwd', '')).resolve() == root
        or root == Path(c.get('cwd', '')).resolve() or root in Path(c.get('cwd', '')).resolve().parents
    ) for c in commands)
    if not shared:
        return argv, cwd, env, output, None
    count, size = 0, 0
    excluded = {'.git', '.venv', '.conda', '__pycache__'}
    for path in root.rglob('*'):
        if excluded.intersection(path.relative_to(root).parts):
            continue
        if path.is_symlink():
            raise ValueError('源码含符号链接，不能安全隔离纠错副本')
        if path.is_file():
            count += 1
            size += path.stat().st_size
        if count > 8000 or size > 2_000_000_000:
            raise ValueError('源码及数据超过隔离纠错复制上限，需审查实验输入')
    target = workspace / 'variants' / f'repair-{job.id}-{time.time_ns()}'
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(root, target, ignore=shutil.ignore_patterns(*excluded))
    def remap(value):
        return str(value).replace(str(root), str(target))
    mapped_argv = [remap(value) for value in argv]
    mapped_cwd = target / cwd.relative_to(root)
    updated = json.loads(json.dumps(execution))
    current = next(c for c in updated['commands'] if c.get('dataset') == dataset)
    current.update(argv=mapped_argv, cwd=str(mapped_cwd), pythonpath=str(target))
    updated['display'] = '\n'.join(f"[{c.get('dataset')}] {' '.join(c['argv'])}" for c in updated['commands'])
    job.execution_command = updated
    return mapped_argv, mapped_cwd, {**env, 'PYTHONPATH': str(target)}, remap(output), {
        'dataset': dataset, 'type': 'isolated_repair_source', 'source': str(root), 'copy': str(target),
        'reason': '运行失败后隔离当前数据集源码及数据；其他命令和原源码不变',
    }
