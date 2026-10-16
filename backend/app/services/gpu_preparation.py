"""Reviewable TensorFlow GPU preparation, never installs or edits source here."""
import ast
import os
import re
from urllib.parse import urlsplit, unquote
from itertools import islice
from pathlib import Path

TF_WINDOWS_PROFILE = 'tensorflow-windows-2.10'


def migration_snapshot_matches(previous, current):
    """Resume after approved CUDA installation without overwriting the original snapshot."""
    if not isinstance(previous, dict) or set(previous) != set(current):
        return False
    def normalized(value, kind):
        rows = []
        for line in str(value).splitlines():
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if kind == 'conda_explicit':
                filename = unquote(urlsplit(line).path.rsplit('/', 1)[-1])
                if filename.startswith(('cudatoolkit-', 'cudnn-')):
                    continue
            rows.append(line)
        return sorted(rows)
    return all(normalized(previous[key], key) == normalized(value, key) for key, value in current.items())


def cuda_dependencies_ready(explicit_inventory):
    """Read installed versions, not a stale completion flag or cached archive."""
    filenames = [unquote(urlsplit(line.strip()).path.rsplit('/', 1)[-1])
                 for line in explicit_inventory.splitlines() if not line.startswith(('#', '@'))]
    return all(any(re.fullmatch(pattern, filename) for filename in filenames) for pattern in [
        r'cudatoolkit-11\.2(?:\.\d+)?-[A-Za-z0-9_]+\.(?:conda|tar\.bz2)',
        r'cudnn-8\.1\.0(?:\.\d+)?-[A-Za-z0-9_]+\.(?:conda|tar\.bz2)',
    ])


def protected_conda_specs(explicit_inventory):
    """Turn the pre-migration snapshot into exact existing package pins."""
    specs = []
    for line in explicit_inventory.splitlines():
        if not line.strip() or line.startswith(('#', '@')):
            continue
        filename = unquote(urlsplit(line.strip()).path.rsplit('/', 1)[-1])
        match = re.fullmatch(r'([A-Za-z0-9_.-]+)-([A-Za-z0-9_.+!]+)-([A-Za-z0-9_]+)\.(?:conda|tar\.bz2)', filename)
        if not match:
            raise ValueError('现有 Conda 快照包含无法锁定的包记录，不会猜测基础依赖版本')
        name, version, build = match.groups()
        if name not in {'cudatoolkit', 'cudnn'}:
            specs.append(f'{name}={version}={build}')
    if not any(spec.startswith('python=') for spec in specs):
        raise ValueError('现有 Conda 快照缺少 Python 的精确版本，不会修改环境')
    return specs


def gpu_preparation_state(readiness):
    """One read-only transition policy for chat, revision approval and reruns.

    A model may propose changes, but cannot turn an unknown runtime or an
    unsupported dependency plan into permission to install or train.
    """
    framework = readiness.get('framework')
    runtime = readiness.get('runtime') or readiness.get('torch_runtime') or {}
    migration = readiness.get('gpu_environment_plan') or {}
    if framework not in {'tensorflow', 'pytorch'}:
        return {'action': 'blocked', 'reason': '当前框架未识别或混用框架；需先核对完整源码，不能盲目配置 GPU 环境'}
    missing_framework = (runtime.get('error_kind') == 'framework_missing' and
                         runtime.get('interpreter_checked') is True)
    if not runtime.get('checked') and not missing_framework:
        return {'action': 'blocked', 'reason': '项目 GPU 运行时探测未完成：' +
                str(runtime.get('error') or '需核验项目解释器和框架；尚未安装依赖')}
    if not runtime.get('cuda_available') and not migration.get('supported'):
        return {'action': 'blocked', 'reason': migration.get('reason') or
                '当前环境没有经过兼容性核验的 GPU 依赖方案；不能盲目安装，需核对解释器、驱动和框架约束'}
    issues = list(readiness.get('gpu_code_issues') or [])
    if migration.get('source_changes_needed') or issues:
        return {'action': 'prepare_code', 'reason': '先生成并审查源码兼容修改：' +
                ('；'.join(issues) if issues else '、'.join(migration.get('source_files') or []) or '保留原框架和算法的兼容入口') +
                '。之后再批准项目环境配置；本次未修改源码或安装依赖'}
    if not runtime.get('cuda_available'):
        return {'action': 'approve_environment', 'reason':
                '源码检查通过，但项目环境尚不能使用 GPU；可批准所示依赖方案，安装后验证真实 GPU 运算，失败停止'}
    return {'action': 'ready', 'reason': '项目环境已枚举 GPU；执行前仍验证真实 GPU 运算，不回退 CPU'}


def requested_framework(instruction):
    """Identify an explicit target, not a historical mention of the source framework."""
    text = instruction.lower()
    targets = []
    for match in re.finditer(r'(?:迁移(?:到|为)?|改成|改为|切换(?:到|为)?|用|使用)\s*(pytorch|torch|tensorflow)', text):
        clause = re.split(r'[，,。；;！!？?\n]|而是|但是|然后', text[:match.start()])[-1]
        clause = re.sub(r'可不可以|能不能|不影响|不改变', '保持', clause)
        if re.search(r'(?:不要|不必|不用|无需|别|禁止|不能|不|未|没有|没)[^，,。；;]{0,20}$', clause):
            continue
        targets.append(match.group(1))
    return {'torch': 'pytorch'}.get(targets[-1], targets[-1]) if targets else ''


def source_bodies(source: Path, revisions=None, entrypoints=None):
    if os.name == 'nt' and not str(source).startswith('\\\\?\\'):
        absolute = str(source.resolve())
        source = Path('\\\\?\\UNC\\' + absolute[2:] if absolute.startswith('\\\\') else '\\\\?\\' + absolute)
    result = {}
    if entrypoints:
        from .source_selection import entry_sources
        paths=entry_sources(source,entrypoints,limit=501)
        paths += [source/name for name in (revisions or {}) if source/name not in paths]
    else:
        paths=islice(source.rglob('*.py'),501)
    for index, path in enumerate(paths):
        if index == 500:
            raise ValueError('源码超过 GPU 检查范围；请缩小源码目录')
        relative = path.relative_to(source)
        if (path.is_symlink() or not path.is_file() or path.stat().st_size > 60_000
                or any(p in {'.git', '.venv', '.conda', '__pycache__', 'data', 'data_set'} for p in relative.parts)):
            continue
        result[relative.as_posix()] = (revisions or {}).get(relative.as_posix(), path.read_text(encoding='utf-8', errors='replace'))
    return result


def source_framework(bodies):
    modules = set()
    for body in bodies.values():
        try:
            tree = ast.parse(body)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules.update(alias.name.split('.')[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                modules.add((node.module or '').split('.')[0])
    if 'tensorflow' in modules and 'torch' not in modules:
        return 'tensorflow'
    if 'torch' in modules and 'tensorflow' not in modules:
        return 'pytorch'
    if 'torch' in modules and 'tensorflow' in modules:
        return 'mixed'
    return 'unknown'


def tensorflow_compat_changes(bodies):
    """Only rewrite exact TF root imports; retain graph semantics and parameters."""
    changes = []
    for name, body in bodies.items():
        if re.search(r'\btf\.contrib\b|from\s+tensorflow\.contrib', body):
            raise ValueError(f'{name} 使用已移除的 tf.contrib，需要单独审查，不能自动替换兼容入口')
        matches = list(re.finditer(r'(?m)^import tensorflow as tf[ \t]*(?:#.*)?$', body))
        for match in matches:
            changes.append({'file': name, 'old_text': match.group(0),
                            'new_text': 'import tensorflow.compat.v1 as tf\ntf.disable_v2_behavior()',
                            'rationale': '保留 TensorFlow 1 图模式 API，供 TensorFlow 2.10 GPU 环境运行；不改变算法或超参数'})
        # Non-canonical imports need review, not a silently incomplete migration.
        root_imports = 0
        for node in ast.walk(ast.parse(body)):
            if isinstance(node, ast.Import) and any(a.name == 'tensorflow' for a in node.names):
                root_imports += 1
            if isinstance(node, ast.Import) and any(a.name == 'tensorflow' and a.asname != 'tf' for a in node.names):
                raise ValueError(f'{name} 的 TensorFlow 导入需要单独审查')
            if isinstance(node, ast.ImportFrom) and node.module == 'tensorflow':
                raise ValueError(f'{name} 的 TensorFlow 根模块导入需要单独审查')
        if root_imports != len(matches):
            raise ValueError(f'{name} 的 TensorFlow 根模块导入不匹配安全修改规则，需要单独审查')
    return changes


def validate_cuda_dependency_plan(output):
    """A successful solver exit is not permission to replace the shared environment."""
    import json
    try:
        plan = json.loads(output)
    except (TypeError, ValueError) as exc:
        raise ValueError('Conda 预检未返回完整 JSON；不会安装或修改环境') from exc
    if not isinstance(plan, dict) or plan.get('success') is not True:
        raise ValueError('Conda 预检未确认成功；不会安装或修改环境')
    actions = plan.get('actions')
    if actions is None and 'already installed' in str(plan.get('message', '')).lower():
        return
    if not isinstance(actions, dict):
        raise ValueError('Conda 预检缺少可核验的依赖变更计划；不会修改环境')
    for key in ('UNLINK', 'LINK'):
        rows = actions.get(key, [])
        if not isinstance(rows, list) or any(not isinstance(row, dict) or not row.get('name') for row in rows):
            raise ValueError('Conda 预检的依赖变更格式无效；不会修改环境')
    replaced = sorted({str(row['name']) for row in actions.get('UNLINK', [])
                       if row['name'] not in {'cudatoolkit', 'cudnn'}})
    if replaced:
        raise ValueError('Conda 预检将替换未批准的现有依赖：' + ', '.join(replaced)
                         + '；已阻止安装，需重新审查，不会自动降级 Python 或替换基础依赖')


def tensorflow_gpu_plan(source, runtime, environment):
    if runtime.get('checked') and runtime.get('cuda_available'):
        return {'needed': False, 'supported': True}
    plan = {'needed': True, 'supported': False,
            'warning': '迁移会更新本项目共享 Conda 环境，可能影响旧源码再次运行；旧实验结果保留。需单独批准，不创建新环境。',
            'reference': 'https://www.tensorflow.org/install/pip#windows-native'}
    if not runtime.get('checked') and not (runtime.get('error_kind') == 'framework_missing' and
                                          runtime.get('interpreter_checked') is True):
        return {**plan, 'reason': '运行时探测未完成，不能自动选择依赖版本'}
    if runtime.get('platform') != 'win32':
        return {**plan, 'reason': '当前仅支持已明确验证版本范围的 Windows 原生 TensorFlow 迁移方案'}
    version = runtime.get('python_version') or []
    if len(version) < 2 or not (version[0] == 3 and 7 <= version[1] <= 10):
        return {**plan, 'reason': 'TensorFlow 2.10 的 Windows GPU 包要求 Python 3.7–3.10；不会自行更换解释器或新建环境'}
    conda = Path(str(environment.get('conda_executable') or ''))
    prefix = Path(str(environment.get('prefix') or ''))
    if not conda.is_file() or not (prefix / 'conda-meta').is_dir():
        return {**plan, 'reason': '未核验现有 Conda 安装或环境目录，不能安装 CUDA 依赖'}
    try:
        changes = tensorflow_compat_changes(source_bodies(source))
    except (ValueError, SyntaxError) as exc:
        return {**plan, 'reason': str(exc)}
    return {**plan, 'supported': True, 'profile': TF_WINDOWS_PROFILE,
            'packages': ['tensorflow==2.10.0', 'cudatoolkit=11.2', 'cudnn=8.1.0'],
            'source_changes_needed': bool(changes), 'source_files': sorted({c['file'] for c in changes}),
            'reason': '先审查 TF1 兼容入口，再单独批准环境迁移和运行；安装后复检 GPU，不可用则停止'}
