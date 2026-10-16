"""Host-aware binary compatibility checks before spending time creating Conda envs.

Author pins remain evidence. Missing host wheels are not silently unpinned.
"""
import asyncio
import ast
import os
import platform
import re
from urllib.parse import unquote, urlparse
from pathlib import Path

import httpx
from packaging.requirements import Requirement, InvalidRequirement
from packaging.specifiers import SpecifierSet
from packaging.utils import parse_wheel_filename, InvalidWheelFilename
from packaging.version import Version, InvalidVersion
from .source_selection import entry_sources, native_path

BINARY_PACKAGES = {'torch', 'torchvision', 'torchaudio', 'numpy', 'scipy', 'scikit-learn',
                   'tensorflow', 'tensorflow-gpu', 'bitsandbytes', 'triton', 'xformers', 'flash-attn',
                   'pandas','pyarrow','numba','llvmlite','pytomlpp','catboost','lightgbm','xgboost','onnxruntime'}


def spawn_safety_issues(source, revisions=None, entrypoints=None, system=None):
    if (system or platform.system()) != 'Windows':
        return []
    root=native_path(source)
    paths=entry_sources(root,entrypoints)
    if not paths:return []
    trees=[]
    for path in paths:
        name=path.relative_to(root).as_posix()
        try:tree=ast.parse((revisions or {}).get(name,path.read_text(encoding='utf-8',errors='replace')))
        except (SyntaxError,OSError):continue
        trees.append((name,tree))
    workers=any(isinstance(node,ast.Call) and (
        isinstance(node.func,ast.Attribute) and node.func.attr=='DataLoader' or
        isinstance(node.func,ast.Name) and node.func.id=='DataLoader') and
        any(k.arg=='num_workers' and isinstance(k.value,ast.Constant) and isinstance(k.value.value,int)
            and k.value.value>0 for k in node.keywords) for _,tree in trees for node in ast.walk(tree))
    if not workers:return []
    def unguarded(nodes):
        for node in nodes:
            if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)):continue
            if isinstance(node,ast.If) and re.fullmatch(r"__name__ == ['\"]__main__['\"]",ast.unparse(node.test)):
                yield from unguarded(node.orelse)
                continue
            if isinstance(node,(ast.For,ast.While)) and any(isinstance(call,ast.Call) and
                (isinstance(call.func,ast.Name) and re.fullmatch(r'train|test|fit|evaluate|main',call.func.id,re.I)
                 or isinstance(call.func,ast.Attribute) and call.func.attr in {'backward','step'}) for call in ast.walk(node)):
                yield node.lineno
            yield from unguarded([child for child in ast.iter_child_nodes(node) if isinstance(child,ast.stmt)])
    name,tree=trees[0]
    return [f'{name}:{line}: Windows 多进程 DataLoader(num_workers>0) 与模块导入时执行的训练循环冲突；'
            '需要主入口保护，或采用仅当前主机的单进程数据加载。保留完整数据、batch、训练轮数与 GPU 计算。'
            for line in unguarded(tree.body)]


def wheel_support(filename, python, system=None, machine=None):
    system = system or platform.system()
    machine = (machine or platform.machine()).lower()
    try: tags = parse_wheel_filename(filename)[3]
    except InvalidWheelFilename: return False
    major, minor = map(int, python.split('.')[:2])
    cp = f'cp{major}{minor}'
    for tag in tags:
        runtime = tag.interpreter in {cp, f'py{major}', f'py{major}{minor}'}
        if tag.abi == 'abi3' and tag.interpreter.startswith('cp'):
            runtime |= int(tag.interpreter[2:]) <= int(cp[2:])
        host = (tag.platform == 'any' or
                system == 'Windows' and tag.platform == ('win_arm64' if machine in {'arm64','aarch64'} else 'win_amd64') or
                system == 'Linux' and ('linux' in tag.platform and tag.platform.endswith('aarch64' if machine in {'arm64','aarch64'} else 'x86_64')) or
                system == 'Darwin' and tag.platform.startswith('macosx_') and tag.platform.endswith(('universal2', 'arm64' if machine == 'arm64' else 'x86_64')))
        if runtime and host: return True
    return False


def python_requirement_matches(requirement, python):
    if not requirement:
        return True
    constraint = SpecifierSet(requirement)
    if len(python.split('.')) >= 3:
        return constraint.contains(python)
    # A Conda major.minor plan is not an installed x.y.0 interpreter. Check
    # whether some patch in this minor line can satisfy all declared bounds.
    major, minor = map(int, python.split('.')[:2])
    patches = {0}
    for bound in constraint:
        try:
            release = Version(bound.version.removesuffix('.*')).release
        except InvalidVersion:
            continue
        if release[:2] == (major, minor) and len(release) >= 3:
            patches.update({max(0, release[2]-1), release[2], release[2]+1})
    return any(constraint.contains(f'{major}.{minor}.{patch}') for patch in patches)


def check_release(requirement, payload, python, system=None, machine=None):
    published = payload.get('urls', [])
    files = [row for row in published if not row.get('yanked')]
    native_required = Requirement(requirement).name.lower() in {'bitsandbytes','triton','xformers','torch','flash-attn'}
    if (system or platform.system()) == 'Windows' and native_required:
        # Older GPU libraries shipped Linux .so inside misleading py3-none-any wheels.
        files = [row for row in files if not row.get('filename','').endswith('-any.whl')]
    compatible = [row['filename'] for row in files if wheel_support(row.get('filename',''), python, system, machine)
                  and python_requirement_matches(row.get('requires_python'), python)]
    candidates = ['3.11', '3.10', '3.12', '3.9', '3.8', '3.7']
    runtimes = [version for version in candidates if any(wheel_support(row.get('filename',''), version, system, machine)
                and python_requirement_matches(row.get('requires_python'), version) for row in files)]
    return {'requirement': requirement, 'compatible_wheels': compatible[:4], 'compatible_python': runtimes,
            'compatible': bool(compatible), 'python_check_scope': 'actual_patch_version' if len(python.split('.')) >= 3 else 'minor_version_candidate',
            'available_files': [row.get('filename') for row in files][:80],
            'release_state': 'missing' if not published else 'withdrawn' if all(row.get('yanked') for row in published) else 'published',
            'withdrawn_files': [{'filename':row.get('filename'), 'reason':str(row.get('yanked_reason') or '维护者未说明原因')[:400]}
                                for row in published if row.get('yanked')][:20]}


def release_issue(row, python, system=None, machine=None):
    """Repository availability and maintainer withdrawal are not OS diagnoses."""
    system, machine = system or platform.system(), machine or platform.machine()
    state = row.get('release_state')
    if state == 'withdrawn':
        reasons = '；'.join(dict.fromkeys(item['reason'] for item in row.get('withdrawn_files', [])))
        reason = '固定版本的发行文件已被维护者撤回（yanked）：' + reasons + '。不能据此认定必须使用 Linux；未尝试安装。'
        options = ['核对维护者撤回原因及当前 Python/API 兼容性', '审查可获取版本与源码的联合适配方案，再独立验证实验']
    elif state == 'missing':
        reason = '已查询的官方发行源没有列出该固定版本的安装文件；这不是已证实的操作系统不兼容。'
        options = ['核验作者提供的官方历史发行地址', '审查可获取版本与源码的联合适配方案']
    else:
        reason = f'作者固定版本没有当前 {system}/{machine}/Python {python} 的可用 wheel'
        options = []
        if row.get('compatible_python'):
            options.append('在本机使用发行文件支持的 Python：' + '、'.join(row['compatible_python']))
        if any(wheel_support(filename, python, 'Linux', machine) for filename in row.get('available_files', [])):
            options.append('发行源有对应 Linux wheel，可选择兼容的 Linux/WSL 执行主机并另外验证 GPU')
        options.append('制定当前主机的依赖与源码联合适配方案并重新验证指标')
    return {'requirement': row['requirement'], 'reason': reason, 'options': options}


async def inspect_host_requirements(source, spec):
    python = str(spec.get('python') or '3.11')
    pins = []
    for name in spec.get('dependency_files', []):
        path = Path(source) / name
        if path.name.lower() not in {'requirements.txt','requirement.txt'} or not path.is_file(): continue
        lines = spec.get('resolved_requirements') or path.read_text(encoding='utf-8', errors='replace').splitlines()
        for line in lines:
            try: req = Requirement(line.strip())
            except InvalidRequirement: continue
            if req.marker and not req.marker.evaluate({'sys_platform': 'win32' if os.name == 'nt' else 'linux', 'python_version': python}): continue
            exact = [s.version for s in req.specifier if s.operator == '==' and '*' not in s.version]
            if req.name.lower().replace('_','-') in BINARY_PACKAGES and len(exact) == 1:
                pins.append((str(req), req.name, exact[0]))
    rows, errors = [], []
    async with httpx.AsyncClient(timeout=httpx.Timeout(12, connect=6), follow_redirects=True) as client:
        semaphore = asyncio.Semaphore(4)
        async def inspect(pin):
            requirement, name, version = pin
            async with semaphore:
                try:
                    response = await client.get(f'https://pypi.org/pypi/{name}/{version}/json')
                    if response.status_code == 404 and name.lower() in {'torch', 'torchvision', 'torchaudio'}:
                        indexes = await asyncio.gather(client.get(f'https://download.pytorch.org/whl/{name}/'),client.get('https://download.pytorch.org/whl/torch_stable.html'),return_exceptions=True)
                        successful = [index for index in indexes if isinstance(index,httpx.Response) and index.is_success]
                        if not successful: response.raise_for_status()
                        filenames = [Path(urlparse(unquote(link)).path).name for index in successful for link in re.findall(r'href=["\']([^"\']+)', index.text)]
                        payload = {'urls': [{'filename': filename} for filename in filenames if filename.startswith(f'{name}-{version}-') or filename.startswith(f'{name}-{version}+') ]}
                        return check_release(requirement, payload, python)
                    response.raise_for_status()
                    return check_release(requirement, response.json(), python)
                except (httpx.HTTPError, ValueError) as exc:
                    return {'requirement': requirement, 'compatible': None, 'error': type(exc).__name__}
        rows = await asyncio.gather(*(inspect(pin) for pin in dict.fromkeys(pins)))
    common = set.intersection(*(set(row['compatible_python']) for row in rows if row.get('compatible_python') is not None)) if rows and all('compatible_python' in row for row in rows) else set()
    recommended = next((version for version in ['3.11','3.10','3.12','3.9','3.8','3.7'] if version in common), '')
    issues = [release_issue(row, python) for row in rows if row['compatible'] is False]
    for wheel in spec.get('incompatible_requirements') or []:
        issues.append({'requirement':wheel,'reason':'作者 wheel 的操作系统或 Python ABI 与当前主机不一致；同名现代包只是迁移候选，不能静默替换后认定兼容。',
                       'options':['协调源码 API 与当前平台依赖进行迁移','选择与作者 wheel 匹配的执行主机']})
    if os.name == 'nt':
        bodies = '\n'.join(path.read_text(encoding='utf-8',errors='replace') for path in entry_sources(Path(source),spec.get('source_entrypoints')))
        if re.search(r'init_process_group\([^)]*backend\s*=\s*["\']nccl["\']',bodies,re.S) and not re.search(r'["\']gloo["\']',bodies):
            issues.append({'requirement':'NCCL distributed backend', 'reason':'所选入口的本地依赖固定使用 NCCL，当前 Windows 执行主机需要核对单卡通信路径或迁移至 Linux。',
                           'options':['单卡模式下联合修改分布式初始化和通信调用并验证','使用兼容的 Linux CUDA 主机']})
    for path in entry_sources(Path(source),spec.get('source_entrypoints'))[:1]:
        try: ast.parse(path.read_text(encoding='utf-8',errors='replace'))
        except SyntaxError as exc:
            if 'Missing parentheses' in exc.msg or 'multiple exception types' in exc.msg:
                issues.append({'requirement':'Python 2 author entry', 'reason':f'作者入口使用 Python 2 语法，与当前规划 Python {python} 不兼容；不能用创建空 Python 3 环境代替源码与框架迁移。',
                               'options':['保留源码并选择兼容的旧运行主机','依据作者算法进行 Python/框架联合迁移后回归验证']})
    errors = [row for row in rows if row['compatible'] is None]
    return {'checked': not errors, 'system': platform.system(), 'machine': platform.machine(), 'python': python,
            'packages': rows, 'issues': issues, 'lookup_errors': errors, 'compatible_python': sorted(common), 'recommended_python': recommended,
            'note': 'wheel 可用性预检；不是完整依赖求解、GPU 运算或作者算法兼容性证明。联网失败不被当作版本不兼容。'}
