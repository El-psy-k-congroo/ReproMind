"""Deterministic environment recovery. Planning/validation never installs packages.

Profiles are bounded, pinned official wheel combinations, not arbitrary LLM repairs.
Reference: https://pytorch.org/get-started/previous-versions/
"""
import re
import hashlib
from pathlib import Path
from urllib.parse import urlparse, unquote
from urllib.request import url2pathname

TORCH_PACKAGES = {'torch', 'torchvision', 'torchaudio'}


def package_name(name):
    return re.sub(r'[-_.]+', '-', name).lower()


def failure_kind(code, output):
    # Network errors may also end in "no matching distribution": check transport first.
    if re.search(r'ReadTimeout|Connection reset|NewConnectionError|Temporary failure|'
                 r'ConnectionError|ProxyError|Cannot connect to proxy|SSLError|CERTIFICATE_VERIFY_FAILED|\b(?:429|502|503)\b', output, re.I):
        return 'network'
    if code == 124:
        return 'timeout'
    if re.search(r'No matching distribution|Could not find a version|Requires-Python|PackagesNotFoundError', output, re.I):
        return 'incompatible_package'
    if re.search(r'ResolutionImpossible|conflicting dependencies', output, re.I):
        return 'dependency_conflict'
    return 'command_error'


def pytorch_cuda_plan(runtime, installed=None, declared=None):
    """Use the actual interpreter ABI; never change Python or downgrade installed APIs."""
    plan = {'needed': not (runtime.get('checked') and runtime.get('cuda_available')),
            'supported': False, 'reference': 'https://pytorch.org/get-started/previous-versions/',
            'warning': '复用项目环境；预检保留已有非 PyTorch 依赖版本，不升级 Python，不新建环境，不回退 CPU。'}
    if not plan['needed']:
        return {**plan, 'supported': True}
    version = runtime.get('python_version') or []
    if (len(version) < 2 or version[0] != 3 or not 7 <= version[1] <= 13
            or runtime.get('platform') not in {'win32', 'linux'}
            or runtime.get('machine', '').lower() not in {'amd64', 'x86_64'}):
        return {**plan, 'reason': '解释器版本或架构不在已核验的 CUDA wheel 范围；不会盲目安装或更换环境'}
    minor = version[1]
    if minor == 7:
        versions, indexes = ('1.13.1', '0.14.1', '0.13.1'), ['cu117']
    elif minor == 8:
        versions, indexes = ('2.4.1', '0.19.1', '2.4.1'), ['cu121', 'cu118']
    else:
        versions, indexes = ('2.6.0', '0.21.0', '2.6.0'), ['cu126', 'cu118']
    installed = installed or {}
    pins = dict(zip(('torch', 'torchvision', 'torchaudio'), versions))
    current_torch=str(installed.get('torch') or runtime.get('version') or '').split('+')[0]
    if current_torch in {'1.13.0','1.13.1'} and (minor<=10 or minor==11 and runtime.get('platform')=='linux'):
        # Official archived wheels preserve the installed API on old projects.
        # Windows cp311 is absent from this release; never substitute Linux wheels.
        versions=(current_torch,'0.14.'+current_torch[-1],'0.13.'+current_torch[-1])
        pins=dict(zip(('torch','torchvision','torchaudio'),versions))
        indexes=['cu117']
    if minor>=9 and re.fullmatch(r'\d+\.\d+\.\d+',current_torch) and tuple(map(int,current_torch.split('.')))>(2,6,0):
        # Preserve an already installed newer API. The official index and pip
        # dry-run must verify that the exact same release has a CUDA build.
        pins['torch']=current_torch
        for name in ('torchvision','torchaudio'):
            if name in installed:pins[name]=str(installed[name]).split('+')[0]
        indexes=['cu126']
        versions=(current_torch,pins['torchvision'],pins['torchaudio'])
    for name in TORCH_PACKAGES:
        current = installed.get(name) or (runtime.get('version') if name == 'torch' else '')
        if current and str(current).split('+')[0] != pins[name]:
            return {**plan, 'reason': f'现有 {name}=={current} 与已核验组合 {pins[name]} 不同；需审查版本约束，不能自动降级或替换 API'}
    wanted = ['torch'] + [name for name in ('torchvision', 'torchaudio')
                          if name in (declared or []) or name in installed]
    candidates = [{'profile': f'pytorch-{versions[0]}-{index}', 'index': index,
                   'index_url': f'https://download.pytorch.org/whl/{index}',
                   'packages': [f'{name}=={pins[name]}+{index}' for name in wanted]}
                  for index in indexes]
    return {**plan, 'supported': True, 'candidates': candidates,
            'profile': candidates[0]['profile'], 'packages': candidates[0]['packages'],
            'reason': f'按当前 Python 3.{minor} 选择兼容 CUDA 包；先预检，最多两个兼容方案，网络错误有限重试'}


def protected_constraints(installed):
    result = []
    for name, version in sorted(installed.items()):
        if package_name(name) in TORCH_PACKAGES:
            continue
        if not re.fullmatch(r'[A-Za-z0-9_.-]+', name) or not re.fullmatch(r'[A-Za-z0-9.+!-]+', version):
            raise ValueError('现有依赖版本不能生成安全约束；未修改环境')
        result.append(f'{name}=={version}')
    return result


def validate_pip_report(report, installed, candidate, wheelhouse=None):
    """Reject replacements of old TensorFlow/Numpy/etc and non-official downloads."""
    if not isinstance(report, dict) or not isinstance(report.get('install'), list):
        raise ValueError('pip 预检缺少可核验的安装计划；未修改环境')
    installs = []
    actual = {}
    for item in report['install']:
        meta = item.get('metadata', {})
        name, version = str(meta.get('name', '')), str(meta.get('version', ''))
        if not re.fullmatch(r'[A-Za-z0-9_.-]+', name) or not re.fullmatch(r'[A-Za-z0-9.+!-]+', version):
            raise ValueError('pip 预检依赖格式无效；未修改环境')
        key = package_name(name)
        if key in actual:
            raise ValueError('pip 预检出现重复依赖；未修改环境')
        actual[key] = version
        previous = installed.get(key)
        if key not in TORCH_PACKAGES and previous is not None and previous != version:
            raise ValueError(f'预检将替换未批准的现有依赖 {name}=={previous} → {version}；已阻止安装')
        url = urlparse(str(item.get('download_info', {}).get('url', '')))
        allowed = {'download.pytorch.org', 'download-r2.pytorch.org'} if key in TORCH_PACKAGES else {
            'files.pythonhosted.org', 'download.pytorch.org', 'download-r2.pytorch.org'}
        if url.scheme == 'file' and wheelhouse is not None:
            local = Path(url2pathname(unquote(url.path))).resolve()
            root = Path(wheelhouse).resolve()
            if url.netloc or local.parent != root or local.is_symlink() or not local.is_file() or local.suffix != '.whl':
                raise ValueError('本地 wheel 不属于本次官方来源下载目录')
            digest = hashlib.sha256()
            with local.open('rb') as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                    digest.update(chunk)
            archive = item.get('download_info', {}).get('archive_info', {})
            expected = archive.get('hashes', {}).get('sha256') or str(archive.get('hash', '')).removeprefix('sha256=')
            if digest.hexdigest() != expected:
                raise ValueError('本地 wheel SHA256 校验失败')
            installs.append(str(local))
            continue
        if url.scheme != 'https' or url.hostname not in allowed or not url.path.endswith('.whl'):
            raise ValueError('预检包含未核验来源或源码包；未修改环境')
        installs.append(f'{name}=={version}')
    for requirement in candidate['packages']:
        name, version = requirement.split('==', 1)
        if actual.get(name, installed.get(name)) != version:
            raise ValueError(f'pip 预检未确认 {requirement}；未修改环境')
    return installs
