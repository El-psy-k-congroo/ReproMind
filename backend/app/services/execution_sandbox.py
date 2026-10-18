"""Linux process/file isolation for multi-user author-code execution.

Only the selected project and its environment are writable. Service databases,
other projects, host process environment and API credentials are not mounted.
Local mode keeps its existing execution behavior.
"""
import os
import shutil
from contextvars import ContextVar
from pathlib import Path

from ..config import get_settings

current_workspace: ContextVar[Path | None] = ContextVar('execution_workspace', default=None)


def environment_root():
    return get_settings().paper_projects_dir.resolve().parent / 'environments'


def interpreter_workspace(python):
    prefix = Path(python).parent.parent
    if prefix.is_symlink() or prefix.parent.resolve() != environment_root().resolve():
        raise ValueError('项目解释器不属于服务器独立环境目录')
    suffix = prefix.name.rsplit('-', 1)[-1]
    matches = [p for p in get_settings().paper_projects_dir.iterdir()
               if p.is_dir() and not p.is_symlink() and p.name.endswith('-'+suffix)]
    if len(matches) != 1:
        raise ValueError('无法唯一确认项目解释器的工作区归属')
    return matches[0]


def sandbox_command(argv, cwd, env, workspace=None):
    if not getattr(get_settings(), 'execution_sandbox', False):
        return list(argv), env
    if os.name != 'posix' or not shutil.which('bwrap'):
        raise RuntimeError('服务器执行隔离不可用，请维护者检查 bubblewrap 配置')
    root = get_settings().paper_projects_dir.resolve()
    # Preserve the lexical project boundary before resolving an author-created
    # symlink. Resolving cwd first could accidentally select another project.
    workspace = Path(workspace or current_workspace.get() or cwd).absolute()
    relative = workspace.relative_to(root)
    workspace = root / relative.parts[0]
    if workspace.is_symlink() or not workspace.is_dir() or workspace.name.startswith('.'):
        raise ValueError('执行工作区无效')
    cwd = Path(cwd).resolve()
    if cwd != workspace and workspace not in cwd.parents:
        raise ValueError('运行目录超出当前项目')
    private = workspace / '.runtime'
    for name in ('home', 'packages', 'cache'):
        (private/name).mkdir(parents=True, exist_ok=True)
    mounts = ['/usr', '/bin', '/sbin', '/lib', '/lib64', '/etc/ssl', '/etc/resolv.conf',
              '/etc/hosts', '/etc/nsswitch.conf', '/etc/passwd', '/etc/group',
              '/opt/repromind-python', '/opt/repromind-api-venv', '/opt/repromind-conda']
    command = ['bwrap', '--die-with-parent', '--new-session', '--unshare-all', '--share-net',
               '--cap-drop', 'ALL', '--clearenv', '--proc', '/proc', '--dev', '/dev', '--tmpfs', '/tmp']
    for value in mounts:
        if Path(value).exists():
            command += ['--ro-bind', value, value]
    command += ['--bind', str(workspace), str(workspace)]
    # Prefix is selected by a validated project suffix, not arbitrary user paths.
    env_root = environment_root()
    prefix = None
    for flag in ('--name', '-n'):
        if flag in argv:
            name = str(argv[argv.index(flag)+1])
            if '/' in name or '\\' in name or not name.endswith('-'+workspace.name.rsplit('-',1)[-1]):
                raise ValueError('环境名称与当前项目不匹配')
            prefix = env_root/name
    for value in [argv[0], *(argv[argv.index('--prefix')+1:argv.index('--prefix')+2] if '--prefix' in argv else [])]:
        path = Path(value)
        try:
            parts = path.relative_to(env_root).parts
        except ValueError:
            continue
        if parts:
            prefix = env_root/parts[0]
    if prefix:
        if prefix.is_symlink() or not prefix.name.endswith('-'+workspace.name.rsplit('-',1)[-1]):
            raise ValueError('环境目录不属于当前项目')
        prefix.mkdir(parents=True, exist_ok=True)
        command += ['--bind', str(prefix), str(prefix)]
    safe = {key: str(value) for key,value in env.items() if key in {
        'PATH','LANG','LC_ALL','TZ','HF_ENDPOINT','PYTHONUNBUFFERED','PYTHONUTF8','PYTHONPATH',
        'OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','CUDA_VISIBLE_DEVICES',
        'CONDA_PREFIX','CONDA_DEFAULT_ENV','PIP_INDEX_URL','PIP_EXTRA_INDEX_URL','PIP_DISABLE_PIP_VERSION_CHECK'}}
    safe.update(HOME=str(private/'home'), XDG_CACHE_HOME=str(private/'cache'),
                HF_HOME=str(private/'cache'/'huggingface'), TORCH_HOME=str(private/'cache'/'torch'),
                CONDA_PKGS_DIRS=str(private/'packages'), CONDA_ENVS_PATH=str(env_root),
                PATH=safe.get('PATH','/usr/bin:/bin'), PYTHONNOUSERSITE='1')
    for key,value in safe.items():
        command += ['--setenv', key, value]
    command += ['--chdir', str(cwd), '--', *map(str,argv)]
    return command, safe
