"""Bounded, read-only framework and run-log evidence; hardware is not execution proof."""
import asyncio
import json
import os
import re
import subprocess
from pathlib import Path

_FRAMEWORK_PROBES = {}
MODEL_DEVICE = re.compile(r'\b(?:training|train|model(?:[_\s-]+(?:param(?:eter)?s?|body|encoder|backbone))?|encoder|decoder|body|backbone)[_\s-]+device\s*[:=]\s*(cuda(?::\d+)?|cpu|/device:(?:gpu|cpu):\d+)', re.I)


def execution_logs(workspace, job_id, limit=3):
    root = Path(workspace).resolve()
    log_dir = (root / 'runs' / job_id / 'telemetry').resolve()
    if root not in log_dir.parents or not log_dir.is_dir():
        return []
    paths = [p for p in log_dir.glob('*.log') if p.is_file() and not p.is_symlink()
             and log_dir in p.resolve().parents]
    result = []
    for path in sorted(paths, key=lambda p: p.stat().st_mtime, reverse=True)[:limit]:
        with path.open('rb') as reader:
            startup = reader.read(65536).decode('utf-8', errors='replace')
            head = startup[:1800]
            reader.seek(max(0, path.stat().st_size - 1800))
            tail = reader.read().decode('utf-8', errors='replace')
        # Framework startup warnings can push first-step placement beyond the
        # ordinary context head. Keep only bounded device lines, not 64KB of noise.
        device_lines = '\n'.join(line[:350] for line in startup.splitlines()
                                 if re.search(r'/device:(?:gpu|cpu):\d+', line, re.I) or MODEL_DEVICE.search(line))[:5600]
        result.append({'log': path.name, 'head': head, 'tail': tail,
                       'device_lines': device_lines,
                       'updated_at': path.stat().st_mtime})
    return result


def device_log_evidence(logs):
    findings = []
    for item in logs:
        for line in (item.get('head', '') + '\n' + item.get('device_lines', '')
                     + '\n' + item.get('tail', '')).splitlines():
            lower = line.lower()
            kind = ''
            explicit = MODEL_DEVICE.search(lower)
            if explicit:
                kind = 'training_device_cpu' if explicit[1].startswith(('cpu','/device:cpu')) else 'training_device_gpu'
            elif re.search(r'(?:training|train|model|using)\s+(?:device\s*[:=]?\s*)?(?:cuda(?::\d+)?|/device:gpu:\d+)', lower):
                kind = 'training_device_gpu'
            elif re.search(r'(?:training|train|model|using)\s+device\s*[:=]?\s*(?:cpu|/device:cpu:\d+)', lower):
                kind = 'training_device_cpu'
            elif re.search(r'\b(?:matmul|conv2d|sparsematmul|sparsetensordensematmul)(?:_\d+)?(?=\s|:).*?/device:gpu:\d+', lower):
                kind = 'gpu_compute_placement'
            elif 'created tensorflow device' in lower and '/device:gpu:' in lower:
                kind = 'gpu_initialized_not_training_proof'
            if kind:
                finding = {'log': item['log'], 'kind': kind, 'line': line[:350]}
                if finding not in findings:
                    findings.append(finding)
    return findings[:8]


def gpu_visibility_environment(required, configured, inherited=None):
    """Make an approved GPU choice visible to author entrypoints that require it."""
    if not required: return {}
    inherited = os.environ if inherited is None else inherited
    value = configured.get('CUDA_VISIBLE_DEVICES', inherited.get('CUDA_VISIBLE_DEVICES'))
    if value is None or not str(value).strip(): value = '0'
    value = str(value).strip()
    if value == '-1':
        raise ValueError('已要求 GPU，但运行环境显式禁用了全部 CUDA 设备；未启动 CPU 实验')
    return {'CUDA_VISIBLE_DEVICES':value}


async def probe_framework(python, framework, verify_compute=False, visible_devices=None):
    python = Path(python)
    if not python.is_file():
        return {'checked': False, 'framework': framework, 'error': '当前项目解释器不存在'}
    if framework == 'pytorch':
        script = ("import json,sys,platform,torch\ncompute_checked=False\n"
                  + ("if torch.cuda.is_available():\n"
                     " a=torch.tensor([[1.,2.],[3.,4.]],device='cuda')\n"
                     " b=a@a\n torch.cuda.synchronize()\n"
                     " assert abs(b[0,0].item()-7.)<0.0001\n compute_checked=True\n" if verify_compute else "") +
                  "print(json.dumps({'framework':'pytorch',"
                  "'platform':sys.platform,'machine':platform.machine(),'python_version':list(sys.version_info[:2]),"
                  "'torch':torch.__version__,'version':torch.__version__,'cuda_build':torch.version.cuda,"
                  "'cuda_available':torch.cuda.is_available(),'device_count':torch.cuda.device_count(),'compute_checked':compute_checked}))")
    elif framework == 'tensorflow':
        script = ("import json,sys,tensorflow as tf\n"
                  "from tensorflow.python.client import device_lib\n"
                  "config=tf.compat.v1.ConfigProto()\n"
                  "config.gpu_options.allow_growth=True\n"
                  "config.gpu_options.per_process_gpu_memory_fraction=0.01\n"
                  "devices=device_lib.list_local_devices(session_config=config) if tf.test.is_built_with_cuda() else []\n"
                  "gpu=[d.name for d in devices if d.device_type=='GPU']\n"
                  "compute_checked=False\n"
                  + ("if gpu:\n"
                     " with tf.Graph().as_default():\n"
                     "  with tf.device('/device:GPU:0'):\n"
                     "   a=tf.constant([[1.,2.],[3.,4.]])\n"
                     "   b=tf.matmul(a,a)\n"
                     "  config.allow_soft_placement=False\n"
                     "  with tf.compat.v1.Session(config=config) as session:\n"
                     "   result=session.run(b)\n"
                     "   assert abs(float(result[0,0])-7.)<0.0001\n"
                     "   compute_checked=True\n" if verify_compute else "") +
                  "print(json.dumps({'framework':'tensorflow','version':tf.__version__,'platform':sys.platform,'python_version':list(sys.version_info[:2]),"
                  "'cuda_build':bool(tf.test.is_built_with_cuda()),'cuda_available':bool(gpu),"
                  "'device_count':len(gpu),'gpu_devices':gpu,'compute_checked':compute_checked}))")
    else:
        return {'checked': False, 'framework': framework, 'error': '此框架尚无运行时探测器'}
    def probe():
        env = os.environ.copy()
        prefix = python.parent
        env['PATH'] = os.pathsep.join(str(prefix / p) for p in ['Library/bin', 'Scripts', '']) + os.pathsep + env.get('PATH', '')
        if visible_devices is not None:
            env['CUDA_VISIBLE_DEVICES'] = str(visible_devices)
        try:
            process = subprocess.run([str(python), '-c', script], capture_output=True,
                                     text=True, timeout=90 if verify_compute else 30, check=False, env=env)
            parsed = json.loads(process.stdout.strip().splitlines()[-1]) if process.returncode == 0 else {}
            return {'checked': process.returncode == 0, **parsed,
                    'error_kind': 'framework_missing' if process.returncode and re.search(
                        r"No module named ['\"](?:torch|tensorflow)['\"]", process.stderr) else '',
                    'error': process.stderr[-300:] if process.returncode else ''}
        except (OSError, subprocess.TimeoutExpired, ValueError, IndexError) as exc:
            return {'checked': False, 'framework': framework, 'error': str(exc)[:200]}
    # Concurrent UI/chat requests share one runtime subprocess. Client timeout
    # must not start another GPU context while the previous probe is still alive.
    key = (asyncio.get_running_loop(), str(python.resolve()), framework, verify_compute, visible_devices)
    task = _FRAMEWORK_PROBES.get(key)
    if task is None:
        task = asyncio.create_task(asyncio.to_thread(probe))
        _FRAMEWORK_PROBES[key] = task
        def release(completed):
            if _FRAMEWORK_PROBES.get(key) is completed:
                _FRAMEWORK_PROBES.pop(key, None)
        task.add_done_callback(release)
    return await asyncio.shield(task)


async def probe_interpreter(python):
    """Read ABI even when torch is missing; importing a framework must not hide Python."""
    def probe():
        try:
            process = subprocess.run([str(python), '-c',
                "import json,sys,platform;print(json.dumps({'python_version':list(sys.version_info[:2]),"
                "'platform':sys.platform,'machine':platform.machine()}))"],
                capture_output=True, text=True, timeout=12, check=False)
            if process.returncode:
                return {}
            return json.loads(process.stdout.strip().splitlines()[-1])
        except (OSError, subprocess.TimeoutExpired, ValueError, IndexError):
            return {}
    return await asyncio.to_thread(probe)
