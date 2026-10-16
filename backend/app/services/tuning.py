"""Evidence-bound, opt-in, bounded hyperparameter search. Never use test scores to select."""
import ast
import hashlib
import json
import math
import os
from pathlib import Path
from .source_selection import entry_sources, native_path
from .hyperparameter_contract import declared_domains, dataclass_defaults, command_config, flat_values, configuration_domains

GROUPS = {
    'learning_rate': ('learning_rate', 'learning-rate', 'lr'),
    'dropout': ('dropout', 'dropout_rate', 'dropout-rate'),
    'weight_decay': ('weight_decay', 'weight-decay', 'l2_reg', 'decay', 'wd', 'wd1', 'wd2'),
    'hidden_size': ('hidden1', 'hidden_dim', 'hidden_size', 'nhid', 'hidden'),
    'batch_size': ('batch_size', 'batch-size'),
}


def numeric_flags(text):
    """Read explicit CLI declarations, never guess a parameter from a variable name."""
    found = {}
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return found
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        default = None
        name = ''
        if node.func.attr in {'DEFINE_float', 'DEFINE_integer'} and len(node.args) >= 2:
            if isinstance(node.args[0], ast.Constant):
                name = str(node.args[0].value)
            if isinstance(node.args[1], ast.Constant):
                default = node.args[1].value
        elif node.func.attr in {'add_argument', 'option'} and node.args:
            options = [arg.value for arg in node.args if isinstance(arg, ast.Constant)
                       and isinstance(arg.value, str) and arg.value.startswith('--')]
            name = options[0].removeprefix('--') if options else ''
            default = next((item.value.value for item in node.keywords if item.arg == 'default'
                            and isinstance(item.value, ast.Constant)), None)
        if type(default) in {int, float} and math.isfinite(default):
            found[name] = default
    return found


def parameter_argv(argv, parameters):
    argv = list(argv)
    for name, value in parameters.items():
        flag = '--' + name
        for index, argument in enumerate(argv):
            if argument == flag and index + 1 < len(argv):
                argv[index + 1] = str(value)
                break
            if argument.startswith(flag + '='):
                argv[index] = flag + '=' + str(value)
                break
        else:
            argv.extend([flag, str(value)])
    return argv


def _current_value(argv, name, default):
    for index, value in enumerate(argv):
        if value == '--' + name and index + 1 < len(argv):
            return float(argv[index + 1])
        if value.startswith('--' + name + '='):
            return float(value.split('=', 1)[1])
    return default


def _values(group, base):
    if group == 'learning_rate' and 0 < base <= 1:
        return [base / 2, min(1, base * 2)]
    if group == 'dropout' and 0 <= base < 1:
        return [max(0, round(base - .1, 6)), min(.9, round(base + .1, 6))]
    if group == 'weight_decay' and 0 <= base <= 1:
        return [base / 10, min(1, base * 10)] if base else [1e-6, 1e-4]
    if group in {'hidden_size', 'batch_size'} and 1 <= base <= 4096 and float(base).is_integer():
        return [max(1, int(base) // 2), min(4096, int(base) * 2)]
    return []


def command_fingerprint(command, source):
    """Bind approval to the actual command and Python source, not only the dataset name."""
    root = Path(source).resolve()
    if os.name == 'nt' and not str(root).startswith('\\\\?\\'):
        absolute = str(root)
        root = Path('\\\\?\\UNC\\' + absolute[2:] if absolute.startswith('\\\\') else '\\\\?\\' + absolute)
    files = []
    for path in sorted(root.rglob('*.py')):
        if len(files) >= 5000:
            return ''  # Avoid approving a partially fingerprinted large repository.
        if not path.is_file() or root not in path.resolve().parents or '.git' in path.parts:
            continue
        if path.stat().st_size > 250_000:
            return ''
        files.append((path.relative_to(root).as_posix(), hashlib.sha256(path.read_bytes()).hexdigest()))
    from .source_selection import native_path
    canonical = native_path(source)
    binding = command_config(canonical, command)
    if binding:
        path = Path(binding['path'])
        files.append((path.relative_to(canonical).as_posix(), hashlib.sha256(path.read_bytes()).hexdigest()))
    return hashlib.sha256(json.dumps({'argv': command.get('argv'), 'files': files},
                                    sort_keys=True).encode()).hexdigest() if files else ''


def build_tuning_plans(job, budget=6):
    """Coordinate exploration followed by a small joint space; code-supported flags only."""
    budget = min(12, max(0, int(budget)))
    execution = job.execution_command or {}
    plans = []
    for command in execution.get('commands', []):
        raw_source = command.get('pythonpath') or execution.get('pythonpath') or (job.code_spec or {}).get('source_path')
        if not raw_source:
            continue
        source = native_path(raw_source)
        workspace = native_path(job.workspace_path)
        if workspace not in source.parents or not source.is_dir():
            continue
        entry = next((source / name for name in execution.get('entrypoint_candidates', [])
                      if (source / name).is_file() and source in (source / name).resolve().parents), None)
        if entry is None or entry.stat().st_size > 250_000:
            continue
        declarations, domains = {}, {}
        for path in entry_sources(source, [entry.relative_to(source).as_posix()]):
            body = path.read_text(encoding='utf-8', errors='replace')
            declarations.update(dataclass_defaults(body))
            declarations.update(numeric_flags(body))
            for name, domain in declared_domains(body).items():
                domains[name] = {**domain, 'source': path.relative_to(source).as_posix()}
        binding = command_config(source, command)
        config_defaults = flat_values(binding['data']) if binding else {}
        domain_warnings = []
        domains.update(configuration_domains(source, binding,domain_warnings))
        space, baseline = [], {}
        for group, aliases in GROUPS.items():
            name = next((alias for alias in aliases if alias in declarations), '')
            config_name = next((key for key in config_defaults if key.split('.')[-1] in aliases), '')
            if not name and config_name: name = 'config:' + config_name
            if not name:
                continue
            try:
                value = config_defaults[name[7:]] if name.startswith('config:') else _current_value(command.get('argv', []), name, declarations[name])
                domain = domains.get(name) or domains.get(name.removeprefix('config:').split('.')[-1])
                values = [float(f'{v:.10g}') if isinstance(v, float) else v
                          for v in (domain['values'] if domain else _values(group, value)) if v != value and type(v) in {int,float} and math.isfinite(v)]
            except (TypeError, ValueError):
                continue
            if not values:
                continue
            if group in {'hidden_size', 'batch_size'}:
                value = int(value)
            baseline[name] = value
            space.append({'parameter': name, 'group': group, 'baseline': value, 'values': values,
                          'source': domain['source'] if domain else Path(binding['path']).relative_to(source).as_posix() if name.startswith('config:') else entry.relative_to(source).as_posix(),
                          'reason': f"作者声明范围：{domain['origin']}，L{domain['line']}" if domain else '围绕当前已声明参数做局部搜索；不是论文指定的搜索范围'})
        represented = {row['parameter'] for row in space}
        for name, domain in domains.items():
            if name in represented: continue
            if name.startswith('config:'):
                if name[7:] not in config_defaults: continue
                value = config_defaults[name[7:]]
            else:
                if name not in declarations: continue
                value = _current_value(command.get('argv',[]),name,declarations[name])
            values = [v for v in domain['values'] if v != value and type(v) in {str,bool,int,float}
                      and (type(v) is not float or math.isfinite(v))]
            if values:
                baseline[name] = value
                space.append({'parameter':name, 'group':'author_declared', 'baseline':value, 'values':values,
                              'source':domain['source'], 'reason':'作者显式搜索空间：'+domain['origin']})
        trials = []
        # Spread the initial budget across dimensions instead of exhausting learning-rate values first.
        for position in range(2):
            for item in space:
                if position < len(item['values']):
                    trials.append({item['parameter']: item['values'][position]})
        if len(space) >= 2 and budget >= len(space) + 1:
            coordinate_budget = max(len(space), budget - 2)
            # Joint trials cover every supported dimension, not always the first two flags.
            trials = trials[:coordinate_budget] + [
                {item['parameter']: item['values'][index % len(item['values'])]
                 for index, item in enumerate(space)},
                {item['parameter']: item['values'][(index + 1) % len(item['values'])]
                 for index, item in enumerate(space)},
            ]
        unique = []
        for trial in trials:
            if trial not in unique:
                unique.append(trial)
        fingerprint = command_fingerprint(command, source)
        plans.append({'dataset': command.get('dataset'), 'strategy': 'bounded_coordinate_then_joint',
                      'declared_parameters': {name:_current_value(command.get('argv',[]),name,value) for name,value in declarations.items()},
                      'parameters_without_search_domain': [name for name in declarations if name not in {item['parameter'] for item in space}],
                      'config_binding': {key:value for key,value in binding.items() if key!='data'} if binding else None,
                      'space': space, 'baseline_parameters': baseline, 'trials': unique[:budget] if fingerprint else [],
                      'max_trials': budget, 'max_seconds': 1800, 'fingerprint': fingerprint,
                      'unsupported_domains':domain_warnings,
                      'selection_rule': '仅验证集；测试集不用于触发搜索、选参或提前停止',
                      'status': 'proposed', 'requires_approval': True,
                      'limitation': '预算有限的局部搜索，不保证全局最优；缺少验证指标时不搜索'+('；'+'；'.join(domain_warnings) if domain_warnings else '')})
    return plans


def validation_score(output, metric):
    import re
    aliases = {'accuracy': 'acc(?:uracy)?', 'auc': '(?:auc|auroc)', 'f1': 'f1(?:[-_]score)?'}
    name = aliases.get(metric, re.escape(metric))
    values=[];epochs={}
    for line in output.splitlines():
        segment = re.search(r'\b(?:val(?:idation)?|valid)\b([^|\r\n]*)',line,re.I)
        direct = re.search(r'\b(?:val(?:idation)?|valid)[_\s-]*'+name+r'\s*[:=]\s*(-?\d+(?:\.\d+)?(?:e[-+]?\d+)?)',line,re.I)
        match = direct or (re.search(r'\b'+name+r'\s*[:=]\s*(-?\d+(?:\.\d+)?(?:e[-+]?\d+)?)',segment[1],re.I) if segment else None)
        if not match: continue
        value=float(match[1])
        if not math.isfinite(value): continue
        if metric in {'accuracy','auc','f1'} and value>1: value/=100
        if metric in {'accuracy','auc','f1'} and not 0<=value<=1: continue
        values.append(value)
        epoch=re.search(r'\bepoch\s*[:=]?\s*(\d+)',line,re.I)
        if epoch: epochs[int(epoch[1])]=value
    restored=re.findall(r'\bload(?:ing)?\s+(\d+)(?:st|nd|rd|th)?\s+epoch',output,re.I)
    if restored and int(restored[-1]) in epochs: return epochs[int(restored[-1])]
    return values[-1] if values else None


def best_trial(records, baseline, metric):
    lower = metric in {'loss', 'mae', 'mse', 'rmse'}
    winner, score = None, baseline
    for index, record in enumerate(records):
        value = record.get('validation_score')
        if record.get('return_code') == 0 and value is not None and math.isfinite(value):
            if (value < score if lower else value > score):
                winner, score = index, value
    return winner


def recovery_datasets(results, selected):
    """A user-reviewed failed baseline permits recovery, not a blanket search approval."""
    return [row['dataset'] for row in results if row.get('dataset') in selected
            and row.get('return_code') == 0 and row.get('verdict') == 'deviated'
            and not row.get('tuning_selected')]


def tuning_gate(policy, row, plan, fingerprint):
    if row.get('return_code') != 0:
        return '运行错误需要先纠错，不进行超参数搜索'
    if row.get('verdict') == 'matched':
        return '作者配置基线已达到论文指标，不需要调参'
    if not policy.get('enabled') or not policy.get('approved_at'):
        return '额外调参未获批准（不影响基线实验）；请在“自动调参与修复”审查恢复搜索计划并确认预算，原基线保留'
    if row.get('dataset') not in policy.get('recovery_datasets', []):
        return '先完成全部作者配置基线；复核未复现原因后另行批准调参'
    if not plan or not plan.get('trials'):
        return '没有可审查的参数搜索空间'
    if not fingerprint or plan.get('fingerprint') != fingerprint:
        return '源码或执行参数已变化，旧调参批准不再适用；需重新确认计划'
    return ''
