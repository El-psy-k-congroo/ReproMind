"""Preserve recorded per-dataset CLI overrides across reviewed reruns."""
import ast
import math
import re
from pathlib import Path

from .tuning import numeric_flags


def _options(argv):
    argv = list(argv or [])
    offset = 3 if len(argv) > 2 and argv[1] == '-m' else min(2, len(argv))
    groups = {}
    current = None
    for token in argv[offset:]:
        token = str(token)
        if re.match(r'^--?[A-Za-z_]', token):
            current = token.split('=', 1)[0].lstrip('-')
            if current in groups:
                raise ValueError(f'历史命令含重复参数 {current}，需要明确核验，未启动实验')
            groups[current] = [token]
        elif current is not None:
            groups[current].append(token)
        else:
            raise ValueError('历史命令含无法安全继承的位置参数，需明确运行参数，未启动实验')
    return argv[:offset], groups


def _cli_defaults(text):
    defaults = {}
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return defaults
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        name, value = '', None
        if node.func.attr.startswith('DEFINE_') and len(node.args) >= 2 and isinstance(node.args[0], ast.Constant):
            name, value = str(node.args[0].value), node.args[1]
        elif node.func.attr == 'add_argument':
            options = [arg.value[2:] for arg in node.args if isinstance(arg, ast.Constant)
                       and isinstance(arg.value, str) and arg.value.startswith('--')]
            name = options[0] if options else ''
            value = next((kw.value for kw in node.keywords if kw.arg == 'default'), None)
        if name and value is not None:
            defaults[name] = ast.dump(value)
    return defaults


def changed_cli_parameters(proposal):
    changed = set()
    for row in (proposal or {}).get('changes', []):
        old = _cli_defaults(str(row.get('old_text', '')))
        new = _cli_defaults(str(row.get('new_text', '')))
        changed.update(key for key in old.keys() | new.keys() if old.get(key) != new.get(key))
    return changed


def effective_numeric_parameters(source, entrypoints, argv):
    """Capture at execution time, not by consulting mutable source later."""
    source = Path(source).resolve()
    values = {}
    argv = list(argv or [])
    script = (str(argv[2]).replace('.', '/') + '.py' if len(argv) > 2 and argv[1] == '-m'
              else str(argv[1]) if len(argv) > 1 and not str(argv[1]).startswith('-') else '')
    candidates = [script] if script else []
    candidates.extend(str(name) for name in entrypoints or [] if Path(str(name)).name == Path(script).name)
    entry = next(((source / name).resolve() for name in candidates
                  if (source / name).is_file() and source in (source / name).resolve().parents), None)
    dynamic = set()
    if entry is not None and entry.stat().st_size <= 250_000:
        text = entry.read_text(encoding='utf-8', errors='replace')
        values.update(numeric_flags(text))
        try:
            tree = ast.parse(text)
        except SyntaxError:
            tree = ast.Module(body=[], type_ignores=[])
        calls = {node.func.id for node in ast.walk(tree) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Name)}
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                dynamic.update(target.attr for target in node.targets
                               if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name)
                               and target.value.id in {'args', 'FLAGS', 'flags'})
            if not isinstance(node, ast.ImportFrom) or not node.module:
                continue
            called = {name.name for name in node.names if (name.asname or name.name) in calls}
            if not called:
                continue
            relative = Path(*node.module.split('.')).with_suffix('.py')
            module = next((path.resolve() for path in [entry.parent / relative, source / relative]
                           if path.is_file() and source in path.resolve().parents), None)
            if module is None or module.stat().st_size > 250_000:
                continue
            try:
                imported = ast.parse(module.read_text(encoding='utf-8', errors='replace'))
            except SyntaxError:
                continue
            for function in imported.body:
                if isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)) and function.name in called:
                    values.update(numeric_flags(ast.unparse(function)))
    values.update(_numeric_overrides(argv))
    # Dynamic file/model selections can overwrite CLI defaults (e.g. SGC --tuned).
    # Their runtime value is unknown here; do not present a static default as proof.
    for name in dynamic:
        values.pop(name, None)
    return values


def _numeric_overrides(argv):
    values = {}
    _, groups = _options(argv)
    for key, tokens in groups.items():
        raw = tokens[0].split('=', 1)[1] if '=' in tokens[0] else tokens[1] if len(tokens) == 2 else None
        try:
            value = float(raw)
            if math.isfinite(value):
                values[key] = int(value) if value.is_integer() else value
        except (TypeError, ValueError):
            pass
    return values


def historical_parameter_context(base):
    evidence = []
    for row in (getattr(base, 'result_json', None) or {}).get('dataset_results', [])[:12]:
        try:
            overrides = _numeric_overrides(row.get('baseline_argv') or [])
        except ValueError:
            overrides = {}
        evidence.append({'dataset':row.get('dataset'), 'recorded_cli_overrides':overrides,
                         'execution_snapshot':row.get('baseline_parameters') or {},
                         'note':'命令参数优先于源码默认值；未保存的历史默认值不能推断'})
    return evidence


def inherit_dataset_parameters(base, planned, source, changed=()):
    """Fresh launch/data paths + recorded arguments; deliberate edits take precedence.

    This applies to any supported dataset/CLI, not just NELL or epoch requests.
    Unknown historical defaults are not filled from another run or a title.
    """
    changed = set(changed)
    original = getattr(base, 'execution_command', None) or {}
    results = (getattr(base, 'result_json', None) or {}).get('dataset_results', [])
    rows = {str(row.get('dataset', '')).casefold(): row for row in results}
    commands = {str(row.get('dataset', '')).casefold(): row for row in original.get('commands', [])}
    old_source = str(original.get('pythonpath') or '')
    inherited = []
    for command in planned.get('commands', []):
        dataset = str(command.get('dataset', '')).casefold()
        row = rows.get(dataset, {})
        previous = row.get('baseline_argv') or commands.get(dataset, {}).get('argv')
        if not previous:
            continue
        _, historical = _options(previous)
        launcher, fresh = _options(command['argv'])
        # Dataset variants/paths must still be checked against the current loader.
        authoritative = {'dataset'} | changed
        kept = []
        for key, tokens in historical.items():
            if key in authoritative:
                continue
            remapped = [token.replace(old_source, str(source)) if old_source else token for token in tokens]
            workspace = str(getattr(base, 'workspace_path', '') or '')
            if workspace and old_source and Path(old_source).resolve() != Path(source).resolve():
                old_data = str(Path(workspace) / 'data')
                remapped = [token.replace(old_data, str(Path(source).parent / 'data')) for token in remapped]
                if any(workspace in token.replace(str(source), '').replace(str(Path(source).parent / 'data'), '') for token in remapped):
                    raise ValueError('历史参数仍指向原工作区的其他可写路径，不能安全隔离，未启动实验')
            fresh[key] = remapped
            kept.append(key)
        for key, value in (row.get('baseline_parameters') or {}).items():
            if key not in authoritative and key not in historical and type(value) in {int, float} and math.isfinite(value):
                fresh[key] = ['--' + key, str(value)]
                kept.append(key)
        command['argv'] = [*launcher, *(token for tokens in fresh.values() for token in tokens)]
        inherited.append({'dataset': command.get('dataset'), 'base_run_id': getattr(base, 'id', ''),
                          'recorded_overrides_preserved': sorted(kept),
                          'explicitly_changed_parameters': sorted(changed),
                          'historical_defaults_snapshotted': bool(row.get('baseline_parameters'))})
    planned['parameter_inheritance'] = inherited
    return planned
