"""Read declared search domains and the configuration actually passed to a task."""
import ast
import json
import math
import re
import tomllib
from pathlib import Path

import yaml
from .source_selection import native_path


def declared_domains(body):
    found = {}
    try: tree = ast.parse(body)
    except SyntaxError: return found
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute): continue
        method = node.func.attr
        if method == 'add_argument':
            names = [arg.value[2:] for arg in node.args if isinstance(arg, ast.Constant)
                     and isinstance(arg.value, str) and arg.value.startswith('--')]
            choices = next((kw.value for kw in node.keywords if kw.arg == 'choices'), None)
            if not names or choices is None: continue
            try: values = ast.literal_eval(choices)
            except (ValueError, TypeError): continue
            if isinstance(values, (tuple, list)) and 1 < len(values) <= 100:
                found[names[0]] = {'values': list(values), 'origin': 'argparse.choices', 'line': node.lineno}
        if method not in {'suggest_float', 'suggest_int', 'suggest_categorical'} or len(node.args) < 2: continue
        try: args = [ast.literal_eval(arg) for arg in node.args]
        except (ValueError, TypeError): continue
        name = args[0]
        if not isinstance(name, str): continue
        if method == 'suggest_categorical' and isinstance(args[1], (list, tuple)):
            values = list(args[1])
        elif len(args) >= 3 and all(type(v) in {int, float} and math.isfinite(v) for v in args[1:3]):
            low, high = args[1:3]
            if low > high: continue
            log = any(kw.arg == 'log' and isinstance(kw.value, ast.Constant) and kw.value.value is True for kw in node.keywords)
            if log and low <= 0: continue
            middle = math.sqrt(low * high) if log else (low + high) / 2
            values = [low, int(round(middle)) if method == 'suggest_int' else middle, high]
        else: continue
        if 1 < len(values) <= 100 and all(type(v) in {str, int, float, bool} for v in values):
            found[name] = {'values': list(dict.fromkeys(values)), 'origin': method, 'line': node.lineno}
    return found


def dataclass_defaults(body):
    try: tree = ast.parse(body)
    except SyntaxError: return {}
    found = {}
    for cls in tree.body:
        if not isinstance(cls, ast.ClassDef) or not any(isinstance(d, ast.Name) and d.id == 'dataclass' for d in cls.decorator_list): continue
        for node in cls.body:
            if not isinstance(node, ast.AnnAssign) or not isinstance(node.target, ast.Name): continue
            value = node.value
            if isinstance(value, ast.Call) and isinstance(value.func, ast.Name) and value.func.id == 'field':
                value = next((kw.value for kw in value.keywords if kw.arg == 'default'), None)
            if isinstance(value, ast.Constant) and type(value.value) in {int, float}:
                found[node.target.id] = value.value
    return found


def load_config(path):
    path = native_path(path)
    body = path.read_text(encoding='utf-8')
    if path.suffix == '.toml': return tomllib.loads(body)
    if path.suffix == '.json': return json.loads(body)
    return yaml.safe_load(body)


def command_config(source, command):
    source = native_path(source)
    argv = command.get('argv') or []
    cwd = Path(command.get('cwd') or source)
    for index, argument in enumerate(argv):
        # Only actual config arguments; never scan and choose an unrelated config.
        value = argument.split('=',1)[-1]
        if Path(value).suffix.lower() not in {'.yaml','.yml','.toml','.json'}: continue
        config_flags = {'--config','--config-file','--config_file','--cfg'}
        if argument.startswith('-'):
            if '=' not in argument or argument.split('=',1)[0] not in config_flags: continue
        elif index and str(argv[index-1]).startswith('-') and argv[index-1] not in config_flags:
            continue
        candidates = [cwd/value, source/value, cwd/'configs'/value]
        candidates = [native_path(p) for p in candidates]
        path = next((p.resolve() for p in candidates if p.is_file() and source.resolve() in p.resolve().parents), None)
        if not path or path.stat().st_size > 100_000: continue
        try: data = load_config(path)
        except (ValueError, yaml.YAMLError): continue
        if isinstance(data, dict): return {'path': str(path), 'argv_index': index, 'argument': argument, 'data': data}
    return None


def flat_values(data, prefix=''):
    found = {}
    for key, value in data.items():
        name = f'{prefix}.{key}' if prefix else str(key)
        if isinstance(value, dict): found.update(flat_values(value, name))
        elif type(value) in {str, bool, int, float}: found[name] = value
    return found


def configuration_domains(source, binding, warnings=None):
    """Read explicit standard domains in an author's tuning configs for this task.

    Custom samplers stay unsupported until their transformation is understood.
    """
    if not binding: return {}
    source = native_path(source)
    baseline = binding['data']; result = {}
    data_identity = (baseline.get('data') or {}).get('path')
    model_path = Path(binding['path']).parent.parent.name
    for path in sorted(Path(binding['path']).parent.parent.rglob('*.toml')):
        if path.stat().st_size > 100_000: continue
        try: data = load_config(path)
        except ValueError: continue
        base = data.get('base_config') or {}
        if not data_identity or (base.get('data') or {}).get('path') != data_identity: continue
        if Path(str(data.get('program') or '')).stem != model_path: continue
        def walk(node, prefix=''):
            for key,value in node.items():
                name = f'{prefix}.{key}' if prefix else key
                if isinstance(value,dict): yield from walk(value,name)
                elif isinstance(value,list) and value: yield name,value
        for name, spec in walk((data.get('optimization') or {}).get('space') or {}):
            kind = spec[0]
            if kind == 'categorical' and len(spec) == 2 and isinstance(spec[1],list): values = spec[1]
            elif kind in {'uniform','loguniform','int'} and len(spec) == 3 and all(type(v) in {int,float} for v in spec[1:]):
                low,high = spec[1:]
                if low > high or kind == 'loguniform' and low <= 0: continue
                middle = math.sqrt(low*high) if kind == 'loguniform' else (low+high)/2
                values = [low,int(round(middle)) if kind == 'int' else middle,high]
            else:
                if warnings is not None: warnings.append(f'{path.relative_to(source).as_posix()}:{name}：不支持的作者采样器 {kind}，该维度未执行搜索')
                continue
            if not all(type(v) in {str,bool,int,float} and (type(v) is not float or math.isfinite(v)) for v in values):
                if warnings is not None: warnings.append(f'{name}：搜索取值包含不能安全序列化的类型，该维度未执行搜索')
                continue
            result['config:'+name] = {'values':list(dict.fromkeys(values)), 'source':path.relative_to(source).as_posix(),
                                     'origin':'optimization.space.'+name, 'line':None}
    return result


def trial_command(argv, parameters, binding, directory):
    from .tuning import parameter_argv
    cli = {name: value for name, value in parameters.items() if not name.startswith('config:')}
    result = parameter_argv(argv, cli)
    config_values = {name[7:]: value for name, value in parameters.items() if name.startswith('config:')}
    if not config_values: return result
    original = Path(binding['path'])
    data = load_config(original)
    for name, value in config_values.items():
        parts = name.split('.'); node = data
        for part in parts[:-1]: node = node[part]
        if parts[-1] not in node: raise ValueError('调参字段已变化')
        node[parts[-1]] = value
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / original.name
    if original.suffix == '.toml':
        # Keep author comments, keys and table structure; replace only a unique numeric assignment.
        body = original.read_text(encoding='utf-8'); section = ''; seen = set(); lines = []
        for line in body.splitlines(keepends=True):
            table = re.match(r'^\s*\[([^\]]+)\]\s*(?:#.*)?$', line.strip())
            if table: section = table[1]
            assignment = re.match(r'''^(\s*)([\w-]+)(\s*=\s*)([-+0-9.eE]+|true|false|"(?:[^"\\]|\\.)*"|'[^']*')(\s*(?:#.*)?)(\r?\n)?$''', line)
            if assignment:
                name = f'{section}.{assignment[2]}' if section else assignment[2]
                if name in config_values:
                    if name in seen: raise ValueError('TOML 调参字段不唯一')
                    seen.add(name)
                    value = config_values[name]
                    literal = json.dumps(value,ensure_ascii=False,allow_nan=False)
                    line = assignment[1] + assignment[2] + assignment[3] + literal + assignment[5] + (assignment[6] or '')
            lines.append(line)
        if seen != set(config_values): raise ValueError('TOML 字段无法无损定位，未运行调参')
        target.write_text(''.join(lines), encoding='utf-8'); tomllib.loads(target.read_text(encoding='utf-8'))
    else:
        target.write_text(json.dumps(data, indent=2) if original.suffix == '.json' else yaml.safe_dump(data, sort_keys=False), encoding='utf-8')
    index = binding['argv_index']
    result[index] = binding['argument'].split('=',1)[0] + '=' + str(target) if '=' in binding['argument'] else str(target)
    return result
