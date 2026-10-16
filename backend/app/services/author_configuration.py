"""Bind dataset/model to an existing author config, without inventing scientific settings."""
import ast
import re
from pathlib import Path

from .hyperparameter_contract import load_config
from .source_selection import entry_sources, native_path


def key(value): return re.sub(r'[^a-z0-9]', '', str(value).casefold())


def choose_configs(source, entry, datasets):
    source = native_path(source)
    model = key(Path(entry).stem)
    found = {}
    paths = sorted(source.rglob('*'))
    # Explicit dataset directories are stronger than a generic config's body.
    # Avoid parsing every published seed/result config on each chat question.
    literal = {label for label in datasets if any(path.suffix in {'.yml','.yaml','.toml','.json'}
        and key(label) in {key(part) for part in path.relative_to(source).with_suffix('').parts} for path in paths)}
    for path in paths:
        if path.suffix not in {'.yml','.yaml','.toml','.json'}: continue
        if path.suffix == '.json' and not re.search(r'config|cfg',path.name,re.I): continue
        if path.suffix == '.toml' and model not in {key(part) for part in path.parts}: continue
        path_keys = {key(part) for part in path.relative_to(source).with_suffix('').parts}
        relevant = [label for label in datasets if label not in literal or key(label) in path_keys]
        if not relevant: continue
        if path.is_symlink() or not path.is_file() or path.stat().st_size > 100_000: continue
        if any(part in {'.git','node_modules','data','datasets'} for part in path.relative_to(source).parts): continue
        try: data = load_config(path)
        except Exception: continue
        if not isinstance(data, dict) or 'optimization' in data: continue
        info = data.get('data') or {}
        identities = [path.stem, *(part for part in path.relative_to(source).parts[:-1])]
        if isinstance(info, dict): identities += [info.get('dataset'), Path(str(info.get('path') or '')).name]
        if path.suffix == '.toml' and model not in {key(part) for part in path.parts} and key(Path(str(data.get('program') or '')).stem) != model: continue
        for label in relevant:
            if key(label) in {key(value) for value in identities if value is not None}:
                found.setdefault(label, []).append(path)
    selected = {}
    for label, candidates in found.items():
        # Multiple seeds are explicit author settings; choose seed zero only when it exists.
        zero = [p for p in candidates if p.stem == '0']
        tuned_zero = [p for p in zero if p.parent.name == 'tuned']
        if len(tuned_zero) == 1: selected[label] = tuned_zero[0]
        elif len(candidates) == 1 or len(zero) == 1: selected[label] = zero[0] if zero else candidates[0]
    return selected


def config_commands(source, entry, base, datasets, examples, run_workspace, *, phase='training'):
    canonical_source = str(source)
    source = native_path(source)
    text = (source/entry).read_text(encoding='utf-8', errors='replace')
    bodies = '\n'.join(p.read_text(encoding='utf-8', errors='replace') for p in entry_sources(source,[entry]))
    flag = re.search(r'add_argument\(\s*["\'](--(?:config|cfg|config-file|config_file))["\']', bodies)
    positional = bool(re.search(r'(?:sys\.argv\[1\]|\b(?:[A-Za-z_]\w*\.)?load_config\()', text))
    if not flag and not positional: return []
    configs = choose_configs(source, entry, datasets)
    commands = []
    for label, config in configs.items():
        output_bindings = {}
        argv = [*base, flag[1], str(config)] if flag else [*base, str(config)]
        if phase == 'sampling':
            sampling = [row for row in examples if '--sample' in row['args']]
            example = next((row for row in sampling if '--use_pretrained' in row['args']),sampling[0] if sampling else None)
        else:
            example = next((row for row in examples if row['args'] and not any(token in row['args'] for token in ['--sample','--test','--fid'])), None)
        if example:
            replacements = {'{DATASET}':config.stem, '{PROJECT_PATH}':str(run_workspace/'output') if run_workspace else '',
                            '{MODEL_NAME}':run_workspace.name if run_workspace else 'repromind'}
            args = []
            from .tuning import numeric_flags
            defaults = numeric_flags(bodies)
            for index,value in enumerate(example['args']):
                original_value = value
                if phase == 'sampling' and value == '--fid': continue
                if re.fullmatch(r'\{[^}]+\}',value) and index and example['args'][index-1].removeprefix('--') in defaults:
                    value = str(defaults[example['args'][index-1].removeprefix('--')])
                for old,new in replacements.items(): value = value.replace(old,new)
                if run_workspace and '{PROJECT_PATH}' in original_value and index and example['args'][index-1].startswith('--'):
                    root = native_path(run_workspace/'output')
                    destination = native_path(value)
                    if Path(value).is_absolute() and (destination == root or root in destination.parents):
                        output_bindings[example['args'][index-1].lstrip('-')] = value
                args.append(value)
            if flag and flag[1] in args:
                index = args.index(flag[1]) + 1
                if index < len(args): args[index] = str(config)
            if not any(re.search(r'\{[^}]+\}|\$[A-Za-z{]', value) for value in args): argv = [*base, *args]
        # A documented sampling command may omit a required bookkeeping option
        # shown in another example of the same entry. Reuse only a fully bound
        # author value; never invent an unresolved scientific parameter.
        from .cli_contract import missing_required_options
        bindings={'{DATASET}':config.stem,'{PROJECT_PATH}':str(run_workspace/'output') if run_workspace else '',
                  '{MODEL_NAME}':run_workspace.name if run_workspace else 'repromind'}
        for option in missing_required_options(bodies,argv):
            for row in examples:
                values=row['args']
                if option not in values: continue
                index=values.index(option)+1
                if index>=len(values) or values[index].startswith('-'): continue
                value=values[index]
                for old,new in bindings.items(): value=value.replace(old,new)
                if not value or re.search(r'\{[^}]+\}|\$[A-Za-z{]',value): continue
                argv.extend([option,value]);break
        # Author result directories in a checkout may already contain DONE/stats files.
        # An explicit supported output option keeps a new run from reusing published results.
        if run_workspace and positional and re.search(r'def load_config\(',bodies) and re.search(r'add_argument\([^\n]*["\']--output["\']',bodies):
            output = run_workspace.parents[1]/'outputs'/run_workspace.name[:8]/key(label)[:32]
            if not any(v in {'-o','--output'} or v.startswith('--output=') for v in argv):
                argv += ['--output',str(output)]
                output_bindings['output'] = str(output)
        commands.append({'argv':argv, 'cwd':canonical_source, 'dataset':label,
                         'config_path':str(config), 'command_source':'author_config_and_documented_cli',
                         'isolated_output_bindings':output_bindings,
                         'config_selection_note':'选用匹配数据集和模型的作者配置；多种子目录仅选择作者 seed 0，不代表多次实验平均'})
    return commands
