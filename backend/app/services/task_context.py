"""Resolve task scope from author evidence; never import the research repository."""
import re
import ast
import os
from pathlib import Path

from .source_selection import entry_sources, native_path

EXCLUDED = {'.git', '.venv', '.conda', '__pycache__', 'node_modules', 'data', 'datasets'}


def terms(text):
    return {s.casefold() for s in (re.findall(r'[A-Za-z][A-Za-z0-9_.-]{2,}', text) + re.findall(r'[A-Za-z][A-Za-z0-9]{2,}', text))
            if s.casefold() not in {'python', 'train', 'model', 'paper', 'dataset', 'with', 'the'}}


def rank_entries(source, manifest, goal):
    """Keep documented commands authoritative, score their local task evidence."""
    source = native_path(source)
    entries = list(manifest.get('entrypoints') or [])
    excluded_text = re.findall(r'(?:不要|不运行|不使用|不用|排除|而非|不选|not\s+use|do\s+not\s+run)\s*([^，,；;。\n]+)', goal, re.I)
    negative = terms(' '.join(excluded_text))
    positive_goal = re.sub(r'(?:不要|不运行|不使用|不用|排除|而非|不选|not\s+use|do\s+not\s+run)\s*[^，,；;。\n]+', '', goal, flags=re.I)
    tokens = terms(positive_goal) - negative
    training = not re.search(r'采样|推理|预测|仅评估|编码|解码|不训练|inference|sampling|decode', positive_goal,re.I)
    for path in ranked_source_files(source, goal):
        if path.suffix != '.py' or path.stat().st_size > 250_000 or any(part in {'tests','test'} for part in path.relative_to(source).parts): continue
        identifier = re.sub(r'[^a-z0-9]', '', path.stem.casefold())
        if training and not any(len(token) >= 4 and re.sub(r'[^a-z0-9]', '', token) in identifier for token in tokens): continue
        body = path.read_text(encoding='utf-8', errors='replace')
        try: tree = ast.parse(body)
        except SyntaxError: continue
        guarded = any(isinstance(node,ast.If) and '__name__' in ast.unparse(node.test) and '__main__' in ast.unparse(node.test) for node in tree.body)
        direct = any(isinstance(node,ast.Expr) and isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Attribute)
                     and node.value.func.attr in {'backward','fit'} for node in tree.body)
        if guarded and not training or (guarded or direct) and re.search(r'\.(?:backward|fit|train_with\w*)\s*\(', body):
            name = path.relative_to(source).as_posix()
            if name not in entries: entries.append(name)
    def score(entry):
        evidence = entry + '\n'
        for command in manifest.get('readme_commands', []):
            if command['entrypoint'] == entry:
                evidence += ' '.join(command['args']) + '\n'
        path = source / entry
        if path.is_file():
            evidence += path.read_text(encoding='utf-8', errors='replace')[:100_000]
        # Filename/model identifiers outweigh an incidental import or mention.
        normalized = re.sub(r'[^a-z0-9]', '', entry.casefold())
        stem = re.sub(r'[^a-z0-9]', '', Path(entry).stem.casefold())
        value = sum(36 if re.sub(r'[^a-z0-9]', '', token) == stem else
                    12 if re.sub(r'[^a-z0-9]', '', token) in normalized else
                    2 if token in evidence.casefold() else 0 for token in tokens)
        value -= sum(100 for token in negative if re.sub(r'[^a-z0-9]', '', token) == stem)
        if not training:
            value += 25 if re.search(r'(?:^|[_/])(?:eval|test|sample|sampling|encode|decode|predict|infer|compressor)(?:[_/.]|$)', entry,re.I) else 0
            value -= 15 if re.search(r'(?:^|[_/])train(?:ing)?[^/]*\.py$', entry,re.I) else 0
        if training and value:
            value += 8 if re.search(r'\.backward\s*\(|optimizer\.step\s*\(',evidence) else 0
            value -= 20 if re.search(r'(?:^|[_/])(?:analysis|eval|ensemble|decode|predict|infer)(?:[_/.]|$)',entry,re.I) else 0
        return value
    # When the question has no model/entry evidence, keep the selected workflow.
    # A shorter unrelated example must not win a tie with the recorded entry.
    return sorted(entries, key=lambda entry: -score(entry))


def task_contract(job):
    execution = getattr(job,'execution_command',None) or {}
    environment = getattr(job,'environment_spec',None) or {}
    code = getattr(job, 'code_spec', None) or {}
    return {'run_id': job.id, 'source_root': execution.get('pythonpath') or code.get('source_path'),
            'entrypoints': code.get('entrypoints') or execution.get('entrypoint_candidates') or [],
            'commands': execution.get('commands') or [], 'cwd': execution.get('cwd'),
            'data_root': execution.get('data_dir') or (str(Path(job.workspace_path)/'data') if getattr(job,'workspace_path',None) else None), 'datasets': (getattr(job, 'dataset_spec', None) or {}).get('dataset_mentions') or [],
            'output_root':str(Path(job.workspace_path)/'outputs'/job.id[:8]) if getattr(job,'workspace_path',None) else None,
            'environment_name': environment.get('environment_name'), 'environment_prefix': environment.get('prefix'),
            'python': environment.get('python'), 'hardware': environment.get('hardware_plan'),
            'gpu_requested': environment.get('gpu_requested'), 'runtime': environment.get('gpu_runtime'),
            'runtime_environment':execution.get('runtime_environment') or {},
            'goal': getattr(job,'goal','') or (getattr(job,'result_json',None) or {}).get('goal', '')}


def question_entrypoints(source, manifest, question):
    """Reading a current experiment doesn't select a new execution workflow.

    Re-rank only if the question explicitly names another entry/model identifier;
    generic words such as forward/optimizer or a read-only 'do not run' suffix
    must not switch to whichever unrelated example mentions them.
    """
    entries=list(manifest.get('entrypoints') or [])
    tokens=terms(question)
    explicit=any(Path(entry).name.casefold() in question.casefold() or
        any(len(token)>=3 and re.sub(r'[^a-z0-9]','',token)==re.sub(r'[^a-z0-9]','',Path(entry).stem.casefold())
            for token in tokens) for entry in entries)
    return rank_entries(source,manifest,question) if explicit else entries


def ranked_source_files(source, question='', entrypoints=None, preferred=''):
    """Import closure first, then explicitly named files and task-local configs."""
    source = native_path(source)
    closure = entry_sources(source, entrypoints) if entrypoints else []
    closure_set = set(closure)
    tokens = terms(question)
    entry_dir = Path((entrypoints or [''])[0]).parent
    files = []
    for directory, dirs, names in os.walk(source, followlinks=False):
        parent = Path(directory)
        dirs[:] = [name for name in dirs if name not in EXCLUDED and not (parent/name).is_symlink()]
        for name in names:
            path = parent/name
            if path.suffix.lower() not in {'.py','.yaml','.yml','.toml','.json','.md','.txt'} or name.startswith('.repromind'): continue
            if path.suffix == '.json' and not re.search(r'config|cfg|hyper',name,re.I) and path.relative_to(source).as_posix() not in question and path.relative_to(source).as_posix() != preferred: continue
            if not path.is_symlink(): files.append(path)
    def score(path):
        name = path.relative_to(source).as_posix()
        explicit = name == preferred or name.casefold() in question.casefold()
        identifier = re.sub(r'[^a-z0-9]', '', path.stem.casefold())
        matched = sum(1 for token in tokens if re.sub(r'[^a-z0-9]', '', token) in identifier)
        return (-int(explicit), -matched, -int(path in closure_set),
                -int(entry_dir != Path('.') and entry_dir in Path(name).parents), len(Path(name).parts), name)
    return sorted(files, key=score)
