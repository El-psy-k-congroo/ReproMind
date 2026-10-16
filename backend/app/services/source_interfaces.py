"""Small, evidence-only cross-module checks. Not a substitute for executing the code."""
import ast
import os
from pathlib import Path


def _modules(root, revisions):
    if os.name=='nt' and not str(root).startswith('\\\\?\\'):
        absolute=str(root.resolve())
        root=Path('\\\\?\\UNC\\'+absolute[2:] if absolute.startswith('\\\\') else '\\\\?\\'+absolute)
    result = {}
    for path in sorted(root.rglob('*.py'))[:300]:
        if path.is_symlink() or path.stat().st_size > 250_000 or '.git' in path.parts:
            continue
        name = path.relative_to(root).as_posix()
        text = revisions.get(name, path.read_text(encoding='utf-8', errors='replace'))
        try:
            result[name] = (text, ast.parse(text))
        except SyntaxError:
            continue
    return result


def _provider(modules, filename, function):
    tree = modules[filename][1]
    local = next((n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == function), None)
    if local:
        return filename, local
    for node in tree.body:
        if not isinstance(node, ast.ImportFrom):
            continue
        alias = next((a for a in node.names if (a.asname or a.name) == function or a.name == '*' and not function.startswith('_')), None)
        if not alias:
            continue
        suffix = (node.module or '').replace('.', '/') + '.py'
        candidates = [name for name in modules if name == suffix or name.endswith('/' + suffix)]
        if len(candidates) == 1:
            return _provider_local(modules, candidates[0], function if alias.name == '*' else alias.name)
    return None


def _provider_local(modules, filename, function):
    node = next((n for n in modules[filename][1].body if isinstance(n, ast.FunctionDef) and n.name == function), None)
    return (filename, node) if node else None


def _kind(modules, provider, seen=None):
    if not provider:
        return None
    filename, function = provider
    seen = set(seen or [])
    identity = (filename, function.name)
    if identity in seen:
        return None
    seen.add(identity)
    def own_nodes(node):
        yield node
        for child in ast.iter_child_nodes(node):
            if not isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                yield from own_nodes(child)
    returns = [n.value for n in own_nodes(function) if isinstance(n, ast.Return)]
    if len(returns) != 1:
        return None  # Dynamic/branching return contracts are unknown, not guessed.
    value = returns[0]
    if isinstance(value, ast.Tuple):
        return 'tuple'
    if isinstance(value, ast.List):
        return 'list'
    if isinstance(value, ast.Call) and isinstance(value.func, ast.Name):
        local = next((n for n in function.body if isinstance(n, ast.FunctionDef) and n.name == value.func.id), None)
        return _kind(modules, (filename, local) if local else _provider(modules, filename, value.func.id), seen)
    if isinstance(value, ast.Name):
        for statement in reversed(function.body):
            if isinstance(statement, ast.Assign) and any(isinstance(t, ast.Name) and t.id == value.id for t in statement.targets):
                if isinstance(statement.value, (ast.Tuple, ast.List)):
                    return 'tuple' if isinstance(statement.value, ast.Tuple) else 'list'
            # Recognize a generic list/tuple converter only when both branches are
            # explicit: list input is kept a list, all other inputs become a tuple.
            if isinstance(statement, ast.If) and len(statement.orelse) == 1:
                condition, other = statement.test, statement.orelse[0]
                if not (isinstance(condition, ast.Call) and isinstance(condition.func, ast.Name)
                        and condition.func.id == 'isinstance' and len(condition.args) == 2
                        and isinstance(condition.args[0], ast.Name) and condition.args[0].id == value.id
                        and isinstance(condition.args[1], ast.Name) and condition.args[1].id == 'list'
                        and isinstance(other, ast.Assign) and len(other.targets) == 1
                        and isinstance(other.targets[0], ast.Name) and other.targets[0].id == value.id
                        and isinstance(other.value, ast.Call) and isinstance(other.value.func, ast.Name)):
                    continue
                if any(isinstance(n, ast.Name) and n.id == value.id and isinstance(n.ctx, ast.Store)
                       for child in statement.body for n in ast.walk(child)):
                    continue
                local = next((n for n in function.body if isinstance(n, ast.FunctionDef) and n.name == other.value.func.id), None)
                if _kind(modules, (filename, local) if local else _provider(modules, filename, other.value.func.id), seen) == 'tuple':
                    return 'sequence'
    return None


def interface_issues(root: Path, revisions=None):
    modules = _modules(root, revisions or {})
    issues = []
    for filename, (_, tree) in modules.items():
        for block in [tree, *(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef))]:
            assignments = {}
            for node in block.body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    continue
                for call in ast.walk(node):
                    if isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute) and isinstance(call.func.value, ast.Name):
                        kind = _kind(modules, assignments.get(call.func.value.id))
                        if kind in {'tuple', 'list', 'sequence'} and call.func.attr in {'toarray', 'tocoo', 'tocsr', 'todense'}:
                            provider = assignments[call.func.value.id]
                            issues.append(f'{filename}:{call.lineno}：{provider[0]} 的 {provider[1].name} 明确返回 {kind}，'
                                          f'调用方却使用 .{call.func.attr}()；需协调数据接口')
                # Evaluate RHS consumers using the preceding value before assigning LHS.
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        for name in ast.walk(target):
                            if isinstance(name, ast.Name) and isinstance(name.ctx, ast.Store):
                                assignments.pop(name.id, None)
                        if isinstance(target, ast.Name):
                            call = node.value
                            assignments[target.id] = (_provider(modules, filename, call.func.id)
                                if isinstance(call, ast.Call) and isinstance(call.func, ast.Name) else None)
                elif isinstance(node, (ast.If, ast.For, ast.While, ast.Try, ast.With, ast.AugAssign, ast.AnnAssign)):
                    # Conditional conversions invalidate a simple return-kind proof.
                    for name in ast.walk(node):
                        if isinstance(name, ast.Name) and isinstance(name.ctx, ast.Store):
                            assignments.pop(name.id, None)
    return list(dict.fromkeys(issues))[:12]


def provider_context(root: Path, filename: Path):
    """Bounded related definitions for the repair prompt, rather than only a traceback line."""
    modules = _modules(root, {})
    relative = filename.relative_to(root).as_posix()
    if relative not in modules:
        return {}
    context = {}
    for call in ast.walk(modules[relative][1]):
        if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name):
            continue
        provider = _provider(modules, relative, call.func.id)
        if provider and provider[0] != relative:
            name, function = provider
            definition = ast.get_source_segment(modules[name][0], function) or ''
            context[name + ':' + function.name] = definition if len(definition) <= 2200 else definition[:700] + '\n# ... excerpt ...\n' + definition[-1500:]
            if sum(len(value) for value in context.values()) >= 7000:
                break
    # Include the actual helper a preprocessing function returns, not just its caller.
    for identity in list(context):
        name, function_name = identity.rsplit(':', 1)
        provider = _provider_local(modules, name, function_name)
        if provider:
            for node in ast.walk(provider[1]):
                if isinstance(node, ast.Return) and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name):
                    helper = _provider(modules, name, node.value.func.id)
                    if helper and sum(len(v) for v in context.values()) < 9000:
                        definition = ast.get_source_segment(modules[helper[0]][0], helper[1]) or ''
                        context[helper[0] + ':' + helper[1].name] = definition[:2200]
    return context
