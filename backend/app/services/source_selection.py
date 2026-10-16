"""Read the selected author entry and its local imports without executing code."""
import ast
import os
from pathlib import Path


def native_path(path):
    root = Path(path).resolve()
    if os.name == 'nt' and not str(root).startswith('\\\\?\\'):
        absolute = str(root)
        return Path('\\\\?\\UNC\\' + absolute[2:] if absolute.startswith('\\\\') else '\\\\?\\' + absolute)
    return root


def _declared_search_paths(tree,path,root):
    def value(node):
        if isinstance(node,ast.Constant) and isinstance(node.value,str):return Path(node.value)
        if isinstance(node,ast.Name) and node.id=='__file__':return path
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute):
            if node.func.attr=='dirname' and len(node.args)==1:
                found=value(node.args[0]);return found.parent if found is not None else None
            if node.func.attr=='join':
                parts=[value(n) for n in node.args]
                if parts and all(p is not None for p in parts):return parts[0].joinpath(*parts[1:])
        return None
    result=[]
    for call in ast.walk(tree):
        if not (isinstance(call,ast.Call) and isinstance(call.func,ast.Attribute)
                and call.func.attr in {'insert','append'} and isinstance(call.func.value,ast.Attribute)
                and isinstance(call.func.value.value,ast.Name) and call.func.value.value.id=='sys'
                and call.func.value.attr=='path' and call.args):continue
        candidate=value(call.args[-1])
        if candidate is None:continue
        candidate=(candidate if candidate.is_absolute() else path.parent/candidate).resolve()
        if (candidate==root or root in candidate.parents) and candidate.is_dir():result.append(candidate)
    return result


def entry_sources(source: Path, entrypoints=None, limit=200):
    root=source.resolve()
    if os.name=='nt' and not str(root).startswith('\\\\?\\'):
        absolute=str(root);root=Path('\\\\?\\UNC\\'+absolute[2:] if absolute.startswith('\\\\') else '\\\\?\\'+absolute)
    starts=[root/name for name in (entrypoints or [])[:1] if (root/name).is_file()]
    if not starts:
        return [p for p in root.rglob('*.py') if p.is_file()][:limit]
    pending=starts[:];seen=set();result=[]
    while pending and len(result)<limit:
        path=pending.pop(0).resolve()
        if path in seen or root not in path.parents or not path.is_file() or path.stat().st_size>250_000:
            continue
        seen.add(path);result.append(path)
        try:tree=ast.parse(path.read_text(encoding='utf-8',errors='replace'))
        except (SyntaxError,OSError):continue
        search_paths=_declared_search_paths(tree,path,root)
        for node in ast.walk(tree):
            names=[];bases=[path.parent,*search_paths,root,root/'src']
            if isinstance(node,ast.Import):names=[alias.name for alias in node.names]
            elif isinstance(node,ast.ImportFrom):
                if node.level:
                    base=path.parent
                    for _ in range(node.level-1):base=base.parent
                    bases=[base]
                names=[node.module or '']
                names += ['.'.join(filter(None,[node.module,alias.name])) for alias in node.names if alias.name!='*']
            for name in names:
                if not name:
                    continue
                relative=Path(*name.split('.')) if name else Path()
                for base in bases:
                    for candidate in [base/relative.with_suffix('.py'),base/relative/'__init__.py']:
                        if candidate.is_file() and candidate.resolve() not in seen:pending.append(candidate)
    return result
