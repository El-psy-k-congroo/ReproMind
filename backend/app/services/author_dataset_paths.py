"""Resolve literal torchvision input roots from author code, without running it."""
import ast
from pathlib import Path
from .source_selection import entry_sources, native_path


def torchvision_roots(source, entrypoints, cwd, dataset, workspace):
    source=native_path(source); workspace=native_path(workspace); cwd=native_path(cwd)
    found=[]
    for path in entry_sources(source,entrypoints):
        try:tree=ast.parse(path.read_text(encoding='utf-8',errors='replace'))
        except (OSError,SyntaxError):continue
        names=set()
        for node in ast.walk(tree):
            if isinstance(node,ast.Import):
                for alias in node.names:
                    if alias.name=='torchvision':names.add((alias.asname or 'torchvision')+'.datasets.'+dataset)
            if isinstance(node,ast.ImportFrom):
                for alias in node.names:
                    if node.module=='torchvision' and alias.name=='datasets':names.add((alias.asname or 'datasets')+'.'+dataset)
                    if node.module=='torchvision.datasets' and alias.name==dataset:names.add(alias.asname or dataset)
        for call in ast.walk(tree):
            if not isinstance(call,ast.Call) or ast.unparse(call.func) not in names:continue
            root=next((k.value for k in call.keywords if k.arg=='root'),call.args[0] if call.args else None)
            if not isinstance(root,ast.Constant) or not isinstance(root.value,str):continue
            raw=Path(root.value); resolved=native_path(raw if raw.is_absolute() else cwd/raw)
            if workspace not in resolved.parents:
                raise ValueError('作者数据根目录不在当前项目工作区，未向外部目录写入数据：'+str(resolved))
            if not any(row['path']==str(resolved) for row in found):
                found.append({'path':str(resolved),'dataset':dataset,'file':path.relative_to(source).as_posix(),
                              'line':call.lineno,'literal_root':root.value})
    return found
