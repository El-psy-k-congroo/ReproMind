"""Read documented repository paths as data; never execute README shell snippets."""
import ast
import re
from pathlib import Path
from .source_selection import native_path, entry_sources

RESERVED = {'HOME','USERPROFILE','PATH','PYTHONPATH','CONDA_PREFIX','CODEX_HOME','SYSTEMROOT','COMSPEC'}


def documented_environment(source, entries):
    source = native_path(source)
    required = set()
    for path in entry_sources(source,entries):
        try: tree = ast.parse(path.read_text(encoding='utf-8',errors='replace'))
        except SyntaxError: continue
        for node in ast.walk(tree):
            if isinstance(node,ast.Subscript) and isinstance(node.value,ast.Attribute) and isinstance(node.value.value,ast.Name) and node.value.value.id=='os' and node.value.attr=='environ' and isinstance(node.slice,ast.Constant) and isinstance(node.slice.value,str):
                required.add(node.slice.value)
    variables, evidence = {}, []
    docs = [p for p in source.rglob('README*') if p.is_file() and '.git' not in p.parts and len(p.relative_to(source).parts)<=3][:30]
    for path in sorted(docs,key=lambda p:len(p.parts)):
        text = path.read_text(encoding='utf-8',errors='replace')[:100_000]
        for match in re.finditer(r'(?m)^\s*(?:export\s+|\$env:)([A-Z][A-Z0-9_]*)\s*=\s*(.+)$',text):
            name,value=match.groups();value=value.strip().strip('"\'')
            if name not in required or name in RESERVED: continue
            repo_root = bool(re.fullmatch(r'\$\(pwd\)|`pwd`|\$PWD|\$\{PWD\}',value) or
                             re.fullmatch(r'<[^>]*(?:repository|project|repo)[^>]*(?:root|directory)[^>]*>',value,re.I))
            if not repo_root: continue
            # A documentation placeholder explicitly denotes this checkout, not its author's machine.
            variables.setdefault(name,str(source))
            evidence.append({'name':name,'value':str(source),'source':path.relative_to(source).as_posix(),'declaration':match.group(0).strip()})
    return {'variables':variables,'evidence':evidence,'unresolved_required_names':sorted(required-set(variables))}
