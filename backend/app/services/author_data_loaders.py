"""Find explicit author Hub dataset loaders; a plan is not a downloaded dataset."""
import ast
import re
from pathlib import Path


def declared_hub_loaders(files, source: Path):
    found={}; parsed=[]; literals=set()
    for file in files:
        if not file.is_file() or file.stat().st_size>200_000:continue
        try:tree=ast.parse(file.read_text(encoding='utf-8',errors='replace'))
        except (OSError,SyntaxError):continue
        parsed.append((file,tree))
        literals.update(n.value for n in ast.walk(tree) if isinstance(n,ast.Constant)
                        and isinstance(n.value,str) and re.fullmatch(r'[a-z][a-z0-9_-]{1,50}',n.value))
        for node in ast.walk(tree):
            if not isinstance(node,ast.If) or not isinstance(node.test,ast.Compare):continue
            test=node.test
            if len(test.ops)!=1 or not isinstance(test.ops[0],ast.Eq) or len(test.comparators)!=1:continue
            alias=test.comparators[0]
            if not isinstance(alias,ast.Constant) or not isinstance(alias.value,str):continue
            variable=test.left
            name=variable.id if isinstance(variable,ast.Name) else variable.attr if isinstance(variable,ast.Attribute) else ''
            if name not in {'dataset','dataset_name','dataset_id'}:continue
            for call in ast.walk(ast.Module(body=node.body,type_ignores=[])):
                if not isinstance(call,ast.Call) or not isinstance(call.func,ast.Name) or call.func.id!='load_dataset' or not call.args:continue
                value=call.args[0]
                if isinstance(value,ast.Constant) and isinstance(value.value,str) and re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+',value.value):
                    found[re.sub(r'[-_ ]','',alias.value.casefold())]={'repository':value.value,'repository_name':alias.value,
                        'loader_file':file.relative_to(source).as_posix(),'loader_line':call.lineno}
                    break
    # Resolve only an explicit author namespace and aliases actually present in
    # the author's source. Arbitrary formatted URLs are not dataset evidence.
    for file,tree in parsed:
        for function in ast.walk(tree):
            if not isinstance(function,(ast.FunctionDef,ast.AsyncFunctionDef)):continue
            parameters={arg.arg for arg in function.args.args}
            for call in ast.walk(function):
                if not isinstance(call,ast.Call) or not isinstance(call.func,ast.Name) or call.func.id!='load_dataset' or not call.args:continue
                value=call.args[0]
                if not isinstance(value,ast.JoinedStr) or len(value.values)!=2:continue
                prefix,variable=value.values
                if not (isinstance(prefix,ast.Constant) and isinstance(prefix.value,str)
                        and re.fullmatch(r'[A-Za-z0-9_.-]+/',prefix.value)
                        and isinstance(variable,ast.FormattedValue) and variable.conversion==-1
                        and variable.format_spec is None and isinstance(variable.value,ast.Name)
                        and variable.value.id in parameters and variable.value.id in {'dataset','dataset_name','dataset_id'}):continue
                for alias in sorted(literals):
                    key=re.sub(r'[-_ ]','',alias.casefold())
                    found.setdefault(key,{'repository':prefix.value+alias,'repository_name':alias,
                        'loader_file':file.relative_to(source).as_posix(),'loader_line':call.lineno,
                        'resolution':'author_namespace_and_literal_alias'})
    return found
