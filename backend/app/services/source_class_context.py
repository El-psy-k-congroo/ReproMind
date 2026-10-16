"""Static constructor reachability for repositories containing multiple models."""
import ast
import re


def call_name(node):
    return node.func.id if isinstance(node.func, ast.Name) else node.func.attr if isinstance(node.func, ast.Attribute) else ''


def class_bindings(trees, entrypoints, selectors=()):
    classes = {}
    for tree in trees.values():
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                classes.setdefault(node.name, []).append(node)
    normalized = lambda value: re.sub(r'[^a-z0-9]', '', value.casefold())
    roots = {name for name in classes if normalized(name) in selectors}
    if not roots:
        class EntryCalls(ast.NodeVisitor):
            def visit_ClassDef(self, node):
                pass  # Constructor calls inside model definitions are descendants.
            def visit_Call(self, node):
                if call_name(node) in classes:
                    roots.add(call_name(node))
                self.generic_visit(node)
        for filename, tree in trees.items():
            if filename.replace('\\', '/').casefold() in {str(name).replace('\\', '/').casefold() for name in entrypoints}:
                EntryCalls().visit(tree)
    active = set(roots)
    pending = list(roots)
    while pending:
        name = pending.pop()
        for definition in classes.get(name, []):
            for node in ast.walk(definition):
                if isinstance(node, ast.Call) and call_name(node) in classes and call_name(node) not in active:
                    active.add(call_name(node)); pending.append(call_name(node))
    # Ambiguous roots or duplicate class names cannot justify excluding evidence.
    return {'roots': sorted(roots), 'reachable': sorted(active),
            'unambiguous': len(roots) == 1 and all(len(classes[name]) == 1 for name in active),
            'scope': '静态构造引用闭包，不代表分支已执行或张量形状已验证'}


def enclosing_class(tree, line):
    owners = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef) and node.lineno <= line <= node.end_lineno]
    return min(owners, key=lambda node: node.end_lineno - node.lineno).name if owners else ''
