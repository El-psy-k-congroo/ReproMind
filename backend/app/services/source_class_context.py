"""Static constructor reachability for repositories containing multiple models."""
import ast
import re


def call_name(node):
    return node.func.id if isinstance(node.func, ast.Name) else node.func.attr if isinstance(node.func, ast.Attribute) else ''


FACTORIES = {'from_pretrained','from_config','load_from_checkpoint'}


def factory_class(node):
    return (node.func.value.id if isinstance(node.func,ast.Attribute) and node.func.attr in FACTORIES
            and isinstance(node.func.value,ast.Name) else '')


def class_bindings(trees, entrypoints, selectors=(), resolved_factories=()):
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
                name = factory_class(node) or call_name(node)
                if name in classes:
                    roots.add(name)
                self.generic_visit(node)
        for filename, tree in trees.items():
            if filename.replace('\\', '/').casefold() in {str(name).replace('\\', '/').casefold() for name in entrypoints}:
                EntryCalls().visit(tree)
    active = set(roots)
    pending = list(roots)
    while pending:
        name = pending.pop()
        for definition in classes.get(name, []):
            descendants = []
            for method in definition.body:
                options = [row['literal_options'] for row in resolved_factories
                           if row['class'] == name and isinstance(method, ast.FunctionDef)
                           and method.name.lstrip('_') == row['factory']]
                # Multiple unresolved factory branches cannot prove one option.
                values = options[0] if options and all(option == options[0] for option in options) else {}
                class Calls(ast.NodeVisitor):
                    def visit_If(self, node):
                        condition = _literal(node.test, values)
                        self.visit(node.test)
                        children = node.body if condition is True else node.orelse if condition is False else [*node.body, *node.orelse]
                        for child in children: self.visit(child)
                    def visit_Call(self, node):
                        descendants.append(node)
                        self.generic_visit(node)
                Calls().visit(method)
            descendants.extend(base for base in definition.bases if isinstance(base, ast.Name))
            for node in descendants:
                called = node.id if isinstance(node, ast.Name) else call_name(node)
                if called in classes and called not in active:
                    active.add(called); pending.append(called)
    # Ambiguous roots or duplicate class names cannot justify excluding evidence.
    return {'roots': sorted(roots), 'reachable': sorted(active),
            'unambiguous': len(roots) == 1 and all(len(classes[name]) == 1 for name in active),
            'complete_definitions': bool(roots) and all(len(classes[name]) == 1 for name in active),
            'scope': '静态构造引用闭包，不代表分支已执行或张量形状已验证'}


UNKNOWN = object()


def _literal(node, values):
    if isinstance(node, ast.Name):
        return values.get(node.id, UNKNOWN)
    if isinstance(node,ast.Attribute) and isinstance(node.value,ast.Name) and node.value.id=='args':
        return values.get(node.attr,UNKNOWN)
    if isinstance(node,ast.Compare) and len(node.ops)==1:
        left,right=_literal(node.left,values),_literal(node.comparators[0],values)
        if left is UNKNOWN or right is UNKNOWN:return UNKNOWN
        op=node.ops[0]
        if isinstance(op,ast.Eq):return left==right
        if isinstance(op,ast.NotEq):return left!=right
    if isinstance(node,ast.UnaryOp) and isinstance(node.op,ast.Not):
        value=_literal(node.operand,values)
        return UNKNOWN if value is UNKNOWN else not value
    try:return ast.literal_eval(node)
    except (ValueError,TypeError):return UNKNOWN


def cli_literal_values(tree, command):
    from .cli_contract import argument_options, option_values
    options = argument_options(ast.unparse(tree))
    values = {option['dest']: option['default'] for option in options.values() if option['default'] is not None}
    argv = command.get('argv', [])
    for name, option in options.items():
        supplied = option_values(argv, '--' + name)
        if supplied:
            try:
                convert = {'int': int, 'float': float}.get(option['kind'], str)
                converted = [convert(value) for value in supplied]
                values[option['dest']] = converted if option['nargs'] in {'+', '*'} or isinstance(option['nargs'], int) else converted[0]
            except ValueError:
                values.pop(option['dest'], None)
        elif '--' + name in argv:
            values.pop(option['dest'], None)
    # Entry code can rewrite parsed arguments. CLI defaults alone cannot then
    # decide a branch, even if the assignment happens later or conditionally.
    for node in ast.walk(tree):
        if (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
                and node.value.id == 'args' and not isinstance(node.ctx, ast.Load)):
            values.pop(node.attr, None)
    return values


def factory_bindings(trees, entrypoints, commands):
    """Resolve literal CLI guards and factory defaults, without evaluating code."""
    classes={}
    for filename,tree in trees.items():
        for node in ast.walk(tree):
            if isinstance(node,ast.ClassDef):classes.setdefault(node.name,[]).append((filename,node))
    results=[]
    for filename in entrypoints[:1]:
        tree=trees.get(filename)
        if tree is None:continue
        for command in commands[:3] or [{}]:
            values=cli_literal_values(tree,command)
            class Calls(ast.NodeVisitor):
                def __init__(self):self.guards=[]
                def visit_ClassDef(self,node):pass
                def visit_If(self,node):
                    condition=_literal(node.test,values)
                    branches = [('else',node.orelse)] if condition is False else [('then',node.body)] if condition is True else [('then',node.body),('else',node.orelse)]
                    for branch,children in branches:
                        self.guards.append({'condition':ast.unparse(node.test),'value':condition if type(condition) is bool else 'unknown','branch':branch})
                        for child in children:self.visit(child)
                        self.guards.pop()
                def visit_Call(self,node):
                    name=factory_class(node)
                    definitions=classes.get(name,[])
                    if len(definitions)==1:
                        owner,definition=definitions[0]
                        method=next((item for item in definition.body if isinstance(item,ast.FunctionDef) and item.name in {node.func.attr,'_'+node.func.attr}),None)
                        effective={};unknown={}
                        if method:
                            args=[*method.args.posonlyargs,*method.args.args]
                            for arg,default in zip(args[-len(method.args.defaults):],method.args.defaults):
                                value=_literal(default,{})
                                if value is not UNKNOWN:effective[arg.arg]=value
                            for arg,default in zip(method.args.kwonlyargs,method.args.kw_defaults):
                                if default is None:continue
                                value=_literal(default,{})
                                if value is not UNKNOWN:effective[arg.arg]=value
                        for keyword in node.keywords:
                            value=_literal(keyword.value,values)
                            if keyword.arg is None:
                                effective.clear();unknown['**kwargs']=ast.unparse(keyword.value)
                            elif value is UNKNOWN:
                                effective.pop(keyword.arg,None);unknown[keyword.arg]=ast.unparse(keyword.value)
                            else:effective[keyword.arg]=value
                        results.append({'file':filename,'line':node.lineno,'class':name,'factory':node.func.attr,
                            'definition_file':owner,'literal_options':effective,'unknown_options':unknown,
                            'guards':list(self.guards),
                            'dataset':command.get('dataset'),'cli_defaults_and_overrides':values,
                            'scope':'仅根据本次 CLI 和工厂签名静态核对可选分支，不证明工厂已调用、权重已下载或实际模型类型'})
                    self.generic_visit(node)
            Calls().visit(tree)
    return results[:6]


def enclosing_class(tree, line):
    owners = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef) and node.lineno <= line <= node.end_lineno]
    return min(owners, key=lambda node: node.end_lineno - node.lineno).name if owners else ''
