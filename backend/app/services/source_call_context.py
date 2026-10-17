"""Bound selected entry calls to local signatures and inherited method bodies."""
import ast
import re

from .source_class_context import UNKNOWN, _literal, cli_literal_values, factory_class
from .source_calculations import numbered_code


def selected_call_packets(trees, bodies, entrypoints, commands):
    classes = {}
    for filename, tree in trees.items():
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                classes.setdefault(node.name, []).append((filename, node))

    def method(owner, name, visited=None):
        visited = set(visited or ())
        if owner in visited or len(classes.get(owner, [])) != 1:
            return None
        visited.add(owner)
        filename, definition = classes[owner][0]
        found = [node for node in definition.body if isinstance(node, ast.FunctionDef) and node.name == name]
        if len(found) == 1:
            return filename, owner, found[0]
        inherited = [method(base.id, name, visited) for base in definition.bases if isinstance(base, ast.Name)]
        inherited = [row for row in inherited if row]
        return inherited[0] if len(inherited) == 1 else None

    packets, inactive = [], {}
    for filename in entrypoints[:1]:
        tree = trees.get(filename)
        if tree is None:
            continue
        for command in commands[:3] or [{}]:
            values = cli_literal_values(tree, command)
            instances = {}
            command_inactive = set()
            class Calls(ast.NodeVisitor):
                def __init__(self):
                    self.guards = []
                    self.active = True
                def visit_ClassDef(self, node): pass
                def visit_If(self, node):
                    previous = self.active
                    result = _literal(node.test, values)
                    self.visit(node.test)
                    for label, children, selected in [('then',node.body,result is not False),('else',node.orelse,result is not True)]:
                        self.active = previous and selected
                        self.guards.append({'condition':ast.unparse(node.test), 'value':result if type(result) is bool else 'unknown', 'branch':label})
                        if not self.active:
                            for child in children:
                                command_inactive.update(range(child.lineno, child.end_lineno + 1))
                        for child in children: self.visit(child)
                        self.guards.pop()
                    self.active = previous
                def visit_Assign(self, node):
                    if self.active:
                        owner = (factory_class(node.value) or (node.value.func.id if isinstance(node.value.func,ast.Name) else '')) if isinstance(node.value,ast.Call) else ''
                        for target in node.targets:
                            if isinstance(target,ast.Name):
                                if owner in classes or target.id in instances:
                                    instances.setdefault(target.id,set()).add(owner or '<unknown>')
                    self.generic_visit(node)
                def visit_Call(self, node):
                    if not self.active:
                        return
                    owner, name = '', ''
                    if isinstance(node.func,ast.Name) and node.func.id in classes:
                        owner,name = node.func.id,'__init__'
                    elif isinstance(node.func,ast.Attribute) and isinstance(node.func.value,ast.Name):
                        receivers = instances.get(node.func.value.id,set())
                        if len(receivers) == 1:
                            owner,name = next(iter(receivers)),node.func.attr
                    resolved = method(owner,name) if owner else None
                    if resolved and len(packets) < 12:
                        definition_file,definition_owner,definition = resolved
                        positional = [*definition.args.posonlyargs,*definition.args.args]
                        defaults = {}
                        for arg,default in zip(positional[-len(definition.args.defaults):],definition.args.defaults):
                            value = _literal(default,{})
                            if value is not UNKNOWN: defaults[arg.arg] = value
                        for arg,default in zip(definition.args.kwonlyargs,definition.args.kw_defaults):
                            value = _literal(default,{})
                            if value is not UNKNOWN: defaults[arg.arg] = value
                        unknown = {}
                        effective = dict(defaults)
                        parameters = positional[1:] if positional and positional[0].arg in {'self','cls'} else positional
                        if any(isinstance(value,ast.Starred) for value in node.args):
                            effective.clear()
                            unknown['*args'] = ast.unparse(node)
                            positional_supplied = []
                        else:
                            positional_supplied = [(arg.arg,value) for arg,value in zip(parameters,node.args)]
                        supplied = positional_supplied + [(kw.arg,kw.value) for kw in node.keywords]
                        for key,value_node in supplied:
                            value = _literal(value_node,values)
                            if key is None:
                                effective.clear();unknown['**kwargs'] = ast.unparse(value_node)
                            elif value is UNKNOWN:
                                effective.pop(key,None);unknown[key] = ast.unparse(value_node)
                            else: effective[key] = value
                        statements = [child for child in definition.body if not (isinstance(child,ast.Expr) and isinstance(child.value,ast.Constant) and isinstance(child.value.value,str))]
                        begin = statements[0].lineno - 1 if statements else definition.lineno - 1
                        lines = bodies.get(definition_file,'').splitlines()
                        end = min(len(lines),definition.end_lineno,begin+55)
                        code = numbered_code(lines,begin,end,2300)
                        quoted_lines = [int(match[1]) for match in re.finditer(r'(?m)^L(\d+):',code)]
                        packets.append({'file':filename,'line':node.lineno,'call':ast.unparse(node),
                            'receiver_class':owner,'resolved_symbol':definition_owner+'.'+name,
                            'definition_file':definition_file,'definition_line':definition.lineno,
                            'literal_parameters':effective,'unknown_parameters':unknown,
                            'signature':ast.unparse(definition.args),'guards':list(self.guards),
                            'definition':{'file':definition_file,'line':definition.lineno,
                                          'code':code},
                            'dataset':command.get('dataset'),
                            'excerpt_complete':bool(quoted_lines) and max(quoted_lines) >= definition.end_lineno,
                            'scope':'静态调用/继承签名证据；不证明运行已到达此处或动态分支取值'})
                    self.generic_visit(node)
            Calls().visit(tree)
            if filename in inactive:
                inactive[filename].intersection_update(command_inactive)
            else:
                inactive[filename] = command_inactive
    return packets, inactive


def current_path_only(question):
    positive = re.sub(r'(?:不要|不需要|不用|不展开|do not|don.t)\s*[^，,；;。\n]+', '', question, flags=re.I)
    return not re.search(r'比较|对照|假如|如果|另一个|其他分支|未启用|未选中|alternative|compare', positive, re.I)


def inactive_excerpt(packet, inactive):
    excluded = inactive.get(packet.get('file'), set())
    if not excluded:
        return False
    texts = [packet.get('code',''), packet.get('excerpt',''), *packet.get('excerpts',[])]
    return any(int(match[1]) in excluded for text in texts for match in re.finditer(r'(?m)^L(\d+):',text))
