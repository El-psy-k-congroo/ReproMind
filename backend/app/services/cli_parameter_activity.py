"""Conservative literal-guard analysis of direct argparse parameter uses."""
import ast

from .cli_contract import argument_options, option_values
from .source_class_context import UNKNOWN


def parameter_activity(body, argv):
    try:
        tree = ast.parse(body)
    except SyntaxError:
        return {}
    options = argument_options(body)
    namespaces = {'args'}
    local_functions = {node.name: node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
    for node in ast.walk(tree):
        if (isinstance(node, ast.Assign) and isinstance(node.value, ast.Call)
                and ((isinstance(node.value.func, ast.Attribute) and node.value.func.attr == 'parse_args')
                     or isinstance(node.value.func, ast.Name) and node.value.func.id == 'parse_args')):
            namespaces.update(target.id for target in node.targets if isinstance(target, ast.Name))
    values = {option['dest']: option['default'] for option in options.values() if option['default'] is not None}
    for name, option in options.items():
        supplied = option_values(argv, '--' + name)
        if len(supplied) == 1:
            try:
                values[option['dest']] = {'int': int, 'float': float}.get(option['kind'], str)(supplied[0])
            except ValueError:
                values.pop(option['dest'], None)
        elif '--' + name in argv or supplied:
            values.pop(option['dest'], None)

    def literal(node):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id in namespaces:
            return values.get(node.attr, UNKNOWN)
        if isinstance(node, ast.Compare) and len(node.ops) == 1:
            left, right = literal(node.left), literal(node.comparators[0])
            if left is UNKNOWN or right is UNKNOWN:
                return UNKNOWN
            if isinstance(node.ops[0], ast.Eq):
                return left == right
            if isinstance(node.ops[0], ast.NotEq):
                return left != right
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            value = literal(node.operand)
            return UNKNOWN if value is UNKNOWN else not value
        try:
            return ast.literal_eval(node)
        except (ValueError, TypeError):
            return UNKNOWN

    parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
    # If the namespace escapes into unknown code or is mutated, branch proofs
    # cannot establish that a CLI parameter has no effect.
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id in namespaces:
            if not isinstance(node.ctx, ast.Load):
                return {}
        if not isinstance(node, ast.Name) or node.id not in namespaces:
            continue
        parent = parents.get(node)
        if isinstance(node.ctx, ast.Store):
            call = parent.value if isinstance(parent, ast.Assign) else None
            if (isinstance(call, ast.Call) and (
                    isinstance(call.func, ast.Attribute) and call.func.attr == 'parse_args'
                    or isinstance(call.func, ast.Name) and call.func.id == 'parse_args')):
                continue
            return {}
        if not isinstance(node.ctx, ast.Load):
            continue
        if isinstance(parent, ast.Attribute) and parent.value is node:
            continue
        if isinstance(parent, ast.Return):
            owner = parents.get(parent)
            while owner is not None and not isinstance(owner, ast.FunctionDef):
                owner = parents.get(owner)
            if owner is not None and owner.name == 'parse_args':
                continue
            return {}
        if (isinstance(parent, ast.Call) and isinstance(parent.func, ast.Name)
                and parent.func.id in local_functions and node in parent.args):
            function = local_functions[parent.func.id]
            index = parent.args.index(node)
            positional = [*function.args.posonlyargs, *function.args.args]
            if index < len(positional) and positional[index].arg == node.id:
                continue
        return {}

    uses = {}
    class Uses(ast.NodeVisitor):
        def __init__(self):
            self.reachable = True
        def visit_If(self, node):
            self.visit(node.test)
            condition, previous = literal(node.test), self.reachable
            for children, selected in [(node.body, condition is not False), (node.orelse, condition is not True)]:
                self.reachable = previous and selected
                for child in children:
                    self.visit(child)
            self.reachable = previous
        def visit_Attribute(self, node):
            if isinstance(node.value, ast.Name) and node.value.id in namespaces:
                uses.setdefault(node.attr, []).append({'line': node.lineno, 'reachable': self.reachable})
            self.generic_visit(node)
    Uses().visit(tree)
    return {name: {'inactive': bool(uses.get(option['dest'])) and not any(
                    row['reachable'] for row in uses[option['dest']]),
                   'uses': uses.get(option['dest'], [])}
            for name, option in options.items()}
