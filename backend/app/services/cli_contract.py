"""Inspect literal CLI requirements without importing or executing author code."""
import ast


def missing_required_options(body, argv):
    try:
        tree=ast.parse(body)
    except SyntaxError:
        return []
    supplied={str(arg).split('=',1)[0] for arg in argv[1:] if str(arg).startswith('-')}
    missing=[]
    for call in ast.walk(tree):
        if not (isinstance(call,ast.Call) and isinstance(call.func,ast.Attribute)
                and call.func.attr=='add_argument'
                and any(k.arg=='required' and isinstance(k.value,ast.Constant) and k.value.value is True for k in call.keywords)):
            continue
        names=[arg.value for arg in call.args if isinstance(arg,ast.Constant)
               and isinstance(arg.value,str) and arg.value.startswith('-')]
        if names and not supplied.intersection(names):
            missing.append(next((name for name in names if name.startswith('--')),names[0]))
    return list(dict.fromkeys(missing))
