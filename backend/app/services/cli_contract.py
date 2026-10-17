"""Inspect literal CLI requirements without importing or executing author code."""
import ast
import re


def option_values(argv, option):
    """Read both --flag=value and --flag value forms, including nargs lists."""
    for index, token in enumerate(argv):
        if token == option:
            end = index + 1
            while end < len(argv) and not re.match(r'^--?[A-Za-z_]', str(argv[end])):
                end += 1
            return list(argv[index+1:end])
        if str(token).startswith(option + '='):
            return [str(token).split('=',1)[1]]
    return []


def argument_options(body):
    """Literal argparse metadata only; no author imports or expression evaluation."""
    try:
        tree = ast.parse(body)
    except SyntaxError:
        return {}
    found = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute) or node.func.attr != 'add_argument':
            continue
        flags = [arg.value for arg in node.args if isinstance(arg,ast.Constant)
                 and isinstance(arg.value,str) and arg.value.startswith('--')]
        if not flags:
            continue
        name = flags[0][2:]
        values = {item.arg:item.value for item in node.keywords}
        def literal(key, fallback=None):
            try:return ast.literal_eval(values[key])
            except (KeyError,ValueError,TypeError):return fallback
        kind = values.get('type')
        kind = kind.id if isinstance(kind,ast.Name) else ''
        default = literal('default')
        if not kind and type(default) in {str,int,float}:
            kind = type(default).__name__
        found[name] = {'kind':kind,'nargs':literal('nargs'),'dest':literal('dest',name.replace('-','_')),
                       'default':default,'choices':literal('choices')}
    return found


def replace_option_values(argv, option, values):
    argv = list(argv)
    for index, token in enumerate(argv):
        if token == option:
            end = index + 1 + len(option_values(argv,option))
            return [*argv[:index],option,*values,*argv[end:]]
        if str(token).startswith(option+'='):
            return [*argv[:index],option,*values,*argv[index+1:]]
    return [*argv,option,*values]


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
