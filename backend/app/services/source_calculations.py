"""Read-only, question-bound calculation evidence from the selected import closure."""
import ast
import re


def numbered_code(lines, start, end, limit=2200):
    """Keep whole source lines, omitting standalone comments rather than formulas."""
    result = []
    used = 0
    for index in range(start, end):
        line = lines[index]
        if not line.strip() or line.lstrip().startswith('#'):
            continue
        rendered = f'L{index + 1}: {line}'
        if used + len(rendered) + 1 > limit:
            break
        result.append(rendered)
        used += len(rendered) + 1
    return '\n'.join(result)


def calculation_packets(body, question, filename):
    """Retrieve metric expressions and preprocessing even if function names differ.

    The expressions remain author code, never evaluated or treated as observed
    shapes/results. A function called evaluate_dataset can define RMSE without
    mentioning RMSE in its own name.
    """
    terms = set(re.findall(r'[a-zA-Z][a-zA-Z_0-9]+', question.lower()))
    aliases = [
        (('均方根',), {'rmse'}), (('压缩率', '压缩比'), {'cr', 'compression_ratio'}),
        (('立方根',), {'cbrt'}), (('归一化',), {'normalize', 'normalization'}),
        (('标准化',), {'standardize', 'scaler'}),
    ]
    for labels, names in aliases:
        if any(label in question for label in labels): terms.update(names)
    terms -= {'code', 'forward', 'model', 'train', 'test', 'loss', 'gpu', 'cuda', 'epoch'}
    if not terms: return []
    try: tree = ast.parse(body)
    except SyntaxError: return []
    lines = body.splitlines(); ranked = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign, ast.Return, ast.Expr, ast.If)):
            continue
        # A method name can occur only in a controlling flag, while its crop,
        # correction and loss statements use unrelated local variable names.
        searchable = node.test if isinstance(node, ast.If) else node
        identifiers = {child.id.lower() if isinstance(child, ast.Name) else child.attr.lower()
                       for child in ast.walk(searchable) if isinstance(child, (ast.Name, ast.Attribute))}
        identifiers.update(part for name in list(identifiers) for part in name.split('_') if len(part) >= 3)
        labels = {token for child in ast.walk(node) if isinstance(child, ast.Constant) and isinstance(child.value, str)
                  for token in re.findall(r'\b([A-Za-z][A-Za-z_0-9]+)\s*[:=]', child.value)}
        score = len(identifiers & terms) + len({label.lower() for label in labels} & terms)
        if score and (not isinstance(node, ast.If) or node.end_lineno - node.lineno < 40):
            ranked.append((score, node.lineno - 1, getattr(node, 'end_lineno', node.lineno)))
    packets, selected = [], set()
    for _, begin, finish in sorted(ranked, key=lambda row: (-row[0], row[1])):
        if begin in selected: continue
        start = max(0, begin - 3); end = min(len(lines), finish + 3)
        selected.update(range(start, end))
        packets.append({'file': filename, 'line': begin + 1,
                        'code': numbered_code(lines, start, end, 2600),
                        'scope': '静态原文计算片段；未执行，不推断输入形状或实际指标'})
        if len(packets) >= 4: break
    return packets


def configured_model_names(command):
    """Explicit execution selectors outrank other imported architectures."""
    names = []
    for item in (command or {}).get('commands') or [command or {}]:
        argv = item.get('argv') or []
        for index, token in enumerate(argv):
            flag, _, inline = str(token).partition('=')
            if flag.lstrip('-').replace('-', '_') not in {'arch', 'architecture', 'model', 'model_name', 'net_type', 'backbone'}:
                continue
            value = inline or (str(argv[index + 1]) if index + 1 < len(argv) else '')
            if value and not value.startswith('-'):
                names.append(re.sub(r'[^a-z0-9]', '', value.casefold()))
    return list(dict.fromkeys(names))
