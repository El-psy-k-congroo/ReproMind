"""Resolve a requested run scope from recorded commands, independent of UI state."""
import re


def resolve_scope(command, instruction):
    rows = (command or {}).get('commands') or []
    labels = list(dict.fromkeys(str(row['dataset']) for row in rows if row.get('dataset')))
    short_names = {}
    for label in labels:
        first = re.split(r'\s+',label.casefold())[0]
        if len(first)>=3 and first not in {'the','dataset','benchmark','english','large','small','challenge'}:
            short_names.setdefault(first,set()).add(label)
    selected, excluded = [], []
    text = instruction.casefold()
    for label in labels:
        aliases = {label.casefold()}
        aliases.update(name for name, matches in short_names.items() if matches=={label})
        if re.fullmatch(r'cifar-?\d+', label.casefold()):
            aliases.update({label.casefold().replace('-', ''), re.sub(r'cifar-?', 'cifar-', label.casefold())})
        for row in rows:
            if row.get('dataset') != label:
                continue
            argv = row.get('argv') or []
            for index, value in enumerate(argv):
                if str(value).startswith('--dataset='):
                    aliases.add(str(value).split('=', 1)[1].casefold())
                elif value == '--dataset' and index + 1 < len(argv):
                    aliases.add(str(argv[index + 1]).casefold())
        for alias in aliases:
            if not alias:
                continue
            for match in re.finditer(r'(?<![a-z0-9_])' + re.escape(alias) + r'(?![a-z0-9_])', text):
                before = re.split(r'[，,。；;\n]|而是|但是', text[:match.start()])[-1]
                after = text[match.end():].split('，')[0].split(',')[0]
                negative = bool(re.search(r'(?:不要|不跑|不运行|不重跑|跳过|排除|除了|除去)[^，,。；;]{0,15}$', before)
                                or re.match(r'\s*(?:不要跑|不用跑|不重跑|不运行)', after))
                bucket = excluded if negative else selected
                if label not in bucket:
                    bucket.append(label)
    selected = [label for label in labels if label in selected and label not in excluded]
    if not selected and excluded:
        selected = [label for label in labels if label not in excluded]
    explicit = bool(selected or excluded)
    # A named but unknown dataset must never silently turn into a full run.
    unknown_target = bool(re.search(r'(?:重跑|重新跑|运行|跑一下|跑一遍)\s*(?:一下)?\s*[a-z0-9_.-]+\s*(?:这个)?数据集', text)
                          or re.search(r'(?:重跑|重新跑|只跑)\s*(?:一下)?\s*(?!gpu\b|cuda\b)[a-z][a-z0-9_.-]+', text))
    reason = ''
    if not explicit and unknown_target:
        reason = '指令中的数据集没有匹配到已规划命令，请明确选择运行范围；不会默认运行全部数据集'
    elif not explicit:
        selected = labels
    if not selected and not reason:
        reason = '没有可执行的数据集范围，请重新选择'
    return {'selected_datasets': selected, 'datasets': labels,
            'scope_explicit': explicit, 'scope_reason': reason}


def revision_scope(base, proposal, requested=None):
    scope = resolve_scope(getattr(base, 'execution_command', {}), str(proposal.get('instruction') or ''))
    if scope['scope_reason']:
        return scope
    if requested:
        lookup = {label.casefold(): label for label in scope['datasets']}
        unknown = [label for label in requested if label.casefold() not in lookup]
        chosen = list(dict.fromkeys(lookup[label.casefold()] for label in requested if label.casefold() in lookup))
        if unknown:
            scope['scope_reason'] = '没有已验证的可执行数据集：' + '、'.join(unknown)
        elif scope['scope_explicit'] and any(label not in scope['selected_datasets'] for label in chosen):
            scope['scope_reason'] = '提交的运行范围超出原指令，请重新生成包含新范围的提案'
        else:
            scope['selected_datasets'] = chosen
    return scope
