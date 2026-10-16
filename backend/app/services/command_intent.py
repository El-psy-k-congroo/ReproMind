"""Apply explicit numeric requests through declared CLI options, not source edits."""
import math
import re
from .source_selection import entry_sources
from .tuning import numeric_flags, parameter_argv


def _chinese_count(match):
    digits = {char:value for value,chars in enumerate(['零〇','一','二两','三','四','五','六','七','八','九']) for char in chars}
    total=current=0
    for char in match[0]:
        if char in digits: current=digits[char]
        else: total+=(current or 1)*{'十':10,'百':100,'千':1000}[char];current=0
    return str(total+current)


def apply_goal_parameters(source, entries, argv, goal):
    goal = re.sub(r'[零〇一二两三四五六七八九十百千]+(?=\s*(?:个|轮)?\s*(?:完整(?:的)?(?:训练)?\s*)?(?:epochs?|轮训练))',_chinese_count,goal,flags=re.I)
    goal = '；'.join(clause for clause in re.split(r'[，,；;。]',goal)
                    if not re.match(r'\s*(?:不要(?:修改|改变|更改)|不(?:改|改变)|别(?:改|调整))',clause))
    declarations = {}
    for path in entry_sources(source, entries):
        declarations.update(numeric_flags(path.read_text(encoding='utf-8', errors='replace')))
    changes = {}
    number = r'([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)'
    for name, default in declarations.items():
        aliases = [re.escape(name), re.escape(name.replace('_','-'))]
        if name in {'epochs','epoch','n_epochs','num_train_epochs','max_epoch'}:
            aliases += [r'epochs?', '训练轮数', '轮数']
        if name in {'seed','random_seed'}:
            aliases += ['随机种子']
        term = '(?:'+'|'.join(aliases)+')'
        matches = re.findall(r'(?<![A-Za-z0-9_])'+term+r'\s*(?:=|:|：|改为|改成|设为|设置为)\s*'+number, goal, re.I)
        if name in {'epochs','epoch','n_epochs','num_train_epochs','max_epoch'}:
            matches += re.findall(number+r'\s*(?:个|轮)?\s*(?:完整(?:的)?(?:训练)?\s*)?(?:epochs?|轮(?:训练|完整训练))', goal, re.I)
        if not matches:
            continue
        value = float(matches[-1])
        if not math.isfinite(value) or type(default) is int and not value.is_integer():
            raise ValueError(f'用户指定 {name}={matches[-1]} 与作者参数类型不符')
        if name in {'epochs','epoch','n_epochs','num_train_epochs','max_epoch'} and value < 1:
            raise ValueError('完整实验的训练轮数必须大于零')
        changes[name] = int(value) if type(default) is int else value
    return parameter_argv(argv, changes), changes
