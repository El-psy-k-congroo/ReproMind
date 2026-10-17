"""Apply explicit numeric requests through declared CLI options, not source edits."""
import math
import re
import ast
from .source_selection import entry_sources
from .tuning import numeric_flags, parameter_argv
from .cli_contract import argument_options, replace_option_values


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
                    if not re.match(r'\s*(?:不要(?:修改|改变|更改|使用|运行)|不(?:改|改变|使用|运行)|别(?:改|调整))',clause))
    declarations, options, bodies = {}, {}, []
    for path in entry_sources(source, entries):
        body = path.read_text(encoding='utf-8', errors='replace')
        bodies.append(body)
        declarations.update(numeric_flags(body))
        options.update(argument_options(body))
    for name, option in options.items():
        if option['kind'] in {'int','float'}:
            declarations.setdefault(name, 0 if option['kind']=='int' else 0.0)
    changes = {}
    number = r'([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)'
    for name, default in declarations.items():
        aliases = [re.escape(name), re.escape(name.replace('_','-'))]
        if name in {'epochs','epoch','n_epochs','num_train_epochs','max_epoch'}:
            aliases += [r'epochs?', '训练轮数', '轮数']
        if name in {'seed','random_seed','seeds','random_seeds'}:
            aliases += ['随机种子',r'seed']
        term = '(?:'+'|'.join(aliases)+')'
        matches = re.findall(r'(?<![A-Za-z0-9_])'+term+r'\s*(?:=|:|：|改为|改成|设为|设置为)\s*'+number, goal, re.I)
        if name in {'epochs','epoch','n_epochs','num_train_epochs','max_epoch'}:
            matches += re.findall(number+r'\s*(?:个|轮)?\s*(?:完整(?:的)?(?:训练)?\s*)?(?:epochs?|轮(?:训练|完整训练))', goal, re.I)
        if name in {'sample_sizes','samples_per_class','num_samples_per_class','n_shot','k_shot','shots_per_class'}:
            matches += re.findall(r'每类\s*'+number+r'\s*(?:个)?\s*(?:训练)?样本', goal, re.I)
            matches += re.findall(number+r'\s*(?:个)?\s*(?:训练)?样本\s*(?:每|/)\s*类',goal,re.I)
        if not matches:
            continue
        value = float(matches[-1])
        if not math.isfinite(value) or type(default) is int and not value.is_integer():
            raise ValueError(f'用户指定 {name}={matches[-1]} 与作者参数类型不符')
        if name in {'epochs','epoch','n_epochs','num_train_epochs','max_epoch'} and value < 1:
            raise ValueError('完整实验的训练轮数必须大于零')
        changes[name] = int(value) if type(default) is int else value
        choices = options.get(name,{}).get('choices')
        if isinstance(choices,(list,tuple)) and changes[name] not in choices:
            raise ValueError(f'用户指定 {name} 不在作者声明的 choices 中')
    # A unique hub identifier can bind only to a declared model option that
    # the author actually passes into from_pretrained, never another dataset.
    identifiers = set(match[1] for match in re.finditer(r'(?<![\w:/.-])([A-Za-z0-9][\w.-]*/[A-Za-z0-9][\w.-]*)(?![\w/.-])',goal)
        if not re.search(r'(?:数据集|dataset|dataset_id|dataset-id)\s*(?:=|:|：)?\s*$',goal[max(0,match.start()-24):match.start()],re.I))
    model_options = set()
    for body in bodies:
        try:tree=ast.parse(body)
        except SyntaxError:continue
        for node in ast.walk(tree):
            if not isinstance(node,ast.Call) or not isinstance(node.func,ast.Attribute) or node.func.attr!='from_pretrained':continue
            values = [*node.args, *(item.value for item in node.keywords)]
            for value in values:
                if isinstance(value,ast.Attribute) and isinstance(value.value,ast.Name) and value.value.id=='args':
                    model_options.update(name for name,option in options.items() if option['dest']==value.attr
                        and option['kind'] in {'','str'} and name in {'model','model_name','model-name','model_name_or_path','pretrained_model','backbone','encoder','encoder_name'})
    if len(identifiers)==1 and len(model_options)==1:
        changes[next(iter(model_options))] = next(iter(identifiers))
    for name,value in changes.items():
        if options.get(name,{}).get('nargs') in {'+','*'}:
            argv = replace_option_values(argv,'--'+name,[str(value)])
        else:
            argv = parameter_argv(argv,{name:value})
    return argv, changes


def unbound_goal_parameters(goal, overrides):
    """Report explicit seed requirements rather than silently run other seeds."""
    positive = '；'.join(clause for clause in re.split(r'[，,；;。]',goal)
        if not re.match(r'\s*(?:不要|不改|不改变|别改)',clause))
    requests = re.findall(r'(?<![A-Za-z_])(?:seed|random_seed|随机种子)\s*(?:=|:|：|设为|设置为)\s*(\d+)',positive,re.I)
    if requests and not any(key in overrides for key in {'seed','random_seed','seeds','random_seeds'}):
        return [{'parameter':'seed','requested':int(requests[-1]),'reason':'当前入口没有可绑定的种子参数；不能把作者默认或循环种子冒充用户指定值'}]
    return []
