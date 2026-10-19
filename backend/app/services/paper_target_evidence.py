"""Bind target numbers to an actual method row and an explicit dataset column.

Ambiguous tables remain unknown; this never blocks running author source.
"""
import re
import json
from .metric_contract import dataset_targets,metric_name

TARGET_EVIDENCE_PROMPT = (
    'target_evidence 是结果表证据列表，每项包含 dataset、metric、reported、method_label、columns、header_quote、row_quote。'
    'header_quote 必须逐字包含数据集列名所在表头行，不能只引用 Table 标题；row_quote 必须逐字摘自论文。columns 是表头按原顺序排列的全部数据集；'
    'row_quote 是作者当前方法的完整同一结果行，包含所有数据集结果；method_label 是该行中的方法标签。'
    'reported 保留该数据集对应单元格的均值、±标准差和百分号，例如 82.3 ±0.6%。'
    '若单元格另附括号内的层数等元数据，额外给出解释括号含义的论文原文 annotation_quote，reported 不包含该元数据。'
    '不要取基线、消融或其他模型的数字。不能确认方法行、数据集列或指标时 target_evidence 留空；'
    '没有目标数值也允许继续准备作者源码实验。'
)


def _compact(text):return re.sub(r'\s+','',str(text)).casefold()
def _key(text):return re.sub(r'[^a-z0-9]','',str(text).casefold())
CELL=re.compile(r'(?<![\w.])[+-]?\d+(?:\.\d+)?(?:\s*(?:±|\+/-)\s*\d+(?:\.\d+)?)?\s*%?(?![\w.])')


def validate_target_evidence(spec,paper):
    accepted={};proof=[];rejected=[]
    labels=[str(x) for x in spec.get('datasets',[])]
    metrics={metric_name(x) for x in spec.get('metrics',[])}
    model=_key(spec.get('model_name',''));text=_compact(paper)
    for row in (spec.get('target_evidence') or [])[:40]:
        try:
            dataset=next(x for x in labels if _key(x)==_key(row['dataset']))
            metric=metric_name(row['metric']);reported=str(row['reported']).strip()
            header=str(row['header_quote']);line=str(row['row_quote']);method=str(row['method_label'])
            columns=row['columns']
            if not model or metric not in metrics or not reported or not method or not isinstance(columns,list) or not 1<=len(columns)<=20:
                raise ValueError('方法、指标或表头结构无效')
            if not header.strip() or not line.strip() or _compact(header) not in text or _compact(line) not in text:
                raise ValueError('表头或方法行不能回查论文原文')
            if model not in _key(method) or _compact(method) not in _compact(line):
                raise ValueError('当前作者方法与结果行标签不匹配')
            keys=[_key(x) for x in columns];heading=_key(header)
            positions=[heading.find(key) for key in keys]
            if len(set(keys))!=len(keys) or any(not key for key in keys) or min(positions)<0 or positions!=sorted(positions):
                raise ValueError('数据集列顺序与原始表头不一致')
            column=keys.index(_key(dataset))
            # Only accept a row with exactly one metric cell per dataset. This
            # excludes layer numbers/ablations and mixed-metric tables instead
            # of guessing their column alignment.
            normalized_line=' '.join(line.split());normalized_method=' '.join(method.split())
            start=normalized_line.casefold().find(normalized_method.casefold())
            if start<0:raise ValueError('方法标签无法在原行定位')
            tail=normalized_line[start+len(normalized_method):]
            if tail.startswith(('*','+')):
                raise ValueError('该结果行属于额外方法变体，不能当作当前方法')
            annotation=str(row.get('annotation_quote') or '')
            if annotation:
                if _compact(annotation) not in text or not re.search(r'parenthes.{0,150}(?:layer|epoch|sample|parameter)',annotation,re.I|re.S):
                    raise ValueError('括号注释缺少可回查的元数据解释')
                tail=re.sub(r'\(\s*\d+\s*\)','',tail)
            cells=[m.group(0).strip() for m in CELL.finditer(tail)]
            if len(cells)!=len(columns) or _compact(cells[column])!=_compact(reported):
                raise ValueError('目标值与对应数据集单元格不一致或表格结构含糊')
            parsed=dataset_targets({'datasets':[dataset],'metrics':[metric],
                'target_results':{dataset:{metric:reported}}})
            if not parsed:raise ValueError('目标单元格数值无法解析')
            accepted.setdefault(dataset,{}).update(parsed[dataset]);proof.append(row)
        except (KeyError,TypeError,ValueError,StopIteration) as exc:
            rejected.append(str(exc))
    return accepted, {'version':1,'status':'verified' if accepted else 'unverified',
        'evidence':proof,'rejected':rejected,'note':'仅核验可明确对齐的方法行和数据集列，不代表完整复现协议一致'}


def bind_targets(spec,paper):
    result=dict(spec)
    # Models sometimes quote a table caption instead of the immediately
    # following column header. Recover only a unique literal header in that
    # same local table; validation still checks every column and result cell.
    evidence=[]
    lines=paper.splitlines()
    for original in result.get('target_evidence') or []:
        row=dict(original)
        header=str(row.get('header_quote') or '')
        keys=[_key(x) for x in row.get('columns', [])]
        if keys and any(key not in _key(header) for key in keys) and re.match(r'\s*Table\s+\d+',header,re.I):
            candidates=[]
            for i,line in enumerate(lines):
                if _compact(line)!=_compact(header):continue
                following=lines[i+1:i+21]
                if not any(_compact(value)==_compact(row.get('row_quote','')) for value in following):continue
                for value in following:
                    if not re.match(r'\s*(?:Method|Model|Approach|Dataset)\b',value,re.I):continue
                    positions=[_key(value).find(key) for key in keys]
                    if min(positions)>=0 and positions==sorted(positions):candidates.append(value)
            if len(set(candidates))==1:row['header_quote']=candidates[0]
        elif keys and any(key not in _key(header) for key in keys):
            # A model may accidentally repeat the result row as its header.
            # Recover only from the nearest literal labelled header above that
            # exact row, never infer a column order from dataset request order.
            candidates=[]
            for i,line in enumerate(lines):
                if _compact(line)!=_compact(row.get('row_quote','')):continue
                for value in reversed(lines[max(0,i-20):i]):
                    if not re.match(r'\s*(?:Method|Model|Approach|Dataset)\b',value,re.I):continue
                    positions=[_key(value).find(key) for key in keys]
                    if min(positions)>=0 and positions==sorted(positions):candidates.append(value)
                    break
            if len(set(candidates))==1:row['header_quote']=candidates[0]
        evidence.append(row)
    result['target_evidence']=evidence
    targets,validation=validate_target_evidence(result,paper)
    if result.get('target_results') and not result.get('unverified_target_results'):
        result['unverified_target_results']=result['target_results']
    result['target_results']=targets
    result['target_validation']=validation
    # Preserve uncertainty and original units only from accepted cells.
    result['target_results_raw']={}
    for row in validation['evidence']:
        dataset=next(x for x in result['datasets'] if _key(x)==_key(row['dataset']))
        result['target_results_raw'].setdefault(dataset,{})[metric_name(row['metric'])]=row['reported']
    return result


def target_table_context(paper,model,datasets):
    lines=paper.splitlines();ranked=[]
    for i,line in enumerate(lines):
        key=_key(line)
        named=bool(_key(model) and _key(model) in key)
        numeric=len(CELL.findall(line))
        score=12*named+3*('±' in line)+sum(_key(d) in key for d in datasets)
        if score and (numeric>=2 or named):ranked.append((score,i))
    selected=set()
    for _,i in sorted(ranked,reverse=True)[:10]:selected.update(range(max(0,i-20),min(len(lines),i+5)))
    return '\n'.join(lines[i] for i in sorted(selected))[:14000]


def explicit_table_targets(spec,paper):
    """Use only a single unambiguous row under a labelled dataset header."""
    model=str(spec.get('model_name') or '');metrics=spec.get('metrics') or []
    if not model or len(metrics)!=1:return bind_targets({**spec,'target_evidence':[]},paper)
    lines=paper.splitlines();candidates=[]
    for i,line in enumerate(lines):
        if not re.search(r'(?<![\w])'+re.escape(model)+r'(?![\w*+])',line,re.I):continue
        match=re.search(re.escape(model),line,re.I);tail=line[match.end():]
        first=CELL.search(tail)
        descriptor=tail[:first.start()].strip(' :|\t') if first else ''
        if descriptor and not re.fullmatch(r'\(?\s*(?:ours|this paper|proposed)\s*\)?',descriptor,re.I):continue
        context='\n'.join(lines[max(0,i-20):i]);annotation=''
        if re.search(r'parenthes.{0,150}(?:layer|epoch|sample|parameter)',context,re.I|re.S):
            annotation=context
            tail=re.sub(r'\(\s*\d+\s*\)','',tail)
        cells=[m.group(0).strip() for m in CELL.finditer(tail)]
        if not cells:continue
        for header in reversed(lines[max(0,i-20):i]):
            title=re.search(r'\b(?:method|model|approach|dataset)\s+(.+)',header,re.I)
            if not title:continue
            columns=title.group(1).split()
            if len(columns)!=len(cells):continue
            for dataset in spec.get('datasets') or []:
                keys=[_key(x) for x in columns]
                if _key(dataset) not in keys:continue
                candidates.append({'dataset':dataset,'metric':metrics[0],
                    'reported':cells[keys.index(_key(dataset))],'method_label':match.group(0),
                    'columns':columns,'header_quote':header,'row_quote':line,
                    **({'annotation_quote':annotation} if annotation else {})})
            break
    # Multiple rows for a requested dataset may be variants/ablations; ask for
    # evidence instead of silently taking the first or best number.
    unique=[row for row in candidates if sum(_key(other['dataset'])==_key(row['dataset']) for other in candidates)==1]
    return bind_targets({**spec,'target_evidence':unique},paper)


async def refresh_target_evidence(paper,spec,model_id=''):
    from ..core.model_router import get_model_router,ModelRole
    from .code_agent import paper_context
    result=explicit_table_targets(spec,paper)
    if result['target_results']:return result
    try:
        response=await get_model_router().complete_json(ModelRole.paper,
            '核验论文目标数值，只返回JSON {target_evidence:[]}。'+TARGET_EVIDENCE_PROMPT,
            json.dumps({'model_name':spec.get('model_name'),'datasets':spec.get('datasets'),
                        'metrics':spec.get('metrics'),'paper':target_table_context(paper,spec.get('model_name',''),spec.get('datasets') or []) or paper_context(paper,limit=12000)},ensure_ascii=False),
            max_tokens=2200,model_override=model_id,non_thinking=True,timeout_seconds=45)
        if response.error:
            result['target_validation']['lookup_error']=str(response.error)[:300]
        else:
            result=bind_targets({**spec,'target_evidence':response.data.get('target_evidence') or []},paper)
    except Exception as exc:
        result['target_validation']['lookup_error']=f'{type(exc).__name__}: {str(exc)[:200]}'
    return result
