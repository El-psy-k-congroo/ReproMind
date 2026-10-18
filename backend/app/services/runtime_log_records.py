"""Bounded scalar evidence from author-printed JSON, never independent probes."""
import json
import math
import re

SECRET = re.compile(r'password|secret|token|api.?key|access.?key|credential|authorization', re.I)
RELEVANT = re.compile(r'label|prediction|sample|train_size|test_size|learning_rate|batch_size|num_epochs|epoch|seed|model_source|device|solver|loss',re.I)


def printed_json_values(text, log, window):
    rows = []
    for number, line in enumerate(text.splitlines(),1):
        start,end = line.find('{'),line.rfind('}')
        if start < 0 or end < start or end-start > 32000:
            continue
        try: record = json.loads(line[start:end+1])
        except (ValueError,TypeError,RecursionError): continue
        if not isinstance(record,dict): continue
        leaves = []
        visited = 0
        def flatten(value,path='',depth=0):
            nonlocal visited
            visited += 1
            if visited > 1000 or depth > 9 or SECRET.search(path): return
            if isinstance(value,dict):
                for key,child in list(value.items())[:200]:
                    flatten(child,(path+'.' if path else '')+str(key),depth+1)
            elif RELEVANT.search(path) and len(path) <= 220:
                if type(value) in {float,int}:
                    try:
                        if not math.isfinite(value):return
                    except OverflowError:return
                elif type(value) not in {bool,str}: return
                if isinstance(value,str) and len(value)>180:return
                leaves.append({'path':path,'value':value})
        flatten(record)
        rows.extend({'log':log,'window':window,'window_line':number,**row} for row in leaves[:100])
    return rows[-200:]


def select_printed_values(logs,question,budget=2000):
    terms=[]
    for triggers,keys in [
        (('学习率','learning rate','lr'),['learning_rate']),
        (('编码器','encoder'),['embedding','encoder']),
        (('设备','gpu','cuda'),['device']),
        (('标签','抽样','采样','样本','label','sample'),['label','sample','train_size','test_size','prediction']),
        (('模型','权重','model'),['model_source']),
    ]:
        if any(word in question.lower() for word in triggers):terms.extend(keys)
    rows=[row for log in logs for row in reversed(log.get('printed_json_values',[]))]
    rows.sort(key=lambda row:-(sum(4 for term in terms if term in row['path'].lower())
        + (3 if re.search(r'effective|runtime|actual|observed',row['path'],re.I) else 0)
        - (3 if re.search(r'cli_args|requested',row['path'],re.I) else 0)))
    result,records,seen=[],[],set()
    for row in rows:
        identity=(row['log'],row['path'])
        if identity in seen:continue
        seen.add(identity)
        origin={key:row[key] for key in ('log','window','window_line')}
        added=origin not in records
        origins=[*records,origin] if added else records
        compact={'record':origins.index(origin),'path':row['path'],'value':row['value']}
        if len(json.dumps({'records':origins,'values':[*result,compact]},ensure_ascii=False)) <= budget:
            records=origins;result.append(compact)
    return {'records':records,'values':result,'scope':'作者进程实际打印的JSON值；不是独立环境探测、算子轨迹或论文证据'}


def pack_printed_values(evidence,budget):
    result={'values':[], 'records':[], 'scope':'作者打印的JSON日志；非独立探测或论文证据'}
    if not isinstance(evidence,dict) or not isinstance(evidence.get('values'),list):return result
    result['records']=evidence.get('records',[])[:20]
    for row in evidence['values']:
        candidate={**result,'values':[*result['values'],row]}
        if len(json.dumps(candidate,ensure_ascii=False,default=str)) <= budget:
            result=candidate
    return result
