"""Canonical metric identity, units and unambiguous dataset target binding."""
import math
import re

RATIOS = {'accuracy', 'auc', 'aupr', 'f1', 'precision', 'recall', 'top1error', 'top5error'}
LOWER = {'loss', 'rmse', 'mae', 'mse', 'prd', 'top1error', 'top5error', 'fid', 'perplexity'}


def metric_name(value):
    key = re.sub(r'[^a-z0-9]', '', str(value).lower())
    return {'acc': 'accuracy', 'auroc': 'auc', 'rocauc': 'auc', 'aucroc': 'auc', 'prauc': 'aupr',
            'aucpr': 'aupr', 'f1score': 'f1', 'top1err': 'top1error', 'top5err': 'top5error'}.get(key, key)


def metric_number(value, name):
    if isinstance(value, dict):
        value = value.get('mean', value.get('value'))
    if isinstance(value, str):
        # Accept a reported mean ± uncertainty, not a number buried in prose.
        match = re.fullmatch(r'\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*(%)?(?:\s*(?:±|\+/-)\s*\d*\.?\d+\s*%?)?\s*', value)
        if not match:
            return None
        value = float(match[1])
    if type(value) not in {int, float} or not math.isfinite(value):
        return None
    return value / 100 if metric_name(name) in RATIOS and value > 1 else float(value)


def dataset_targets(spec):
    raw = spec.get('target_results') or {}
    labels = spec.get('datasets') or []
    if not isinstance(raw, dict):
        return {}
    # A flat metric map is assignable only for a single declared dataset.
    metrics = {metric_name(key) for key in spec.get('metrics', [])}
    if len(labels) == 1 and raw and all(metric_name(key) in metrics for key in raw):
        raw = {labels[0]: raw}
    result = {}
    for label in labels:
        value = next((v for k, v in raw.items() if str(k).casefold() == str(label).casefold()), None)
        if type(value) in {int, float, str} and len(metrics) == 1:
            value = {next(iter(metrics)): value}
        if not isinstance(value, dict):
            continue
        row = {metric_name(key): metric_number(v, key) for key, v in value.items()}
        row = {key: v for key, v in row.items() if v is not None}
        if row:
            result[str(label)] = row
    return result


def target_tolerance(spec, dataset, metric):
    """No silent percentage-point margin. Preserve reported uncertainty."""
    raw = spec.get('target_results_raw') or spec.get('target_results') or {}
    if isinstance(raw, dict):
        row = next((v for k,v in raw.items() if str(k).casefold() == dataset.casefold()), None)
        if row is None and len(spec.get('datasets') or []) == 1:
            row = raw
        if isinstance(row,(str,int,float)) and len(spec.get('metrics') or []) == 1:
            row = {spec['metrics'][0]:row}
        if isinstance(row, dict):
            value = next((v for k,v in row.items() if metric_name(k) == metric_name(metric)), None)
            uncertainty = value.get('std') if isinstance(value,dict) else None
            if isinstance(value,dict) and type(uncertainty) in {int,float} and type(value.get('mean')) in {int,float}:
                if value['mean'] > 1 and metric_name(metric) in RATIOS: uncertainty /= 100
            if isinstance(value,str):
                match = re.search(r'(?:±|\+/-)\s*(\d*\.?\d+)', value)
                if match:
                    uncertainty = float(match[1])
                    mean = re.match(r'\s*(\d*\.?\d+)', value)
                    if mean and float(mean[1]) > 1 and metric_name(metric) in RATIOS:
                        uncertainty /= 100
            if type(uncertainty) in {int,float} and math.isfinite(uncertainty) and uncertainty >= 0:
                return uncertainty, '使用论文报告的一个标准差作数值比较，不能替代多次运行或协议验收'
    return 0.0, '未核验论文不确定性；按目标数值比较，不自动放宽容差'
