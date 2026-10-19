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
    percent = False
    if isinstance(value, dict):
        value = value.get('mean', value.get('value'))
    if isinstance(value, str):
        # Accept a reported mean ± uncertainty, not a number buried in prose.
        match = re.fullmatch(r'\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*(%)?(?:\s*(?:±|\+/-)\s*\d*\.?\d+\s*%?)?\s*', value)
        if not match:
            return None
        percent = bool(match[2])
        value = float(match[1])
    if type(value) not in {int, float} or not math.isfinite(value):
        return None
    return value / 100 if metric_name(name) in RATIOS and (value > 1 or percent) else float(value)


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


def acceptance_policy():
    from ..config import get_settings
    settings=get_settings()
    return {'mode':settings.reproduction_comparison_mode,
            'ratio_tolerance':settings.reproduction_ratio_tolerance,
            'relative_tolerance':settings.reproduction_relative_tolerance}


def target_tolerance(spec, dataset, metric):
    """Apply a declared practical margin, retaining paper uncertainty and strict opt-in."""
    policy=spec.get('acceptance_policy') or acceptance_policy()
    practical=policy.get('mode')!='strict'
    targets=dataset_targets(spec)
    row=next((v for k,v in targets.items() if k.casefold()==str(dataset).casefold()), {})
    target=row.get(metric_name(metric))
    floor=(float(policy.get('ratio_tolerance',0.01)) if metric_name(metric) in RATIOS else
           abs(target or 0)*float(policy.get('relative_tolerance',0.05))) if practical else 0.0
    basis=(f"实用复现容差：{floor*100:g} 个百分点" if metric_name(metric) in RATIOS else
           f"实用复现容差：目标值的 {float(policy.get('relative_tolerance',0.05))*100:g}%") if practical else '严格比较'

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
                    if mean and (float(mean[1]) > 1 or '%' in value) and metric_name(metric) in RATIOS:
                        uncertainty /= 100
            if type(uncertainty) in {int,float} and math.isfinite(uncertainty) and uncertainty >= 0:
                return max(floor,uncertainty), basis + '；同时保留论文报告的不确定性，取较宽者；原始指标与单次/多次运行范围另列'
    return floor, basis + '；显示实际差值，小幅波动不直接视为失败；不代表全部论文结论已验证' if practical else '严格比较；论文未报告不确定性时采用零数值容差'
