"""Normalize bounded scalar dictionaries printed by common training frameworks."""
import ast
import json
import math
import re

NUMBER = re.compile(r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?\Z')
METRICS = {'loss', 'embedding_loss', 'accuracy', 'acc', 'auc', 'auroc', 'aupr',
           'f1', 'precision', 'recall', 'rmse', 'mae', 'mse', 'prd', 'cr',
           'fid', 'perplexity', 'bleu', 'rouge', 'top1error', 'top5error'}


def scalar_number(value):
    try:
        if isinstance(value, str) and NUMBER.fullmatch(value.strip()):
            value = float(value)
        return float(value) if type(value) in {int, float} and math.isfinite(value) else None
    except (ValueError, OverflowError):
        return None


def normalize_scalar_line(line):
    start, end = line.find('{'), line.rfind('}')
    if start < 0 or end < start or end - start > 32000:
        return line
    body = line[start:end + 1]
    try:
        record = json.loads(body)
    except (ValueError, TypeError, RecursionError):
        try:
            record = ast.literal_eval(body)
        except (ValueError, SyntaxError, TypeError, RecursionError):
            return line
    if not isinstance(record, dict):
        return line
    epoch = scalar_number(record.get('epoch'))
    metrics = []
    for key, value in record.items():
        if not isinstance(key, str):
            continue
        split, bare = '', key.lower()
        for prefix, label in [('train_', 'train'), ('eval_', 'val'), ('validation_', 'val'),
                              ('val_', 'val'), ('test_', 'test')]:
            if bare.startswith(prefix):
                split, bare = label, bare[len(prefix):]
                break
        numeric = scalar_number(value)
        if bare not in METRICS or numeric is None:
            continue
        if not split and (bare == 'embedding_loss' or epoch is not None):
            split = 'train'
        metrics.append(f'{split + " " if split else ""}{bare}: {numeric!r}')
    if not metrics:
        return line
    axis = f'Epoch: {epoch!r} ' if epoch is not None else ''
    return line[:start] + axis + ' | '.join(metrics)


def normalize_scalar_logs(output):
    return '\n'.join(normalize_scalar_line(line) for line in output.splitlines())
