"""Parse author training logs into bounded, source-attributed experiment curves."""

import re
import math
from pathlib import Path

EPOCH = re.compile(r"\b[Ee]poch\s*[:=]?\s*(\d+)(?:\s*/\s*\d+)?", re.IGNORECASE)
FOLD = re.compile(r"\b(?:test[_\s-]*fold|fold)\s*[:=]?\s*(\d+)", re.IGNORECASE)
METRIC = re.compile(
    r"\b((?:(?:train|training|val|valid|validation|test)[_\s]+)?(?:avg[_\s]*loss|loss|acc(?:uracy)?|auc|auroc|aupr|f1|"
    r"rmse|mae|mse|prd|cr|fid|perplexity|top[_ -]?[15][ _-]*(?:err(?:or)?)|bleu|rouge|precision|recall))\s*(?:[:=]|\s)\s*(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)(%?)",
    re.IGNORECASE,
)
LOG_NAME = re.compile(
    r"^(.*?)-(original|data-retry|loader-retry|dependency-retry-\d+|"
    r"agent-repair-\d+|tuning-\d+(?:-retry-\d+)?|lr-[\d.]+)\.log$"
)


def parse_experiment_log(path: Path, *, dataset: str = '', attempt: str = '') -> dict:
    name = LOG_NAME.match(path.name)
    if not name and not dataset:
        return {"dataset": path.stem, "attempt": "unknown", "points": []}
    dataset = dataset or name.group(1)
    attempt = attempt or (name.group(2) if name else 'recorded-baseline')
    if path.stat().st_size > 12_000_000:
        return {"dataset": dataset, "attempt": attempt, "points": [], "note": "日志超过解析限制"}
    text = path.read_text(encoding="utf-8", errors="replace")
    points: list[dict] = []
    for line in text.splitlines():
        epoch = EPOCH.search(line)
        if not epoch:
            continue
        metrics = {}
        split = ''
        for match in METRIC.finditer(line):
            key, value, percent = match.groups()
            preceding = line[:match.start()]
            markers = list(re.finditer(r'\b(train(?:ing)?|val(?:id(?:ation)?)?|test)\b', preceding, re.I))
            if markers:
                split = markers[-1][1].lower()
                split = 'val' if split.startswith('val') else 'train' if split.startswith('train') else 'test'
            normalized = re.sub(r"\s+", "_", key.lower())
            normalized = (normalized.replace("training_", "train_").replace("validation_", "val_")
                          .replace("valid_", "val_").replace("accuracy", "acc").replace("avg_loss", "loss"))
            if normalized == "loss":
                normalized = f'{split or "train"}_loss'
            elif not normalized.startswith(('train_', 'val_', 'test_')) and split:
                normalized = f'{split}_{normalized}'
            numeric = float(value)
            if math.isfinite(numeric):
                # Keep author units; the dashboard already formats fractions
                # and percentage numbers without mixing the two in one key.
                metrics[normalized] = numeric
        if metrics:
            fold = FOLD.search(line)
            points.append({"epoch": int(epoch.group(1)),
                           **({"fold": int(fold.group(1))} if fold else {}), **metrics})
    if len(points) > 500:
        points = [points[round(index * (len(points)-1)/499)] for index in range(500)]
    return {
        "dataset": dataset, "attempt": attempt, "points": points,
        "updated_at": path.stat().st_mtime,
    }


def read_job_telemetry(workspace: Path, job_id: str) -> dict:
    directory = (workspace / "runs" / job_id / "telemetry").resolve()
    if workspace.resolve() not in directory.parents or not directory.is_dir():
        return {"series": []}
    files = sorted((p for p in directory.rglob('*.log') if not p.is_symlink() and directory in p.resolve().parents),
                   key=lambda path: path.stat().st_mtime, reverse=True)[:48]
    # A large search must not evict the original datasets' baseline curves.
    originals = [p for p in directory.glob('*-original.log')
                 if not p.is_symlink() and directory in p.resolve().parents][:12]
    files = list(dict.fromkeys([*originals, *files]))
    series = []
    for path in files:
        item = parse_experiment_log(path)
        if path.parent != directory:
            item['attempt'] = path.parent.name[:8] + ':' + item['attempt']
        series.append(item)
    return {"series": series}
