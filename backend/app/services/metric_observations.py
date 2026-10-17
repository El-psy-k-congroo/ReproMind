"""Select final evaluation observations without mixing train/validation metrics."""
import re
from .scalar_log_records import normalize_scalar_logs

PARTITION = re.compile(r'(?<![A-Za-z0-9])(train(?:ing)?|val(?:id(?:ation)?)?|test)(?=[_\s:\]])', re.I)


def evaluation_observations(output, metric_pattern):
    observations = []
    scoped = False
    active = None
    last_partition = None
    for number, line in enumerate(normalize_scalar_logs(output).splitlines(), 1):
        if re.search(r'\bfinal\s+(?:evaluation|test|results?)\b|\bload(?:ing)?\s+\d+(?:st|nd|rd|th)?\s+epoch', line, re.I):
            observations = []
            active = 'test'
            last_partition = None
            scoped = True
        elif re.search(r'\brunning\s+evaluation\b', line, re.I):
            # An unlabelled evaluation may use validation or test data. Keep
            # the observations without inventing its dataset partition.
            observations = []
            active = 'evaluation'
            last_partition = None
            scoped = True
        elif re.search(r'\bepoch\s*[:=]?\s*\[?\s*\d+', line, re.I):
            active = None
        markers = list(PARTITION.finditer(line))
        def partition(match):
            label = match[1].casefold()
            return 'train' if label.startswith('train') else 'val' if label.startswith('val') else 'test'
        labels = {partition(marker) for marker in markers}
        # Summary lines can end in "Test Loss" after their other metrics.
        sole = next(iter(labels)) if len(labels) == 1 else None
        if sole and (not list(metric_pattern.finditer(line)) or markers[0].start() < 8):
            active = sole
        for metric in metric_pattern.finditer(line):
            preceding = [marker for marker in markers if marker.start() <= metric.start()]
            split = partition(preceding[-1]) if preceding else sole or active
            if split:
                scoped = True
                if split in {'test', 'evaluation'}:
                    if last_partition != split:
                        observations = []
                    observations.append({'partition': 'test' if split == 'test' else 'unscoped',
                                         'line': number, 'text': metric[0]})
                last_partition = split
    return {'scoped': scoped, 'observations': observations,
            'text': '\n'.join(item['text'] for item in observations),
            'partition': 'test' if observations and all(item['partition'] == 'test' for item in observations) else 'unscoped'}
