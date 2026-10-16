"""Read-only, cross-run comparison. Recorded metrics are not causal proof."""
import math
import re
from pathlib import Path

from .runtime_evidence import execution_logs, device_log_evidence
from .tuning import numeric_flags, command_fingerprint
from .telemetry import read_job_telemetry, parse_experiment_log
from .run_parameters import effective_numeric_parameters


def performance_question(question):
    # Merely naming an evaluation metric is not a request for causal diagnosis.
    metric = r'性能|准确率|精度|效果|分数|得分|速度|耗时|吞吐|延迟|显存|内存|收敛|指标|结果|loss|accuracy|precision|recall|f1\b|auc\b|rmse|mae\b|performance|latency|throughput'
    analysis = r'为什么|为何|原因|退化|变慢|下降|提升|差这么|差异|比较|对比|\bwhy\b|\bworse\b|\bcompare\b'
    return bool(re.search(metric, question, re.I) and re.search(analysis, question, re.I))


def _finite(value):
    return type(value) in {int, float} and math.isfinite(value)


def _overrides(argv):
    values = {}
    for index, flag in enumerate(argv):
        if not isinstance(flag, str) or not flag.startswith('--'):
            continue
        name, equal, value = flag[2:].partition('=')
        if not equal:
            value = argv[index + 1] if index + 1 < len(argv) else ''
        if not re.fullmatch(r'[\w-]{1,60}', name) or not isinstance(value, str) or value.startswith('--'):
            continue
        try:
            number = float(value)
            if math.isfinite(number):
                values[name] = number
        except ValueError:
            if name in {'dataset', 'data', 'device', 'config', 'seed'}:
                values[name] = value[:120]
    return values


def _row(job, workspace):
    result = job.result_json or {}
    execution = job.execution_command or {}
    source = Path(str(execution.get('pythonpath') or (job.code_spec or {}).get('source_path') or '')).resolve()
    safe_source = bool(workspace and workspace in source.parents and source.is_dir())
    revision = result.get('revision') or result.get('rerun') or {}
    rows = []
    series = read_job_telemetry(workspace, job.id)['series'] if workspace else []
    for item in (result.get('dataset_results') or [])[:12]:
        if not isinstance(item, dict):
            continue
        argv = item.get('baseline_argv') or []  # preparation commands may have changed later
        parameters = _overrides(argv)
        snapshot = item.get('baseline_parameters') or {}
        if snapshot:
            parameters = {**snapshot, **parameters}
        matched = False
        fingerprint = item.get('baseline_fingerprint')
        if safe_source and argv and fingerprint:
            matched = command_fingerprint({'argv': argv}, source) == fingerprint
        if matched:
            try:
                defaults = effective_numeric_parameters(source, execution.get('entrypoint_candidates'), argv)
            except ValueError:
                defaults = {}
            parameters = {**defaults, **parameters}
        epochs = [point['epoch'] for line in series
                  if line['dataset'].lower() == str(item.get('dataset','')).lower() and line['attempt'] == 'original'
                  for point in line['points']]
        # A successful retry can be the recorded baseline; the failed first
        # attempt's empty telemetry must not hide its actual completed epochs.
        if workspace and item.get('baseline_log') and item.get('return_code') == 0:
            run_root = (workspace / 'runs' / job.id).resolve()
            baseline = (run_root / str(item['baseline_log'])).resolve()
            if run_root in baseline.parents and baseline.is_file():
                points = parse_experiment_log(baseline, dataset=str(item.get('dataset','')), attempt='recorded-baseline')['points']
                if points:
                    epochs = [point['epoch'] for point in points]
        rows.append({
            'dataset': str(item.get('dataset') or '')[:100], 'return_code': item.get('return_code'),
            'metrics': {k: v for k, v in (item.get('metrics') or {}).items() if _finite(v)},
            'baseline_metrics': {k: v for k, v in (item.get('baseline_metrics') or {}).items() if _finite(v)},
            'tuning_selected': bool(item.get('tuning_selected')),
            'wall_time_seconds': item.get('wall_time_seconds') if _finite(item.get('wall_time_seconds')) else None,
            'parameters': parameters, 'source_matches_execution_fingerprint': matched,
            'parameter_evidence': '执行时参数快照及记录的命令覆盖' if snapshot else '执行指纹匹配的源码默认值及记录的命令覆盖' if matched else '仅记录的命令覆盖；源码默认值未核验',
            'execution_parameters_snapshotted': bool(snapshot),
            'baseline_fingerprint': fingerprint,
            'last_logged_epoch': max(epochs, default=None),
        })
    proof = device_log_evidence(execution_logs(workspace, job.id)) if workspace else []
    device = ('GPU计算算子已记录' if any(row['kind'] == 'gpu_compute_placement' for row in proof)
              else 'GPU训练设备已记录' if any(row['kind'] == 'training_device_gpu' for row in proof)
              else 'CPU训练设备已记录' if any(row['kind'] == 'training_device_cpu' for row in proof)
              else '实际训练设备未知：没有该记录的训练设备证明')
    return {'run_id': job.id, 'selected': False, 'status': job.status.value, 'training_device_verdict':device,
            'title': str(revision.get('title') or '原实验')[:160],
            'base_run_id': revision.get('base_run_id'), 'datasets': rows,
            'device_evidence': proof[:3],
            'environment_note': '共享环境当前探测不可倒推历史设备；迁移提案和标题不是执行证明',
            'created_at': str(job.created_at) if getattr(job, 'created_at', None) else None}


def comparison_evidence(jobs, selected, question, workspace_for):
    selected_names = {str(row.get('dataset', '')).lower() for row in (selected.result_json or {}).get('dataset_results', [])}
    mentioned = [job for job in jobs if job.id[:8].lower() in question.lower() or job.id.lower() in question.lower()]
    matching = [job for job in jobs if any(str(row.get('dataset', '')).lower() in selected_names
                                         for row in (job.result_json or {}).get('dataset_results', []))]
    candidates = list(dict.fromkeys([selected.id, *(job.id for job in mentioned), *(job.id for job in matching)]))
    by_id = {job.id: job for job in jobs}
    by_id[selected.id] = selected
    records = [_row(by_id[id], workspace_for(by_id[id])) for id in candidates[:8]]
    for row in records:
        row['selected'] = row['run_id'] == selected.id
        row['datasets'] = [item for item in row['datasets'] if item['dataset'].lower() in selected_names] if selected_names else row['datasets']
    return {'selected_run_id': selected.id, 'records': records,
            'omitted_runs': max(0, len(candidates) - len(records)),
            'rules': ['先回答结果差异，再区分准确率和运行速度；未明确时两者分别报告',
                      '只比较相同数据集和指标；调参结果与原配置基线分开',
                      '比较命令、已核验参数、轮数和设备证据；无记录的配置或历史环境为未知',
                      'wall_time 包括启动/预处理等开销，不等于纯训练吞吐，缺少硬件同条件对照不能归因设备',
                      '配置差异是混杂因素，不证明其就是原因；只能提出可验证假设',
                      '引用实验短ID及实际数值，不把提案说明当作已执行事实；没有对照应明确说明']}


def comparison_facts_answer(comparison):
    """Numbers, devices and configuration provenance are rendered by the server."""
    records = (comparison or {}).get('records') or []
    if not any(record.get('datasets') for record in records):
        return '当前没有可比较的已记录指标，不能判断性能差异。\n\n'
    def safe(value):
        return str(value).replace('|','/').replace('\n',' ')[:120]
    selected = next((record for record in records if record.get('selected')), None)
    differences = []
    if selected:
        for current in selected.get('datasets') or []:
            for other in records:
                if other is selected:
                    continue
                for previous in other.get('datasets') or []:
                    if previous.get('dataset', '').lower() != current.get('dataset', '').lower():
                        continue
                    left, right = current.get('parameters') or {}, previous.get('parameters') or {}
                    changed = sorted(key for key in left.keys() & right.keys()
                                     if key not in {'epochs', 'dataset'} and left[key] != right[key])
                    if changed:
                        differences.append(f"- {safe(current['dataset'])} 与 `{safe(other['run_id'][:8])}` 的"
                                           + '、'.join(safe(key) for key in changed)
                                           + ' 不同；并非仅改变 epoch 的受控对照。')
    lines = (['已核验的配置差异：', *differences[:8], ''] if differences else []) + [
             '已记录的实验对照如下。下文区分已核验差异与待验证原因。', '',
             '| 实验 | 数据集与最终指标 | 总耗时 | 实际日志轮数 / 配置上限 | 设备证据 |',
             '|---|---|---|---|---|']
    details = []
    for record in records:
        for row in record.get('datasets') or []:
            metrics = ', '.join(f'{safe(k)}={v:g}' for k,v in row['metrics'].items()) or '未取得'
            label = safe(record['run_id'][:8]) + ('（当前）' if record['selected'] else '')
            device = record.get('training_device_verdict','未知').split('：')[0]
            lines.append(f"| {label} | {safe(row['dataset'])}: {metrics}{'（调参后）' if row['tuning_selected'] else ''} | "
                         f"{row['wall_time_seconds'] if row['wall_time_seconds'] is not None else '未知'} s | "
                         f"{row['last_logged_epoch'] or '未知'} / {row['parameters'].get('epochs','未知')} | {device} |")
            parameters = ', '.join(f'{safe(k)}={safe(v)}' for k,v in row['parameters'].items() if k not in {'dataset','epochs'})
            details.append(f"- `{record['run_id'][:8]}` {safe(row['dataset'])}：{parameters or '其他参数未核验'}。"
                           + ('执行时参数快照已保存。' if row.get('execution_parameters_snapshotted') else '默认值已由执行指纹核验。' if row['source_matches_execution_fingerprint']
                              else '仅采用历史命令覆盖；其余默认值未核验，不能从标题或父记录补齐。'))
    lines.extend(['', '已核验的配置证据：', *details, '',
                  '以上是记录事实，不是因果结论。配置上限不等于实际轮数；总耗时不等于纯训练吞吐。'
                  '参数、数据划分、随机种子及训练轮数未受控时，不能只凭指标差异确定性能变化的原因。', '',
                  '待验证的原因与下一步：', ''])
    return '\n'.join(lines)


def render_hypotheses(data, comparison):
    allowed = {row['run_id'] for row in (comparison or {}).get('records',[])}
    rows = data.get('hypotheses')
    if not isinstance(rows,list) or not 1 <= len(rows) <= 3:
        raise ValueError('性能分析未返回有效的待验证假设')
    result = []
    for row in rows:
        if not isinstance(row,dict):
            continue
        reason, verify, ids = row.get('possible_reason'), row.get('verify'), row.get('supporting_runs')
        if (not isinstance(reason,str) or not isinstance(verify,str) or not reason.strip() or not verify.strip()
                or len(reason)>250 or len(verify)>250 or not isinstance(ids,list) or not ids
                or any(id not in allowed for id in ids)):
            continue
        # Numeric facts are rendered above from records; the model cannot invent
        # metric/configuration values in the hypothesis text.
        # Identifiers such as hidden1 or beta2 are parameter names, not
        # fabricated numeric configuration values. Values remain forbidden.
        numeric_text = reason + verify
        parameter_names = {name for record in (comparison or {}).get('records', [])
                           for item in record.get('datasets', [])
                           for name in (item.get('parameters') or {})}
        for name in parameter_names:
            if re.fullmatch(r'[A-Za-z_][\w-]*', name):
                numeric_text = re.sub(r'(?<![\w])' + re.escape(name) + r'(?![\w])', '', numeric_text)
        for run_id in allowed:
            for name in (run_id, run_id[:8]):
                numeric_text = re.sub(r'(?<![\w])' + re.escape(name) + r'(?![\w])', '', numeric_text)
        # Users often name their requested epoch limit in the question. Permit
        # that explicit named value only when every cited run has it recorded;
        # this is not permission to introduce scores or infer missing defaults.
        by_id = {record['run_id']: record for record in (comparison or {}).get('records', [])}
        def verified_epoch(match):
            value = int(match.group(1))
            if all(by_id[id].get('datasets') and all(
                    item.get('parameters', {}).get('epochs') == value
                    for item in by_id[id]['datasets']) for id in ids):
                return 'epoch配置上限'
            return match.group(0)
        numeric_text = re.sub(r'(?i)(?<![A-Za-z_])epochs?[^\d\n]{0,8}(\d+)(?![\d.])', verified_epoch, numeric_text)
        if re.search(r'\d',numeric_text):
            continue
        result.append(f"- 可能原因：{reason.strip()} 待验证：{verify.strip()} 相关记录："
                      + '、'.join(f'`{id[:8]}`' for id in dict.fromkeys(ids)) + '。')
    if not result:
        raise ValueError('原因假设未通过格式、项目引用或非数值证据校验')
    if len(result) < len(rows):
        result.append('\n部分候选原因未通过证据校验，已省略；已核验实验对照不受影响。')
    return '\n'.join(result)
