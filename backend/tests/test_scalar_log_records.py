import pytest

from app.services.metric_observations import evaluation_observations
from app.services.reproduction_engine import METRIC_PATTERN, _extract_metrics
from app.services.scalar_log_records import normalize_scalar_line
from app.services.telemetry import parse_experiment_log


@pytest.mark.parametrize('output', [
    "Metrics: {'accuracy': 0.4964305326743548}",
    'Metrics: {"accuracy": "0.4964305326743548"}',
])
def test_dictionary_final_metrics_and_fractional_training_axis(tmp_path, output):
    log = ("{'embedding_loss': '0.3418', 'grad_norm': '2.699', 'epoch': '0.025'}\n"
           "{'train_loss': '0.2284', 'epoch': '1'}\n"
           '***** Running evaluation *****\n' + output)
    assert _extract_metrics(log) == {'accuracy': .4964305326743548}
    evidence = evaluation_observations(log, METRIC_PATTERN)
    assert evidence['partition'] == 'unscoped'  # evaluation alone is not proof of test data
    path = tmp_path / 'example-original.log'
    path.write_text(log, encoding='utf-8')
    assert parse_experiment_log(path)['points'] == [
        {'epoch': .025, 'train_embedding_loss': .3418}, {'epoch': 1, 'train_loss': .2284}]


def test_explicit_train_validation_and_test_dictionaries_are_kept_separate():
    log = ('{"epoch": 1, "loss": 0.1, "accuracy": 0.99}\n'
           '{"epoch": 1, "eval_loss": 0.2, "eval_accuracy": 0.8}\n'
           '{"test_accuracy": 0.7}\n')
    assert _extract_metrics(log) == {'accuracy': .7}
    assert evaluation_observations(log, METRIC_PATTERN)['partition'] == 'test'
    assert _extract_metrics(log.split('{"test_accuracy"')[0]) == {}


def test_dictionary_values_are_bounded_finite_and_never_evaluated():
    for line in ["{'accuracy': __import__('os').getcwd()}", '{"accuracy": true}',
                 '{"accuracy": 1e999}', "{'accuracy': {'mean': 0.9}}"]:
        assert normalize_scalar_line(line) == line
        assert _extract_metrics(line) == {}
