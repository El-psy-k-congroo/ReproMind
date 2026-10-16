"""Numeric formats that occur in real training output."""
import pytest
from app.services.reproduction_engine import _extract_metrics


@pytest.mark.parametrize('text, expected', [
    ('Test RMSE: 1.25e-3', {'rmse': 0.00125}),
    ('Final mse = 2E+2', {'mse': 200.0}),
    ('Final accuracy: 81.0%', {'accuracy': 0.81}),
    ('AUC: .85 loss: -2.3e-2', {'auc': .85, 'loss': -.023}),
    ('accuracy: 0.0', {'accuracy': 0.0}),
    ('accuracy: 1e999', {}),
    ('accuracy: NaN loss: inf', {}),
])
def test_observed_metric_format_is_not_truncated(text, expected):
    for _ in range(2):
        assert _extract_metrics(text) == expected
