import json
from app.api.copilot import _bounded_prompt
from app.services.prompt_budget import whole_source_packets


def test_source_budget_keeps_whole_records_and_marks_omissions():
    packet = {'file': 'network.py', 'code': 'L1: output = weight * branch + (1-weight) * anchor\nL2: return output'}
    result = whole_source_packets({'named_implementations': [packet] * 40}, 700)
    assert result['named_implementations'] and all(row == packet for row in result['named_implementations'])
    assert result['omitted_packets_due_to_budget_not_missing']['named_implementations'] > 0


def test_reserved_source_and_loss_slots_cannot_bypass_total_prompt_limit():
    context = {'job': {'run_id': 'selected', 'status': 'completed', 'dataset_results_count': 1},
               'source_task_evidence': {'named_implementations': [{'file': 'network.py', 'code': 'L1: ' + 'x' * 9000}] * 20},
               'source_loss_definitions': [{'file': 'loss.py', 'excerpt': 'L2: ' + 'y' * 8000}] * 4}
    result = _bounded_prompt('解释当前模型', context, [], '', limit=15000)
    assert len(result) <= 15000
    parsed = json.loads(result)
    assert parsed['question'] == '解释当前模型'
    source = parsed['selected_source_evidence_not_execution_proof']
    for row in source.get('named_implementations', []):
        assert row['code'] == 'L1: ' + 'x' * 9000
