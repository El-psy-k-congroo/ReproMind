from types import SimpleNamespace
import pytest
from app.services.experiment_scope import resolve_scope, revision_scope

COMMAND = {'commands': [{'dataset': name, 'argv': ['python', '--dataset', alias]}
                       for name, alias in [('Citeseer', 'citeseer'), ('Cora', 'cora'),
                                           ('Pubmed', 'pubmed'), ('NELL', 'nell.0.001')]]}

@pytest.mark.parametrize('instruction,expected', [
    ('请将epoch设为58，其余参数保持不变，重新跑一下nell这个数据集', ['NELL']),
    ('请在GPU上重跑nell.0.001', ['NELL']),
    ('重新跑NELL和Cora', ['Cora', 'NELL']),
    ('全部数据集重跑，但不要跑Cora', ['Citeseer', 'Pubmed', 'NELL']),
    ('不要跑Cora，而是重新跑NELL', ['NELL']),
    ('只跑Citeseer，不要跑Cora', ['Citeseer']),
    ('请把实验在gpu上跑一遍', ['Citeseer', 'Cora', 'Pubmed', 'NELL']),
])
def test_scope(instruction, expected):
    assert resolve_scope(COMMAND, instruction)['selected_datasets'] == expected

@pytest.mark.parametrize('instruction', ['请重新跑一下unknown这个数据集', '只跑mnist', '请重跑unknown'])
def test_unknown_dataset_is_not_all(instruction):
    scope = resolve_scope(COMMAND, instruction)
    assert scope['scope_reason'] and not scope['selected_datasets']

def test_backend_blocks_scope_expansion_and_empty_payload_uses_intent():
    base = SimpleNamespace(execution_command=COMMAND)
    proposal = {'instruction': '请将epoch设为58，重新跑NELL'}
    assert revision_scope(base, proposal, [])['selected_datasets'] == ['NELL']
    assert revision_scope(base, proposal, ['Cora', 'NELL'])['scope_reason']
    assert not revision_scope(base, proposal, ['nell'])['scope_reason']

def test_scope_works_for_other_repository_labels_and_cli_aliases():
    command = {'commands': [{'dataset': 'Custom-Benchmark', 'argv': ['python', '--dataset=custom_v2']},
                            {'dataset': 'Other'}]}
    assert resolve_scope(command, '重新跑custom_v2')['selected_datasets'] == ['Custom-Benchmark']
