from types import SimpleNamespace
import pytest
from app.services.experiment_scope import resolve_scope, resolve_run_scope, revision_scope

COMMAND = {'commands': [{'dataset': name, 'argv': ['python', '--dataset', alias]}
                       for name, alias in [('Citeseer', 'citeseer'), ('Cora', 'cora'),
                                           ('Pubmed', 'pubmed'), ('NELL', 'nell.0.001')]]}


@pytest.mark.parametrize('instruction,expected', [
    ('请在gpu上跑完这四个数据集', ['Citeseer', 'Cora', 'Pubmed', 'NELL']),
    ('请在 GPU 上执行这4个数据集', ['Citeseer', 'Cora', 'Pubmed', 'NELL']),
    ('请跑完全部数据集', ['Citeseer', 'Cora', 'Pubmed', 'NELL']),
    ('只跑Cora和NELL', ['Cora', 'NELL']),
    ('请跑完全部数据集，但不要跑Cora', ['Citeseer', 'Pubmed', 'NELL']),
    ('请重新运行当前实验', ['NELL']),
])
def test_project_scope_is_distinct_from_selected_subset(instruction, expected):
    job = SimpleNamespace(execution_command={'commands': [COMMAND['commands'][-1]]},
                          paper_spec={'datasets': ['Citeseer', 'Cora', 'Pubmed', 'NELL']})
    scope = resolve_run_scope(job, instruction)
    assert scope['selected_datasets'] == expected and not scope['scope_reason']
    assert len(job.execution_command['commands']) == 1


@pytest.mark.parametrize('instruction', ['请跑完这三个数据集', '请重跑unknown'])
def test_ambiguous_count_and_unknown_names_cannot_fall_back_to_last_dataset(instruction):
    job = SimpleNamespace(execution_command={'commands': [COMMAND['commands'][-1]]},
                          paper_spec={'datasets': ['Citeseer', 'Cora', 'Pubmed', 'NELL']})
    scope = resolve_run_scope(job, instruction)
    assert not scope['selected_datasets'] and scope['scope_reason']

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
