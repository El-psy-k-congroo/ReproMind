import asyncio
import hashlib
import io
import json
import subprocess
import sys
from types import SimpleNamespace

import httpx
import pytest

from app.services import reproduction_engine as engine
from app.services.code_agent import CodingFailure, generate_project
from app.services.official_datasets import discover_official, normalize_tabular, official_identity
from app.services.paper_evidence import paper_context, related_evidence, core_implementation_concerns


def reply(data=None, text=''):
    return SimpleNamespace(data=data, text=text, error='', usage={}, finish_reason='stop')


def test_appendix_settings_survive_long_method_and_bibliography():
    text = ('1 I NTRODUCTION\n' + 'background\n' * 3000 +
            '3 P ROPOSED METHOD\n' + 'algorithm fine-tuning narrative\n' * 1500 +
            '4 E XPERIMENTS\nSPLIT_70_30\n' + 'result table\n' * 500 +
            'REFERENCES\n' + 'reference bibliography\n' * 1500 +
            'A D ETAILED SETTINGS\nA.3 F INE - TUNING\n'
            'Distill-GPT2 is fine-tuned for 200 or 300 epochs with learning rate 5e-5.\n'
            'C P ROMPT TEMPLATES\nACTUAL_PROMPT_EXAMPLE\n')
    context = paper_context(text)
    assert len(context) <= 24000
    assert '200 or 300 epochs' in context and '5e-5' in context
    assert 'ACTUAL_PROMPT_EXAMPLE' in context and 'SPLIT_70_30' in context
    assert context.count('reference bibliography') < 30
    assert '200 or 300 epochs' in related_evidence(text, ['fine-tuning training procedure'])


def test_review_protocol_retains_original_issue_and_fetches_exact_paper_evidence():
    quote = 'Fine-tune for 200 epochs with learning rate 5e-5.'
    paper = 'background\n' * 3000 + '\nA.3 Implementation\n' + quote
    reviews, writes = [], []

    async def review(_role, _system, prompt, **kwargs):
        request = json.loads(prompt)
        reviews.append(request)
        if len(reviews) == 1:
            return reply({'issues': [{'file': 'method.py', 'category': 'method',
                                     'reason': 'missing fine-tuning procedure', 'code_evidence': 'return 0'}]})
        if len(reviews) == 2:
            assert quote in request['supplemental_paper_evidence']
            return reply({'issues': [{'file': 'method.py', 'category': 'method', 'reason': 'missing fine-tuning',
                                     'code_evidence': 'return 0', 'paper_evidence': quote}]})
        return reply({'issues': [], 'issue_checks': [{'id': request['pending_issues'][0]['id'],
                      'status': 'resolved', 'reason': 'procedure implemented'}]})

    async def write(*args, **kwargs):
        writes.append(True)
        return reply(text='def train():\n    return 1\n')

    result = asyncio.run(generate_project(SimpleNamespace(complete_json=review, complete_text=write), paper, {},
                        initial_files={'method.py': 'def train():\n    return 0\n'},
                        previous_plan={'files': [{'path': 'method.py'}], 'entrypoint': 'method.py'}))
    assert len(writes) == 1
    event = next(row for row in result['generation_trace'] if row['phase'] == 'review_protocol')
    assert event['review_response']['issues'][0]['reason'] == 'missing fine-tuning procedure'
    assert quote in event['supplemental_paper_evidence']


def test_missing_core_training_cannot_pass_as_a_yellow_limitation():
    async def review(*args, **kwargs):
        return reply({'issues': [], 'limitations': ['The implementation does not include the fine-tuning procedure.']})
    with pytest.raises(CodingFailure, match='核心实现缺失') as failed:
        asyncio.run(generate_project(SimpleNamespace(complete_json=review), 'fine-tune for 200 epochs', {},
                    initial_files={'method.py': 'def train():\n    return 0\n'},
                    previous_plan={'files': [{'path': 'method.py'}], 'entrypoint': 'method.py'}))
    assert failed.value.trace[-1]['review_response']['limitations']
    assert not core_implementation_concerns(['Missing fine-tuned checkpoint; user must provide access.'])


def mock_client(monkeypatch, handler):
    original = httpx.AsyncClient
    monkeypatch.setattr(httpx, 'AsyncClient', lambda **kwargs: original(**kwargs, transport=httpx.MockTransport(handler)))


def uci_metadata():
    return {'data': {'name': 'Default of Credit Card Clients', 'data_url': 'https://archive.ics.uci.edu/static/public/350/data.csv',
                     'target_col': ['Y'], 'index_col': ['ID'], 'num_instances': 2, 'num_features': 1}}


def test_official_metadata_is_not_equated_to_paper_version(monkeypatch):
    mock_client(monkeypatch, lambda request: httpx.Response(200, json=uci_metadata()))
    candidates, request, errors = asyncio.run(discover_official('Default', 'credit card Taiwan classification'))
    assert not errors and candidates[0]['availability'] == 'metadata_verified'
    assert request['dataset_id'] == 350 and request['target_column'] == 'Y'
    assert request['paper_version_verified'] is False and request['download_status'] == 'planned'


@pytest.mark.parametrize('name,paper', [('Diabetes', 'sklearn regression dataset'), ('Default', 'a generic task'), ('Sick', '')])
def test_ambiguous_short_names_do_not_trigger_network_or_download(name, paper, monkeypatch):
    def forbidden(_request):
        raise AssertionError('Ambiguous identity must not fetch a replacement')
    mock_client(monkeypatch, forbidden)
    candidates, request, errors = asyncio.run(discover_official(name, paper))
    assert request is None and not errors and not candidates[0]['identity_verified']


def test_openml_checks_identity_version_target_and_preserves_checksum(monkeypatch):
    mock_client(monkeypatch, lambda request: httpx.Response(200, json={'data_set_description': {
        'id': '37', 'name': 'diabetes', 'version': '1', 'status': 'active',
        'url': 'https://www.openml.org/data/v1/download/37/diabetes.arff',
        'default_target_attribute': 'class', 'md5_checksum': 'abc'}}))
    _, request, errors = asyncio.run(discover_official('Diabetes', 'Pima Indians glucose'))
    assert not errors and request['format'] == 'arff' and request['version'] == '1'
    assert request['md5_checksum'] == 'abc'


def test_official_source_failure_is_not_reported_as_private_data(monkeypatch):
    mock_client(monkeypatch, lambda request: httpx.Response(503))
    candidates, request, errors = asyncio.run(discover_official('Shoppers', ''))
    assert request is None and errors and candidates[0]['availability'] == 'metadata_lookup_failed'


def test_official_metadata_cannot_redirect_to_a_different_dataset(monkeypatch):
    mock_client(monkeypatch, lambda request: httpx.Response(200, json=uci_metadata()))
    candidates, request, errors = asyncio.run(discover_official('Shoppers', ''))
    assert request is None and errors and 'ID 不匹配' in errors[0]


def test_invalid_official_target_is_not_guessed(monkeypatch):
    metadata = uci_metadata()
    metadata['data']['target_col'] = []
    mock_client(monkeypatch, lambda request: httpx.Response(200, json=metadata))
    _, request, errors = asyncio.run(discover_official('Default', 'credit card'))
    assert request is None and '目标列' in errors[0]


def test_dataset_preparation_does_not_auto_adapt_author_source(tmp_path, monkeypatch):
    mock_client(monkeypatch, lambda request: httpx.Response(200, json=uci_metadata()))
    async def run(generated):
        return await engine._dataset_spec('credit card Taiwan', {'datasets': ['Default']}, tmp_path, generated=generated)
    author = asyncio.run(run(False))
    assert author['missing_datasets'] == ['Default'] and not author['download_requests']
    assert author['dataset_diagnostics']['Default']['kind'] == 'loader_adaptation_required'
    generated = asyncio.run(run(True))
    assert not generated['missing_datasets'] and generated['download_requests'][0]['adapter'] == 'official_tabular'
    assert not generated['available_datasets']  # planned != downloaded


def test_tabular_normalization_preserves_missing_values_and_classes_without_resplitting():
    request = {'format': 'csv', 'target_column': 'Y', 'index_columns': ['ID'], 'expected_rows': 2, 'expected_features': 1}
    frame, manifest = normalize_tabular(io.StringIO('ID,X,Y\n1,10,negative\n2,,positive\n'), request)
    assert list(frame.columns) == ['X', 'Y'] and len(frame) == 2
    assert frame['Y'].tolist() == ['negative', 'positive'] and manifest['missing_values'] == 1
    assert manifest['transformations']['split'] == 'none' and manifest['paper_version_verified'] is False
    with pytest.raises(ValueError, match='行数'):
        normalize_tabular(io.StringIO('ID,X,Y\n1,10,0\n'), request)


def test_arff_categorical_labels_are_decoded_and_not_silently_dropped():
    data = '@relation pima\n@attribute glucose numeric\n@attribute class {positive,negative}\n@data\n1,positive\n?,negative\n'
    frame, manifest = normalize_tabular(io.StringIO(data), {'format': 'arff', 'target_column': 'class'})
    assert frame['class'].tolist() == ['positive', 'negative'] and len(frame) == 2
    assert manifest['missing_values'] == 1


def test_approved_official_download_converts_real_payload_in_project_interpreter(tmp_path, monkeypatch):
    payload = b'ID,X,Y\n1,10,0\n2,20,1\n'
    mock_client(monkeypatch, lambda request: httpx.Response(200, content=payload))
    async def run_process(argv, cwd, timeout):
        result = subprocess.run(argv, cwd=cwd, capture_output=True, text=True, timeout=timeout)
        return result.returncode, result.stdout + result.stderr
    monkeypatch.setattr(engine, '_run_process', run_process)
    request = {**uci_metadata()['data'], 'format': 'csv', 'target_column': 'Y', 'index_columns': ['ID'],
               'expected_rows': 2, 'expected_features': 1, 'dataset': 'Default', 'dataset_id': 350,
               'url': uci_metadata()['data']['data_url'], 'source': 'official_uci', 'version': 'current_official'}
    job = SimpleNamespace(id='test-official-download', approved_at=object())
    asyncio.run(engine._download_official_tabular(job, request, engine.Path(sys.executable), tmp_path))
    manifest = json.loads((tmp_path / 'default.manifest.json').read_text(encoding='utf-8'))
    assert request['download_status'] == 'ready' and manifest['raw_sha256'] == hashlib.sha256(payload).hexdigest()
    assert (tmp_path / 'default.csv').read_text().splitlines()[0] == 'X,Y'
    (tmp_path / 'default.csv').write_text('user data\n', encoding='utf-8')
    with pytest.raises(ValueError, match='拒绝覆盖'):
        asyncio.run(engine._download_official_tabular(job, request, engine.Path(sys.executable), tmp_path))
    assert (tmp_path / 'default.csv').read_text() == 'user data\n'


def test_official_download_requires_approval_before_any_network(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match='批准'):
        asyncio.run(engine._download_official_tabular(SimpleNamespace(id='test'), {}, engine.Path(sys.executable), tmp_path))


def test_retry_archives_diagnostics_and_reuses_only_unchanged_reviewed_source(tmp_path):
    source = tmp_path / 'generated' / 'method'
    source.mkdir(parents=True)
    code = 'def train():\n    return 1\n'
    (source / 'train.py').write_text(code)
    job = SimpleNamespace(source_mode='generated_implementation',
        code_spec={'source_path': str(source), 'files': [{'path': 'train.py', 'content': code}],
                   'validation': {'method_fidelity': 'reviewed'}},
        dataset_spec={'missing_datasets': ['Default']}, environment_spec={}, execution_command={},
        result_json={'user_action': 'need data', 'action_required': {'kind': 'dataset_unavailable', 'reason': 'need data'}})
    engine._archive_preparation(job)
    assert job.result_json['preparation_history'][0]['action_required']['reason'] == 'need data'
    assert 'user_action' not in job.result_json
    assert engine._reusable_generated_source(job, tmp_path) == source.resolve()
    job.code_spec['limitations'] = ['The implementation lacks the fine-tuning procedure.']
    assert engine._reusable_generated_source(job, tmp_path) is None
    job.code_spec.pop('limitations')
    (source / 'train.py').write_text('modified code')
    assert engine._reusable_generated_source(job, tmp_path) is None


def test_official_cli_plan_supplies_exact_target_and_csv_path(tmp_path):
    source = tmp_path / 'code'
    source.mkdir()
    (source / 'train.py').write_text('p.add_argument("--dataset")\np.add_argument("--target-col")\ndf=pd.read_csv(args.dataset)\n')
    plan = engine._execution_command(source, {'entrypoints': ['train.py']}, 'full', {'prefix': str(tmp_path / 'env')},
        {'datasets': ['Default']}, {'download_requests': [{'dataset': 'Default', 'repository_name': 'default',
        'adapter': 'official_tabular', 'target_column': 'Y'}]}, tmp_path / 'runs' / 'run-1')
    argv = plan['commands'][0]['argv']
    assert argv[-2:] == ['--target-col', 'Y']
    assert argv[argv.index('--dataset') + 1] == str(tmp_path / 'data' / 'default.csv')
