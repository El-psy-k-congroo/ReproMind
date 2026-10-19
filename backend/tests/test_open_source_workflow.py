import asyncio
import json
from types import SimpleNamespace
from pathlib import Path

import pytest
from fastapi import BackgroundTasks, HTTPException

from app.api import reproductions
from app.models import ArtifactKind, ReproductionStatus
from app.services import reproduction_engine as engine
from app.services.dataset_discovery import PLANETOID_PARTS


def test_discovered_repository_links_use_https_without_www():
    assert engine.extract_repository_urls('Code: http://www.github.com/Author/Model.') == [
        'https://github.com/Author/Model']


def test_same_named_repository_cache_requires_owner_identity(tmp_path, monkeypatch):
    old = tmp_path / 'sources' / 'model'
    (old / '.git').mkdir(parents=True)
    (old / 'train.py').write_text('old_author = True', encoding='utf-8')
    (old / '.git' / 'config').write_text(
        '[remote "origin"]\nurl = https://github.com/other/model.git\n', encoding='utf-8')

    async def clone(argv, *args):
        assert argv[:4] == ['git', 'clone', '--depth', '1']
        target = Path(argv[-1])
        target.mkdir(parents=True)
        (target / 'train.py').write_text('new_author = True', encoding='utf-8')
        return 0, 'clone complete'
    monkeypatch.setattr(engine, '_run_process', clone)
    source, _ = asyncio.run(engine._clone_repository('https://github.com/author/model', tmp_path, tmp_path))
    assert source != old and engine._repository_cache_matches(source, 'https://github.com/author/model.git')
    assert (old / 'train.py').read_text(encoding='utf-8') == 'old_author = True'

    async def forbidden(*args): raise AssertionError('Verified source should be reused')
    monkeypatch.setattr(engine, '_run_process', forbidden)
    reused, _ = asyncio.run(engine._clone_repository('https://github.com/author/model', tmp_path, tmp_path))
    assert reused == source


def test_legacy_git_origin_can_be_reused_but_unidentified_source_cannot(tmp_path):
    (tmp_path / '.git').mkdir()
    config = tmp_path / '.git' / 'config'
    config.write_text('[remote "origin"]\nurl=https://github.com/author/model.git\n', encoding='utf-8')
    assert engine._repository_cache_matches(tmp_path, 'https://github.com/author/model')
    config.unlink()
    assert not engine._repository_cache_matches(tmp_path, 'https://github.com/author/model')


def test_verified_archive_reuse_preserves_inner_source_root(tmp_path, monkeypatch):
    url = 'https://github.com/author/model'
    outer = tmp_path / 'sources' / f'{engine._repository_cache_name(url)}-archive'
    inner = outer / 'model-main'
    inner.mkdir(parents=True)
    (inner / 'train.py').write_text('author_code = True', encoding='utf-8')
    engine._record_repository_identity(outer, url)
    engine._record_repository_identity(inner, url)
    def forbidden(*args, **kwargs): raise AssertionError('Cached archive must not redownload')
    monkeypatch.setattr(engine.httpx, 'AsyncClient', forbidden)
    source, _ = asyncio.run(engine._download_repository_archive(url, tmp_path, tmp_path))
    assert source == inner


def setup_workflow(tmp_path, monkeypatch, *, text='Author code https://github.com/author/model', spec=None):
    source = tmp_path / 'sources' / 'model'
    source.mkdir(parents=True)
    (source / 'train.py').write_text(
        "import tensorflow as tf\nflags=tf.app.flags\n"
        "flags.DEFINE_string('dataset', 'alpha', 'Dataset name')\n"
        "print('accuracy')\n", encoding='utf-8')
    (source / 'README.md').write_text('Datasets: Alpha and Beta. Metric: Accuracy.', encoding='utf-8')
    data = source / 'data'
    data.mkdir()
    for name in ['alpha', 'beta']:
        for part in PLANETOID_PARTS:
            (data / f'ind.{name}.{part}').write_bytes(b'fixture')
    project = SimpleNamespace(id='open-source-fixture', name='Demo', artifacts=[SimpleNamespace(
        kind=ArtifactKind.paper, filename='paper.txt', content_text=text)])
    job = SimpleNamespace(id='open-source-run', project=project, project_id=project.id,
        status=ReproductionStatus.queued, events=[], repository_url='', source_mode='', model_id='test',
        goal='复现论文核心实验', approved_at=None, execution_mode='full', paper_spec=spec or {},
        code_spec={}, dataset_spec={}, environment_spec={}, execution_command={}, result_json={},
        error_message='old failure', logs='')
    if spec:
        from app.services.paper_protocol import specification_context
        job.paper_spec={**spec,'spec_context':specification_context(text,job.goal)}
    values = iter([job, None, None])

    class Session:
        async def __aenter__(self): return self
        async def __aexit__(self, *args): pass
        async def scalar(self, *args): return next(values)
        async def execute(self, *args): pass
        async def commit(self): pass
        async def get(self, *args): return job

    async def clone(url, *args):
        assert url == 'https://github.com/author/model'
        return source, 'acquired'

    async def forbidden(*args, **kwargs):
        raise AssertionError('Preparation must not generate source, install packages, or train')

    async def uploads(*args): return []

    dataset_planner = engine._dataset_spec
    async def datasets(paper, spec, path, **kwargs):
        assert not kwargs.get('generated')
        return await dataset_planner(paper, spec, path, public_lookup=False)

    monkeypatch.setattr(engine, 'AsyncSessionLocal', Session)
    monkeypatch.setattr(engine, '_project_workspace', lambda project: tmp_path)
    monkeypatch.setattr(engine, '_copy_paper_sources', lambda *args: None)
    monkeypatch.setattr(engine, '_clone_repository', clone)
    monkeypatch.setattr(engine, '_generate_scaffold', forbidden)
    monkeypatch.setattr(engine, '_ensure_environment', forbidden)
    monkeypatch.setattr(engine, '_run_process', forbidden)
    monkeypatch.setattr(engine, '_install_uploaded_datasets', uploads)
    monkeypatch.setattr(engine, '_dataset_spec', datasets)
    monkeypatch.setattr(engine, '_environment_plan', lambda *args, **kwargs: {
        'prefix': str(tmp_path / 'env'), 'requires_approval': True, 'hardware_plan': {}})
    return job, source


def test_open_source_without_scope_reaches_approval_and_plans_tf_datasets(tmp_path, monkeypatch):
    spec = {'datasets': ['Alpha', 'Beta'], 'metrics': ['Accuracy'],
            'reproduction_scope': {'required': [{'name': 'method', 'paper_evidence': 'Not in paper'}]}}
    job, source = setup_workflow(tmp_path, monkeypatch, spec=spec)
    original = (source / 'train.py').read_bytes()
    job.result_json = {'source_decision': 'generate'}  # legacy choice must not hijack author source

    async def forbidden(*args, **kwargs):
        raise AssertionError('Valid author targets do not require method quote extraction')
    monkeypatch.setattr(engine, '_paper_spec', forbidden)
    asyncio.run(engine.prepare_reproduction(job.id))
    assert job.status == ReproductionStatus.awaiting_approval and job.source_mode == 'paper_repository'
    assert 'reproduction_scope' not in job.paper_spec and not job.error_message
    assert 'source_decision' not in job.result_json
    commands = job.execution_command['commands']
    assert [row['dataset'] for row in commands] == ['Alpha', 'Beta']
    assert [row['argv'][-2:] for row in commands] == [['--dataset', 'alpha'], ['--dataset', 'beta']]
    assert not job.execution_command['unplanned_datasets']
    assert (source / 'train.py').read_bytes() == original


def test_expanded_scope_reextracts_datasets_before_planning(tmp_path,monkeypatch):
    job,source=setup_workflow(tmp_path,monkeypatch,spec={'datasets':['Alpha'],'metrics':['Accuracy']})
    job.goal='复现全部数据集 Alpha 和 Beta'
    async def extract(paper,report,model,**kwargs):
        assert kwargs['goal']==job.goal
        return {'datasets':['Alpha','Beta'],'metrics':['Accuracy']}
    monkeypatch.setattr(engine,'_paper_spec',extract)
    asyncio.run(engine.prepare_reproduction(job.id))
    assert job.status==ReproductionStatus.awaiting_approval
    assert [row['dataset'] for row in job.execution_command['commands']]==['Alpha','Beta']
    assert job.paper_spec['acceptance_policy']['mode']=='practical'


def test_author_source_is_available_before_simple_metadata_extraction(tmp_path, monkeypatch):
    job, source = setup_workflow(tmp_path, monkeypatch)

    async def extract(paper, report, model, **kwargs):
        assert job.repository_url == 'https://github.com/author/model'
        assert job.code_spec['source_path'] == str(source)
        assert kwargs['require_scope'] is False
        assert 'Dataset name' in kwargs['source_evidence']['files']['train.py']
        await kwargs['progress_callback'](1)
        return {'datasets': ['Alpha', 'Beta'], 'metrics': ['Accuracy'], '_model_usage': {'total_tokens': 5}}
    monkeypatch.setattr(engine, '_paper_spec', extract)
    asyncio.run(engine.prepare_reproduction(job.id))
    assert job.status == ReproductionStatus.awaiting_approval
    assert job.code_spec['model_usage']['total_tokens'] == 5


def test_rejected_author_plan_survives_transaction_rollback(tmp_path,monkeypatch):
    import copy
    job,source=setup_workflow(tmp_path,monkeypatch,spec={'datasets':['Alpha'],'metrics':['Accuracy']})
    job.goal='Alpha 完整训练，seed=42'
    from app.services.paper_protocol import specification_context
    job.paper_spec['spec_context']=specification_context(job.project.artifacts[0].content_text,job.goal)
    session_class=engine.AsyncSessionLocal
    saved={}
    class Session(session_class):
        async def commit(self):
            saved['command']=copy.deepcopy(job.execution_command)
        async def __aexit__(self,kind,*args):
            if kind:
                job.execution_command=copy.deepcopy(saved.get('command',{}))
    monkeypatch.setattr(engine,'AsyncSessionLocal',Session)
    asyncio.run(engine.prepare_reproduction(job.id))
    assert job.status==ReproductionStatus.failed
    assert job.execution_command['commands'][0]['dataset']=='Alpha'
    assert job.execution_command['unresolved_goal_parameters'][0]['requested']==42
    assert '种子' in job.execution_command['blocked_reason']


@pytest.mark.parametrize('cause', ['ReadTimeout', 'Missing dataset names'])
def test_metadata_failure_preserves_readable_author_source_not_fake_implementation(tmp_path, monkeypatch, cause):
    job, source = setup_workflow(tmp_path, monkeypatch)

    async def extract(*args, **kwargs):
        raise engine.PaperSpecificationFailure([{'reason': cause}], {})
    monkeypatch.setattr(engine, '_paper_spec', extract)
    asyncio.run(engine.prepare_reproduction(job.id))
    assert job.status == ReproductionStatus.needs_input
    assert job.current_stage == 'experiment_planning'
    assert job.result_json['action_required']['kind'] == 'paper_spec_failed'
    assert cause in job.error_message
    assert job.code_spec['source_path'] == str(source) and 'train.py' in job.code_spec['files']
    assert job.source_mode == 'paper_repository' and not job.execution_command


def test_missing_repository_stops_without_model_or_codegen(tmp_path, monkeypatch):
    job, source = setup_workflow(tmp_path, monkeypatch, text='No author source link.')
    async def forbidden(*args, **kwargs):
        raise AssertionError('No author source: no model/codegen requests')
    monkeypatch.setattr(engine, '_paper_spec', forbidden)
    asyncio.run(engine.prepare_reproduction(job.id))
    assert job.status == ReproductionStatus.needs_input
    assert job.result_json['action_required']['kind'] == 'source_unavailable'
    assert '暂缓' in job.result_json['action_required']['reason']


def test_repository_download_failure_does_not_trigger_codegen_or_paper_gate(tmp_path, monkeypatch):
    job, source = setup_workflow(tmp_path, monkeypatch)
    async def clone(*args): raise RuntimeError('network unavailable')
    async def forbidden(*args, **kwargs): raise AssertionError('Must not bypass failed source acquisition')
    monkeypatch.setattr(engine, '_clone_repository', clone)
    monkeypatch.setattr(engine, '_paper_spec', forbidden)
    asyncio.run(engine.prepare_reproduction(job.id))
    assert job.status == ReproductionStatus.needs_input and job.source_mode == 'paper_repository'
    assert job.repository_url == 'https://github.com/author/model'
    assert 'network unavailable' in job.result_json['action_required']['detail']
    assert 'generate_code' not in job.result_json['action_required']['options']


def test_uploaded_author_code_remains_supported_without_repository(tmp_path, monkeypatch):
    job, source = setup_workflow(tmp_path, monkeypatch, text='No link.',
                                spec={'datasets': ['Alpha'], 'metrics': ['Accuracy']})
    monkeypatch.setattr(engine, '_acquire_uploaded_code', lambda *args: source)
    async def forbidden(*args): raise AssertionError('Uploaded source must not be replaced by clone')
    monkeypatch.setattr(engine, '_clone_repository', forbidden)
    asyncio.run(engine.prepare_reproduction(job.id))
    assert job.status == ReproductionStatus.awaiting_approval and job.source_mode == 'uploaded_code'


def test_open_source_spec_ignores_generated_code_scope_but_requires_actual_targets(monkeypatch):
    calls = []
    async def complete_json(role, system, prompt, **kwargs):
        calls.append(json.loads(prompt))
        assert '不要求 reproduction_scope' in system
        return SimpleNamespace(data={'datasets': ['Alpha'], 'metrics': ['Accuracy'],
                                     'acceptance_policy': {'mode':'practical','ratio_tolerance':1},
                                     'spec_context': {'goal':'fabricated'},
                                     'reproduction_scope': {'required': [{'paper_evidence': 'wrong'}]}},
                               error='', usage={})
    monkeypatch.setattr(engine, 'get_model_router', lambda: SimpleNamespace(complete_json=complete_json))
    result = asyncio.run(engine._paper_spec('paper', {}, require_scope=False, source_evidence={'files': {'README': 'Alpha'}}))
    assert len(calls) == 1 and 'reproduction_scope' not in result
    assert 'acceptance_policy' not in result and 'spec_context' not in result
    assert result['spec_validation']['scope'] == 'author_source_workflow'


def test_old_generate_api_is_disabled_and_queues_nothing():
    class DB:
        async def scalar(self, statement): return SimpleNamespace()
        async def commit(self): raise AssertionError('Disabled feature must not write')
    tasks = BackgroundTasks()
    with pytest.raises(HTTPException) as caught:
        asyncio.run(reproductions.choose_source('p', 'r', reproductions.SourceChoice(choice='generate'), tasks, DB()))
    assert caught.value.status_code == 409 and '暂缓' in caught.value.detail and not tasks.tasks
