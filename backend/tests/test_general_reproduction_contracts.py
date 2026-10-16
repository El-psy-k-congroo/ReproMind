"""Failure injection and held-out structural contracts; not paper reproduction evidence."""
import asyncio
import hashlib
import json
import sys
from types import SimpleNamespace

import httpx
import pytest

from app.services import download_activity as download
from app.services.host_environment import check_release, wheel_support
from app.services.hyperparameter_contract import trial_command, command_config
from app.services.tuning import build_tuning_plans, command_fingerprint
from app.services.task_context import rank_entries
from app.services.project_coding import inventory
from app.services.author_configuration import config_commands
from app.api.copilot import _recorded_results_answer
from app.services.runtime_compatibility import terminal_exception, compatibility_plan
from app.services.reproduction_engine import _missing_module


def test_host_tags_respect_os_runtime_abi_and_gpu_native_wheels():
    assert wheel_support('numpy-1.26.4-cp311-cp311-win_amd64.whl','3.11','Windows','AMD64')
    assert not wheel_support('numpy-1.26.4-cp311-cp311-manylinux_2_17_x86_64.whl','3.11','Windows','AMD64')
    assert not wheel_support('numpy-1.26.4-cp310-cp310-win_amd64.whl','3.11','Windows','AMD64')
    row = check_release('bitsandbytes==0.40.0',{'urls':[{'filename':'bitsandbytes-0.40.0-py3-none-any.whl'}]},'3.11','Windows','AMD64')
    assert not row['compatible']


def test_large_repository_keeps_named_task_and_its_provider_in_coding_inventory(tmp_path):
    unrelated = tmp_path/'vendor'; unrelated.mkdir()
    for index in range(301): (unrelated/f'example{index:03}.py').write_text('pass\n')
    task = tmp_path/'z_task'; task.mkdir()
    (task/'special_transformer.py').write_text('from data_loader import load\nload()\n')
    (task/'data_loader.py').write_text('def load(): return 1\n')
    bodies, manifest = inventory(tmp_path,'只修改 special_transformer 的数据加载',['z_task/special_transformer.py'])
    assert {'z_task/special_transformer.py','z_task/data_loader.py'} <= set(bodies)
    assert len(manifest) == 250


def test_task_entry_selection_discovers_nonstandard_author_entry(tmp_path):
    (tmp_path/'main.py').write_text('print("another task")\n')
    (tmp_path/'special_transformer.py').write_text('loss.backward()\n')
    entries = rank_entries(tmp_path,{'entrypoints':['main.py']},'special-transformer 的训练')
    assert entries[0] == 'special_transformer.py'


@pytest.mark.parametrize('suffix,body', [('yaml','training:\n  lr: 0.01\n'),('json','{"training":{"lr":0.01}}'),('toml','[training]\nlr = 0.01 # author\n')])
def test_configuration_trials_do_not_mutate_author_file(tmp_path,suffix,body):
    config = tmp_path/f'config.{suffix}'; config.write_text(body)
    argv = ['python','train.py','--config',str(config)]
    binding = command_config(tmp_path,{'argv':argv,'cwd':str(tmp_path)})
    command = trial_command(argv,{'config:training.lr':0.005},binding,tmp_path/'trial')
    assert config.read_text() == body
    assert command[3] != str(config) and '0.005' in open(command[3]).read()


def test_author_toml_domains_override_generic_scaling_and_change_fingerprint(tmp_path):
    source = tmp_path/'source'; source.mkdir()
    (source/'train.py').write_text('import sys\nprint(sys.argv[1])\n')
    tuned = source/'output'/'adult'/'train'/'tuned'; tuned.mkdir(parents=True)
    config = tuned/'0.toml'; config.write_text('[data]\npath="data/adult"\n[training]\nlr=0.01\n')
    tuning = tuned.parent/'tuning'; tuning.mkdir()
    (tuning/'0.toml').write_text('program="bin/train.py"\n[base_config.data]\npath="data/adult"\n[optimization.space.training]\nlr=["loguniform",0.00001,0.001]\n')
    command={'argv':['python','train.py',str(config)],'cwd':str(source),'dataset':'Adult'}
    job=SimpleNamespace(workspace_path=str(tmp_path),code_spec={},execution_command={'pythonpath':str(source),'entrypoint_candidates':['train.py'],'commands':[command]})
    plan=build_tuning_plans(job)[0]
    assert plan['space'][0]['values'] == [1e-5, 0.0001, 0.001]
    assert 'optimization.space' in plan['space'][0]['reason']
    old=command_fingerprint(command,source); config.write_text(config.read_text().replace('0.01','0.02'))
    assert command_fingerprint(command,source)!=old


def test_categorical_config_trials_and_unsupported_sampler_are_explicit(tmp_path):
    from app.services.hyperparameter_contract import configuration_domains
    config=tmp_path/'output'/'adult'/'train'/'tuned'/'0.toml';config.parent.mkdir(parents=True)
    config.write_text('[data]\npath="data/adult"\n[model]\nactivation="ReGLU" # keep comment\n')
    tuning=config.parent.parent/'tuning';tuning.mkdir()
    (tuning/'0.toml').write_text('program="train.py"\n[base_config.data]\npath="data/adult"\n[optimization.space.model]\nactivation=["categorical",["ReGLU","GEGLU"]]\nd_token=["$custom",32,128]\n')
    argv=['python','train.py',str(config)];binding=command_config(tmp_path,{'argv':argv});warnings=[]
    domains=configuration_domains(tmp_path,binding,warnings)
    assert domains['config:model.activation']['values']==['ReGLU','GEGLU'] and '$custom' in warnings[0]
    command=trial_command(argv,{'config:model.activation':'GEGLU'},binding,tmp_path/'trial')
    assert 'activation="GEGLU" # keep comment' in open(command[-1]).read()
    assert 'ReGLU' in config.read_text()


def test_existing_config_completes_documented_template_without_inventing_parameters(tmp_path):
    (tmp_path/'main.py').write_text('parser.add_argument("--config",required=True)\n')
    (tmp_path/'configs').mkdir(); (tmp_path/'configs'/'tiny.yml').write_text('data:\n  dataset: Tiny\n')
    rows=config_commands(tmp_path,'main.py',['python','main.py'],['Tiny'],[{'args':['--config','{DATASET}.yml','--doc','{MODEL_NAME}','--exp','{PROJECT_PATH}']}],tmp_path/'run-1')
    assert rows[0]['dataset']=='Tiny' and '{' not in ' '.join(rows[0]['argv'])
    selected=rows[0]['argv'][rows[0]['argv'].index('--config')+1]
    assert open(selected).read() == 'data:\n  dataset: Tiny\n'


def test_data_json_is_not_a_tuning_configuration(tmp_path):
    (tmp_path/'data.json').write_text('{"lr": 0.5}')
    assert command_config(tmp_path,{'argv':['python','train.py','--data','data.json']}) is None
    assert command_config(tmp_path,{'argv':['python','train.py','--data=data.json']}) is None
    assert command_config(tmp_path,{'argv':['python','train.py','--config=data.json']})['data']['lr']==.5


def test_slow_source_switch_requires_an_alternative_and_meaningful_wait():
    assert not download.slow_mirror(10,10000,1_000_000,True)
    assert not download.slow_mirror(30,10000,1_000_000,False)
    assert not download.slow_mirror(30,10_000_000,1_000_000,True)
    assert download.slow_mirror(30,10000,1_000_000,True)


def test_archive_mirror_search_uses_exact_filename_and_pinned_public_revision(monkeypatch):
    from app.services import torchvision_downloads as mirrors
    queries=[]
    def handler(request):
        if request.url.path=='/api/datasets':
            queries.append(request.url.params['search'])
            return httpx.Response(200,json=[{'id':'any/author-archive'}] if 'python' in request.url.params['search'] else [])
        return httpx.Response(200,json={'sha':'abc123','siblings':[{'rfilename':'nested/cifar-100-python.tar.gz'},{'rfilename':'cifar-100-python.zip'}]})
    original=httpx.AsyncClient
    monkeypatch.setattr(mirrors.httpx,'AsyncClient',lambda **kwargs:original(transport=httpx.MockTransport(handler),**kwargs))
    result=asyncio.run(mirrors.public_archive_mirrors('CIFAR100','cifar-100-python.tar.gz'))
    assert 'cifar-100-python' in queries
    assert result==['https://huggingface.co/datasets/any/author-archive/resolve/abc123/nested/cifar-100-python.tar.gz']


def test_retry_allows_only_source_revisions_in_the_approved_snapshot():
    from app.api.reproductions import _source_changed_after_approval
    job=SimpleNamespace(approved_at='2026-01-01T10:00:00',result_json={'code_revisions':[{'applied_at':'2026-01-01T09:59:59Z'}]})
    assert not _source_changed_after_approval(job)
    job.result_json['code_revisions'][0]['applied_at']='2026-01-01T10:00:01Z'
    assert _source_changed_after_approval(job)


def test_unique_short_dataset_name_preserves_author_distributed_command(tmp_path,monkeypatch):
    from app.services import reproduction_engine as engine
    from app.services.experiment_scope import resolve_scope
    scope=resolve_scope({'commands':[{'dataset':'E2E NLG Challenge'},{'dataset':'WebNLG'}]},'LoRA GPT2 E2E CUDA')
    assert scope['selected_datasets']==['E2E NLG Challenge']
    ambiguous=resolve_scope({'commands':[{'dataset':'Shared Dataset A'},{'dataset':'Shared Dataset B'}]},'Shared')
    assert not ambiguous['scope_explicit']
    task=tmp_path/'task';(task/'src').mkdir(parents=True)
    (task/'src'/'train.py').write_text('import argparse\np=argparse.ArgumentParser()\np.add_argument("--train_data",required=True)\np.add_argument("--valid_data",required=True)\n')
    (task/'README.md').write_text('python -m torch.distributed.launch --nproc_per_node=1 src/train.py --train_data data/e2e/train.jsonl --valid_data data/e2e/valid.jsonl\n')
    (task/'requirement.txt').write_text('torch==1.7.1\n')
    manifest=engine._repository_manifest(tmp_path)
    assert 'task/requirement.txt' in manifest['dependency_files']
    command=engine._execution_command(tmp_path,manifest,'full',{'prefix':str(tmp_path/'env')},{'datasets':['E2E NLG Challenge']},{'dataset_mentions':['E2E NLG Challenge']})
    assert not command['blocked_reason']
    assert command['argv'][1:5]==['-m','torch.distributed.launch','--nproc_per_node=1','src/train.py']
    assert command['commands'][0]['dataset']=='E2E NLG Challenge'


def test_task_pip_index_directives_keep_the_author_requirements_file(tmp_path,monkeypatch):
    from app.services import reproduction_engine as engine
    (tmp_path/'train.py').write_text('import torch\n')
    (tmp_path/'requirement.txt').write_text('--find-links https://download.pytorch.org/whl/torch_stable.html\ntorch==1.7.1+cu101\ntransformers==3.3.1\n')
    monkeypatch.setattr(engine,'_conda_executable',lambda:'conda')
    monkeypatch.setattr(engine,'_named_environment_prefix',lambda *_:tmp_path/'env')
    monkeypatch.setattr(engine,'_hardware_plan',lambda *_:{'requirement':'optional','available':False})
    spec=engine._environment_plan(tmp_path,engine._repository_manifest(tmp_path),tmp_path,'model')
    assert not spec['excluded_task_requirements']
    assert spec['install_commands'][0][-2:]==['-r',str(tmp_path/'requirement.txt')]


def test_documented_repo_environment_and_new_output_do_not_reuse_author_results(tmp_path):
    from app.services.author_runtime import documented_environment
    (tmp_path/'train.py').write_text('import os\nfrom util import load_config\nroot=os.environ["PROJECT_DIR"]\nload_config()\n')
    (tmp_path/'util.py').write_text('import argparse\ndef load_config():\n p=argparse.ArgumentParser()\n p.add_argument("-o", "--output", metavar="DIR")\n')
    (tmp_path/'README.md').write_text('export PROJECT_DIR=<ABSOLUTE path to the repository root>\nexport PATH=$(pwd)\nexport SECRET_TOKEN=$(cat secret)\n')
    (tmp_path/'config-adult.yml').write_text('data:\n dataset: Adult\n')
    env=documented_environment(tmp_path,['train.py'])
    assert set(env['variables'])=={'PROJECT_DIR'} and env['variables']['PROJECT_DIR'].endswith(str(tmp_path))
    run=tmp_path/'runs'/'12345678-uuid'
    rows=config_commands(tmp_path,'train.py',['python','train.py'],['Adult'],[],run)
    assert rows[0]['argv'][-2:]==['--output',str(tmp_path/'outputs'/'12345678'/'adult')]


def test_tuning_copy_cannot_overwrite_baseline_checkpoint(tmp_path):
    from app.services.tuning_isolation import isolate_trial
    source=tmp_path/'author';source.mkdir();(source/'checkpoint.pt').write_bytes(b'author baseline')
    (source/'train.py').write_text('print("train")')
    argv,cwd,env,record=isolate_trial(['python','train.py','--output','old-output'],source,{'PYTHONPATH':str(source),'PROJECT_DIR':str(source)},tmp_path/'trial'/'source')
    (cwd/'checkpoint.pt').write_bytes(b'trial checkpoint')
    assert (source/'checkpoint.pt').read_bytes()==b'author baseline'
    assert env['PROJECT_DIR']==str(cwd) and argv[-1]!= 'old-output'
    assert record['file_count']==2


def test_validation_score_follows_restored_epoch_and_rejects_test_only():
    from app.services.tuning import validation_score
    log='Epoch:0001 train loss:1.3 acc:40.0 | val loss:1.5 acc:76.2\nEpoch:0002 train loss:1.1 acc:50.0 | val loss:1.6 acc:75.0\nLoad 1st epoch\nTest acc.:99.9\n'
    assert validation_score(log,'accuracy')==.762
    assert validation_score('Test accuracy=0.99','accuracy') is None


def test_tar_archive_uses_native_paths_for_nested_dataset(tmp_path):
    import tarfile,io
    from app.services.reproduction_engine import _extract_dataset_archive
    archive=tmp_path/'data.tar.gz'
    with tarfile.open(archive,'w:gz') as writer:
        data=b'original dataset';member=tarfile.TarInfo('data/nested/train/file.txt');member.size=len(data)
        writer.addfile(member,io.BytesIO(data))
    _extract_dataset_archive(archive,tmp_path/'extracted')
    assert (tmp_path/'extracted/data/nested/train/file.txt').read_bytes()==data


def test_result_and_device_compound_request_keeps_both_answers():
    answer=_recorded_results_answer({'job':{'run_id':'r','dataset_results':[{'dataset':'D','metrics':{'accuracy':.8}}]},
        'gpu_readiness':{'device_log_evidence':[{'kind':'training_device_gpu','log':'run.log','line':'model_device=cuda:0'}]}},'总结最终指标并说明GPU设备')
    assert 'accuracy=0.8' in answer and 'model_device=cuda:0' in answer


def test_optional_missing_import_does_not_hide_terminal_dependency_api_error():
    output = ('ModuleNotFoundError: No module named \'tensorflow\'\n'
        'During handling of the above exception, another exception occurred:\n'
        '  File "site-packages/tensorboard_logger/tf_protobuf/summary_pb2.py", line 36\n'
        'TypeError: Descriptors cannot be created directly.\n'
        'Downgrade the protobuf package to 3.20.x or lower.\n')
    assert terminal_exception(output)[0]=='TypeError' and not _missing_module(output)
    assert compatibility_plan(output)['package']=='protobuf'
    assert compatibility_plan(output)['import_probe']=='tensorboard_logger.tf_protobuf.summary_pb2'
    assert _missing_module('ImportError: numpy failed\nModuleNotFoundError: No module named \'actual\'')=='actual'


def test_resumable_download_validates_range_validator_and_checksum(tmp_path,monkeypatch):
    data=b'x'*65536+b'y'*65536; requests=[]
    class Broken(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield data[:65536]
            raise httpx.ReadError('injected connection lost')
    def handler(request):
        requests.append(dict(request.headers))
        if len(requests)==1:return httpx.Response(200,headers={'etag':'"v1"','content-length':str(len(data))},stream=Broken())
        assert request.headers['range']=='bytes=65536-' and request.headers['if-range']=='"v1"'
        return httpx.Response(206,headers={'etag':'"v1"','content-range':f'bytes 65536-{len(data)-1}/{len(data)}'},content=data[65536:])
    original=httpx.AsyncClient
    monkeypatch.setattr(download.httpx,'AsyncClient',lambda **kwargs:original(transport=httpx.MockTransport(handler),**kwargs))
    job=SimpleNamespace(id='r',result_json={},logs=''); target=tmp_path/'dataset.zip'
    asyncio.run(download.download_file(None,job,tmp_path,'AnyData',['https://example.org/data'],target,lambda _:None,resume=True,checksum='sha256:'+hashlib.sha256(data).hexdigest()))
    assert target.read_bytes()==data and job.result_json['last_operation']['resumed_bytes']==65536


def test_source_switch_preserves_old_partial_until_verified_replacement(tmp_path,monkeypatch):
    data=b'author-original-data'*65536; visited=[]
    class Slow(httpx.AsyncByteStream):
        async def __aiter__(self):
            await asyncio.sleep(1.05)
            yield data[:65536]
    def handler(request):
        visited.append(request.url.host)
        if request.url.host=='slow.example':
            return httpx.Response(200,headers={'etag':'"author-v1"','content-length':str(len(data))},stream=Slow())
        archived=list((tmp_path/'.repro-download-partials').glob('*.part'))
        assert len(archived)==1 and archived[0].stat().st_size==65536
        assert 'range' not in request.headers  # A different mirror's bytes are never concatenated.
        return httpx.Response(200,content=data)
    original=httpx.AsyncClient
    monkeypatch.setattr(download.httpx,'AsyncClient',lambda **kwargs:original(transport=httpx.MockTransport(handler),**kwargs))
    monkeypatch.setattr(download,'slow_mirror',lambda elapsed,received,remaining,alternative:alternative)
    job=SimpleNamespace(id='r',result_json={},logs='');target=tmp_path/'dataset.tar.gz'
    asyncio.run(download.download_file(None,job,tmp_path,'AnyData',['https://slow.example/file','https://mirror.example/file'],target,lambda _:None,resume=True,checksum='sha256:'+hashlib.sha256(data).hexdigest()))
    assert visited==['slow.example','mirror.example'] and target.read_bytes()==data
    assert not list((tmp_path/'.repro-download-partials').glob('*.part'))


@pytest.mark.parametrize('kind',['html','checksum'])
def test_bad_content_never_published(tmp_path,monkeypatch,kind):
    original=httpx.AsyncClient
    monkeypatch.setattr(download.httpx,'AsyncClient',lambda **kwargs:original(transport=httpx.MockTransport(lambda _:httpx.Response(200,content=b'<html>login</html>',headers={'content-type':'text/html' if kind=='html' else 'application/octet-stream'})),**kwargs))
    job=SimpleNamespace(id='r',result_json={},logs=''); target=tmp_path/'dataset.zip'
    with pytest.raises(RuntimeError):asyncio.run(download.download_file(None,job,tmp_path,'Data',['https://example.org/data'],target,lambda _:None,resume=True,checksum='md5:'+'0'*32 if kind=='checksum' else ''))
    assert not target.exists()


def test_repeated_execution_preserves_previous_log(tmp_path,monkeypatch):
    from app.services import reproduction_engine as engine
    monkeypatch.setattr(engine,'get_settings',lambda:SimpleNamespace(paper_projects_dir=tmp_path,hf_endpoint='',reproduction_use_system_proxy=True,reproduction_log_chars=1000))
    project=tmp_path/'project';project.mkdir()
    log=project/'runs'/'job'/'telemetry'/'original.log';log.parent.mkdir(parents=True);log.write_text('previous failure\n')
    code,output=asyncio.run(engine._run_process([sys.executable,'-c','print("new attempt")'],project,10,output_path=log))
    assert code==0 and 'new attempt' in output
    archived=list((log.parent/'history').glob('*.log'))
    assert len(archived)==1 and archived[0].read_text()=='previous failure\n'
