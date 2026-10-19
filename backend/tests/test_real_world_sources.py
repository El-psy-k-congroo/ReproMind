import io
import os
import zipfile
from pathlib import Path

import pytest

from app.services import reproduction_engine as engine
from app.api import copilot


def test_pdf_repository_urls_survive_scheme_and_hyphen_line_breaks():
    text='Code is available at https:\n//github.com/chennnM/GCNII.\nBaseline https://github.com/DropEdge/DropEdge'
    assert engine.extract_repository_urls(text)[0]=='https://github.com/chennnM/GCNII'
    assert engine.extract_repository_urls('We provide code at https://github.com/lukasruff/Deep-\nSVDD.')==['https://github.com/lukasruff/Deep-SVDD']


def test_explicit_author_code_sentence_precedes_baseline_repository():
    text='We compare to https://github.com/baseline/model.\nOur code is available at https://github.com/author/ours.'
    assert engine.extract_repository_urls(text)[0]=='https://github.com/author/ours'


def test_documented_execute_entry_is_found_without_python_command(tmp_path):
    (tmp_path/'README.md').write_text('Finally, `execute.py` puts the above together and executes a full training run on Cora.',encoding='utf-8')
    (tmp_path/'execute.py').write_text('import torch\nmodel.train()\nloss.backward()\n',encoding='utf-8')
    assert engine._repository_manifest(tmp_path)['entrypoints'][0]=='execute.py'


@pytest.mark.skipif(os.name!='nt',reason='Windows MAX_PATH regression')
def test_author_archive_with_long_template_and_result_paths_is_complete(tmp_path):
    root=tmp_path/('workspace-'+'x'*70)/'sources'/('repository-'+'y'*50)
    relative='Repo-main/examples/NLU/templates/adding_a_new_example_script/{{cookiecutter.directory_name}}/run_{{cookiecutter.example_shortcut}}.py'
    data=io.BytesIO()
    with zipfile.ZipFile(data,'w') as archive:
        archive.writestr(relative,'# public template\n')
        archive.writestr('Repo-main/train.py','print("author")\n')
    with zipfile.ZipFile(data) as archive:
        engine._extract_source_archive(archive,root)
    path=str((root/relative).resolve())
    assert Path('\\\\?\\'+path).read_text(encoding='utf-8')=='# public template\n'
    assert 'train.py' in engine._repository_manifest(root/'Repo-main')['entrypoints']
    from app.services.gpu_preparation import source_bodies
    assert 'train.py' in source_bodies(root/'Repo-main')


def test_explanation_with_source_references_is_not_a_full_file_dump():
    question='请依据本项目论文和作者源码说明核心方法、损失函数、真实训练入口和数据来源。请引用具体文件；无法确认的内容明确说未知。'
    assert not copilot._source_read_requested(question)
    assert copilot._source_read_requested('请给我看看核心模型的代码')


def test_click_numeric_options_and_hyphenated_argparse_flags():
    from app.services.tuning import numeric_flags, GROUPS
    assert numeric_flags("import click\n@click.option('--lr', type=float, default=0.0001)\ndef main(lr): pass") == {'lr':0.0001}
    assert 'batch-size' in GROUPS['batch_size']
    assert 'decay' in GROUPS['weight_decay']


def test_readme_positional_command_in_documented_subdirectory(tmp_path):
    source=tmp_path/'source';(source/'src').mkdir(parents=True)
    (source/'src/main.py').write_text('import click\n',encoding='utf-8')
    (source/'README.md').write_text('cd src\npython main.py mnist mnist_LeNet ../log/test ../data --lr 0.0001;\n',encoding='utf-8')
    manifest=engine._repository_manifest(source)
    assert manifest['readme_commands'][0]['entrypoint']=='src/main.py'
    execution=engine._execution_command(source,manifest,'full',{'prefix':str(tmp_path/'env')},{'datasets':['MNIST']},{'available_datasets':['MNIST'],'dataset_names':{'mnist':'mnist'}},tmp_path/'run')
    assert execution['commands'][0]['argv'][1:]==['main.py','mnist','mnist_LeNet','../log/test','../data','--lr','0.0001']


def test_cuda_tensor_metric_is_parsed_without_guessing_unlabelled_values():
    assert engine._extract_metrics("Average accuracy: tensor([0.8214], device='cuda:0')\ntensor(82.1360, device='cuda:0')") == {'accuracy':0.8214}


def test_root_training_scope_uses_shell_dataset_parameters(tmp_path):
    (tmp_path/'train.py').write_text("import argparse\np=argparse.ArgumentParser()\np.add_argument('--data', default='cora')\n",encoding='utf-8')
    (tmp_path/'PyG').mkdir();(tmp_path/'PyG/arxiv.py').write_text('print(1)')
    (tmp_path/'README.md').write_text('Usage: sh semi.sh\nReference implementation:\npython arxiv.py\n')
    (tmp_path/'semi.sh').write_text('python -u train.py --data cora --layer 64\npython -u train.py --data citeseer --layer 32\n')
    manifest=engine._repository_manifest(tmp_path)
    assert manifest['entrypoints'][0]=='train.py'
    execution=engine._execution_command(tmp_path,manifest,'full',{'prefix':str(tmp_path/'env')},{'datasets':['Cora','Citeseer']},{'available_datasets':['Cora','Citeseer']},tmp_path/'run')
    assert execution['commands'][1]['argv'][-2:]==['--layer','32']


def test_user_scope_matches_cifar_spelling_without_running_other_datasets():
    from app.services.experiment_scope import resolve_scope
    scope=resolve_scope({'commands':[{'dataset':'CIFAR10'},{'dataset':'CIFAR100'},{'dataset':'ImageNet'}]}, '本次仅运行 CIFAR-10')
    assert scope['selected_datasets']==['CIFAR10'] and scope['scope_explicit']


def test_selected_pytorch_entry_does_not_inherit_unrelated_tensorflow_runtime(tmp_path,monkeypatch):
    (tmp_path/'train.py').write_text('import torch\nfrom helper import forward\n')
    (tmp_path/'helper.py').write_text('def forward(): pass\n')
    (tmp_path/'tabnet.py').write_text('import tensorflow as tf\nx=tf.placeholder()\n')
    monkeypatch.setattr(engine,'_conda_executable',lambda:'conda')
    monkeypatch.setattr(engine,'_named_environment_prefix',lambda *_:tmp_path/'env')
    plan=engine._environment_plan(tmp_path,{'entrypoints':['train.py'],'dependency_files':[]},tmp_path/'paper-12345678','model')
    assert plan['python']=='3.11' and 'tensorflow' not in plan['inferred_dependencies']


def test_root_package_metadata_precedes_unrelated_example_requirements(tmp_path,monkeypatch):
    (tmp_path/'scripts/setfit').mkdir(parents=True);(tmp_path/'scripts/tfew').mkdir()
    (tmp_path/'scripts/setfit/run.py').write_text('import torch\n')
    (tmp_path/'scripts/tfew/requirements.txt').write_text('deepspeed==0.5.10\n')
    (tmp_path/'setup.py').write_text('from setuptools import setup\nsetup(name="example")\n')
    monkeypatch.setattr(engine,'_conda_executable',lambda:'conda')
    monkeypatch.setattr(engine,'_named_environment_prefix',lambda *_:tmp_path/'env')
    plan=engine._environment_plan(tmp_path,{'entrypoints':['scripts/setfit/run.py'],
        'dependency_files':['setup.py','scripts/tfew/requirements.txt']},tmp_path/'paper-12345678','model')
    assert plan['dependency_source']=='setup.py'
    assert plan['ignored_dependency_files']==['scripts/tfew/requirements.txt']


def test_explicit_hub_loader_maps_only_its_dataset_alias(tmp_path):
    from app.services.author_data_loaders import declared_hub_loaders
    path=tmp_path/'train.py'
    path.write_text('''def load(dataset_name):
    if dataset_name == 'alpaca':
        return load_dataset('tatsu-lab/alpaca')
    elif dataset_name == 'other':
        return load_dataset('author/other')
''')
    found=declared_hub_loaders([path],tmp_path)
    assert found['alpaca']['repository']=='tatsu-lab/alpaca'
    assert found['other']['repository']=='author/other'


def test_author_hub_namespace_uses_literal_alias_and_rejects_arbitrary_format(tmp_path):
    from app.services.author_data_loaders import declared_hub_loaders
    path=tmp_path/'train.py'
    path.write_text("aliases=['sst2']\ndef load_data_splits(dataset):\n return load_dataset(f'SetFit/{dataset}', split='train')\ndef other(url):\n return load_dataset(f'https://{url}')\n")
    found=declared_hub_loaders([path],tmp_path)
    assert found['sst2']['repository']=='SetFit/sst2'
    assert all(not row['repository'].startswith('https:') for row in found.values())


def test_selected_entry_follows_src_package_import(tmp_path):
    from app.services.source_selection import entry_sources
    (tmp_path/'src/pkg').mkdir(parents=True)
    (tmp_path/'train.py').write_text('from pkg.utils import load_data\n')
    (tmp_path/'src/pkg/utils.py').write_text('def load_data(): pass\n')
    assert any(p.name=='utils.py' for p in entry_sources(tmp_path,['train.py']))


def test_author_declared_sys_path_and_optional_imports_are_not_pip_packages(tmp_path):
    (tmp_path/'scripts/model').mkdir(parents=True)
    (tmp_path/'scripts/helper.py').write_text('import torch\n')
    (tmp_path/'scripts/model/run.py').write_text("import os,sys\nfrom typing import TYPE_CHECKING\nsys.path.insert(0,os.path.join(os.path.dirname(__file__),'..'))\nfrom helper import value\nif TYPE_CHECKING:\n import spacy\ntry:\n import optuna\nexcept ImportError:\n pass\ndef optional():\n import cuda_extension\n")
    assert engine._infer_python_dependencies(tmp_path,['scripts/model/run.py'])==['torch']


def test_training_child_inherits_operator_proxy_policy(tmp_path,monkeypatch):
    import asyncio,sys
    from app.config import Settings
    settings=Settings(_env_file=None,paper_projects_dir=tmp_path)
    monkeypatch.setattr(engine,'get_settings',lambda:settings)
    monkeypatch.setenv('HTTPS_PROXY','http://proxy.example:8080')
    code,output=asyncio.run(engine._run_process([sys.executable,'-c','import os; print(bool(os.environ.get("HTTPS_PROXY")))'],tmp_path,20))
    assert code==0 and output.strip()=='True'


def test_bare_percent_accuracy_and_final_acc_dot_are_normalized():
    assert engine._extract_metrics('Epoch:0001 train acc:99.0 | val acc:82.0\nTest acc.:82.2')=={'accuracy':pytest.approx(0.822)}
    assert engine._extract_metrics('accuracy: 822.0')=={}
    assert 0.822 < 0.855-0.03


def test_output_path_and_cuda_dependency_failures_are_not_missing_data():
    failure="os.makedirs(output_path, exist_ok=True)\nFileNotFoundError: [WinError 3] path: 'C:/"+'long/'*60+"results'"
    assert engine._diagnose_failure(failure,dataset='SST2')['type']=='filesystem_path_error'
    failure='bitsandbytes CUDA Setup failed: loading libbitsandbytes_cpu.so; WindowsPath is not iterable'
    result=engine._diagnose_failure(failure)
    assert result['type']=='dependency_platform_incompatible' and '源码补丁' in result['suggestion']


def test_gpu_request_and_cpu_fallback_repair_are_identified():
    assert engine._goal_requires_gpu('本次仅运行 Cora，使用 GPU')
    assert not engine._goal_requires_gpu('本次不使用 GPU')
    assert engine._cpu_device_assignment('cudaid="cuda:0" if torch.cuda.is_available() else "cpu"')
    assert not engine._cpu_device_assignment('labels=labels.detach().cpu().numpy()')


def test_gpu_guard_uses_selected_nonstandard_entry_instead_of_other_examples(tmp_path):
    from app.api.experiment_branches import _gpu_preview_issues
    (tmp_path/'scripts').mkdir()
    (tmp_path/'scripts/run_fewshot.py').write_text("import torch\nassert torch.cuda.is_available()\nmodel.to('cuda')\n")
    (tmp_path/'train.py').write_text('import tensorflow as tf\n')
    assert _gpu_preview_issues(tmp_path,{},['scripts/run_fewshot.py'])==[]


def test_cuda_recovery_preserves_newer_installed_torch_api():
    from app.services.environment_recovery import pytorch_cuda_plan
    plan=pytorch_cuda_plan({'version':'2.14.1+cpu','python_version':[3,11,14],'platform':'win32','machine':'AMD64','checked':True,'cuda_available':False}, {'torch':'2.14.1+cpu'})
    assert plan['supported'] and plan['candidates'][0]['packages']==['torch==2.14.1+cu126']


def test_author_required_config_options_block_incomplete_execution(tmp_path):
    (tmp_path/'main.py').write_text("import argparse\np=argparse.ArgumentParser()\np.add_argument('--config', required=True)\np.add_argument('-d', '--doc', required=True)\n")
    command=engine._execution_command(tmp_path,{'entrypoints':['main.py']},'full',{'prefix':str(tmp_path/'env')},{'datasets':['CIFAR10']},{'available_datasets':['CIFAR10']})
    assert '--config' in command['blocked_reason'] and '--doc' in command['blocked_reason']
    from app.services.cli_contract import missing_required_options
    assert missing_required_options((tmp_path/'main.py').read_text(),['python','main.py','--config=x.yml','-d','qa'])==[]


@pytest.mark.parametrize('goal', [
    '当前服务器无 GPU，请根据本机情况选择兼容的 CPU 环境',
    '没有可用 GPU，使用 CPU', '无需 CUDA，按 CPU 运行',
    'No GPU is available; run on CPU', 'run without CUDA',
    'GPU 不可用，请用 CPU',
])
def test_host_gpu_absence_does_not_request_gpu(goal):
    assert not engine._goal_requires_gpu(goal)


@pytest.mark.parametrize('goal', ['使用 GPU', '要求 CUDA 运行',
    '本机没有 GPU；请在远程 GPU 上运行', 'Use GPU; do not fall back to CPU'])
def test_positive_gpu_request_is_preserved(goal):
    assert engine._goal_requires_gpu(goal)
