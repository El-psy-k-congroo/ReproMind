import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from app.api import experiment_branches as branches
from app.api.experiment_branches import _apply_text, _code_file, _copy_bounded, _remap_command
from app.services import reproduction_engine as engine
from app.services.reproduction_engine import _validate_generated_code, _write_generated_files, _repository_manifest


def test_branch_patch_requires_unique_valid_python():
    original = "value = 1\nprint(value)\n"
    assert _apply_text(original, "value = 1", "value = 2", "train.py").startswith("value = 2")
    with pytest.raises(HTTPException):
        _apply_text("x=1\nx=1\n", "x=1", "x=2", "train.py")
    with pytest.raises(HTTPException):
        _apply_text(original, "value = 1", "value = (", "train.py")


def test_branch_copy_and_command_relocation_preserve_original(tmp_path: Path):
    source = tmp_path / "sources" / "paper"
    source.mkdir(parents=True)
    (source / "train.py").write_text("print(1)\n", encoding="utf-8")
    branch = tmp_path / "branches" / "new" / "source"
    command = {"cwd": str(source), "pythonpath": str(source), "argv": ["python", "train.py"],
               "commands": [{"cwd": str(source), "argv": ["python", "train.py"], "dataset": "sample"}]}
    relocated = _remap_command(command, source, branch, tmp_path / "branches" / "new" / "data")
    _copy_bounded(source, branch)
    _code_file(branch, "train.py").write_text("print(2)\n", encoding="utf-8")
    assert (source / "train.py").read_text(encoding="utf-8") == "print(1)\n"
    assert relocated["commands"][0]["cwd"] == str(branch)
    with pytest.raises(HTTPException):
        _code_file(branch, "../source/train.py")


def test_branch_copy_preserves_author_files_with_long_windows_paths(tmp_path):
    root=engine._native_path(tmp_path/'source');path=root/('long-segment-'*5)/('nested-'*10)/('ratings-'*15+'.jsonl')
    path.parent.mkdir(parents=True);path.write_text('{"score":1}\n')
    target=tmp_path/'branches/new/source'
    _copy_bounded(tmp_path/'source',target)
    copied=engine._native_path(target)/path.relative_to(root)
    assert copied.read_text()=='{"score":1}\n'


def test_agent_selects_relevant_code_file(tmp_path: Path):
    (tmp_path / "run.py").write_text("print('start')\n", encoding="utf-8")
    (tmp_path / "model.py").write_text("hidden_dim = 64\n", encoding="utf-8")
    assert branches._select_code_file(tmp_path, "将 hidden_dim 改为 128").name == "model.py"


def test_generated_dependencies_become_environment_evidence(tmp_path: Path):
    payload = {"entrypoint": "train.py", "files": [{"path": "train.py", "content": "import torch\nprint(torch.__version__)\n"}],
               "dependencies": ["torch>=2.0", "numpy"]}
    assert _validate_generated_code(payload) == ""
    source = _write_generated_files(tmp_path, "demo", "run-1", payload)
    assert (source / "requirements.txt").read_text(encoding="utf-8").splitlines() == payload["dependencies"]
    assert "requirements.txt" in _repository_manifest(source)["dependency_files"]
    assert _validate_generated_code({**payload, "dependencies": ["--extra-index-url=https://evil.test"]})


def test_generated_torch_environment_prefers_gpu_only_when_available(tmp_path: Path, monkeypatch):
    source = _write_generated_files(tmp_path, "demo", "run-2", {
        "files": [{"path": "train.py", "content": "import torch\nprint(torch.cuda.is_available())\n"}],
        "entrypoint": "train.py", "dependencies": ["torch>=2.0", "numpy"],
    })
    monkeypatch.setattr(engine, "_conda_executable", lambda: "C:/conda/conda.exe")
    monkeypatch.setattr(engine, "_named_environment_prefix", lambda conda, name: tmp_path / "envs" / name)
    monkeypatch.setattr(engine, "_hardware_plan", lambda path, *_: {
        "requirement": "optional", "available": True, "framework": "pytorch", "backend": "cuda",
    })
    plan = engine._environment_plan(source, _repository_manifest(source), tmp_path / "project-12345678", "demo")
    assert plan["dependency_source"].startswith("requirements.txt")
    assert any("https://download.pytorch.org/whl/cu126" in command for command in plan["install_commands"])
    monkeypatch.setattr(engine, "_hardware_plan", lambda path, *_: {
        "requirement": "optional", "available": False, "framework": "pytorch", "backend": "cpu",
    })
    cpu_plan = engine._environment_plan(source, _repository_manifest(source), tmp_path / "project-12345678", "demo")
    assert all("https://download.pytorch.org/whl/cu126" not in command for command in cpu_plan["install_commands"])


def test_no_repository_mnist_is_planned_for_download_not_upload(tmp_path: Path):
    spec = asyncio.run(engine._dataset_spec("The experiment uses MNIST.", {"datasets": ["MNIST"]}, tmp_path))
    assert spec["automatic_download"] is True
    assert spec["missing_datasets"] == []
    assert spec["available_datasets"] == ["MNIST"]


def test_revision_reuses_project_owned_environment_without_conda_clone(tmp_path: Path, monkeypatch):
    conda = tmp_path / "conda.exe"
    conda.touch()
    original = tmp_path / "envs" / "demo-12345678"
    original.mkdir(parents=True)
    (original / "python.exe").touch()
    (original / ".repromind-ready").touch()
    monkeypatch.setattr(branches, "_named_environment_prefix", lambda executable, name: tmp_path / "envs" / name)
    monkeypatch.setattr(engine, "_named_environment_prefix", lambda executable, name: tmp_path / "envs" / name)
    base = SimpleNamespace(project_id="12345678-0000-0000-0000-000000000000", environment_spec={
        "conda_executable": str(conda), "environment_name": original.name, "prefix": str(original), "ready": True,
    })
    plan = branches._branch_environment(base, "abcdef12-0000-0000-0000-000000000000")
    assert plan["environment_name"] == original.name
    assert plan["prefix"] == str(original)
    assert plan["create_commands"] == []
    assert plan["recheck_on_run"] is True


def test_completed_cpu_install_can_enter_approved_gpu_recovery_without_ready_marker(tmp_path,monkeypatch):
    conda=tmp_path/'conda.exe';conda.touch()
    prefix=tmp_path/'demo-12345678';prefix.mkdir();(prefix/'python.exe').touch()
    monkeypatch.setattr(branches,'_named_environment_prefix',lambda *_:prefix)
    monkeypatch.setattr(branches,'_environment_owned_by_job',lambda *_:True)
    base=SimpleNamespace(project_id='12345678-0000',environment_spec={'conda_executable':str(conda),
        'environment_name':prefix.name,'prefix':str(prefix),'dependencies_installed':True})
    with pytest.raises(HTTPException):branches._branch_environment(base,'branch')
    plan=branches._branch_environment(base,'branch',True)
    assert plan['gpu_recovery_pending'] and plan['recheck_on_run'] and not plan['install_commands']


def test_multi_file_candidates_include_related_modules(tmp_path: Path):
    (tmp_path / "run.py").write_text("import torch\ndevice = 'cpu'\n", encoding="utf-8")
    (tmp_path / "model.py").write_text("import torch\nself.device = 'cpu'\n", encoding="utf-8")
    (tmp_path / "notes.py").write_text("print('unrelated')\n", encoding="utf-8")
    files = branches._candidate_code_files(tmp_path, "迁移到 GPU，修改 device 和模型")
    assert {item.name for item in files[:2]} == {"run.py", "model.py"}


def test_no_repository_generated_code_reaches_full_experiment_plan(tmp_path: Path, monkeypatch):
    source = _write_generated_files(tmp_path, "paper-model", "run-1", {
        "entrypoint": "train.py", "dependencies": ["torch", "torchvision"],
        "files": [{"path": "train.py", "content":
                   "import argparse\nfrom torchvision import datasets\n"
                   "parser = argparse.ArgumentParser()\nparser.add_argument('--dataset')\n"
                   "print('accuracy: 0.8')\n"}],
    })
    monkeypatch.setattr(engine, "_conda_executable", lambda: "C:/conda/conda.exe")
    monkeypatch.setattr(engine, "_named_environment_prefix", lambda conda, name: tmp_path / "envs" / name)
    monkeypatch.setattr(engine, "_hardware_plan", lambda path, *_: {
        "requirement": "optional", "available": False, "framework": "pytorch", "backend": "cpu",
    })
    manifest = _repository_manifest(source)
    workspace = tmp_path / "project-12345678"
    environment = engine._environment_plan(source, manifest, workspace, "paper-model")
    dataset = asyncio.run(engine._dataset_spec("MNIST benchmark", {"datasets": ["MNIST"]}, source))
    command = engine._execution_command(source, manifest, "full", environment,
                                        {"datasets": ["MNIST"]}, dataset, workspace / "runs" / "run-1")
    assert not dataset["missing_datasets"]
    assert not command["blocked_reason"]
    assert command["commands"][0]["dataset"] == "MNIST"
    assert "--dataset" in command["commands"][0]["argv"]
