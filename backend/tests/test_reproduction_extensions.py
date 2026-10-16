import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import BackgroundTasks

from app.services import ablations
from app.services import reproduction_engine as engine
from app.api import projects
from app.api.reproductions import SourceChoice, choose_source
from app.config import Settings
from app.core.model_router import ModelRole, ModelRouter
from app.models import ArtifactKind, ReproductionStatus
from app.services.reproduction_engine import _extract_metrics, _primary_metric, _target_for_dataset


def test_abstract_only_experiment_spec_requires_full_paper_enrichment():
    assert engine._vague_paper_spec({'datasets': ['UCI机器学习库中的多个数据集'],
                                   'metrics': ['具体指标未在摘要中明确']})
    assert not engine._vague_paper_spec({'datasets': ['Default', 'Shoppers'], 'metrics': ['F1', 'bAcc']})


def test_disabled_generation_keeps_old_paper_plan_and_drafts(tmp_path: Path, monkeypatch):
    workspace = tmp_path / 'demo'
    draft = workspace / 'generated' / 'demo' / 'run-1-draft'
    draft.mkdir(parents=True)
    original = 'def transform(x):\n    return x\n'
    (draft / 'method.py').write_text(original, encoding='utf-8')
    accepted = workspace / 'generated' / 'demo' / 'run-1'
    accepted.mkdir()
    (accepted / 'method.py').write_text('def old_version():\n    return 0\n', encoding='utf-8')
    plan = {'files': [{'path': 'method.py'}, {'path': 'train.py'}], 'entrypoint': 'train.py'}
    project = SimpleNamespace(id='project-1', artifacts=[SimpleNamespace(
        kind=ArtifactKind.paper, filename='paper.txt', content_text='method evidence')])
    job = SimpleNamespace(id='run-1', project_id='project-1', project=project,
                          status=ReproductionStatus.queued, events=[], repository_url='', model_id='test',
                          dataset_spec={'missing_datasets': ['obsolete']}, environment_spec={'ready': False},
                          execution_command={'display': 'obsolete command'},
                          result_json={'source_decision': 'generate'}, paper_spec={'model_name': 'demo',
                              'datasets': ['demo'], 'metrics': ['F1'],
                              'reproduction_scope': {'required': [{'name': 'demo', 'paper_evidence': 'method evidence'}]}},
                          code_spec={'draft_source': str(draft), 'source_path': str(accepted), 'coding_plan': plan,
                                     'model_usage': {'total_tokens': 10}})
    values = iter([job, None, None])

    class Session:
        async def __aenter__(self): return self
        async def __aexit__(self, *_args): pass
        async def scalar(self, *_args): return next(values)
        async def get(self, *_args): return job
        async def execute(self, *_args): pass
        async def commit(self): pass

    async def must_not_extract(*_args, **_kwargs):
        raise AssertionError('Retry must not discard current paper/code state')

    async def retry(_paper, _spec, _model, **kwargs):
        assert kwargs['initial_files'] == {'method.py': original}
        assert kwargs['previous_plan'] == plan
        await kwargs['checkpoint']({'method.py': original}, [], plan, {'total_tokens': 5})
        raise engine.CodeGenerationFailure([{'attempt': 1, 'reason': 'temporary API failure'}],
                                           {'total_tokens': 5}, {'method.py': original})

    monkeypatch.setattr(engine, 'AsyncSessionLocal', Session)
    monkeypatch.setattr(engine, '_project_workspace', lambda _project: workspace)
    monkeypatch.setattr(engine, '_model_name', lambda *_args: 'demo')
    monkeypatch.setattr(engine, '_copy_paper_sources', lambda *_args: None)
    monkeypatch.setattr(engine, '_paper_spec', must_not_extract)
    monkeypatch.setattr(engine, '_generate_scaffold', retry)
    asyncio.run(engine.prepare_reproduction(job.id))
    assert job.status == ReproductionStatus.needs_input
    assert job.result_json['action_required']['kind'] == 'source_unavailable'
    assert job.code_spec['coding_plan'] == plan
    assert job.code_spec['model_usage']['total_tokens'] == 10
    assert job.dataset_spec == job.environment_spec == job.execution_command == {}
    assert (draft / 'method.py').read_text(encoding='utf-8') == original


def test_repair_scope_accepts_isolated_revision_and_blocks_shared_source(tmp_path: Path):
    workspace = tmp_path / "paper"
    isolated = workspace / "branches" / "run-1" / "source"
    other = workspace / "runs" / "run-1" / "variants" / "drugcombdb"
    isolated.mkdir(parents=True)
    other.mkdir(parents=True)
    source = isolated / "run.py"
    source.write_text("raise ValueError()\n", encoding="utf-8")
    job = SimpleNamespace(execution_command={"commands": [
        {"dataset": "Oncology-Screen", "cwd": str(isolated), "pythonpath": str(isolated)},
        {"dataset": "DrugCombDB", "cwd": str(other), "pythonpath": str(other)},
    ]})
    assert engine._repair_source_scope(job, workspace, isolated, source) == ""
    job.execution_command["commands"][1]["cwd"] = str(isolated)
    job.execution_command["commands"][1]["pythonpath"] = str(isolated)
    assert "多个数据集" in engine._repair_source_scope(job, workspace, isolated, source)
    job.execution_command["commands"][1]["cwd"] = str(other)
    job.execution_command["commands"][1]["pythonpath"] = str(isolated)
    assert "其他数据集" in engine._repair_source_scope(job, workspace, isolated, source)


def test_repair_agent_retries_branch_source_only_after_failure(tmp_path: Path, monkeypatch):
    workspace = tmp_path / "paper"
    source_dir = workspace / "branches" / "run-1" / "source"
    source_dir.mkdir(parents=True)
    source = source_dir / "run.py"
    source.write_text("raise RuntimeError('old')\n", encoding="utf-8")
    job = SimpleNamespace(id="run-1", project_id="project-1", repository_url="repo",
                          execution_command={"commands": [{"dataset": "Oncology-Screen",
                                                            "cwd": str(source_dir)}]},
                          paper_spec={}, code_spec={}, model_id="test-model")

    class Model:
        async def complete_json(self, *args, **kwargs):
            return SimpleNamespace(data={"old_text": "raise RuntimeError('old')",
                                         "new_text": "print('fixed')",
                                         "rationale": "修复本次异常"},
                                   usage={}, model="test-model")

    async def no_history(*args):
        return []

    async def successful_retry(*args, **kwargs):
        return 0, "accuracy: 0.9"

    monkeypatch.setattr(engine, "get_model_router", lambda: Model())
    monkeypatch.setattr(engine, "_prior_repair_experience", no_history)
    monkeypatch.setattr(engine, "_run_process", successful_retry)
    traceback = f'File "{source}", line 1, in <module>\nRuntimeError: old'
    code, output, record = asyncio.run(engine._agent_repair_code(
        job, workspace, "Oncology-Screen", traceback, [sys.executable, "run.py"],
        source_dir, {}, 1,
    ))
    assert code == 0 and output == "accuracy: 0.9"
    assert record["applied"] is True
    assert source.read_text(encoding="utf-8") == "print('fixed')\n"
    source.write_text("raise RuntimeError('old')\n", encoding="utf-8")

    async def paused_retry(*args, **kwargs):
        raise engine.ReproductionPaused("paused")

    monkeypatch.setattr(engine, "_run_process", paused_retry)
    with pytest.raises(engine.ReproductionPaused):
        asyncio.run(engine._agent_repair_code(job, workspace, "Oncology-Screen", traceback,
                                               [sys.executable, "run.py"], source_dir, {}, 2))
    assert source.read_text(encoding="utf-8") == "raise RuntimeError('old')\n"


def test_known_cuda_numpy_failure_is_repaired_without_model_guessing(tmp_path: Path, monkeypatch):
    workspace = tmp_path / "paper"
    source_dir = workspace / "branches" / "run-1" / "source"
    source_dir.mkdir(parents=True)
    source = source_dir / "run.py"
    source.write_text("import torch\n\ndef eval_classification(labels, logits):\n"
                      "    auc = roc_auc_score(y_true=labels, y_score=logits)\n",
                      encoding="utf-8")
    job = SimpleNamespace(id="run-1", project_id="project-1", repository_url="repo",
                          execution_command={"commands": [{"dataset": "Oncology-Screen",
                                                            "cwd": str(source_dir)}]},
                          paper_spec={}, code_spec={}, model_id="test-model")

    async def successful_retry(*args, **kwargs):
        return 0, "Final result: auc 0.90"

    monkeypatch.setattr(engine, "_run_process", successful_retry)
    monkeypatch.setattr(engine, "get_model_router", lambda: pytest.fail("known fix must not depend on model"))
    traceback = (f'File "{source}", line 4, in eval_classification\n'
                 "TypeError: can't convert cuda:0 device type tensor to numpy. "
                 "Use Tensor.cpu() to copy the tensor to host memory first.")
    code, _, record = asyncio.run(engine._agent_repair_code(
        job, workspace, "Oncology-Screen", traceback, [sys.executable, "run.py"],
        source_dir, {}, 1,
    ))
    assert code == 0 and record["type"] == "known_code_repair" and record["applied"]
    assert "logits = logits.detach().cpu().numpy()" in source.read_text(encoding="utf-8")


def test_execution_stops_before_next_dataset_when_repair_fails(tmp_path: Path, monkeypatch):
    workspace = tmp_path / "paper"
    source = workspace / "sources" / "author"
    source.mkdir(parents=True)
    (source / "run.py").write_text("print('run')\n", encoding="utf-8")
    job = SimpleNamespace(
        id="stop-after-failure", project_id="project-1", status=ReproductionStatus.executing,
        current_stage="executing", progress=90, message="", events=[], logs="", result_json={},
        workspace_path=str(workspace), environment_spec={}, dataset_spec={}, paper_spec={}, code_spec={},
        execution_mode="full", model_id="test", repository_url="repo", completed_at=None,
        execution_command={"commands": [
            {"dataset": "first", "argv": [sys.executable, "run.py"], "cwd": str(source)},
            {"dataset": "second", "argv": [sys.executable, "run.py"], "cwd": str(source)},
        ]},
    )
    calls = []

    class Session:
        async def __aenter__(self):
            return self
        async def __aexit__(self, *args):
            return None
        async def get(self, model, key):
            return job
        async def commit(self):
            return None
        async def execute(self, statement):
            return None

    async def advance(session, row, stage, progress, message, *, status=None):
        row.current_stage, row.progress, row.message = stage, progress, message
        row.events.append({"stage": stage, "message": message})
        if status is not None:
            row.status = status

    async def run_process(argv, cwd, timeout, env, *, output_path=None):
        calls.append(str(output_path))
        return 1, 'File "' + str(source / "run.py") + '", line 1\nRuntimeError: test failure'

    async def repair(*args, **kwargs):
        return 1, "RuntimeError: test failure", {"dataset": "first", "applied": False,
                                                 "reason": "补丁未通过校验"}

    async def ensure_env(*args):
        return Path(sys.executable)

    async def empty_async(*args):
        return None

    async def no_uploads(*args):
        return []

    monkeypatch.setattr(engine, "AsyncSessionLocal", Session)
    monkeypatch.setattr(engine, "_advance", advance)
    monkeypatch.setattr(engine, "_ensure_environment", ensure_env)
    monkeypatch.setattr(engine, "_install_uploaded_datasets", no_uploads)
    monkeypatch.setattr(engine, "_download_small_datasets", empty_async)
    monkeypatch.setattr(engine, "_align_dataset_commands", lambda row: [])
    monkeypatch.setattr(engine, "_preflight_source", lambda *args, **kwargs: {"issues": [], "created_output_dirs": []})
    monkeypatch.setattr(engine, "_run_process", run_process)
    monkeypatch.setattr(engine, "_agent_repair_code", repair)
    monkeypatch.setattr(engine, "_write_reproduction_report", lambda *args: None)
    asyncio.run(engine.execute_reproduction(job.id))
    assert len(calls) == 1
    assert job.status == ReproductionStatus.failed
    assert job.result_json["unattempted_datasets"] == ["second"]
    assert len(job.result_json["dataset_results"]) == 1


def test_partial_conda_environment_is_owned_without_ready_marker(tmp_path: Path, monkeypatch):
    name = "demo-12345678"
    prefix = tmp_path / name
    meta = prefix / "conda-meta"
    meta.mkdir(parents=True)
    (meta / "history").write_text(
        f"==> 2026-10-06 17:21:32 <==\n# cmd: C:\\conda\\Scripts\\conda-script.py "
        f"create --yes --name {name} python=3.11 pip\n", encoding="utf-8",
    )
    monkeypatch.setattr(engine, "_named_environment_prefix", lambda conda, env_name: tmp_path / env_name)
    job = SimpleNamespace(
        id="job-1", project_id="12345678-0000-0000-0000-000000000000",
        environment_spec={
            "prefix": str(prefix), "conda_executable": "C:/conda/Scripts/conda.exe",
            "environment_name": name,
            "create_commands": [["C:/conda/Scripts/conda.exe", "create", "--yes", "--name", name, "python=3.11"]],
        },
        events=[{"stage": "environment_building"}],
    )
    assert engine._environment_owned_by_job(job, prefix)
    (meta / "history").write_text("# cmd: conda create --name another-env python=3.11\n", encoding="utf-8")
    assert not engine._environment_owned_by_job(job, prefix)
    (prefix / ".repromind-owner.json").write_text(json.dumps({
        "project_id": job.project_id, "job_id": job.id, "environment_name": name,
    }), encoding="utf-8")
    assert engine._environment_owned_by_job(job, prefix)
    job.id = "job-2"
    assert engine._environment_owned_by_job(job, prefix)
    (prefix / ".repromind-owner.json").write_text(json.dumps({
        "project_id": "wrong-project", "job_id": "job-1", "environment_name": name,
    }), encoding="utf-8")
    (prefix / ".repromind-ready").touch()
    assert not engine._environment_owned_by_job(job, prefix)


def test_delete_project_removes_interrupted_conda_environment(tmp_path: Path, monkeypatch):
    project_id = "12345678-0000-0000-0000-000000000000"
    name = "demo-12345678"
    prefix = tmp_path / "envs" / name
    meta = prefix / "conda-meta"
    meta.mkdir(parents=True)
    (meta / "history").write_text(
        f"# cmd: conda create --yes --name {name} python=3.11 pip\n", encoding="utf-8",
    )
    job = SimpleNamespace(
        id="job-1", project_id=project_id, status=ReproductionStatus.failed, result_json={},
        workspace_path="", environment_spec={
            "prefix": str(prefix), "conda_executable": "C:/conda/Scripts/conda.exe",
            "environment_name": name,
            "create_commands": [["conda", "create", "--name", name, "python=3.11"]],
        },
        events=[{"stage": "environment_building"}],
    )
    monkeypatch.setattr(engine, "_named_environment_prefix", lambda conda, env_name: tmp_path / "envs" / env_name)
    monkeypatch.setattr(projects, "get_settings", lambda: SimpleNamespace(
        paper_projects_dir=tmp_path / "paper-projects", upload_dir=tmp_path / "uploads",
    ))
    monkeypatch.setattr(projects.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(
        returncode=0, stderr="",
    ))

    async def fake_project(db, requested_id):
        assert requested_id == project_id
        return SimpleNamespace(id=project_id)

    class FakeSession:
        calls = 0
        deleted = False
        committed = False

        async def scalars(self, statement):
            self.calls += 1
            return SimpleNamespace(all=lambda: [job] if self.calls == 1 else [])

        async def delete(self, project):
            self.deleted = True

        async def commit(self):
            self.committed = True

    monkeypatch.setattr(projects, "_project_or_404", fake_project)
    session = FakeSession()
    asyncio.run(projects.delete_project(project_id, session))
    assert not prefix.exists()
    assert session.deleted and session.committed


def test_non_accuracy_paper_metric_is_selected_from_observed_results():
    metrics = _extract_metrics("Epoch 2 AUC: 0.881 AUPR: 0.704 RMSE: 0.35")
    assert metrics == {"auc": 0.881, "aupr": 0.704, "rmse": 0.35}
    assert _primary_metric({"metrics": ["AUPR", "AUC"]}, metrics) == "aupr"
    assert _target_for_dataset({"target_results": {"DrugBank": {"AUPR": 70.0}}}, "drugbank", "aupr") == 0.7


def test_execution_repair_creates_only_missing_model_output_folder(tmp_path: Path):
    output = "File 'run.py', line 8, in train\n torch.save(model, 'model/checkpoint.pkl')\n" \
             "FileNotFoundError: [Errno 2] No such file or directory: 'model/checkpoint.pkl'"
    created = engine._repair_missing_output_directory(tmp_path, output)
    assert created == str(tmp_path / "model")
    assert (tmp_path / "model").is_dir()
    assert engine._repair_missing_output_directory(tmp_path, output) == ""
    assert engine._repair_missing_output_directory(tmp_path, output.replace("model/checkpoint.pkl", "../outside.pkl")) == ""
    assert engine._diagnose_failure(output, tmp_path, "example")["type"] == "missing_output_directory"


def test_generated_code_requires_real_entrypoint_and_valid_syntax():
    assert engine._validate_generated_code({"files": [], "entrypoint": "train.py"})
    assert engine._validate_generated_code({
        "files": [{"path": "train.py", "content": "def train(:\n    pass"}], "entrypoint": "train.py",
    }).startswith("train.py 第")
    assert engine._validate_generated_code({
        "files": [{"path": "train.py", "content": "def train():\n    return 1\n"}], "entrypoint": "train.py",
    }) == ""
    assert "占位" in engine._validate_generated_code({
        "files": [{"path": "train.py", "content": "def train():\n    return 1\n"}],
        "entrypoint": "train.py", "limitations": ["This is a simplified proxy; the LITO method is not implemented."],
    })
    mapped = engine._normalize_generated_code({
        "files": {"train.py": "def train():\n    return 1\n"}, "limitations": "需要真实数据",
    })
    assert mapped["entrypoint"] == "train.py"
    assert mapped["limitations"] == ["需要真实数据"]
    assert engine._validate_generated_code(mapped) == ""


def test_no_repository_code_agent_retries_with_extended_non_thinking_output(monkeypatch):
    calls = []
    plan = {"files": [{"path": "method.py"}, {"path": "train.py"}], "entrypoint": "train.py"}

    async def complete_json(*_args, **_kwargs):
        return SimpleNamespace(data=plan if not calls else {"issues": [], "limitations": ["需要真实数据"]},
                               usage={}, error="")

    async def complete_text(_role, _system, prompt, **kwargs):
        request = json.loads(prompt)
        calls.append({"request": request, **kwargs})
        target = request["target_file"]["path"]
        if target == "method.py":
            text = "def transform(:\n    return 1" if len(calls) == 1 else "def transform(x):\n    return x\n"
        else:
            text = ("from method import transform\n" if request["validation_feedback"] else
                    "from method import missing\n")
        return SimpleNamespace(text=text, usage={"completion_tokens": 500}, error="", finish_reason="stop")

    monkeypatch.setattr(engine, "get_model_router", lambda: SimpleNamespace(
        complete_json=complete_json, complete_text=complete_text))
    result = asyncio.run(engine._generate_scaffold("论文内容", {"model_name": "Example"}, "test-code"))
    assert result["entrypoint"] == "train.py"
    assert result["limitations"] == ["需要真实数据"]
    assert result["_model_usage"]["completion_tokens"] == 2000
    assert len(calls) == 4
    assert "第 1 行" in calls[1]["request"]["validation_feedback"]
    assert "不存在的接口" in calls[3]["request"]["validation_feedback"]
    assert "method.py" in calls[2]["request"]["existing_files"]
    assert result["validation"]["local_imports"] is True


def test_no_repository_code_agent_reports_each_failed_attempt(monkeypatch):
    async def complete_json(*_args, **_kwargs):
        return SimpleNamespace(data={"files": [{"path": "train.py"}], "entrypoint": "train.py"},
                               usage={}, error="")

    async def complete_text(*_args, **_kwargs):
        return SimpleNamespace(text="def train(:\n    pass", usage={"completion_tokens": 100}, error="")

    monkeypatch.setattr(engine, "get_model_router", lambda: SimpleNamespace(
        complete_json=complete_json, complete_text=complete_text))
    with pytest.raises(engine.CodeGenerationFailure) as error:
        asyncio.run(engine._generate_scaffold("论文内容", {"model_name": "Example"}, "test-code"))
    assert len(error.value.attempts) == 3
    assert all("train.py 第 1 行" in attempt["reason"] for attempt in error.value.attempts)
    assert error.value.usage["completion_tokens"] == 300
    assert "train.py" in error.value.drafts


def test_failed_generated_project_can_retry_generation_without_running_experiment():
    job = SimpleNamespace(id="run-1", project_id="project-1", status=ReproductionStatus.failed,
                          source_mode="generated_implementation", code_spec={}, approved_at=None,
                          result_json={}, current_stage="failed", progress=36, message="复现准备失败",
                          error_message="旧的生成失败")

    class Session:
        def __init__(self):
            self.calls = 0

        async def scalar(self, _statement):
            self.calls += 1
            return job if self.calls == 1 else None

        async def commit(self):
            return None

        async def refresh(self, _job):
            return None

    tasks = BackgroundTasks()
    job.approved_at = None
    result = asyncio.run(choose_source("project-1", "run-1", SourceChoice(choice="repository", repository_url="https://github.com/author/model"),
                                       tasks, Session()))
    assert result.status == ReproductionStatus.queued
    assert result.result_json["source_decision"] == "repository"
    assert result.repository_url == 'https://github.com/author/model'
    assert result.error_message == ""
    assert len(tasks.tasks) == 1  # Only preparation is queued; no execution approval.


def test_extended_code_model_call_gets_longer_read_timeout(monkeypatch):
    settings = Settings(_env_file=None, model_provider="openai_compatible", LLM_API_KEY="dummy",
                        llm_model_coder="test-code", llm_timeout_seconds=60,
                        llm_max_tokens_per_call=1600)
    router = ModelRouter(settings)
    seen = []

    async def fake_request(*_args, **kwargs):
        seen.append(kwargs)
        return {"ok": True}, {}

    monkeypatch.setattr(router, "_request_json", fake_request)
    asyncio.run(router.complete_json(ModelRole.code, "system", "user", max_tokens=2200,
                                     extended_output=True, non_thinking=True))
    assert seen[0]["timeout_seconds"] == 120
    assert seen[0]["max_tokens"] == 2200
    asyncio.run(router.complete_json(ModelRole.code, "system", "user", max_tokens=2200))
    assert seen[1]["timeout_seconds"] == 60
    assert seen[1]["max_tokens"] == 1600


def test_environment_agent_retries_transient_pip_failure_once(tmp_path: Path, monkeypatch):
    name = "demo-12345678"
    prefix = tmp_path / "envs" / name
    conda = tmp_path / "conda.exe"
    conda.touch()
    python = engine._environment_python(prefix)
    calls = []

    async def fake_run(command, cwd, timeout, **kwargs):
        calls.append(command)
        if "create" in command:
            (prefix / "conda-meta").mkdir(parents=True)
            python.touch()
            return 0, "created"
        return (1, "ReadTimeout while fetching wheel") if len(calls) == 2 else (0, "installed")

    class Session:
        async def execute(self, statement):
            pass

        async def commit(self):
            pass

        async def refresh(self, *_args, **_kwargs):
            pass

    job = SimpleNamespace(
        id="job-1", project_id="12345678-0000-0000-0000-000000000000",
        environment_spec={
            "prefix": str(prefix), "conda_executable": str(conda), "environment_name": name,
            "create_commands": [[str(conda), "create", "--name", name]],
            "install_commands": [[str(python), "-m", "pip", "install", "numpy"]],
        },
        result_json={}, events=[], logs="",
    )
    monkeypatch.setattr(engine, "_named_environment_prefix", lambda c, n: tmp_path / "envs" / n)
    monkeypatch.setattr(engine, "_run_process", fake_run)
    monkeypatch.setattr(engine, "get_settings", lambda: SimpleNamespace(
        reproduction_timeout_seconds=2, reproduction_log_chars=5000,
    ))
    assert asyncio.run(engine._ensure_environment(Session(), job, tmp_path)) == python
    assert len(calls) == 3
    assert (prefix / ".repromind-owner.json").is_file()
    assert (prefix / ".repromind-ready").is_file()


def test_ablation_discovery_requires_author_component_switch(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(ablations, "get_settings", lambda: SimpleNamespace(paper_projects_dir=tmp_path))
    workspace = tmp_path / "example-12345678"
    source = workspace / "sources"
    source.mkdir(parents=True)
    (source / "train.py").write_text(
        "parser.add_argument('--no_attention', action='store_true')\n"
        "parser.add_argument('--no_cuda', action='store_true')\n"
        "parser.add_argument('--dropout', type=float, default=0.5)\n", encoding="utf-8",
    )
    job = SimpleNamespace(
        workspace_path=str(workspace), project_id="12345678-0000-0000-0000-000000000000",
        execution_command={"commands": [{"dataset": "demo", "cwd": str(source),
                                          "argv": ["python", "train.py"]}]},
    )
    assert ablations.discover_ablations(job) == [
        {"dataset": "demo", "flag": "--no_attention", "label": "no-attention"}
    ]


def test_pausing_stops_only_the_registered_job_process(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(engine, "get_settings", lambda: SimpleNamespace(
        paper_projects_dir=tmp_path, hf_endpoint="https://hf-mirror.com",
    ))

    async def scenario():
        job_id = "pause-test-job"
        token = engine.CURRENT_JOB_ID.set(job_id)
        try:
            running = asyncio.create_task(engine._run_process(
                [sys.executable, "-c", "import time; time.sleep(8)"], tmp_path, 20,
            ))
            await asyncio.sleep(0.5)
            assert await asyncio.to_thread(engine.request_process_pause, job_id)
            with pytest.raises(engine.ReproductionPaused):
                await running
            assert not engine.has_active_process(job_id)
        finally:
            engine.clear_process_pause(job_id)
            engine.CURRENT_JOB_ID.reset(token)

    asyncio.run(scenario())
