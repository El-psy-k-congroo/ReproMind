from pathlib import Path
from types import SimpleNamespace
import asyncio

from app.api.reproductions import _verified_report_ready
from app.services.telemetry import parse_experiment_log, read_job_telemetry
from app.services.reproduction_engine import _preflight_source, _repair_preflight_syntax, _repair_missing_output_directory, _diagnose_failure, _extract_metrics, _hardware_plan
from app.api.copilot import (
    _code_action_answer, _code_action_requested, _identity_answer, _identity_question, _matching_excerpts,
    _message_text, _model_failure_message, _model_unavailable_answer, _query_terms,
    _gpu_capability_answer, _gpu_capability_question, _rerun_action_plan, _rerun_answer, _rerun_requested,
    _source_read_requested, _source_read_answer, _experiment_control_requested,
    _experiment_control_plan, _experiment_control_answer, _status_question, _status_answer,
    _restart_project_requested,
    _generation_diagnosis_question, _generation_diagnosis_answer,
)
import httpx


def test_dialogue_experiment_control_requires_explicit_instruction_and_confirmation():
    assert _experiment_control_requested("可以暂停当前实验吗？") == ""
    assert _experiment_control_requested("为什么实验暂停了") == ""
    assert _experiment_control_requested("请暂停当前实验") == "pause"
    assert _experiment_control_requested("恢复实验") == "resume"
    active = {"job": {"run_id": "run-1", "status": "executing"}}
    plan = _experiment_control_plan(active, "pause")
    assert plan["kind"] == "pause_confirm"
    assert "尚未执行暂停" in _experiment_control_answer(plan)
    assert _experiment_control_plan(active, "resume")["kind"] == "unavailable"
    assert _experiment_control_plan({"job": {"run_id": "run-1", "status": "paused"}}, "resume")["kind"] == "resume_confirm"


def test_status_questions_are_read_from_fold_log_without_model(tmp_path: Path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    (source / "run.py").write_text("for cv in range(1):\n    test_fold = 0\n"
                                   "    for i in range(5):\n        print(test_fold)\n",
                                   encoding="utf-8")
    telemetry = tmp_path / "runs" / "run-1" / "telemetry"
    telemetry.mkdir(parents=True)
    (telemetry / "drugcombdb-original.log").write_text(
        "[test_fold 2,epoch 17],avg_loss=0.135\n", encoding="utf-8")
    job = SimpleNamespace(id="run-1", status=SimpleNamespace(value="paused"), message="用户已暂停",
                          result_json={"dataset_results": [{"dataset": "Oncology-Screen", "verdict": "failed"}],
                                       "repair_attempts": [{"dataset": "Oncology-Screen", "applied": False,
                                                            "reason": "模型未给出可唯一定位的最小修改"}]},
                          execution_command={"commands": [{"dataset": "DrugCombDB", "cwd": str(source),
                                                              "argv": ["python", "run.py"]}]})

    class Session:
        async def scalar(self, statement):
            return job

    monkeypatch.setattr("app.api.copilot._safe_workspace", lambda value: tmp_path)
    assert _status_question("当前数据集跑到第几折了")
    answer = asyncio.run(_status_answer(Session(), SimpleNamespace(id="project-1"), "run-1",
                                        "当前数据集跑到第几折了，纠错了吗"))
    assert "第 3/5 折" in answer
    assert "epoch 17" in answer
    assert "未应用" in answer
    assert not _restart_project_requested("能不能重新启动这个项目？")
    assert _restart_project_requested("请重新启动这个项目")


def test_preparation_failure_status_does_not_invent_training_progress(monkeypatch):
    monkeypatch.setattr("app.api.copilot._safe_workspace", lambda _job: None)
    job = SimpleNamespace(
        id="run-1", status=SimpleNamespace(value="failed"), current_stage="failed", progress=36,
        message="复现准备失败", error_message="Code Agent 没有生成源码文件",
        source_mode="generated_implementation", code_spec={}, result_json={}, execution_command={},
        events=[{"stage": "source_acquisition"}, {"stage": "failed"}],
    )

    class Session:
        async def scalar(self, _statement):
            return job

    assert _status_question("现在到哪一步了")
    answer = asyncio.run(_status_answer(Session(), SimpleNamespace(id="project-1"), "run-1", "现在到哪一步了"))
    assert "source_acquisition" in answer
    assert "进度 36%" in answer
    assert "环境构建和实验执行都未开始" in answer
    assert "最近运行的数据集" not in answer


def test_report_is_hidden_until_actual_result_verification():
    job = SimpleNamespace(
        current_stage="awaiting_data",
        events=[{"stage": "data_planning"}],
        result_json={"report_path": "old-report.md", "finished_at": "2026-10-06"},
    )
    assert not _verified_report_ready(job)
    job.events.append({"stage": "result_verification"})
    assert _verified_report_ready(job)
    job.result_json.pop("finished_at")
    assert not _verified_report_ready(job)


def test_telemetry_is_parsed_only_from_the_jobs_own_run_directory(tmp_path: Path):
    directory = tmp_path / "runs" / "run-a" / "telemetry"
    directory.mkdir(parents=True)
    log = directory / "cora-original.log"
    log.write_text(
        "Epoch 1 train_loss: 0.91 val_acc: 0.42\nEpoch 2 train_loss: 0.73 val_acc: 0.51\n",
        encoding="utf-8",
    )
    parsed = parse_experiment_log(log)
    assert parsed["dataset"] == "cora"
    assert parsed["points"][1]["val_acc"] == 0.51
    assert len(read_job_telemetry(tmp_path, "run-a")["series"]) == 1
    assert read_job_telemetry(tmp_path, "run-b")["series"] == []


def test_author_kgansynergy_epoch_loss_is_visible_without_claiming_test_result(tmp_path: Path):
    log = tmp_path / "oncology-screen-agent-repair-1.log"
    log.write_text("[test_fold 0,epoch 0],avg_loss=0.8993\n[test_fold 0,epoch 1],avg_loss=0.7410\n", encoding="utf-8")
    points = parse_experiment_log(log)["points"]
    assert points == [{"epoch": 0, "fold": 0, "train_loss": 0.8993},
                      {"epoch": 1, "fold": 0, "train_loss": 0.741}]
    assert _extract_metrics(log.read_text(encoding="utf-8")) == {}
    assert _extract_metrics("Final result: auc 0.8521 | aupr 0.7731 | accuracy 0.7190") == {
        "auc": 0.8521, "aupr": 0.7731, "accuracy": 0.719,
    }


def test_multiple_epoch_metrics_keep_training_and_validation_scopes(tmp_path: Path):
    log = tmp_path / "cora-original.log"
    log.write_text("Epoch 1 Train Loss: 0.85 Validation Accuracy: 0.72 Val AUC: 0.81\n"
                   "Epoch 2 train_loss=0.61 val_acc=0.78 val_auc=0.84\n", encoding="utf-8")
    points = parse_experiment_log(log)["points"]
    assert points[0] == {"epoch": 1, "train_loss": 0.85, "val_acc": 0.72, "val_auc": 0.81}
    assert points[1] == {"epoch": 2, "train_loss": 0.61, "val_acc": 0.78, "val_auc": 0.84}


def test_preflight_scans_source_and_creates_only_local_output_directories(tmp_path: Path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "run.py").write_text("import torch\ntorch.save({}, 'model/{}_decoder.pkl'.format(1))\n", encoding="utf-8")
    (source / "broken.py").write_text("def broken(:\n", encoding="utf-8")
    dry = _preflight_source(source)
    assert dry["scanned_python_files"] == 2
    assert any(item["kind"] == "syntax_error" for item in dry["issues"])
    assert not (source / "model").exists()
    actual = _preflight_source(source, create_output_dirs=True)
    assert actual["created_output_dirs"] == ["model"]
    assert (source / "model").is_dir()


def test_torch_runtime_parent_directory_repair_is_project_local(tmp_path: Path):
    output = "File run.py, line 130, in train\ntorch.save(net.state_dict(), 'model/file.pkl')\nRuntimeError: Parent directory model does not exist."
    assert _repair_missing_output_directory(tmp_path, output) == str(tmp_path / "model")
    assert _diagnose_failure("命令超过 900 秒，已终止。")["type"] == "training_timeout"


def test_copilot_retrieves_paper_code_terms_from_question():
    terms = _query_terms("论文中的hidden_dim设为了多少")
    assert "e_dim" in terms
    assert _matching_excerpts("other\n--e_dim default=64\nfollowing", terms) == ["other\n--e_dim default=64\nfollowing"]
    assert "超时" in _model_failure_message(httpx.ReadTimeout("timeout"))
    assert "def forward" in _query_terms("它和当前代码实现一致吗？")


def test_gpu_capability_question_does_not_confuse_host_gpu_with_project_environment():
    assert _gpu_capability_question("现在这个代码以及环境可以跑gpu吗？？")
    assert _gpu_capability_question("机器有3090 Ti，是不是训练已经用上GPU了？")
    assert not _gpu_capability_question("请用gpu跑一下DrugCombDB数据集")
    answer = _gpu_capability_answer({
        "job": {"run_id": "abc12345", "status": "paused", "dataset_results_count": 1},
        "host_gpu": {"available": True, "name": "RTX 3090 Ti"},
        "gpu_readiness": {
            "gpu_code_ready": False,
            "gpu_code_issues": ["训练入口没有将模型迁移到 CUDA"],
            "torch_runtime": {"checked": True, "torch": "2.14.1+cpu",
                              "cuda_build": None, "cuda_available": False},
        },
    })
    assert "不能认定" in answer
    assert "2.14.1+cpu" in answer
    assert "已有 1 组最终数据集结果" in answer
    assert "这是错误推断" in answer


def test_gpu_word_alone_does_not_claim_optional_gpu_support(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("app.services.reproduction_engine.shutil.which", lambda name: None)
    (tmp_path / "run.py").write_text("import torch\n# This model might use a GPU someday.\n", encoding="utf-8")
    assert _hardware_plan(tmp_path)["requirement"] == "not_detected"
    (tmp_path / "run.py").write_text("import torch\nuse_cuda = torch.cuda.is_available()\n", encoding="utf-8")
    assert _hardware_plan(tmp_path)["requirement"] == "optional"


def test_copilot_action_request_does_not_claim_code_or_experiment_changed():
    question = "请将复现代码移到 GPU 上，并重新运行 DrugCombDB 实验"
    assert _code_action_requested(question)
    assert not _code_action_requested("为什么要将复现代码移到 GPU 上？")
    paused = _code_action_answer({"job": {"status": "paused"}})
    assert "变更预览" in paused
    assert "尚未修改代码或启动实验" in paused


def test_questions_about_changes_never_start_a_code_preview():
    questions = [
        "现在的代码可以迁移到gpu上吗",
        "现在代码可以迁移到 GPU 上",
        "能不能把训练脚本改为GPU版本？",
        "请问修复模型代码会影响原实验吗？",
        "如果修改model.py需要重新运行吗？",
        "请用GPU重新跑DrugCombDB可以吗？",
    ]
    assert all(not _code_action_requested(item) for item in questions)
    assert all(not _rerun_requested(item) for item in questions)
    assert _code_action_requested("请修改 model.py 中的设备迁移逻辑")
    assert _code_action_requested("把训练代码迁移到 GPU 上")
    assert _code_action_requested("请将训练代码改成可以使用 GPU 的版本")
    assert _code_action_requested("把 hidden_dim 改为 128")
    assert _source_read_requested("你能不能把模型文件怎么写的读出来给我看看")


def test_source_reader_returns_actual_project_file_without_model(tmp_path: Path, monkeypatch):
    source = tmp_path / "sources" / "repo"
    source.mkdir(parents=True)
    (source / "model.py").write_text("class KGANSynergy:\n    hidden_dim = 64\n", encoding="utf-8")
    (source / "run.py").write_text("print('train')\n", encoding="utf-8")
    monkeypatch.setattr("app.api.copilot._safe_workspace", lambda _job: tmp_path)

    class Session:
        async def scalar(self, _statement):
            return SimpleNamespace(execution_command={"pythonpath": str(source)}, code_spec={})

    answer = asyncio.run(_source_read_answer(Session(), SimpleNamespace(id="project-1"),
                                             "run-1", "把模型的主要实现文件读出来给我看"))
    assert "model.py" in answer
    assert "hidden_dim = 64" in answer
    assert "未运行或修改代码" in answer


def test_source_reader_finds_actual_core_implementation_in_retained_drafts(tmp_path: Path, monkeypatch):
    source = tmp_path / "generated" / "draft"
    source.mkdir(parents=True)
    (source / "language_model.py").write_text("class LanguageModel: pass\n", encoding="utf-8")
    (source / "lito.py").write_text("class LITO:\n    puncture_count = 2\n", encoding="utf-8")
    monkeypatch.setattr("app.api.copilot._safe_workspace", lambda _job: tmp_path)

    class Session:
        async def scalar(self, _statement):
            return SimpleNamespace(execution_command={}, paper_spec={"model_name": "LITO"},
                                   code_spec={"draft_source": str(source)})

    answer = asyncio.run(_source_read_answer(Session(), SimpleNamespace(id="project-1"),
                                             "run-1", "把模型的主要实现文件读出来给我看"))
    assert "puncture_count = 2" in answer
    assert "尚未通过全部检查" in answer


def test_dialogue_rerun_plan_uses_actual_command_dataset_and_requires_confirmation(monkeypatch):
    class Session:
        async def scalar(self, _statement):
            return SimpleNamespace(id="run-1", status=SimpleNamespace(value="paused"),
                                   result_json={},
                                   environment_spec={},
                                   execution_command={"commands": [{"dataset": "Oncology-Screen"},
                                                                   {"dataset": "DrugCombDB"}]})

    async def readiness(_job):
        return {"framework": "pytorch", "gpu_environment_plan": {"needed": True, "supported": True},
                "gpu_code_ready": False, "gpu_code_issues": ["model remains on CPU"],
                "torch_runtime": {"checked": True, "cuda_available": False}}

    monkeypatch.setattr("app.api.copilot.inspect_run_readiness", readiness)
    project = SimpleNamespace(id="project-1")
    question = "请将代码迁移到 GPU，并重新跑 DrugCombDB"
    assert _rerun_requested(question)
    plan = asyncio.run(_rerun_action_plan(Session(), project, "run-1", question))
    assert plan["kind"] == "gpu_revision"
    assert plan["dataset"] == "DrugCombDB"
    assert plan["gpu_requested"] is True
    assert "当前没有改代码或运行实验" in _rerun_answer(plan)
    ambiguous = asyncio.run(_rerun_action_plan(Session(), project, "run-1", "请重新运行数据集"))
    assert ambiguous["kind"] == "needs_dataset"
    assert "DrugCombDB" in _rerun_answer(ambiguous)
    all_plan = asyncio.run(_rerun_action_plan(Session(), project, "run-1", "请重新跑全部数据集"))
    assert all_plan["kind"] == "rerun_confirm"
    assert all_plan["selected_datasets"] == ["Oncology-Screen", "DrugCombDB"]
    assert _rerun_requested("请重新开始实验")
    restart_plan = asyncio.run(_rerun_action_plan(Session(), project, "run-1", "请重新开始实验"))
    assert restart_plan["kind"] == "rerun_confirm"
    assert restart_plan["selected_datasets"] == ["Oncology-Screen", "DrugCombDB"]
    assert "确认前不会启动实验" in _rerun_answer(restart_plan)
    assert not _rerun_requested("可以重新开始实验吗？")
    running = _code_action_answer({"job": {"status": "executing"}})
    assert "现在不会修改源码" in running


def test_restart_experiment_keeps_gpu_requirement_and_blocks_active_run(monkeypatch):
    class Session:
        def __init__(self, status):
            self.status = status

        async def scalar(self, _statement):
            return SimpleNamespace(id="run-1", status=SimpleNamespace(value=self.status),
                                   result_json={},
                                   environment_spec={"gpu_requested": True},
                                   execution_command={"commands": [{"dataset": "Oncology-Screen"},
                                                                   {"dataset": "DrugCombDB"}]})

    async def readiness(_job):
        return {"framework": "pytorch", "gpu_code_ready": True, "gpu_code_issues": [],
                "torch_runtime": {"checked": True, "cuda_available": True}}

    monkeypatch.setattr("app.api.copilot.inspect_run_readiness", readiness)
    project = SimpleNamespace(id="project-1")
    plan = asyncio.run(_rerun_action_plan(Session("paused"), project, "run-1", "请重新开始实验"))
    assert plan["kind"] == "rerun_confirm"
    assert plan["gpu_requested"] is True
    assert plan["selected_datasets"] == ["Oncology-Screen", "DrugCombDB"]
    active = asyncio.run(_rerun_action_plan(Session("executing"), project, "run-1", "请重新开始实验"))
    assert active["kind"] == "unavailable"
    assert "活动实验" in active["reason"]


def test_copilot_empty_model_output_has_explicit_degraded_answer():
    assert _message_text([{"type": "text", "text": "已核对"}, {"type": "output_text", "text": "日志"}]) == "已核对日志"
    answer = _model_unavailable_answer({"job": {"status": "paused", "stage": "executing"}}, RuntimeError("empty"))
    assert "未生成完整回答" in answer
    assert "paused" in answer
    assert "不能据此推断" in answer


def test_no_repository_code_failure_answer_uses_recorded_attempts():
    assert _generation_diagnosis_question("你是为什么没有生成代码？？")
    assert not _generation_diagnosis_question("请生成代码")
    answer = _generation_diagnosis_answer({"job": {
        "status": "needs_input", "stage": "source_acquisition", "source_mode": "generated_implementation",
        "error_message": "代码生成未通过校验",
        "code_generation_attempts": [
            {"attempt": 1, "reason": "JSON 输出被截断"},
            {"attempt": 2, "reason": "没有生成源码文件"},
        ],
    }})
    assert "第 1 次：JSON 输出被截断" in answer
    assert "第 2 次：没有生成源码文件" in answer
    assert "自动编码功能已暂缓" in answer
    assert "不能说已经生成代码" in answer


def test_generation_diagnosis_distinguishes_existing_drafts_from_no_source():
    answer = _generation_diagnosis_answer({"job": {
        "status": "needs_input", "stage": "source_acquisition", "source_mode": "generated_implementation",
        "error_message": "方法审查待修正", "code_generation": {"draft_files": ["model.py", "main.py"]},
    }})
    assert "model.py" in answer
    assert "已保留草稿不等于完整实现已通过验证" in answer
    assert "不能说已经生成代码" not in answer


def test_copilot_identity_question_has_immediate_truthful_answer():
    assert _identity_question("你是谁？")
    assert _identity_question("你好")
    assert not _identity_question("你是谁？论文用了哪些数据集？")
    answer = _identity_answer(SimpleNamespace(name="KGANSynergy"))
    assert "KGANSynergy" in answer
    assert "分别确认" in answer
    assert "直接执行" in answer


def test_preflight_syntax_agent_only_patches_broken_entrypoint(tmp_path: Path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    entry = source / "run.py"
    entry.write_text("def train(:\n    pass\n", encoding="utf-8")
    (source / "unused.py").write_text("def unused(:\n", encoding="utf-8")

    class Router:
        async def complete_json(self, *args, **kwargs):
            return SimpleNamespace(
                data={"old_text": "def train(:", "new_text": "def train():", "rationale": "修复语法"},
                usage={"total_tokens": 8}, model="test-model",
            )

    monkeypatch.setattr("app.services.reproduction_engine.get_model_router", lambda: Router())
    job = SimpleNamespace(id="job-1", model_id="test-model")
    records = asyncio.run(_repair_preflight_syntax(job, source, {"entrypoints": ["run.py"]}))
    assert records[0]["applied"] is True
    assert "def train():" in entry.read_text(encoding="utf-8")
    assert "def unused(:" in (source / "unused.py").read_text(encoding="utf-8")


def test_source_reader_includes_each_requested_file_and_actual_line_numbers(tmp_path, monkeypatch):
    source=tmp_path/'sources'/'repo';source.mkdir(parents=True)
    (source/'train.py').write_text("import utils\nprint('train')\n",encoding='utf-8')
    (source/'utils.py').write_text('def normalize():\n    return 1\n',encoding='utf-8')
    monkeypatch.setattr('app.api.copilot._safe_workspace',lambda _:tmp_path)
    class Session:
        async def scalar(self,_):
            return SimpleNamespace(execution_command={'pythonpath':str(source)},code_spec={})
    answer=asyncio.run(_source_read_answer(Session(),SimpleNamespace(id='project-1'),'run-1',
        '展示 train.py 和 utils.py 的源码和行号'))
    assert '`train.py`' in answer and '`utils.py`' in answer
    assert 'L2: print' in answer and 'L2:     return 1' in answer
