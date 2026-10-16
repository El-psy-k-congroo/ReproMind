"""Copilot streaming regressions with synthetic evidence and no external model calls."""

import asyncio
import json
from datetime import datetime, timezone
from types import SimpleNamespace
from uuid import uuid4

import httpx
import pytest

from app.api import copilot
from app.core.model_router import ModelSelection
from app.models import CopilotMessage


class FakeSession:
    def __init__(self):
        self.messages = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        return None

    async def scalar(self, *_):
        return None

    def add(self, message):
        if isinstance(message, CopilotMessage):
            message.id = str(uuid4())
            message.created_at = datetime.now(timezone.utc)
            self.messages.append(message)

    async def commit(self):
        return None

    async def refresh(self, *_):
        return None


@pytest.mark.parametrize('first_failure', ['', 'timeout', 'invalid'])
def test_performance_question_reaches_reasoning_model_with_cross_run_facts(monkeypatch, first_failure):
    session = FakeSession()
    project = SimpleNamespace(name='跨实验性能测试')
    conversation = SimpleNamespace(title='新对话', updated_at=None)
    async def fake_project(*_): return project
    async def fake_conversation(*_): return conversation
    async def history(*_): return '', []
    async def context(*_):
        return {'job':{'run_id':'gpu-run'}, 'performance_comparison':{'records':[
            {'run_id':'gpu-run','metrics':{'accuracy':.5}},
            {'run_id':'cpu-run','metrics':{'accuracy':.8}}]}}
    async def forbidden(*_): raise AssertionError('Must not classify analysis as code modification')
    monkeypatch.setattr(copilot,'_project',fake_project)
    monkeypatch.setattr(copilot,'_conversation',fake_conversation)
    monkeypatch.setattr(copilot,'_managed_history',history)
    monkeypatch.setattr(copilot,'_context',context)
    monkeypatch.setattr(copilot,'_semantic_code_requested',forbidden)
    monkeypatch.setattr(copilot,'AsyncSessionLocal',lambda:session)
    selection = ModelSelection('test','test-model','https://example.invalid/v1','dummy','paper')
    received = []
    async def completion(role,system_prompt,user_prompt,**kwargs):
        facts = json.loads(user_prompt)['verified_performance_comparison']
        received.append(facts)
        if first_failure == 'timeout' and len(received) == 1:
            return SimpleNamespace(error='ReadTimeout: temporary network error')
        if first_failure == 'invalid' and len(received) == 1:
            return SimpleNamespace(error='', data={'hypotheses':[
                {'possible_reason':'准确率可能为0.99','verify':'检查参数','supporting_runs':['gpu-run']}]})
        return SimpleNamespace(error='',model='test-model',usage={'total_tokens':10},data={'hypotheses':[
            {'possible_reason':'参数配置可能不同','verify':'对照执行参数后只改变设备进行受控实验',
             'supporting_runs':['gpu-run','cpu-run']}]})
    monkeypatch.setattr(copilot,'get_model_router',lambda:SimpleNamespace(select=lambda _:selection,complete_json=completion))
    async def collect():
        response = await copilot.stream_answer('p','c',copilot.CopilotQuestion(question='为什么迁移到gpu上后性能差这么多'),db=session)
        return ''.join([frame async for frame in response.body_iterator])
    output = asyncio.run(collect())
    assert len(received) == (2 if first_failure else 1) and len(received[0]['records']) == 2
    assert '参数配置可能不同' in output and 'action_plan' not in output


def test_performance_model_timeout_does_not_erase_verified_comparison():
    answer = copilot._model_unavailable_answer({'performance_comparison':{'records':[{}]}}, RuntimeError('timeout'))
    assert '已核验实验指标和参数对照仍然有效' in answer
    assert '当前任务状态' not in answer and '缺失结果' in answer


def test_question_acknowledged_before_context_failure(monkeypatch):
    session = FakeSession()
    project = SimpleNamespace(name='合成测试项目')
    conversation = SimpleNamespace(title='新对话', updated_at=None)
    async def fake_project(*args): return project
    async def fake_conversation(*args): return conversation
    async def broken_context(*args):
        assert session.messages and session.messages[0].context_json['status'] == 'pending'
        raise RuntimeError('offline context failure')
    monkeypatch.setattr(copilot, '_project', fake_project)
    monkeypatch.setattr(copilot, '_conversation', fake_conversation)
    monkeypatch.setattr(copilot, '_context', broken_context)
    monkeypatch.setattr(copilot, 'AsyncSessionLocal', lambda: session)
    async def history(*args): return '', []
    monkeypatch.setattr(copilot, '_managed_history', history)
    async def collect():
        response = await copilot.stream_answer('p', 'c', copilot.CopilotQuestion(
            question='请说明论文使用哪些数据集', request_id='offline'), db=session)
        return ''.join([frame async for frame in response.body_iterator])
    output = asyncio.run(collect())
    assert output.index('event: user') < output.index('event: error')
    assert len(session.messages) == 1 and session.messages[0].context_json['status'] == 'failed'


@pytest.mark.parametrize(
    "stream_text,finish_reason,fallback_text,expected_status,expected_text",
    [
        ("这篇论文使用两个公开数据集进行评估。", "stop", "", "completed", "公开数据集"),
        ("**当前", "length", "完整回复：论文使用两个数据集。", "completed", "完整回复"),
        ("**当前", "stop", "完整回复：论文使用两个数据集。", "completed", "完整回复"),
        ("", "stop", "", "degraded", "未生成完整回答"),
    ],
)
def test_stream_completion_retry_and_empty_fallback(
    monkeypatch, stream_text, finish_reason, fallback_text, expected_status, expected_text,
):
    session = FakeSession()
    project = SimpleNamespace(name="合成测试项目", description="")
    conversation = SimpleNamespace(title="新对话", updated_at=None)

    async def fake_project(*_):
        return project

    async def fake_conversation(*_):
        return conversation

    async def fake_history(*_):
        return "", []

    async def fake_context(*_):
        return {"project": {"name": project.name}, "job": None}

    monkeypatch.setattr(copilot, "_project", fake_project)
    monkeypatch.setattr(copilot, "_conversation", fake_conversation)
    monkeypatch.setattr(copilot, "_managed_history", fake_history)
    monkeypatch.setattr(copilot, "_context", fake_context)
    monkeypatch.setattr(copilot, "AsyncSessionLocal", lambda: session)
    selection = ModelSelection("test", "test-model", "https://example.invalid/v1", "dummy", "coder")
    monkeypatch.setattr(copilot, "get_model_router", lambda: SimpleNamespace(select=lambda _: selection))

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if body.get("stream"):
            chunks = []
            if stream_text:
                chunks.append("data: " + json.dumps({"choices": [{"delta": {"content": stream_text}}]}) + "\n\n")
            chunks.append("data: " + json.dumps({"choices": [{"delta": {}, "finish_reason": finish_reason}]}) + "\n\n")
            chunks.append("data: [DONE]\n\n")
            return httpx.Response(200, text="".join(chunks), headers={"content-type": "text/event-stream"})
        return httpx.Response(200, json={"choices": [{"message": {"content": fallback_text}, "finish_reason": "stop"}]})

    real_client = httpx.AsyncClient
    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(copilot.httpx, "AsyncClient", lambda **kwargs: real_client(transport=transport, **kwargs))

    async def collect():
        response = await copilot.stream_answer(
            "project-1", "conversation-1",
            copilot.CopilotQuestion(question="请说明论文使用哪些数据集", request_id="offline-test"),
            db=session,
        )
        return "".join([chunk async for chunk in response.body_iterator])

    output = asyncio.run(collect())
    done = next(frame for frame in output.split("\n\n") if frame.startswith("event: done"))
    answer = json.loads(done.split("data: ", 1)[1])
    assert answer["context"]["status"] == expected_status
    assert expected_text in answer["content"]
    assert len([message for message in session.messages if message.role == "user"]) == 1
    assert len([message for message in session.messages if message.role == "assistant"]) == 1
    if finish_reason == "length" and fallback_text:
        assert "event: replace" in output
        assert "**当前" not in answer["content"]


def test_history_compaction_does_not_wait_for_another_model(monkeypatch):
    rows = [SimpleNamespace(role="user" if index % 2 == 0 else "assistant",
                            content=f"合成消息 {index}", context_json={"status": "completed"})
            for index in range(16)]

    class HistorySession:
        async def scalars(self, *_):
            return SimpleNamespace(all=lambda: rows)

        async def commit(self):
            return None

    conversation = SimpleNamespace(id="test", summary="", summary_count=0)
    monkeypatch.setattr(copilot, "get_model_router", lambda: (_ for _ in ()).throw(AssertionError("must not call model")))
    summary, history = asyncio.run(copilot._managed_history(HistorySession(), conversation))
    assert conversation.summary_count == 6
    assert "合成消息 0" in summary
    assert len(history) == 8


def test_history_over_300_messages_keeps_latest_turns():
    rows = [SimpleNamespace(role="user" if index % 2 == 0 else "assistant",
                            content=f"message-{index}", context_json={"status": "completed"})
            for index in range(320)]

    class HistorySession:
        def __init__(self):
            self.calls = 0

        async def scalars(self, *_):
            self.calls += 1
            selected = rows[:300] if self.calls == 1 else list(reversed(rows[-8:]))
            return SimpleNamespace(all=lambda: selected)

        async def commit(self):
            return None

    conversation = SimpleNamespace(id="test", summary="", summary_count=0)
    summary, history = asyncio.run(copilot._managed_history(HistorySession(), conversation))
    assert conversation.summary_count == 290
    assert "message-289" in summary
    assert history[-1]["content"] == "message-319"


def test_stream_protocol_error_retries_without_streaming(monkeypatch):
    session = FakeSession()
    project = SimpleNamespace(name="合成测试项目", description="")
    conversation = SimpleNamespace(title="新对话", updated_at=None)

    async def fake_project(*_):
        return project

    async def fake_conversation(*_):
        return conversation

    async def fake_history(*_):
        return "", []

    async def fake_context(*_):
        return {"project": {"name": project.name}, "job": None}

    monkeypatch.setattr(copilot, "_project", fake_project)
    monkeypatch.setattr(copilot, "_conversation", fake_conversation)
    monkeypatch.setattr(copilot, "_managed_history", fake_history)
    monkeypatch.setattr(copilot, "_context", fake_context)
    monkeypatch.setattr(copilot, "AsyncSessionLocal", lambda: session)
    selection = ModelSelection("test", "test-model", "https://example.invalid/v1", "dummy", "coder")
    monkeypatch.setattr(copilot, "get_model_router", lambda: SimpleNamespace(select=lambda _: selection))

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if body.get("stream"):
            raise httpx.RemoteProtocolError("server closed stream")
        return httpx.Response(200, json={"choices": [{"message": {"content": "模型文件位于 model.py。"},
                                                       "finish_reason": "stop"}]})

    real_client = httpx.AsyncClient
    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(copilot.httpx, "AsyncClient", lambda **kwargs: real_client(transport=transport, **kwargs))

    async def collect():
        response = await copilot.stream_answer(
            "project-1", "conversation-1",
            copilot.CopilotQuestion(question="请说明模型文件位于哪里", request_id="offline-test"), db=session,
        )
        return "".join([chunk async for chunk in response.body_iterator])

    output = asyncio.run(collect())
    done = next(frame for frame in output.split("\n\n") if frame.startswith("event: done"))
    answer = json.loads(done.split("data: ", 1)[1])
    assert answer["context"]["status"] == "completed"
    assert "model.py" in answer["content"]
    assert "event: replace" in output


def test_explicit_code_change_gets_preview_plan_not_rerun_answer(monkeypatch):
    session = FakeSession()
    project = SimpleNamespace(name="合成测试项目", description="")
    conversation = SimpleNamespace(title="新对话", updated_at=None)

    async def fake_project(*_):
        return project

    async def fake_conversation(*_):
        return conversation

    async def fake_action_context(*_):
        return {"project": {"name": project.name},
                "job": {"run_id": "run-1", "status": "paused", "stage": "paused"}}

    monkeypatch.setattr(copilot, "_project", fake_project)
    monkeypatch.setattr(copilot, "_conversation", fake_conversation)
    monkeypatch.setattr(copilot, "_action_context", fake_action_context)
    monkeypatch.setattr(copilot, "AsyncSessionLocal", lambda: session)
    selection = ModelSelection("test", "test-model", "https://example.invalid/v1", "dummy", "coder")
    monkeypatch.setattr(copilot, "get_model_router", lambda: SimpleNamespace(select=lambda _: selection))

    async def collect():
        response = await copilot.stream_answer(
            "project-1", "conversation-1",
            copilot.CopilotQuestion(question="请将训练代码改成可以使用 GPU 的版本", request_id="offline-test"),
            db=session,
        )
        return "".join([chunk async for chunk in response.body_iterator])

    output = asyncio.run(collect())
    done = next(frame for frame in output.split("\n\n") if frame.startswith("event: done"))
    answer = json.loads(done.split("data: ", 1)[1])
    assert answer["context"]["action_plan"]["kind"] == "code_revision"
    assert "代码修改请求" in answer["content"]
    assert "暂不能执行重跑" not in answer["content"]


def test_restart_experiment_stream_returns_confirmation_without_model_call(monkeypatch):
    session = FakeSession()
    project = SimpleNamespace(id="project-1", name="合成测试项目", description="")
    conversation = SimpleNamespace(title="新对话", updated_at=None)

    async def fake_project(*_):
        return project

    async def fake_conversation(*_):
        return conversation

    async def fake_action_context(*_):
        return {"project": {"name": project.name},
                "job": {"run_id": "run-1", "status": "paused", "stage": "paused"}}

    async def fake_rerun_plan(*_):
        return {"kind": "rerun_confirm", "run_id": "run-1", "dataset": "Oncology-Screen、DrugCombDB",
                "datasets": ["Oncology-Screen", "DrugCombDB"],
                "selected_datasets": ["Oncology-Screen", "DrugCombDB"],
                "gpu_requested": True, "status": "paused"}

    monkeypatch.setattr(copilot, "_project", fake_project)
    monkeypatch.setattr(copilot, "_conversation", fake_conversation)
    monkeypatch.setattr(copilot, "_action_context", fake_action_context)
    monkeypatch.setattr(copilot, "_rerun_action_plan", fake_rerun_plan)
    monkeypatch.setattr(copilot, "AsyncSessionLocal", lambda: session)
    selection = ModelSelection("test", "test-model", "https://example.invalid/v1", "dummy", "coder")
    monkeypatch.setattr(copilot, "get_model_router", lambda: SimpleNamespace(select=lambda _: selection))

    async def collect():
        response = await copilot.stream_answer(
            "project-1", "conversation-1",
            copilot.CopilotQuestion(question="请重新开始实验", run_id="run-1", request_id="offline-test"),
            db=session,
        )
        return "".join([chunk async for chunk in response.body_iterator])

    output = asyncio.run(collect())
    done = next(frame for frame in output.split("\n\n") if frame.startswith("event: done"))
    answer = json.loads(done.split("data: ", 1)[1])
    assert answer["context"]["action_plan"]["kind"] == "rerun_confirm"
    assert answer["context"]["action_plan"]["selected_datasets"] == ["Oncology-Screen", "DrugCombDB"]
    assert "确认前不会启动实验" in answer["content"]
