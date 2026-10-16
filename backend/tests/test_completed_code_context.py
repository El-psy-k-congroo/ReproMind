import pytest
from app.api.copilot import _code_action_requested, _rerun_requested


@pytest.mark.parametrize('question', [
    '请通过复现助手启动当前所选采样实验。代码修改已经通过工作台应用；请使用现有环境。',
    '代码已经修改完成，请启动当前实验。',
    '请注意代码已修改完成，继续运行当前实验。',
])
def test_completed_code_context_cannot_turn_run_request_into_code_generation(question):
    assert not _code_action_requested(question)
    if '启动' in question:
        assert _rerun_requested(question)


@pytest.mark.parametrize('question', [
    '请修改代码，让最后一个不足批次也正确运行。',
    '代码已经修改完成；请继续修复当前实验的设备错误。',
    '请把train.py迁移到GPU，保留当前实验数据。',
])
def test_new_code_directive_remains_actionable(question):
    assert _code_action_requested(question)
