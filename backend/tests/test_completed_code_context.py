import pytest
from app.api.copilot import _code_action_requested, _rerun_requested, _run_instruction, _gpu_capability_question


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


def test_conditional_prefix_is_not_lost_when_splitting_a_comma():
    assert not _run_instruction('如果环境已经就绪，请启动当前实验。')
    assert _run_instruction('请启动当前实验，如果出错请说明原因。')


@pytest.mark.parametrize('question', [
    '当前兼容性修改已经应用。请使用本次修订后的依赖文件，在本机 CUDA GPU 启动 MNIST 实验，保留作者训练轮数。不要再次修改代码。',
    '请保留当前模型和数据，在现有环境中运行实验。',
    '麻烦使用当前环境，按原参数评估模型。',
])
def test_device_and_environment_clauses_preserve_explicit_run_directive(question):
    assert _run_instruction(question)
    assert _rerun_requested(question)
    assert not _gpu_capability_question(question)
    assert not _code_action_requested(question)


@pytest.mark.parametrize('question', [
    '请解释当前依赖，在 GPU 上运行是否兼容？',
    '如果依赖安装完成，在 GPU 启动实验。',
    '请使用当前代码，不要启动实验。',
])
def test_readonly_conditional_and_negated_continuations_do_not_start_runs(question):
    assert not _run_instruction(question)
