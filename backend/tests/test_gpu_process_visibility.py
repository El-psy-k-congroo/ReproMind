import asyncio
from types import SimpleNamespace
import pytest
from app.services import runtime_evidence as runtime


def test_approved_gpu_is_visible_even_when_author_requires_environment_variable():
    assert runtime.gpu_visibility_environment(True,{}, {}) == {'CUDA_VISIBLE_DEVICES':'0'}
    assert runtime.gpu_visibility_environment(True,{}, {'CUDA_VISIBLE_DEVICES':'2'}) == {'CUDA_VISIBLE_DEVICES':'2'}
    assert runtime.gpu_visibility_environment(True,{'CUDA_VISIBLE_DEVICES':'GPU-existing'}, {}) == {'CUDA_VISIBLE_DEVICES':'GPU-existing'}
    assert runtime.gpu_visibility_environment(False,{}, {}) == {}
    with pytest.raises(ValueError,match='禁用'):
        runtime.gpu_visibility_environment(True,{'CUDA_VISIBLE_DEVICES':'-1'}, {})


def test_gpu_probe_checks_exact_selected_visibility_and_does_not_merge_other_choices(tmp_path,monkeypatch):
    python=tmp_path/'python.exe';python.write_text('fake')
    selections=[]
    def run(*args,**kwargs):
        selections.append(kwargs['env']['CUDA_VISIBLE_DEVICES'])
        return SimpleNamespace(returncode=0,stdout='{"cuda_available":true,"compute_checked":true}',stderr='')
    monkeypatch.setattr(runtime.subprocess,'run',run)
    async def scenario():
        return await asyncio.gather(runtime.probe_framework(python,'pytorch',True,visible_devices='0'),
            runtime.probe_framework(python,'pytorch',True,visible_devices='1'))
    results=asyncio.run(scenario())
    assert all(row['compute_checked'] for row in results)
    assert sorted(selections)==['0','1']
