import json
from types import SimpleNamespace
import asyncio
import httpx
import pytest
from app.core.model_capabilities import generation_options
from app.core.model_router import ModelRouter,ModelSelection


def test_forced_reasoning_reserves_budget_without_publishing_thoughts():
    options=generation_options('glm-5.3-flash',1400,True,8192)
    assert options == {'thinking':{'type':'enabled'},'reasoning_effort':'low','max_tokens':4096}
    with pytest.raises(ValueError,match='预算'):generation_options('glm-5.3',800,True,1600)
    assert generation_options('deepseek-v4.1-flash',800,True)['thinking']['type']=='disabled'
    assert generation_options('other-model',800,True)=={'max_tokens':800}


def test_json_compat_retry_does_not_restore_max_thinking(monkeypatch):
    requests=[]
    def handler(request):
        body=json.loads(request.content);requests.append(body)
        if len(requests)==1:return httpx.Response(400,json={'error':'response_format unsupported'})
        return httpx.Response(200,json={'choices':[{'message':{'content':'{"valid":true}','reasoning_content':'private thoughts'},'finish_reason':'stop'}]})
    real_client=httpx.AsyncClient
    monkeypatch.setattr('app.core.model_router.httpx.AsyncClient',lambda **kwargs:real_client(transport=httpx.MockTransport(handler),**kwargs))
    router=ModelRouter.__new__(ModelRouter);router.settings=SimpleNamespace(llm_max_tokens_per_call=8192,llm_timeout_seconds=10)
    content,_,_=asyncio.run(router._request_text(ModelSelection('test','glm-5.3-flash','https://example.invalid','dummy','chat'),
        'test','test',max_tokens=800,non_thinking=True,json_mode=True))
    assert content=='{"valid":true}' and len(requests)==2
    assert all(row['reasoning_effort']=='low' and row['thinking']['type']=='enabled' for row in requests)
    assert 'response_format' not in requests[1]
