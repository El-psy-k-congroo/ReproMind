import asyncio
import base64
import httpx
from app.services.author_documentation import linked_readme
from app.services import reproduction_engine as engine


def test_api_readme_resolves_author_archive_when_raw_domain_is_unavailable(tmp_path,monkeypatch):
    (tmp_path/'README.md').write_text('Data preprocessing: https://github.com/author/data-repo')
    (tmp_path/'train.py').write_text('print("train")')
    calls=[]
    def respond(request):
        calls.append(str(request.url))
        if request.url.host=='raw.githubusercontent.com':raise httpx.ConnectError('reset',request=request)
        if request.url.path.endswith('/readme'):
            content=base64.b64encode(b'DatasetX files: https://example.org/data/datasetx.tar.gz').decode()
            return httpx.Response(200,json={'encoding':'base64','content':content})
        if '/git/trees/' in request.url.path:return httpx.Response(200,json={'tree':[]})
        return httpx.Response(200,json={'default_branch':'main'})
    cls=httpx.AsyncClient
    monkeypatch.setattr(engine.httpx,'AsyncClient',lambda **kwargs:cls(transport=httpx.MockTransport(respond),**kwargs))
    spec=asyncio.run(engine._dataset_spec('',{'datasets':['DatasetX']},tmp_path,public_lookup=False))
    assert not spec['missing_datasets']
    assert spec['download_requests'][0]['url']=='https://example.org/data/datasetx.tar.gz'
    assert spec['download_requests'][0]['source']=='author_repository_documentation'
    assert not any('raw.githubusercontent.com' in url for url in calls)


def test_raw_readme_fallback_records_api_failure_and_tries_default_branches():
    def respond(request):
        if request.url.host=='api.github.com':return httpx.Response(403)
        return httpx.Response(200,text='Author data docs')
    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            return await linked_readme(client,'author','repo')
    text,errors=asyncio.run(run())
    assert text=='Author data docs' and len(errors)==1 and '403' in errors[0]
