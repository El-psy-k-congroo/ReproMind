import httpx
import pytest

from scripts.check_services import check_services


PROJECT = '04e7ea06-650d-49f2-b41d-d135cfdd5ff0'


def healthy_response(request):
    if request.url.path == '/api/health':
        return httpx.Response(200, json={'status': 'ok'})
    if request.url.path == '/api/projects/' + PROJECT:
        return httpx.Response(200, json={'id': PROJECT})
    if request.url.path == '/src/main.ts':
        return httpx.Response(200, text='import App from "./App.vue"', headers={'content-type': 'text/javascript'})
    return httpx.Response(200, text='<div id="app"></div><script type="module" src="/src/main.ts"></script>',
                          headers={'content-type': 'text/html'})


def test_checks_frontend_backend_script_proxy_and_actual_project():
    with httpx.Client(transport=httpx.MockTransport(healthy_response)) as client:
        results = check_services(client, project_id=PROJECT)
    assert len(results) == 5 and all(row['ok'] for row in results)


def test_healthy_backend_does_not_hide_stopped_frontend():
    def request_handler(request):
        if request.url.port == 5173:
            raise httpx.ConnectError('Connection refused', request=request)
        return healthy_response(request)
    with httpx.Client(transport=httpx.MockTransport(request_handler)) as client:
        results = check_services(client)
    assert results[0]['ok'] and not next(row['ok'] for row in results if row['check'] == 'frontend_page')


@pytest.mark.parametrize('failure', ['proxy', 'script', 'project'])
def test_page_http_200_does_not_hide_missing_script_proxy_or_project(failure):
    def request_handler(request):
        if failure == 'proxy' and request.url.port == 5173 and request.url.path == '/api/health':
            return httpx.Response(502)
        if failure == 'script' and request.url.path == '/src/main.ts':
            return httpx.Response(200, text='<html>not found</html>', headers={'content-type': 'text/html'})
        if failure == 'project' and request.url.path.startswith('/api/projects/'):
            return httpx.Response(404)
        return healthy_response(request)
    with httpx.Client(transport=httpx.MockTransport(request_handler)) as client:
        results = check_services(client, project_id=PROJECT)
    assert next(row['ok'] for row in results if row['check'] == 'frontend_page')
    assert any(not row['ok'] for row in results)
