import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest

from app.config import Settings
from app.core.model_router import ModelRole, ModelRouter
from app.services.code_agent import CodingFailure, generate_project, local_import_errors, paper_context, source_text, file_error
from app.services.code_validation import grounded_issue, local_call_errors, static_risks, unrelated_definition_error


def test_raw_code_transport_preserves_source_and_detects_truncation(monkeypatch):
    settings = Settings(_env_file=None, model_provider="openai_compatible", LLM_API_KEY="test",
                        llm_model_coder="code-test", llm_base_url="https://example.invalid/v1")
    requests = []

    def transport(request):
        body = json.loads(request.content)
        requests.append(body)
        return httpx.Response(200, json={"choices": [{"message": {"content": "def train():\n    return 1"},
                                                      "finish_reason": "length"}], "usage": {"completion_tokens": 10}})

    original = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: original(
        **kwargs, transport=httpx.MockTransport(transport)))
    result = asyncio.run(ModelRouter(settings).complete_text(ModelRole.code, "source only", "write file"))
    assert result.text.startswith("def train")
    assert result.finish_reason == "length"
    assert "response_format" not in requests[0]


def test_reasoning_is_not_accepted_as_final_source(monkeypatch):
    settings = Settings(_env_file=None, model_provider="openai_compatible", LLM_API_KEY="test",
                        llm_model_coder="code-test", llm_base_url="https://example.invalid/v1")
    original = httpx.AsyncClient
    response = {"choices": [{"message": {"content": "", "reasoning_content": "I should write some code"},
                             "finish_reason": "length"}]}
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: original(
        **kwargs, transport=httpx.MockTransport(lambda _: httpx.Response(200, json=response))))
    result = asyncio.run(ModelRouter(settings).complete_text(ModelRole.code, "source only", "write file"))
    assert result.text == ""
    assert "空内容" in result.error


def test_code_loop_continues_truncated_output_and_reuses_existing_modules():
    plan = {"files": [{"path": "method.py"}, {"path": "train.py"}], "entrypoint": "train.py"}
    calls = []
    snapshots = []

    async def complete_json(*_args, **_kwargs):
        return SimpleNamespace(data={"issues": [], "limitations": []}, error="", usage={})

    async def complete_text(_role, _system, prompt, **_kwargs):
        calls.append(json.loads(prompt))
        if len(calls) == 1:
            return SimpleNamespace(text="from method import transform\ndef main():\n    return trans",
                                   finish_reason="length", error="", usage={})
        return SimpleNamespace(text="    return transform([1, 2])\n", finish_reason="stop", error="", usage={})

    async def checkpoint(files, _trace, _plan, _usage):
        snapshots.append(dict(files))

    result = asyncio.run(generate_project(
        SimpleNamespace(complete_json=complete_json, complete_text=complete_text), "paper", {},
        initial_files={"method.py": "def transform(x):\n    return x\n"}, previous_plan=plan,
        checkpoint=checkpoint,
    ))
    assert len(calls) == 2
    assert "existing_source" in calls[1]
    assert calls[1]["unfinished_line"] == "    return trans"
    assert result["validation"]["syntax"]
    assert "return transform([1, 2])" in result["files"][1]["content"]
    assert snapshots[-1]["method.py"] == "def transform(x):\n    return x\n"


def test_local_import_inspection_finds_missing_exports():
    assert local_import_errors({"model.py": "class Model: pass\n",
                                "train.py": "from model import Missing\n"}) == {
        "train.py": "从 model 导入了不存在的接口：Missing"}


def test_unclosed_markdown_fence_does_not_discard_valid_source():
    code = source_text('```python\ndef transform(x):\n    return x')
    assert code.startswith('def transform')
    assert file_error('method.py', code) == ''
    assert source_text('\n    return x\n```', suffix=True) == '\n    return x'
    assert source_text('```python\n    return x', suffix=True) == '    return x'
    assert file_error('method.py', source_text('```python\ndef transform('))


def test_concatenated_fragment_fences_are_recovered_without_changing_docstrings():
    docstring = '\"\"\"Example:\n```python\nx = 1\n```\n\"\"\"\n'
    source = docstring + 'def first():\n    return 1\n```python\ndef second():\n    return 2\n'
    recovered = source_text(source)
    assert recovered.startswith(docstring)
    assert file_error('method.py', recovered) == ''
    assert '\n```python\ndef second' not in recovered


def test_paper_context_includes_method_beyond_introduction():
    text = "Introduction\n" + "background " * 2500 + "\n3 Method\nMASK_AND_IMPUTE\n4 Experiments\nDATA_SPLIT\n"
    context = paper_context(text)
    assert "MASK_AND_IMPUTE" in context
    assert "DATA_SPLIT" in context


def test_pdf_small_cap_headings_include_custom_method_and_appendix():
    text = ("Paper abstract\n1 I NTRODUCTION\n" + "background " * 2500
            + "\n2 B ACKGROUND\n" + "prose " * 1000
            + "\n3 L ANGUAGE -INTERFACED TABULAR OVERSAMPLING\n3.1 M ETHOD\nATTENTION_EQUATION\n"
            + "\n4 E XPERIMENTS\nDATASET_NAMES\n5 C ONCLUSION\nconclusion\n"
            + "\nA D ETAILED EXPERIMENTAL SETTINGS\nREAL_HYPERPARAMETERS\n"
            + "\nC D ETAILED SETTINGS FOR IN-CONTEXT LEARNING\nREAL_PROMPTS\n")
    context = paper_context(text)
    for evidence in ("ATTENTION_EQUATION", "DATASET_NAMES", "REAL_HYPERPARAMETERS", "REAL_PROMPTS"):
        assert evidence in context
    assert context.count("background") < 200


def test_method_review_repairs_only_the_file_with_supported_defects():
    plan = {"files": [{"path": "method.py"}, {"path": "train.py"}], "entrypoint": "train.py",
            "limitations": ["preliminary assumption"]}
    writes = []
    reviews = []
    training = "from method import transform\ndef main():\n    return transform([1])\n"

    async def complete_json(*_args, **_kwargs):
        reviews.append(True)
        request = json.loads(_args[2])
        return SimpleNamespace(data={"issues": [{"file": "method.py", "reason": "Core transformation missing",
                                                 "code_evidence": "return x"}]
                                     if len(reviews) == 1 else [], "limitations": ["actual resource required"],
                                     "issue_checks": [{"id": row["id"], "status": "resolved", "reason": "Transformation implemented"}
                                                      for row in request["pending_issues"]],
                                     "required_resources": [{"name": "weights"}]}, error="", usage={})

    async def complete_text(_role, _system, prompt, **_kwargs):
        writes.append(json.loads(prompt))
        return SimpleNamespace(text="def transform(x):\n    return [v + 1 for v in x]\n",
                               finish_reason="stop", error="", usage={})

    result = asyncio.run(generate_project(
        SimpleNamespace(complete_json=complete_json, complete_text=complete_text), "paper", {},
        initial_files={"method.py": "def transform(x):\n    return x\n", "train.py": training}, previous_plan=plan))
    assert len(writes) == 1
    assert writes[0]["target_file"]["path"] == "method.py"
    assert writes[0]["validation_feedback"] == "Core transformation missing"
    assert result["files"][1]["content"] == training
    assert result["limitations"] == ["actual resource required"]
    assert result["validation"]["method_fidelity"] == "reviewed"


def test_unavailable_method_review_is_not_reported_as_successful_review():
    async def unavailable(*_args, **_kwargs):
        return SimpleNamespace(data={}, error="ReadTimeout", usage={})

    with pytest.raises(CodingFailure, match="方法审查未完成") as failed:
        asyncio.run(generate_project(
            SimpleNamespace(complete_json=unavailable), "paper", {},
            initial_files={"train.py": "def main():\n    return 1\n"},
            previous_plan={"files": [{"path": "train.py"}], "entrypoint": "train.py"}))
    assert failed.value.files["train.py"].startswith("def main")


def test_truncated_entrypoint_repair_preserves_prior_valid_functions():
    prefix = '# module header\n' * 15 + 'def helper():\n    return 7\n\n'
    draft = prefix + 'def main():\n    print(f"Result: \n'
    requests = []

    async def complete_text(_role, _system, prompt, **_kwargs):
        requests.append(json.loads(prompt))
        return SimpleNamespace(text='def main():\n    print(helper())\n',
                               finish_reason='stop', error='', usage={})

    async def complete_json(*_args, **_kwargs):
        return SimpleNamespace(data={'issues': []}, error='', usage={})

    result = asyncio.run(generate_project(
        SimpleNamespace(complete_json=complete_json, complete_text=complete_text), 'paper', {},
        initial_files={'main.py': draft},
        previous_plan={'files': [{'path': 'main.py'}], 'entrypoint': 'main.py'}))
    assert requests[0]['preserved_prefix'] == prefix
    assert requests[0]['rewrite_from_line'] == 19
    assert result['files'][0]['content'] == prefix + 'def main():\n    print(helper())\n'


def test_transport_retry_does_not_forget_original_method_defect():
    calls, reviews = [], []

    async def complete_json(*_args, **_kwargs):
        reviews.append(True)
        request = json.loads(_args[2])
        return SimpleNamespace(data={'issues': [{'file': 'method.py', 'reason': 'preserve required algorithm',
                                                 'code_evidence': 'return x'}]
                                     if len(reviews) == 1 else [],
                                     'issue_checks': [{'id': row['id'], 'status': 'resolved', 'reason': 'Algorithm updated'}
                                                      for row in request['pending_issues']]}, error='', usage={})

    async def complete_text(_role, _system, prompt, **_kwargs):
        calls.append(json.loads(prompt))
        return SimpleNamespace(text='' if len(calls) == 1 else 'def transform(x):\n    return [v + 1 for v in x]\n',
                               error='ReadTimeout' if len(calls) == 1 else '', usage={}, finish_reason='stop')

    asyncio.run(generate_project(SimpleNamespace(complete_json=complete_json, complete_text=complete_text),
                                 'paper', {}, initial_files={'method.py': 'def transform(x):\n    return x\n'},
                                 previous_plan={'files': [{'path': 'method.py'}], 'entrypoint': 'method.py'}))
    assert 'preserve required algorithm' in calls[1]['validation_feedback']
    assert 'ReadTimeout' in calls[1]['validation_feedback']


@pytest.mark.parametrize('source', [
    'def mapping(tokens):\n    return tokenizer.tokenize(tokens)\n',
    'def mapping(tokens):\n    return [missing(x) for x in tokens]\n',
    'class Model:\n    def forward(self, x):\n        return unseen(x)\n',
])
def test_undefined_global_references_are_detected_without_running_code(source):
    assert '未定义的全局名称' in file_error('method.py', source)


@pytest.mark.parametrize('source', [
    'def mapping(tokenizer):\n    return tokenizer.tokenize("x")\n',
    'def outer(x):\n    def inner():\n        return x\n    return inner()\n',
    'def transform(xs):\n    return [x + 1 for x in xs]\n',
    'def forward(x):\n    return helper(x)\ndef helper(x):\n    return x\n',
    'from __future__ import annotations\ndef forward(x: Tensor):\n    return x\n',
    'class Model:\n    def forward(self):\n        return __class__\n',
])
def test_static_name_check_respects_python_scopes(source):
    assert file_error('method.py', source) == ''


def _reply(data=None, text='', error=''):
    return SimpleNamespace(data=data, text=text, error=error, usage={}, finish_reason='stop')


def test_third_repair_is_followed_by_a_final_review_and_tracks_resolution():
    writes, reviews = [], []

    async def review(_role, _system, prompt, **_kwargs):
        request = json.loads(prompt)
        reviews.append(request)
        n = len(reviews) - 1
        issues = [{'file': 'method.py', 'reason': f'Implement step {n + 1}',
                   'code_evidence': f'return {n}'}] if n < 3 else []
        return _reply({'issues': issues, 'issue_checks': [
            {'id': row['id'], 'status': 'resolved', 'reason': 'Checked replacement against the requirement'}
            for row in request['pending_issues']]})

    async def write(*_args, **_kwargs):
        writes.append(True)
        return _reply(text=f'def transform():\n    return {len(writes)}\n')

    result = asyncio.run(generate_project(SimpleNamespace(complete_json=review, complete_text=write),
                         'paper', {}, initial_files={'method.py': 'def transform():\n    return 0\n'},
                         previous_plan={'files': [{'path': 'method.py'}], 'entrypoint': 'method.py'}))
    assert len(writes) == 3 and len(reviews) == 4
    assert len(result['issue_ledger']) == 3
    assert all(row['status'] == 'resolved' for row in result['issue_ledger'])
    assert all(row['status'] == 'resolved' for row in result['generation_trace'] if row['phase'] == 'review')


@pytest.mark.parametrize('issue', [
    {'file': 'invented.py', 'reason': 'wrong file', 'code_evidence': 'return x'},
    {'file': 'method.py', 'reason': 'may be wrong'},
    {'file': 'method.py', 'reason': 'wrong method', 'category': 'method', 'code_evidence': 'return x'},
    {'file': 'method.py', 'reason': 'wrong method', 'category': 'method',
     'code_evidence': 'return x', 'paper_evidence': 'invented paper quote'},
])
def test_ungrounded_review_cannot_modify_source(issue):
    calls = []

    async def review(*_args, **_kwargs):
        calls.append(True)
        return _reply({'issues': [issue]})

    with pytest.raises(CodingFailure, match='证据校验失败') as failed:
        asyncio.run(generate_project(SimpleNamespace(complete_json=review), 'paper', {},
                    initial_files={'method.py': 'def transform(x):\n    return x\n'},
                    previous_plan={'files': [{'path': 'method.py'}], 'entrypoint': 'method.py'}))
    assert len(calls) == 2
    assert failed.value.files['method.py'] == 'def transform(x):\n    return x\n'


def test_omitted_old_issue_is_not_silently_resolved():
    calls = []

    async def review(*_args, **_kwargs):
        calls.append(True)
        return _reply({'issues': [{'file': 'method.py', 'reason': 'wrong value', 'code_evidence': 'return 0'}]
                                  if len(calls) == 1 else []})

    async def write(*_args, **_kwargs):
        return _reply(text='def transform():\n    return 1\n')

    with pytest.raises(CodingFailure, match='未逐项复核') as failed:
        asyncio.run(generate_project(SimpleNamespace(complete_json=review, complete_text=write), 'paper', {},
                    initial_files={'method.py': 'def transform():\n    return 0\n'},
                    previous_plan={'files': [{'path': 'method.py'}], 'entrypoint': 'method.py'}))
    assert failed.value.issues[0]['status'] == 'pending_verification'


def test_semantic_repair_rejects_unrelated_function_regression_and_preserves_draft():
    original = 'def helper():\n    return 7\n\ndef transform():\n    return 0\n'
    writes = []

    async def review(*_args, **_kwargs):
        return _reply({'issues': [{'file': 'method.py', 'reason': 'wrong value', 'code_evidence': 'return 0'}]})

    async def write(*_args, **_kwargs):
        writes.append(True)
        return _reply(text='def helper():\n    return 99\n\ndef transform():\n    return 1\n')

    with pytest.raises(CodingFailure, match='证据范围之外') as failed:
        asyncio.run(generate_project(SimpleNamespace(complete_json=review, complete_text=write), 'paper', {},
                    initial_files={'method.py': original},
                    previous_plan={'files': [{'path': 'method.py'}], 'entrypoint': 'method.py'}))
    assert failed.value.files['method.py'] == original
    assert len(writes) == 3


def test_transport_failure_with_partial_response_preserves_valid_source():
    original = 'def transform():\n    return 0\n'

    async def review(*_args, **_kwargs):
        return _reply({'issues': [{'file': 'method.py', 'reason': 'wrong value', 'code_evidence': 'return 0'}]})

    async def write(*_args, **_kwargs):
        return _reply(text='def transform(', error='RemoteProtocolError')

    with pytest.raises(CodingFailure, match='RemoteProtocolError') as failed:
        asyncio.run(generate_project(SimpleNamespace(complete_json=review, complete_text=write), 'paper', {},
                    initial_files={'method.py': original},
                    previous_plan={'files': [{'path': 'method.py'}], 'entrypoint': 'method.py'}))
    assert failed.value.files['method.py'] == original


def test_unchanged_repair_stops_instead_of_wasting_review_budget():
    original = 'def transform():\n    return 0\n'

    async def review(*_args, **_kwargs):
        return _reply({'issues': [{'file': 'method.py', 'reason': 'wrong value', 'code_evidence': 'return 0'}]})

    async def write(*_args, **_kwargs):
        return _reply(text=original)

    with pytest.raises(CodingFailure, match='未产生源码变化'):
        asyncio.run(generate_project(SimpleNamespace(complete_json=review, complete_text=write), 'paper', {},
                    initial_files={'method.py': original},
                    previous_plan={'files': [{'path': 'method.py'}], 'entrypoint': 'method.py'}))


def test_class_repair_preserves_unrelated_methods():
    source = 'class Model:\n    def helper(self):\n        return 7\n    def forward(self):\n        return 0\n'
    candidate = source.replace('return 7', 'return 99').replace('return 0', 'return 1')
    assert 'helper' in unrelated_definition_error(source, candidate, ['return 0'])
    assert unrelated_definition_error(source, source.replace('return 0', 'return 1'), ['return 0']) == ''


def test_reviewer_hints_cover_outer_generation_loop_and_silent_row_drops():
    risks = static_risks({'method.py': 'def samples(rows, count):\n    while len(rows) < count:\n        for i in range(3):\n            break\n    return rows.dropna()\n'})
    assert len(risks) == 2
    assert risks[0]['line'] == 2 and risks[1]['line'] == 5


def test_changed_open_issue_requires_current_code_evidence():
    async def review(_role, _system, prompt, **_kwargs):
        request = json.loads(prompt)
        return _reply({'issues': [], 'issue_checks': [
            {'id': row['id'], 'status': 'open', 'reason': 'Still wrong'} for row in request['pending_issues']]})

    original = 'def transform():\n    return 0\n'
    issue = grounded_issue({'file': 'method.py', 'reason': 'Wrong value', 'code_evidence': 'return 0'},
                           {'method.py': original}, 'paper')
    with pytest.raises(CodingFailure, match='源码证据'):
        asyncio.run(generate_project(SimpleNamespace(complete_json=review), 'paper', {},
                    initial_files={'method.py': original.replace('return 0', 'return 1')},
                    previous_plan={'files': [{'path': 'method.py'}], 'entrypoint': 'method.py', 'issue_ledger': [issue]}))


def test_dynamic_namespace_is_reported_as_partial_not_full_name_validation():
    async def review(*_args, **_kwargs):
        return _reply({'issues': []})

    result = asyncio.run(generate_project(SimpleNamespace(complete_json=review), 'paper', {},
                         initial_files={'method.py': 'def names():\n    return globals()\n'},
                         previous_plan={'files': [{'path': 'method.py'}], 'entrypoint': 'method.py'}))
    assert result['validation']['undefined_globals'] == 'partial'
    assert any('动态命名空间' in item for item in result['limitations'])


def test_ambiguous_source_quote_requires_more_precise_evidence():
    with pytest.raises(ValueError, match='唯一可定位'):
        grounded_issue({'file': 'method.py', 'reason': 'wrong result', 'code_evidence': 'return 0'},
                       {'method.py': 'def first():\n    return 0\ndef second():\n    return 0\n'}, 'paper')


def test_static_name_error_is_repaired_before_model_method_review():
    events = []

    async def write(_role, _system, prompt, **_kwargs):
        events.append('write')
        assert 'tokenizer' in json.loads(prompt)['validation_feedback']
        return _reply(text='def mapping(tokenizer):\n    return tokenizer.tokenize("x")\n')

    async def review(*_args, **_kwargs):
        events.append('review')
        return _reply({'issues': []})

    asyncio.run(generate_project(SimpleNamespace(complete_json=review, complete_text=write), 'paper', {},
                initial_files={'method.py': 'def mapping(tokens):\n    return tokenizer.tokenize(tokens)\n'},
                previous_plan={'files': [{'path': 'method.py'}], 'entrypoint': 'method.py'}))
    assert events == ['write', 'review']


@pytest.mark.parametrize('call', ['mapping()', 'mapping(1, 2)', 'mapping(unknown=1)'])
def test_local_call_signature_detects_integration_breakages(call):
    errors = local_call_errors({'method.py': 'def mapping(tokenizer):\n    return tokenizer\n',
                               'main.py': f'from method import mapping\ndef main():\n    return {call}\n'})
    assert '参数不匹配' in errors['main.py']


@pytest.mark.parametrize('source', [
    'def mapping(x=1):\n    return x\ndef main():\n    return mapping()\n',
    'def mapping(x, /, *, count=1):\n    return x\ndef main():\n    return mapping(1, count=2)\n',
    'def mapping(*xs, **kw):\n    return xs\ndef main():\n    return mapping(1, 2, count=3)\n',
    'def mapping(x):\n    return x\ndef main(mapping):\n    return mapping()\n',
    'def mapping(x):\n    return x\ndef main(xs):\n    return mapping(*xs)\n',
    'def mapping(x):\n    return x\nmapping = print\nmapping()\n',
    'def mapping(x):\n    return x\ndef main(xs):\n    return [mapping() for mapping in xs]\n',
])
def test_call_binding_does_not_guess_dynamic_or_shadowed_interfaces(source):
    assert local_call_errors({'method.py': source}) == {}


def test_generation_failure_summary_preserves_current_reason_not_resolved_history(monkeypatch):
    from app.services import reproduction_engine as engine

    async def fail(*_args, **_kwargs):
        raise CodingFailure('two current blockers: A and B', {'main.py': 'def main():\n    return 1'}, [
            {'phase': 'write', 'reason': 'temporary disconnect', 'status': 'resolved'},
            {'phase': 'review', 'reason': 'old fixed issue', 'status': 'resolved'},
            {'phase': 'review', 'reason': 'B', 'status': 'open'},
        ], {}, [{'id': 'b', 'file': 'main.py', 'reason': 'B', 'status': 'open'}])

    monkeypatch.setattr('app.services.code_agent.generate_project', fail)
    with pytest.raises(engine.CodeGenerationFailure) as caught:
        asyncio.run(engine._generate_scaffold('paper', {}))
    assert 'A and B' in str(caught.value)
    assert len(caught.value.attempts) == 1
    assert caught.value.issues[0]['id'] == 'b'
