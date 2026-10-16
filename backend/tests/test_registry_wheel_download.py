import hashlib
from io import BytesIO
from types import SimpleNamespace

import pytest
from app.services.registry_wheel_download import download, validate_range, wheel_record


def record(payload=b'wheel content'):
    return {'filename': 'custom-1.0-cp37-cp37m-win_amd64.whl', 'size': len(payload),
            'url': 'https://files.pythonhosted.org/packages/wheel.whl',
            'digests': {'sha256': hashlib.sha256(payload).hexdigest()}}


def test_abi_and_trusted_registry_are_required():
    row = record()
    assert wheel_record({'urls': [row]}, {('cp37', 'cp37m', 'win_amd64')}) == row
    with pytest.raises(ValueError):
        wheel_record({'urls': [row]}, {('cp311', 'cp311', 'win_amd64')})
    with pytest.raises(ValueError):
        wheel_record({'urls': [{**row, 'url': 'https://untrusted.invalid/wheel.whl'}]}, {('cp37', 'cp37m', 'win_amd64')})


def test_reject_full_response_or_wrong_range():
    validate_range(206, 'bytes 0-9/100', 0, 9, 100)
    for status, header in [(200, 'bytes 0-9/100'), (206, 'bytes 0-99/100')]:
        with pytest.raises(ValueError):
            validate_range(status, header, 0, 9, 100)


def test_compressed_wheel_tags_match_current_interpreter():
    row = {**record(), 'filename': 'custom-1.0-py2.py3-none-win_amd64.whl'}
    assert wheel_record({'urls': [row]}, {('py3', 'none', 'win_amd64')}) == row


def test_curl_transport_requires_range_headers_and_hash(tmp_path, monkeypatch):
    def curl(command, **kwargs):
        from pathlib import Path
        Path(command[command.index('--dump-header') + 1]).write_text('HTTP/1.1 206 Partial Content\nContent-Range: bytes 0-12/13\n')
        Path(command[command.index('--output') + 1]).write_bytes(b'wheel content')
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr('app.services.registry_wheel_download.subprocess.run', curl)
    assert download(record(), tmp_path, None, curl='curl.exe').read_bytes() == b'wheel content'


def test_checksum_before_publication_and_reusable_cache(tmp_path):
    class Response(BytesIO):
        status = 206
        headers = {'Content-Range': 'bytes 0-12/13'}
    calls = []
    opener = SimpleNamespace(open=lambda request, timeout: (calls.append(request) or Response(b'wheel content')))
    path = download(record(), tmp_path, opener)
    assert path.read_bytes() == b'wheel content'
    assert download(record(), tmp_path, opener) == path
    assert len(calls) == 1


def test_bad_hash_does_not_publish_wheel(tmp_path):
    class Response(BytesIO):
        status = 206
        headers = {'Content-Range': 'bytes 0-12/13'}
    opener = SimpleNamespace(open=lambda request, timeout: Response(b'wrong content'))
    with pytest.raises(ValueError, match='SHA-256'):
        download(record(), tmp_path, opener)
    assert not (tmp_path / record()['filename']).exists()
