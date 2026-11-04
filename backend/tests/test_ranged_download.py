import asyncio
import re

import httpx
import pytest

from app.services.ranged_download import transfer_ranges


def test_concurrent_ranges_preserve_prefix_and_resume_failed_chunk(tmp_path):
    payload = bytes(range(256)) * (16*1024)
    prefix = 65536
    partial = tmp_path/'dataset.part'
    partial.write_bytes(payload[:prefix])
    active = 0
    peak = 0
    failed = False
    resumes = []
    updates = []

    class Body(httpx.AsyncByteStream):
        def __init__(self, start, end): self.start, self.end = start, end
        async def __aiter__(self):
            nonlocal active, peak, failed
            active += 1; peak = max(active, peak)
            try:
                for start in range(self.start, self.end+1, 65536):
                    await asyncio.sleep(.001)
                    yield payload[start:min(start+65536, self.end+1)]
                    if self.start == prefix and not failed:
                        failed = True
                        raise httpx.ReadError('temporary failure after saved bytes')
            finally:
                active -= 1

    def handle(request):
        start, end = map(int, re.fullmatch(r'bytes=(\d+)-(\d+)', request.headers['range']).groups())
        resumes.append(start)
        assert request.headers['if-range'] == '"version-1"'
        return httpx.Response(206, headers={'content-range': f'bytes {start}-{end}/{len(payload)}',
            'etag': '"version-1"'}, stream=Body(start, end))

    async def progress(*args): updates.append(args)
    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            assert await transfer_ranges(client, 'https://author.example/data', partial,
                len(payload), prefix, '"version-1"', 4, progress, lambda: None)
    asyncio.run(run())
    assert partial.read_bytes() == payload
    assert peak >= 3 and prefix+65536 in resumes
    assert updates[-1][0] == len(payload)
    assert active == 0


@pytest.mark.parametrize('status,etag,range_header', [
    (200, '"v1"', ''), (206, '"v2"', 'bytes 0-1048575/4194304'),
    (206, '"v1"', 'bytes 1-1048576/4194304'),
])
def test_unconfirmed_ranges_do_not_replace_original_partial(tmp_path, status, etag, range_header):
    partial = tmp_path/'data.part'
    partial.write_bytes(b'old-prefix')
    async def progress(*args): pass
    async def run():
        handler = lambda request: httpx.Response(status, headers={'etag': etag, 'content-range': range_header}, content=b'bad')
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            assert not await transfer_ranges(client, 'https://author.example/data', partial,
                4194304, 10, '"v1"', 4, progress, lambda: None)
    asyncio.run(run())
    assert partial.read_bytes() == b'old-prefix'
