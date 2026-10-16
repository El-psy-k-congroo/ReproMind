"""Isolated pip preflight: metadata only, fail closed instead of fetching a huge wheel.

Executed by the project's interpreter. Hooks exist only in this short-lived subprocess;
no installed pip files or project environment packages are modified.
"""
import sys
import re


def main(argv):
    if '--dry-run' not in argv or '--report' not in argv or '--only-binary=:all:' not in argv:
        return 86
    try:
        from pip._internal.cli.main import main as pip_main
        from pip._internal.operations.prepare import RequirementPreparer
        from pip._internal.network import download
        Downloader = download.Downloader
        BatchDownloader = getattr(download, 'BatchDownloader', None)
        from pip._internal.network.lazy_wheel import LazyZipOverHTTP
        if not hasattr(RequirementPreparer, 'prepare_linked_requirements_more'):
            return 86
        # Pip 22.x completes partial wheel downloads even for --dry-run. Suppress only
        # this completion step: the resolver still checks actual wheel metadata/deps.
        RequirementPreparer.prepare_linked_requirements_more = lambda self, *args, **kwargs: None
        def no_full_download(*args, **kwargs):
            raise RuntimeError('REPROMIND_METADATA_ONLY: server/pip requires full wheel; use verified wheelhouse fallback')
        Downloader.__call__ = no_full_download
        if BatchDownloader is not None:
            BatchDownloader.__call__ = no_full_download
        if hasattr(Downloader, 'batch'):
            Downloader.batch = no_full_download
        original_range = LazyZipOverHTTP._stream_response
        transferred = [0]
        def bounded_range(self, start, end):
            size = end - start + 1
            transferred[0] += size
            if size > 2_000_000 or transferred[0] > 8_000_000:
                raise RuntimeError('REPROMIND_METADATA_ONLY: metadata transfer limit exceeded')
            response = original_range(self, start, end)
            expected = re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)', response.headers.get('Content-Range', ''))
            # Old pip trusts the server's Accept-Ranges header. Verify the actual
            # streamed response before reading it, so HTTP 200 cannot fetch 2 GB.
            if response.status_code != 206 or not expected or tuple(map(int, expected.groups()[:2])) != (start, end):
                response.close()
                raise RuntimeError('REPROMIND_METADATA_ONLY: server ignored bounded Range request')
            return response
        LazyZipOverHTTP._stream_response = bounded_range
        return pip_main(['install', '--use-feature=fast-deps', *argv])
    except Exception as exc:
        print('REPROMIND_METADATA_ONLY: ' + str(exc), flush=True)
        return 86


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
