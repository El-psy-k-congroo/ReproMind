"""Download a registry-verified wheel in bounded, resumable ranges (Python 3.7+).

No dependency installation or interpreter changes. Only HTTPS PyPI artifacts;
chunks are reusable, and the registry SHA-256 must match before publishing.
"""
import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import urllib.request
from urllib.parse import urlsplit

CHUNK = 8 * 1024 * 1024


def wheel_record(metadata, tags):
    for row in metadata.get('urls', []):
        name = row.get('filename', '')
        if not name.endswith('.whl') or not re.fullmatch(r'[A-Za-z0-9_.+!-]+', name):
            continue
        parts = name[:-4].split('-')
        if len(parts) < 5 or not any((py, abi, platform) in tags
                for py in parts[-3].split('.') for abi in parts[-2].split('.') for platform in parts[-1].split('.')):
            continue
        url = urlsplit(row.get('url', ''))
        digest = row.get('digests', {}).get('sha256', '')
        if (url.scheme != 'https' or url.hostname != 'files.pythonhosted.org'
                or url.username or url.password or not re.fullmatch(r'[0-9a-f]{64}', digest)
                or not isinstance(row.get('size'), int) or not 0 < row['size'] < 4_000_000_000):
            raise ValueError('Registry wheel metadata is not trusted or bounded')
        return row
    raise ValueError('No verified wheel matches the current interpreter ABI')


def validate_range(status, content_range, start, end, total):
    if status != 206 or content_range != 'bytes {}-{}/{}'.format(start, end, total):
        raise ValueError('Server did not honor the exact bounded Range request')


def registry_metadata(opener, package, version):
    if not all(re.fullmatch(r'[A-Za-z0-9_.+!-]+', value) for value in [package, version]):
        raise ValueError('Invalid pinned package name/version')
    with opener.open('https://pypi.org/pypi/{}/{}/json'.format(package, version), timeout=20) as response:
        payload = response.read(2_000_001)
    if len(payload) > 2_000_000:
        raise ValueError('Registry metadata limit exceeded')
    return json.loads(payload.decode('utf-8'))


def install_wheel(wheel, directory, opener, tags, curl):
    # Transport hooks live only in this subprocess. Pip still resolves and
    # validates the actual dependency graph; installed pip files are untouched.
    from pip._internal.cli.main import main as pip_main
    from pip._internal.network.download import Downloader, BatchDownloader
    from pip._vendor.packaging.utils import parse_wheel_filename

    def fetch(_downloader, link, location):
        url = urlsplit(link.url)
        if url.scheme != 'https' or url.hostname != 'files.pythonhosted.org' or not link.filename.endswith('.whl'):
            raise ValueError('Dependency download must be a verified official PyPI wheel')
        name, version, _, _ = parse_wheel_filename(link.filename)
        metadata = registry_metadata(opener, str(name), str(version))
        rows = [row for row in metadata.get('urls', []) if row.get('filename') == link.filename]
        record = wheel_record({'urls': rows}, tags)
        if urlsplit(record['url'])._replace(fragment='') != url._replace(fragment=''):
            raise ValueError('Resolver and registry disagree on the artifact URL')
        return str(download(record, directory, opener, curl=curl)), 'application/octet-stream'

    def batch(downloader, links, location):
        for link in links:
            yield link, fetch(downloader, link, location)
    Downloader.__call__ = fetch
    BatchDownloader.__call__ = batch
    return pip_main(['install', str(wheel), '--only-binary=:all:', '--retries', '2', '--timeout', '30',
                     '--progress-bar', 'on', '--disable-pip-version-check'])


def download(row, directory, opener, curl=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / row['filename']
    digest = row['digests']['sha256']

    def verified(path):
        if not path.is_file() or path.stat().st_size != row['size']:
            return False
        checksum = hashlib.sha256()
        with path.open('rb') as reader:
            for data in iter(lambda: reader.read(CHUNK), b''):
                checksum.update(data)
        return checksum.hexdigest() == digest

    if verified(destination):
        print('Verified wheel cache: ' + str(destination), flush=True)
        return destination
    if destination.exists():
        raise ValueError('Existing wheel fails checksum; it will not be overwritten')
    chunks = directory / (row['filename'] + '.parts-' + digest[:12])
    chunks.mkdir(exist_ok=True)
    ranges = [(start, min(start + CHUNK, row['size']) - 1) for start in range(0, row['size'], CHUNK)]

    def fetch(pair):
        start, end = pair
        target = chunks / str(start)
        if target.is_file() and target.stat().st_size == end - start + 1:
            return target
        request = urllib.request.Request(row['url'], headers={
            'Range': 'bytes={}-{}'.format(start, end), 'Accept-Encoding': 'identity'})
        last_error = None
        for _ in range(2):
            try:
                if curl:
                    header = chunks / (str(start) + '.headers')
                    result = subprocess.run([curl, '--noproxy', '*', '--silent', '--show-error', '--fail',
                        '--connect-timeout', '10', '--max-time', '45', '--max-filesize', str(end - start + 1),
                        '--range', '{}-{}'.format(start, end), '--dump-header', str(header),
                        '--output', str(target), row['url']], capture_output=True, timeout=50)
                    if result.returncode:
                        raise OSError(result.stderr.decode('utf-8', errors='replace')[-300:])
                    headers = header.read_text(encoding='iso-8859-1')
                    status = re.findall(r'^HTTP/\S+ (\d+)', headers, re.M)
                    content = re.findall(r'^Content-Range:\s*(.+)$', headers, re.M | re.I)
                    validate_range(int(status[-1]) if status else 0, content[-1].strip() if content else '', start, end, row['size'])
                    if target.stat().st_size != end - start + 1:
                        raise ValueError('Partial or oversized wheel range')
                else:
                    with opener.open(request, timeout=30) as response:
                        validate_range(response.status, response.headers.get('Content-Range'), start, end, row['size'])
                        payload = response.read(end - start + 2)
                        if len(payload) != end - start + 1:
                            raise ValueError('Partial or oversized wheel range')
                    with target.open('wb') as writer:
                        writer.write(payload)
                return target
            except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
                last_error = exc
        raise RuntimeError('Bounded wheel range failed: ' + str(last_error))

    completed = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(fetch, pair): pair for pair in ranges}
        for future in concurrent.futures.as_completed(futures):
            path = future.result()
            completed += path.stat().st_size
            print('Wheel download: {} / {} bytes ({:.1f}%)'.format(completed, row['size'], completed * 100 / row['size']), flush=True)
    partial = directory / (row['filename'] + '.verified-pending')
    checksum = hashlib.sha256()
    with partial.open('wb') as writer:
        for start, _ in ranges:
            with (chunks / str(start)).open('rb') as reader:
                for payload in iter(lambda: reader.read(CHUNK), b''):
                    checksum.update(payload)
                    writer.write(payload)
    if checksum.hexdigest() != digest:
        raise ValueError('Wheel SHA-256 mismatch; no installation permitted')
    os.replace(str(partial), str(destination))
    print('SHA-256 verified: ' + digest, flush=True)
    return destination


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('package')
    parser.add_argument('version')
    parser.add_argument('directory', type=Path)
    parser.add_argument('--install', action='store_true', help='Install only after registry checksum verification')
    args = parser.parse_args()
    if not all(re.fullmatch(r'[A-Za-z0-9_.+!-]+', value) for value in [args.package, args.version]):
        raise ValueError('Invalid pinned package name/version')
    from pip._vendor.packaging.tags import sys_tags
    tags = {(tag.interpreter, tag.abi, tag.platform) for tag in sys_tags()}
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    record = wheel_record(registry_metadata(opener, args.package, args.version), tags)
    print('Official wheel: {} ({} bytes)'.format(record['filename'], record['size']), flush=True)
    curl = shutil.which('curl.exe') or shutil.which('curl')
    wheel = download(record, args.directory, opener, curl=curl)
    if args.install:
        return install_wheel(wheel, args.directory, opener, tags, curl)


if __name__ == '__main__':
    try:
        sys.exit(main() or 0)
    except Exception as exc:
        print('Verified wheel download failed: ' + str(exc), flush=True)
        sys.exit(1)
