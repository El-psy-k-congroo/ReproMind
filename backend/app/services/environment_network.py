"""Scoped recovery from a failed local proxy, with the approved index unchanged."""
from urllib.parse import urlparse
from urllib.request import getproxies
from pathlib import Path


def direct_pip_retry(command, output):
    pip_install = list(command[1:4]) == ['-m', 'pip', 'install']
    registry_install = (len(command) >= 6 and command[-1] == '--install'
                        and Path(command[1]).resolve() == Path(__file__).with_name('registry_wheel_download.py').resolve())
    if not (pip_install or registry_install) or 'ProxyError' not in output:
        return None
    proxies = getproxies()
    endpoints = [urlparse(value if '://' in value else 'http://'+value)
                 for key, value in proxies.items() if key in {'http', 'https', 'all'}]
    if not endpoints or any(endpoint.hostname not in {'127.0.0.1', 'localhost', '::1'} for endpoint in endpoints):
        return None
    return {'NO_PROXY': '*', 'no_proxy': '*', 'HTTP_PROXY': '', 'HTTPS_PROXY': '',
            'ALL_PROXY': '', 'http_proxy': '', 'https_proxy': '', 'all_proxy': ''}
