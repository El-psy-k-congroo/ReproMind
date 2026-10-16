"""Let installed torchvision declare file names and hashes; HTTP transport owns resume."""
import json
import asyncio
import re
from urllib.parse import quote
from pathlib import Path
import httpx

from .download_activity import download_file, verify_file

SUPPORTED = {'MNIST', 'CIFAR10', 'CIFAR100'}
VERIFIED_ARCHIVES = {'cifar-10-python.tar.gz':(
    'md5:c58f30108f718f92721af3b95e74349a',
    'https://huggingface.co/datasets/MIT-OL-AI-D/cifar-10-python/resolve/a48007227f9e2cd0af96175f4afb5ac4965e261b/cifar-10-python.tar.gz')}


async def public_archive_mirrors(dataset, filename):
    """Accept only the exact original archive name at a public, pinned revision.

    These are transport candidates; torchvision's original checksum is mandatory.
    """
    urls = []
    try:
        async with httpx.AsyncClient(timeout=8,follow_redirects=True) as client:
            archive_name = re.sub(r'\.(?:tar\.gz|tar\.bz2|tar\.xz|tgz|zip)$', '', filename, flags=re.I)
            queries = list(dict.fromkeys([archive_name, dataset]))
            async def search(query):
                try:
                    response = await client.get('https://huggingface.co/api/datasets',params={'search':query,'limit':6})
                    response.raise_for_status()
                    return [row['id'] for row in response.json() if not row.get('private') and row.get('id')]
                except (httpx.HTTPError,ValueError,TypeError): return []
            repos = list(dict.fromkeys(repo for rows in await asyncio.gather(*(search(q) for q in queries)) for repo in rows))[:12]
            semaphore = asyncio.Semaphore(4)
            async def inspect(repo):
                try:
                    async with semaphore:
                        result = await client.get('https://huggingface.co/api/datasets/'+repo)
                    result.raise_for_status(); data = result.json()
                    if data.get('private') or data.get('gated') not in (None,False) or not data.get('sha'): return []
                    return [f'https://huggingface.co/datasets/{repo}/resolve/{data["sha"]}/'+quote(row['rfilename'],safe='/')
                            for row in data.get('siblings',[]) if Path(row.get('rfilename','')).name==filename]
                except (httpx.HTTPError,ValueError,TypeError): return []
            for values in await asyncio.gather(*(inspect(repo) for repo in repos)): urls.extend(values)
    except (httpx.HTTPError,ValueError,TypeError): pass
    return list(dict.fromkeys(urls))[:3]


def metadata_script(dataset):
    if dataset not in SUPPORTED: raise ValueError('不支持的数据下载元数据类型')
    return (
        'import json; from torchvision import datasets; '
        f'cls=getattr(datasets,{dataset!r}); '
        'items=([{"filename":name,"checksum":"md5:"+md5,"urls":[base+name for base in cls.mirrors],"directory":cls.__name__+"/raw"} for name,md5 in cls.resources] '
        'if hasattr(cls,"resources") else [{"filename":cls.filename,"checksum":"md5:"+cls.tgz_md5,"urls":[cls.url],"directory":""}]); '
        'print("REPROMIND_DOWNLOAD_METADATA="+json.dumps(items))'
    )


def parse_metadata(output):
    marker = 'REPROMIND_DOWNLOAD_METADATA='
    line = next((line[len(marker):] for line in output.splitlines() if line.startswith(marker)), None)
    if line is None: raise ValueError('数据包未返回下载元数据')
    return json.loads(line)


async def fetch_resources(session, job, workspace, dataset, root, resources, check_pause):
    for resource in resources:
        relative = Path(resource['directory']) / resource['filename']
        if relative.is_absolute() or '..' in relative.parts: raise ValueError('数据包返回的路径越界')
        path = root / relative
        checksum = resource['checksum']
        if path.is_file():
            verify_file(path, checksum)
            continue
        mirrors = await public_archive_mirrors(dataset,resource['filename']) if resource['filename'].endswith(('.tar.gz','.tgz','.zip')) else []
        known = VERIFIED_ARCHIVES.get(resource['filename'])
        if known and checksum == known[0]: mirrors.insert(0,known[1])
        resource['urls'] = [*mirrors,*resource['urls']]
        await download_file(session, job, workspace, dataset, resource['urls'], path,
                            check_pause, resume=True, checksum=checksum)
