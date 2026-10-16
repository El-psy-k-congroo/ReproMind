import asyncio
import hashlib
from types import SimpleNamespace

import pytest
from app.services.author_dataset_paths import torchvision_roots
from app.services.source_selection import native_path
from app.services import torchvision_downloads as transport


@pytest.mark.parametrize('statement,call', [
    ('from torchvision import datasets','datasets.CIFAR100'),
    ('from torchvision import datasets as ds','ds.CIFAR100'),
    ('from torchvision.datasets import CIFAR100 as Images','Images'),
    ('import torchvision as tv','tv.datasets.CIFAR100'),
])
def test_literal_author_loader_path_is_resolved_without_execution(tmp_path,statement,call):
    source=tmp_path/'sources'/'repo';source.mkdir(parents=True)
    (source/'train.py').write_text(statement+f'\ntrain={call}("../inputs",train=True,download=True)\n')
    rows=torchvision_roots(source,['train.py'],source,'CIFAR100',tmp_path)
    assert len(rows)==1 and rows[0]['path']==str(native_path(source.parent/'inputs'))
    assert rows[0]['literal_root']=='../inputs'


def test_loader_path_cannot_escape_workspace_or_be_guessed_from_other_library(tmp_path):
    source=tmp_path/'source';source.mkdir()
    (source/'train.py').write_text('from torchvision import datasets\nx=datasets.CIFAR100("../../external")\n')
    with pytest.raises(ValueError,match='外部目录'):
        torchvision_roots(source,['train.py'],source,'CIFAR100',tmp_path)
    (source/'train.py').write_text('from other import datasets\nx=datasets.CIFAR100("../inputs")\n')
    assert not torchvision_roots(source,['train.py'],source,'CIFAR100',tmp_path)


def test_verified_archive_reuse_preserves_partial_failed_copy(tmp_path,monkeypatch):
    cache=tmp_path/'cache';target=tmp_path/'author-inputs';cache.mkdir();target.mkdir()
    data=b'complete author archive';(cache/'dataset.tar.gz').write_bytes(data)
    (target/'dataset.tar.gz').write_bytes(b'partial')
    resource={'directory':'','filename':'dataset.tar.gz','checksum':'md5:'+hashlib.md5(data).hexdigest(),'urls':[]}
    async def no_network(*args,**kwargs):raise AssertionError('Reuse must not access network')
    monkeypatch.setattr(transport,'download_file',no_network)
    monkeypatch.setattr(transport,'public_archive_mirrors',no_network)
    asyncio.run(transport.fetch_resources(None,SimpleNamespace(id='r'),tmp_path,'Images',target,[resource],lambda _:None,cache_root=cache))
    assert (target/'dataset.tar.gz').read_bytes()==data
    assert list(target.glob('*.repromind-invalid-*'))[0].read_bytes()==b'partial'
    assert (cache/'dataset.tar.gz').read_bytes()==data
    assert resource['verified_cache_reused']==str(cache/'dataset.tar.gz')
