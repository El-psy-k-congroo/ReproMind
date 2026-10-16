"""Separate author inference resources from unavailable full training datasets."""
import re
from pathlib import PurePosixPath
from .author_resources import public_archive_resources
from .source_selection import entry_sources


def requested_phase(goal):
    if re.search(r'采样|\bsampl(?:e|ing)\b',goal,re.I) and not re.search(r'完整训练|训练并|先训练',goal):
        return 'sampling'
    if re.search(r'预训练|公开权重|编码|解码|仅评估|不训练|\binference\b',goal,re.I) and not re.search(r'完整训练|训练并|先训练',goal):
        return 'evaluation'
    return 'training'


def archive_resource_plan(source, entries, datasets):
    catalog = public_archive_resources(source)
    if not catalog or not datasets: return None
    body = '\n'.join(path.read_text(encoding='utf-8',errors='replace') for path in entry_sources(source,entries))
    candidates = re.findall(r'''["'](\.?/[^"'\n]+\.(?:pt|pth|npy|npz|bin|safetensors))["']''',body)
    files=[]
    for resource in catalog['files']:
        paths=[]
        for candidate in candidates:
            path=PurePosixPath(candidate)
            if path.name==resource['filename'] and not path.is_absolute() and '..' not in path.parts:
                paths.append(path.as_posix())
        if paths:
            files.append({**resource,'relative_path':min(paths,key=lambda p:(len(PurePosixPath(p).parts),p))})
    if not files: return None
    return {'dataset_mentions':datasets,'missing_datasets':[], 'available_datasets':[],
            'all_datasets_resolved':True,'automatic_download':True,'download_driver':'author_archive_resources',
            'download_requests':[{'dataset':datasets[0],'adapter':'author_archive_resources','files':files,
                                  'record_url':catalog['record_url'],'paper_version_verified':False}],
            'resource_scope':'公开作者权重与示例测试文件；不代表完整训练数据可用',
            'reason':'非训练任务仅准备作者代码所引用的公开归档文件；逐文件校验，不要求私有原始训练集。'}
