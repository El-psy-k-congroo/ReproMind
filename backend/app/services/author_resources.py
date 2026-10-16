"""Public author archive metadata. Listing is not a download or dataset proof."""
import json
from pathlib import Path
from .source_selection import native_path


def public_archive_resources(source):
    path = native_path(source)/'zenodo-record.json'
    if not path.is_file() or path.is_symlink() or path.stat().st_size > 500_000:
        return {}
    try:
        record = json.loads(path.read_text(encoding='utf-8'))
        identifier = str(record.get('id') or '')
        if not identifier.isdigit(): return {}
        files = []
        for row in record.get('files',[])[:200]:
            name = str(row.get('key') or '')
            if not name or Path(name).name != name or '\\' in name or name in {'.','..'}: continue
            files.append({'filename': name, 'bytes':row.get('size'), 'checksum':row.get('checksum'),
                          'url':f'https://zenodo.org/api/records/{identifier}/files/{name}/content'})
        return {'record_url': f'https://zenodo.org/records/{identifier}', 'files':files,
                'description':str((record.get('metadata') or {}).get('description') or '')[:14000],
                'downloaded':False, 'training_data_availability_not_implied':True}
    except (ValueError,OSError,TypeError):
        return {}
