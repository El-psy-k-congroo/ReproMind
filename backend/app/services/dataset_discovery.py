"""Evidence-first discovery of datasets in a paper's own source tree."""

import os
import re
from itertools import islice
from pathlib import Path
from urllib.parse import quote

import httpx


DATA_ROOTS = {"data", "dataset", "datasets", "data_set", "data-set", "datafiles"}
IGNORED_DIRS = {".git", ".conda", "node_modules", "__pycache__", ".venv", "runs"}
PLANETOID_PARTS = {"x", "y", "tx", "ty", "allx", "ally", "graph", "test.index"}


def _key(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


def _aliases(label: str) -> set[str]:
    parts = re.findall(r"[A-Za-z0-9]+", label)
    aliases = {_key(label)}
    if len(parts) >= 2:
        aliases.add("".join(part[0] for part in parts).lower())
    return aliases - {""}


def _real_payload(path: Path) -> bool:
    if path.is_symlink() or not path.is_file() or path.stat().st_size == 0 or path.name.startswith("."):
        return False
    if path.stat().st_size < 300:
        try:
            if path.read_bytes().startswith(b"version https://git-lfs.github.com/spec/v1"):
                return False
        except OSError:
            return False
    return path.suffix.lower() not in {".md", ".rst", ".html"}


def _source_paths(source: Path) -> tuple[list[Path], list[Path]]:
    roots: list[Path] = []
    python_files: list[Path] = []
    scanned = 0
    for base, dirs, files in os.walk(source, followlinks=False):
        root = Path(base)
        dirs[:] = [name for name in dirs if name not in IGNORED_DIRS and not (root / name).is_symlink()]
        if root.name.lower() in DATA_ROOTS:
            roots.append(root)
            # Data files are inspected separately; avoid traversing large datasets.
            dirs[:] = []
            continue
        python_files.extend(root / name for name in files if name.endswith(".py"))
        scanned += len(files)
        if scanned > 5000:
            break
    return roots[:30], python_files[:120]


def _loader_references(source: Path, python_files: list[Path]) -> set[str]:
    references: set[str] = set()
    expression = re.compile(r"[\"']((?:data|data_set|data-set|dataset|datasets|datafiles)/[^\"'\n]+)[\"']", re.I)
    for path in python_files:
        try:
            content = path.read_text(encoding="utf-8", errors="replace")[:120_000]
        except OSError:
            continue
        for match in expression.finditer(content):
            value = match.group(1).replace("\\", "/")
            if "{" not in value and "}" not in value and ".." not in Path(value).parts:
                references.add(value)
    return references


def inspect_local_datasets(source: Path, requested: list[str]) -> dict[str, dict]:
    """Return only datasets with real files, including nested and aliased folders."""
    roots, python_files = _source_paths(source)
    references = _loader_references(source, python_files)
    found: dict[str, dict] = {}
    for label in requested:
        aliases = _aliases(label)
        candidates: list[dict] = []
        for root in roots:
            for directory in root.iterdir():
                if not directory.is_dir() or directory.is_symlink() or _key(directory.name) not in aliases:
                    continue
                files = list(islice((path for path in directory.rglob("*") if _real_payload(path)), 100))
                if not files:
                    continue
                relative = directory.relative_to(source).as_posix()
                related = sorted(ref for ref in references if ref.startswith(relative + "/"))
                missing = [ref for ref in related if not (source / ref).is_file()]
                candidates.append({
                    "path": relative, "repository_name": directory.name,
                    "files": [path.relative_to(source).as_posix() for path in files[:20]],
                    "file_count": len(files), "loader_references": related,
                    "missing_loader_files": missing,
                    "source": "author_repository", "ready": not missing,
                })
        if candidates:
            candidates.sort(key=lambda item: (not item["ready"], -len(item["loader_references"]), -item["file_count"]))
            found[label] = candidates[0]
            continue

        # Planetoid/GCN datasets are flat groups of ind.<name>.* files.
        for root in roots:
            files = {path.name.lower(): path for path in root.iterdir() if path.is_file()}
            groups: dict[str, set[str]] = {}
            for name, path in files.items():
                match = re.fullmatch(r'ind\.(.+)\.(test\.index|allx|ally|graph|tx|ty|x|y)', name)
                if match and _real_payload(path):
                    dataset, part = match.groups()
                    # NELL has several real groups, e.g. nell.0.001. Never mix their parts.
                    exact = _key(dataset) in aliases
                    generic = '.' not in label and _key(dataset.split('.')[0]) in aliases
                    if exact or generic:
                        groups.setdefault(dataset, set()).add(part)
            candidates = []
            for dataset, parts in sorted(groups.items()):
                prefix = f'ind.{dataset}.'
                candidates.append({
                    "path": root.relative_to(source).as_posix(), "repository_name": dataset,
                    "dataset_variant": dataset,
                    "files": [files[prefix + part].relative_to(source).as_posix() for part in sorted(parts)[:20]],
                    "file_count": len(parts), "loader_references": [],
                    "missing_loader_files": sorted(PLANETOID_PARTS - parts),
                    "source": "author_repository", "ready": PLANETOID_PARTS <= parts,
                })
            if candidates:
                candidates.sort(key=lambda row: (not row['ready'], -row['file_count'], row['dataset_variant']))
                found[label] = candidates[0]
            if label in found:
                break
    return found


def document_dataset_links(text: str, dataset: str) -> list[str]:
    """Find URLs in dataset-specific README/paper context, not unrelated links."""
    aliases = _aliases(dataset)
    urls: list[str] = []
    lines = text.splitlines()
    for index, line in enumerate(lines):
        context = " ".join(lines[max(0, index - 1):min(len(lines), index + 2)])
        if not any(alias in _key(context) for alias in aliases):
            continue
        for url in re.findall(r"https?://[^\s)\]>]+", line):
            cleaned = url.rstrip(".,;:)]}")
            if cleaned not in urls:
                urls.append(cleaned)
    return urls


def directly_downloadable(url: str) -> bool:
    path = url.split("?", 1)[0].lower()
    return path.endswith((".zip", ".tar", ".tar.gz", ".tgz", ".gz")) or "/resolve/" in path or bool(re.match(r'https://drive\.google\.com/(?:file/d/|uc\?)',url))


def shared_data_archive(text):
    """A repository-wide data archive explicitly documented by its authors."""
    links = []
    for line in text.splitlines():
        if not re.search(r'data|dataset', line, re.I) or not re.search(r'download|wget|curl|下载',line,re.I): continue
        for raw in re.findall(r'https?://[^\s`<>]+',line):
            url = raw.rstrip(').,;')
            if directly_downloadable(url) and not re.search(r'checkpoint|pretrain|weight|model',url,re.I): links.append(url)
    links = list(dict.fromkeys(links))
    return links[0] if len(links) == 1 else None


async def search_public_dataset(dataset: str, mirror_url: str) -> tuple[list[dict], dict | None, list[str]]:
    """Search public Hub metadata; auto-fetch only a complete, loader-compatible file group."""
    candidates: list[dict] = []
    errors: list[str] = []
    for endpoint in ("https://huggingface.co", mirror_url.rstrip("/")):
        if endpoint not in {"https://huggingface.co", "https://hf-mirror.com"}:
            continue
        try:
            async with httpx.AsyncClient(follow_redirects=True, timeout=7) as client:
                response = await client.get(f"{endpoint}/api/datasets", params={"search": dataset, "limit": 8})
                response.raise_for_status()
                results = response.json()
                if not isinstance(results, list):
                    raise ValueError("数据集搜索响应格式不正确")
                for result in results:
                    repository = str(result.get("id") or "") if isinstance(result, dict) else ""
                    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository):
                        continue
                    if _key(repository.rsplit("/", 1)[-1]) != _key(dataset):
                        continue
                    page = f"https://huggingface.co/datasets/{repository}"
                    candidates.append({"url": page, "source": "public_huggingface", "repository": repository,
                                       "paper_version_verified": False})
                    details = await client.get(f"{endpoint}/api/datasets/{repository}")
                    if details.status_code != 200:
                        continue
                    siblings = details.json().get("siblings") or []
                    files = [str(item.get("rfilename") or "") for item in siblings if isinstance(item, dict)]
                    groups: dict[str, dict[str, str]] = {}
                    for filename in files:
                        match = re.search(r"(?:^|/)ind\.([A-Za-z0-9_.-]+)\.(x|y|tx|ty|allx|ally|graph|test\.index)$", filename, re.I)
                        if match:
                            groups.setdefault(match.group(1).lower(), {})[match.group(2).lower()] = filename
                    def variant_order(value: str) -> tuple[float, str]:
                        suffix = value[len(dataset) + 1:] if value.lower().startswith(dataset.lower() + ".") else ""
                        return (float(suffix) if re.fullmatch(r"\d+(?:\.\d+)?", suffix) else float("inf"), value)

                    for variant in sorted(groups, key=variant_order):
                        parts = groups[variant]
                        matches = _key(variant) == _key(dataset) or variant.lower().startswith(dataset.lower() + ".")
                        if matches and PLANETOID_PARTS <= parts.keys():
                            base = "https://hf-mirror.com" if mirror_url.rstrip("/") == "https://hf-mirror.com" else endpoint
                            request = {
                                "dataset": dataset, "repository_name": variant,
                                "source": "public_huggingface_unverified_paper_version",
                                "paper_version_verified": False,
                                "files": [{"filename": Path(path).name,
                                           "url": f"{base}/datasets/{repository}/resolve/main/{quote(path, safe='/')}"}
                                          for path in parts.values()],
                            }
                            return candidates, request, errors
            break
        except (httpx.HTTPError, ValueError, TypeError, KeyError) as exc:
            errors.append(f"{endpoint}: {type(exc).__name__}: {exc}")
    # Zenodo is a metadata lead unless the paper itself identifies a record.
    # A title match alone must never be treated as the paper's exact version.
    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=7) as client:
            response = await client.get("https://zenodo.org/api/records/", params={"q": f'"{dataset}"', "size": 5})
            response.raise_for_status()
            payload = response.json()
            records = payload.get("hits", {}).get("hits", []) if isinstance(payload, dict) else payload
            for record in records[:5] if isinstance(records, list) else []:
                if not isinstance(record, dict):
                    continue
                title = str((record.get("metadata") or {}).get("title") or "")
                record_id = str(record.get("id") or "")
                if not record_id.isdigit() or _key(dataset) not in _key(title):
                    continue
                candidates.append({
                    "url": f"https://zenodo.org/records/{record_id}",
                    "source": "public_zenodo", "title": title,
                    "paper_version_verified": False,
                })
    except (httpx.HTTPError, ValueError, TypeError, KeyError) as exc:
        errors.append(f"https://zenodo.org: {type(exc).__name__}: {exc}")
    return candidates, None, errors
