import csv
import io
import json
import zipfile
from pathlib import Path
from typing import Any

import yaml
from pypdf import PdfReader


TEXT_EXTENSIONS = {
    ".txt", ".md", ".py", ".js", ".ts", ".tsx", ".vue", ".java", ".go", ".rs",
    ".toml", ".ini", ".cfg", ".sh", ".ps1", ".sql", ".xml",
}
STRUCTURED_EXTENSIONS = {".yaml", ".yml", ".json", ".csv", ".tsv"}
MAX_TEXT_CHARS = 400_000
MAX_ZIP_FILES = 200


def _decode(data: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    if isinstance(value, dict):
        for key, item in value.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            flattened.update(_flatten(item, child))
    elif isinstance(value, list):
        flattened[prefix] = value
    else:
        flattened[prefix] = value
    return flattened


def parse_bytes(filename: str, data: bytes) -> tuple[str, dict[str, Any]]:
    suffix = Path(filename).suffix.lower()
    metadata: dict[str, Any] = {"extension": suffix}

    if suffix == ".pdf":
        reader = PdfReader(io.BytesIO(data))
        pages: list[str] = []
        page_offsets: list[dict[str, int]] = []
        cursor = 0
        for index, page in enumerate(reader.pages, start=1):
            page_text = page.extract_text() or ""
            pages.append(f"\n--- Page {index} ---\n{page_text}")
            page_offsets.append({"page": index, "start": cursor, "length": len(page_text)})
            cursor += len(page_text)
        text = "".join(pages)
        metadata.update({"page_count": len(reader.pages), "page_offsets": page_offsets})
        return text[:MAX_TEXT_CHARS], metadata

    if suffix == ".zip":
        chunks: list[str] = []
        names: list[str] = []
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            safe_files = [entry for entry in archive.infolist() if not entry.is_dir()][:MAX_ZIP_FILES]
            for entry in safe_files:
                entry_suffix = Path(entry.filename).suffix.lower()
                if entry.file_size > 1_000_000 or entry_suffix not in TEXT_EXTENSIONS | STRUCTURED_EXTENSIONS:
                    continue
                raw = archive.read(entry)
                content = _decode(raw)
                names.append(entry.filename)
                chunks.append(f"\n--- File: {entry.filename} ---\n{content}")
                if sum(len(chunk) for chunk in chunks) >= MAX_TEXT_CHARS:
                    break
        metadata.update({"file_count": len(names), "files": names})
        return "".join(chunks)[:MAX_TEXT_CHARS], metadata

    text = _decode(data)
    if suffix in {".yaml", ".yml"}:
        parsed = yaml.safe_load(text) or {}
        metadata["values"] = _flatten(parsed)
    elif suffix == ".json":
        parsed = json.loads(text)
        metadata["values"] = _flatten(parsed)
    elif suffix in {".csv", ".tsv"}:
        delimiter = "\t" if suffix == ".tsv" else ","
        rows = list(csv.DictReader(io.StringIO(text), delimiter=delimiter))
        metadata.update({"columns": list(rows[0].keys()) if rows else [], "row_count": len(rows)})
    elif suffix not in TEXT_EXTENSIONS:
        metadata["warning"] = "当前文件类型未进行正文解析，仅保留原文件。"
        text = ""
    return text[:MAX_TEXT_CHARS], metadata
