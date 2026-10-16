"""Official tabular sources; public availability is not paper-version equivalence."""

import re
from urllib.parse import urlparse

import httpx


CATALOG = (
    {"aliases": {"default", "defaultofcreditcardclients"}, "provider": "uci", "id": 350,
     "qualifiers": ("credit", "taiwan"), "slug": "default+of+credit+card+clients"},
    {"aliases": {"shoppers", "onlineshopperspurchasingintentiondataset", "onlineshoppers"},
     "provider": "uci", "id": 468, "slug": "online+shoppers+purchasing+intention+dataset"},
    {"aliases": {"obesity", "obesitylevels"}, "provider": "uci", "id": 544,
     "slug": "estimation+of+obesity+levels+based+on+eating+habits+and+physical+condition"},
    {"aliases": {"satimage", "statloglandsatsatellite"}, "provider": "uci", "id": 146,
     "slug": "statlog+landsat+satellite"},
    {"aliases": {"sick"}, "provider": "openml", "id": 38, "name": "sick",
     "qualifiers": ("thyroid",)},
    {"aliases": {"diabetes", "pimaindiansdiabetes", "pimadiabetes"}, "provider": "openml",
     "id": 37, "name": "diabetes", "qualifiers": ("pima", "pregnan", "glucose")},
)


def dataset_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


def official_identity(dataset: str, paper: str) -> dict | None:
    key = dataset_key(dataset)
    entry = next((row for row in CATALOG if key in row["aliases"]), None)
    if entry is None:
        return None
    # Short names such as Diabetes are ambiguous (classification vs sklearn's
    # regression dataset). A name match is a lead, not sufficient identity.
    qualifiers = entry.get("qualifiers", ()) if key in {"default", "diabetes", "sick"} else ()
    verified = not qualifiers or any(word in paper.lower() for word in qualifiers)
    return {**entry, "identity_verified": verified}


async def discover_official(dataset: str, paper: str) -> tuple[list[dict], dict | None, list[str]]:
    entry = official_identity(dataset, paper)
    if entry is None:
        return [], None, []
    provider, identifier = entry["provider"], entry["id"]
    page = (f"https://archive.ics.uci.edu/dataset/{identifier}/{entry['slug']}" if provider == "uci"
            else f"https://www.openml.org/search?type=data&id={identifier}")
    candidate = {"url": page, "source": f"official_{provider}", "dataset_id": identifier,
                 "identity_verified": entry["identity_verified"], "paper_version_verified": False,
                 "availability": "public_source_identified", "version_status": "unverified"}
    if not entry["identity_verified"]:
        candidate["diagnosis"] = "简称存在歧义；需要论文中的完整名称或特征描述确认，不能自动替换。"
        return [candidate], None, []
    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=12) as client:
            if provider == "uci":
                response = await client.get("https://archive.ics.uci.edu/api/dataset", params={"id": identifier})
                response.raise_for_status()
                data = response.json().get("data") or {}
                url = str(data.get("data_url") or "")
                targets = data.get("target_col") or []
                if not isinstance(targets, list) or len(targets) != 1:
                    raise ValueError("官方元数据未提供唯一目标列，不能猜测标签")
                details = {"format": "csv", "target_column": targets[0],
                           "index_columns": data.get("index_col") or [],
                           "expected_rows": data.get("num_instances"),
                           "expected_features": data.get("num_features"), "version": "current_official"}
            else:
                response = await client.get(f"https://www.openml.org/api/v1/json/data/{identifier}")
                response.raise_for_status()
                data = response.json().get("data_set_description") or {}
                if str(data.get("id")) != str(identifier) or data.get("name") != entry["name"] or data.get("status") != "active":
                    raise ValueError("OpenML 数据集身份或活动状态不匹配")
                url = str(data.get("url") or "")
                target = str(data.get("default_target_attribute") or "")
                if not target or "," in target:
                    raise ValueError("OpenML 元数据未提供唯一目标列")
                details = {"format": "arff", "target_column": target, "index_columns": [],
                           "version": str(data.get("version") or ""), "md5_checksum": data.get("md5_checksum")}
        parsed = urlparse(url)
        if parsed.scheme != "https" or parsed.hostname not in {"archive.ics.uci.edu", "www.openml.org", "openml.org", "api.openml.org"}:
            raise ValueError("官方元数据下载地址不符合来源安全规则")
        if provider == "uci" and (parsed.hostname != "archive.ics.uci.edu"
                                  or not parsed.path.startswith(f"/static/public/{identifier}/")):
            raise ValueError("UCI 下载地址与请求的数据集 ID 不匹配")
        candidate.update(availability="metadata_verified", **details)
        return [candidate], {**candidate, **details, "url": url, "source_url": page,
                             "dataset": dataset, "repository_name": dataset_key(dataset),
                             "adapter": "official_tabular", "download_status": "planned",
                             "preprocessing_status": "requires_paper_verification"}, []
    except (httpx.HTTPError, ValueError, TypeError, KeyError) as exc:
        candidate["availability"] = "metadata_lookup_failed"
        return [candidate], None, [f"{page}: {type(exc).__name__}: {exc}"]


def normalize_tabular(payload, request: dict):
    """Parse real official files. No generated rows, imputation or random split."""
    import pandas as pd
    if request["format"] == "csv":
        frame = pd.read_csv(payload)
    elif request["format"] == "arff":
        from scipy.io import arff
        records, _ = arff.loadarff(payload)
        frame = pd.DataFrame(records)
        for name in frame.columns:
            if frame[name].dtype == object:
                frame[name] = frame[name].map(lambda value: value.decode("utf-8") if isinstance(value, bytes) else value)
                frame[name] = frame[name].replace("?", pd.NA)
    else:
        raise ValueError("不支持的官方数据格式")
    target = request["target_column"]
    if target not in frame or frame.empty or frame.columns.duplicated().any():
        raise ValueError("数据为空、目标列不存在或列名重复")
    if request.get("expected_rows") and len(frame) != int(request["expected_rows"]):
        raise ValueError("下载数据行数与官方元数据不一致")
    index_columns = [name for name in request.get("index_columns", []) if name in frame and name != target]
    frame = frame.drop(columns=index_columns)
    if request.get("expected_features") and len(frame.columns) - 1 != int(request["expected_features"]):
        raise ValueError("下载数据特征数与官方元数据不一致")
    return frame, {"rows": len(frame), "columns": list(frame.columns), "target_column": target,
                   "missing_values": int(frame.isna().sum().sum()),
                   "transformations": {"removed_official_id_columns": index_columns,
                                       "imputation": "none", "encoding": "none", "split": "none"},
                   "paper_version_verified": False, "preprocessing_status": "requires_paper_verification"}
