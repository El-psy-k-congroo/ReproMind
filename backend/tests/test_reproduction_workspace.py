import asyncio
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.services import dataset_discovery
from app.services.dataset_discovery import PLANETOID_PARTS
from app.services.reproduction_engine import (
    _environment_plan,
    _dataset_spec,
    _execution_command,
    _infer_python_dependencies,
    _model_name,
    _requirements_compatibility,
    _choose_dataset_variant,
    _align_dataset_commands,
    _repair_known_nell_loader,
    _install_uploaded_datasets,
    _materialize_extracted_dataset,
    _extract_dataset_archive,
)


def test_model_environment_name_is_stable_and_path_safe():
    project = SimpleNamespace(id="12345678-abcd", name="FiLM 论文复现")
    assert _model_name(project, {"model_name": "FiLM"}) == "film"
    assert _model_name(project, {}) == "film"


def test_environment_plan_creates_named_conda_environment(tmp_path: Path):
    workspace = tmp_path / "film-fixture"
    source = workspace / "sources" / "implementation-a"
    source.mkdir(parents=True)
    (source / "train.py").write_text(
        "import torch\nimport numpy as np\nfrom pathlib import Path\n",
        encoding="utf-8",
    )
    manifest = {
        "dependency_files": [],
        "entrypoints": ["train.py"],
    }

    inferred = _infer_python_dependencies(source)
    plan = _environment_plan(source, manifest, workspace, "film")

    assert inferred == ["numpy", "torch"]
    assert Path(plan["prefix"]).name == "film-fixture"
    assert Path(plan["prefix"]).parent.name == "envs"
    assert plan["environment_name"] == "film-fixture"
    assert plan["activation_command"] == "conda activate film-fixture"
    assert "--name" in plan["create_commands"][0]
    assert plan["dependency_source"] == "agent_inferred_imports"
    assert plan["install_commands"][0][-2:] == ["numpy", "torch"]


def test_legacy_linux_wheel_is_replaced_for_current_platform(tmp_path: Path):
    requirements = tmp_path / "requirements.txt"
    requirements.write_text(
        "http://example.test/torch-0.1.11-cp35-cp35m-linux_x86_64.whl\nnumpy\n",
        encoding="utf-8",
    )

    resolved, incompatible = _requirements_compatibility(requirements)

    assert resolved == ["torch", "numpy"]
    assert incompatible == [
        "http://example.test/torch-0.1.11-cp35-cp35m-linux_x86_64.whl"
    ]


def test_small_dataset_is_eligible_for_approved_automatic_download(tmp_path: Path):
    mnist = asyncio.run(_dataset_spec("Experiments use MNIST.", {"datasets": ["MNIST"]}, tmp_path))
    clevr = asyncio.run(_dataset_spec("Experiments use CLEVR.", {"datasets": ["CLEVR"]}, tmp_path, public_lookup=False))

    assert mnist["automatic_download"] is True
    assert mnist["download_driver"] == "torchvision"
    assert clevr["automatic_download"] is False


def test_repository_datasets_take_priority_and_missing_data_is_reported(tmp_path: Path):
    data = tmp_path / "paper-code" / "data"
    data.mkdir(parents=True)
    for part in PLANETOID_PARTS:
        (data / f"ind.alpha.{part}").write_bytes(b"fixture")

    spec = asyncio.run(_dataset_spec(
        "Experiments use Alpha and Beta.",
        {"datasets": ["Alpha", "Beta"]},
        tmp_path, public_lookup=False,
    ))

    assert spec["available_datasets"] == ["Alpha"]
    assert spec["missing_datasets"] == ["Beta"]


def test_nested_author_datasets_and_loader_paths_are_found(tmp_path: Path):
    source = tmp_path / "KGANSynergy-main"
    for name in ("OS", "DrugCombDB"):
        folder = source / "data_set" / name
        folder.mkdir(parents=True)
        (folder / "comb_final.txt").write_text("1,2,3,1,0\n", encoding="utf-8")
        (folder / "kg_final2.txt").write_text("1,2,3\n", encoding="utf-8")
    (source / "load_gat.py").write_text(
        "def readKGData(path='data_set/OS/kg_final2.txt'): pass\n"
        "def readRecData(path='data_set/OS/comb_final.txt'): pass\n", encoding="utf-8",
    )
    (source / "README.md").write_text(
        "DrugCombDB and Oncology-Screen experiments use processed data in data_set.\n",
        encoding="utf-8",
    )
    spec = asyncio.run(_dataset_spec("", {"datasets": ["DrugCombDB", "Oncology-Screen"]}, source))
    assert spec["missing_datasets"] == []
    assert spec["available_datasets"] == ["DrugCombDB", "Oncology-Screen"]
    assert spec["local_dataset_evidence"]["Oncology-Screen"]["path"] == "data_set/OS"
    assert len(spec["local_dataset_evidence"]["Oncology-Screen"]["loader_references"]) == 2

    (source / "run.py").write_text("import load_gat\n", encoding="utf-8")
    plan = _execution_command(
        source, {"entrypoints": ["run.py"]}, "full",
        {"prefix": str(tmp_path / "envs" / "demo")},
        {"datasets": ["DrugCombDB", "Oncology-Screen"]},
        spec, tmp_path / "runs" / "job-1",
    )
    assert {item["dataset"] for item in plan["commands"]} == {"DrugCombDB", "Oncology-Screen"}
    assert plan["unplanned_datasets"] == []
    variant = next(item for item in plan["commands"] if item["dataset"] == "DrugCombDB")
    assert "data_set/DrugCombDB/kg_final2.txt" in (Path(variant["cwd"]) / "load_gat.py").read_text(encoding="utf-8")
    assert "data_set/OS/kg_final2.txt" in (source / "load_gat.py").read_text(encoding="utf-8")
    assert plan["adaptations"][0]["hyperparameters_review_required"] is True


def test_incomplete_author_dataset_is_not_declared_runnable(tmp_path: Path):
    source = tmp_path / "code"
    folder = source / "data_set" / "OS"
    folder.mkdir(parents=True)
    (folder / "comb_final.txt").write_text("1,2,3,1,0\n", encoding="utf-8")
    (source / "load_gat.py").write_text(
        "path='data_set/OS/comb_final.txt'\nother='data_set/OS/kg_final2.txt'\n", encoding="utf-8",
    )
    spec = asyncio.run(_dataset_spec("", {"datasets": ["Oncology-Screen"]}, source, public_lookup=False))
    assert spec["available_datasets"] == []
    assert spec["missing_datasets"] == ["Oncology-Screen"]
    assert spec["local_dataset_evidence"]["Oncology-Screen"]["missing_loader_files"] == [
        "data_set/OS/kg_final2.txt"
    ]


def test_uploaded_archive_fills_exact_missing_author_loader_file(tmp_path: Path):
    workspace = tmp_path / "project"
    source = workspace / "sources" / "code"
    source.mkdir(parents=True)
    archive_path = tmp_path / "uploaded.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("data_set/OS/kg_final2.txt", "1,2,3\n")
    job = SimpleNamespace(
        project_id="12345678-0000-0000-0000-000000000000",
        workspace_path=str(workspace),
        execution_command={"cwd": str(source), "pythonpath": str(source)},
        dataset_spec={"local_dataset_evidence": {
            "Oncology-Screen": {"missing_loader_files": ["data_set/OS/kg_final2.txt"]},
        }},
    )

    class Session:
        async def scalars(self, statement):
            return SimpleNamespace(all=lambda: [SimpleNamespace(storage_path=str(archive_path))])

    installed = asyncio.run(_install_uploaded_datasets(Session(), job))
    assert [item.replace("\\", "/") for item in installed] == ["data_set/OS/kg_final2.txt"]
    assert (source / "data_set" / "OS" / "kg_final2.txt").read_text(encoding="utf-8") == "1,2,3\n"


def test_author_archive_is_mapped_to_nested_loader_paths(tmp_path: Path):
    workspace = tmp_path / "paper"
    source = workspace / "sources" / "code"
    (source / "data_set" / "OS").mkdir(parents=True)
    (source / "data_set" / "OS" / "comb_final.txt").write_text("1,2,3,1,0\n", encoding="utf-8")
    (source / "load_gat.py").write_text(
        "a='data_set/OS/comb_final.txt'\nb='data_set/OS/kg_final2.txt'\n", encoding="utf-8",
    )
    payload = tmp_path / "data.zip"
    with zipfile.ZipFile(payload, "w") as archive:
        archive.writestr("release/data_set/OS/kg_final2.txt", "1,2,3\n")
    extracted = workspace / "data" / "author" / "extract"
    _extract_dataset_archive(payload, extracted)
    installed, ready = _materialize_extracted_dataset(
        extracted, source, source, workspace, "Oncology-Screen",
        {"missing_loader_files": ["data_set/OS/kg_final2.txt"]},
    )
    assert ready
    assert "data_set/OS/kg_final2.txt" in installed
    assert (source / "data_set" / "OS" / "kg_final2.txt").read_text(encoding="utf-8") == "1,2,3\n"


def test_author_archive_rejects_traversal(tmp_path: Path):
    payload = tmp_path / "bad.zip"
    with zipfile.ZipFile(payload, "w") as archive:
        archive.writestr("../outside.txt", "unsafe")
    with pytest.raises(ValueError, match="越界"):
        _extract_dataset_archive(payload, tmp_path / "extract")


def test_paper_and_readme_download_links_have_provenance(tmp_path: Path):
    (tmp_path / "README.md").write_text(
        "Download Alpha dataset: https://example.org/alpha.zip\n", encoding="utf-8",
    )
    spec = asyncio.run(_dataset_spec(
        "Beta data is at https://example.org/beta.tar.gz", {"datasets": ["Alpha", "Beta"]}, tmp_path,
    ))
    assert [(item["dataset"], item["source"]) for item in spec["download_requests"]] == [
        ("Alpha", "author_repository_documentation"), ("Beta", "paper"),
    ]
    assert spec["missing_datasets"] == []


def test_public_lookup_only_auto_selects_complete_loader_compatible_group(tmp_path: Path, monkeypatch):
    class Response:
        status_code = 200

        def __init__(self, data):
            self.data = data

        def raise_for_status(self):
            pass

        def json(self):
            return self.data

    class Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def get(self, url, params=None):
            if url.endswith("/api/datasets"):
                return Response([{"id": "lab/Alpha"}, {"id": "other/Alpha-extra"}])
            return Response({"siblings": [{"rfilename": f"data/ind.alpha.{part}"} for part in PLANETOID_PARTS]})

    monkeypatch.setattr(dataset_discovery.httpx, "AsyncClient", lambda **kwargs: Client())
    spec = asyncio.run(_dataset_spec("", {"datasets": ["Alpha"]}, tmp_path))
    assert spec["missing_datasets"] == []
    request = spec["download_requests"][0]
    assert request["source"] == "public_huggingface_unverified_paper_version"
    assert request["paper_version_verified"] is False
    assert len(request["files"]) == len(PLANETOID_PARTS)


def test_public_lookup_does_not_equate_name_match_with_usable_data(tmp_path: Path, monkeypatch):
    async def candidate(dataset, mirror):
        return ([{"url": "https://huggingface.co/datasets/lab/Beta", "source": "public_huggingface"}], None, [])

    monkeypatch.setattr("app.services.reproduction_engine.search_public_dataset", candidate)
    spec = asyncio.run(_dataset_spec("", {"datasets": ["Beta"]}, tmp_path))
    assert spec["missing_datasets"] == ["Beta"]
    assert spec["dataset_source_candidates"]["Beta"][0]["source"] == "public_huggingface"


def test_zenodo_title_match_is_only_a_candidate(monkeypatch):
    class Response:
        def __init__(self, data):
            self.data = data

        def raise_for_status(self):
            pass

        def json(self):
            return self.data

    class Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def get(self, url, params=None):
            if "zenodo.org" in url:
                return Response({"hits": {"hits": [{"id": 1234, "metadata": {"title": "Beta dataset"}}]}})
            return Response([])

    monkeypatch.setattr(dataset_discovery.httpx, "AsyncClient", lambda **kwargs: Client())
    candidates, request, errors = asyncio.run(dataset_discovery.search_public_dataset("Beta", "https://hf-mirror.com"))
    assert request is None and errors == []
    assert candidates == [{
        "url": "https://zenodo.org/records/1234", "source": "public_zenodo",
        "title": "Beta dataset", "paper_version_verified": False,
    }]


def test_unroutable_multiple_datasets_are_not_silently_marked_planned(tmp_path: Path):
    (tmp_path / "run.py").write_text("# dataset details are in README\nprint('training')\n", encoding="utf-8")
    spec = {"available_datasets": ["Alpha", "Beta"], "dataset_names": {}, "download_requests": [],
            "local_dataset_evidence": {}}
    plan = _execution_command(
        tmp_path, {"entrypoints": ["run.py"]}, "full", {"prefix": str(tmp_path / "env")},
        {"datasets": ["Alpha", "Beta"]}, spec, tmp_path / "runs" / "job",
    )
    assert [item["dataset"] for item in plan["commands"]] == ["default"]
    assert plan["unplanned_datasets"] == ["Alpha", "Beta"]


def test_nell_variant_uses_author_smallest_split_and_repairs_cli(tmp_path: Path):
    data = tmp_path / "sources" / "gcn" / "data"
    data.mkdir(parents=True)
    for name in ("nell.0.001", "nell.0.01", "nell.0.1"):
        for part in PLANETOID_PARTS:
            (data / f"ind.{name}.{part}").write_bytes(b"fixture")
    assert _choose_dataset_variant({"nell.0.1", "nell.0.01", "nell.0.001"}, "nell", "NELL") == "nell.0.001"
    job = SimpleNamespace(execution_command={"commands": [
        {"dataset": "NELL", "cwd": str(data.parent), "argv": ["python", "train.py", "--dataset", "nell"]}
    ]})
    changes = _align_dataset_commands(job)
    assert changes == [{"dataset": "NELL", "from": "nell", "to": "nell.0.001"}]
    assert job.execution_command["commands"][0]["argv"][-1] == "nell.0.001"


def test_known_nell_loader_patch_is_scoped_and_keeps_backup(tmp_path: Path):
    source = tmp_path / "sources" / "gcn"
    source.mkdir(parents=True)
    loader = source / "utils.py"
    original = """    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]
"""
    loader.write_text("def load_data(dataset_str):\n" + original, encoding="utf-8")
    job = SimpleNamespace(repository_url="https://github.com/tkipf/gcn", workspace_path=str(tmp_path))
    assert _repair_known_nell_loader(job, source, "ValueError: row index 10098 out of bounds")
    assert "ReproMind NELL adapter" in loader.read_text(encoding="utf-8")
    assert (source / "utils.py.repromind-original").read_text(encoding="utf-8") == "def load_data(dataset_str):\n" + original
    compile(loader.read_text(encoding="utf-8"), str(loader), "exec")
    assert not _repair_known_nell_loader(job, tmp_path, "ValueError: row index 10098 out of bounds")
