from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from app.api.experiment_branches import _complete_known_gpu_patches, _fresh_run_command, _gpu_preview_issues


def _base(source: Path):
    return SimpleNamespace(
        execution_command={"entrypoint_candidates": ["run.py"]},
        paper_spec={"datasets": ["Oncology-Screen", "DrugCombDB"]},
        dataset_spec={"local_dataset_evidence": {
            "Oncology-Screen": {"ready": True, "path": "data_set/OS",
                                "loader_references": ["data_set/OS/train.csv"]},
            "DrugCombDB": {"ready": True, "path": "data_set/DrugCombDB",
                           "loader_references": []},
        }},
    )


def test_rerun_rebuilds_variant_from_current_source_and_selects_one_dataset(tmp_path: Path):
    source = tmp_path / "source"
    (source / "data_set" / "OS").mkdir(parents=True)
    (source / "data_set" / "DrugCombDB").mkdir(parents=True)
    (source / "data_set" / "OS" / "train.csv").write_text("value", encoding="utf-8")
    (source / "data_set" / "DrugCombDB" / "train.csv").write_text("value", encoding="utf-8")
    (source / "run.py").write_text("# revised code\nDATA = 'data_set/OS/train.csv'\n", encoding="utf-8")
    environment = {"prefix": str(tmp_path / "conda")}
    planned = _fresh_run_command(_base(source), source, tmp_path / "runs" / "new-run",
                                 environment, ["DrugCombDB"])
    assert [item["dataset"] for item in planned["commands"]] == ["DrugCombDB"]
    variant = Path(planned["commands"][0]["cwd"])
    assert "new-run" in str(variant)
    assert "# revised code" in (variant / "run.py").read_text(encoding="utf-8")
    assert "data_set/DrugCombDB/train.csv" in (variant / "run.py").read_text(encoding="utf-8")


def test_rerun_rejects_unplanned_dataset(tmp_path: Path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "run.py").write_text("print('ok')\n", encoding="utf-8")
    with pytest.raises(HTTPException, match="没有已验证的可执行数据集"):
        _fresh_run_command(_base(source), source, tmp_path / "runs" / "new-run",
                           {"prefix": str(tmp_path / "conda")}, ["Missing"])


def test_cli_rerun_preview_resolves_author_file_variant_without_changing_source(tmp_path):
    source = tmp_path / 'source'
    data = source / 'data'
    data.mkdir(parents=True)
    entry = "import argparse\np = argparse.ArgumentParser()\np.add_argument('--dataset')\n"
    (source / 'run.py').write_text(entry, encoding='utf-8')
    for part in ['x', 'tx', 'allx', 'y', 'ty', 'ally', 'graph', 'test.index']:
        (data / f'ind.custom.0.01.{part}').write_text('author data', encoding='utf-8')
    base = SimpleNamespace(execution_command={'entrypoint_candidates': ['run.py']},
        paper_spec={'datasets': ['Custom']}, dataset_spec={'available_datasets': ['Custom']})
    planned = _fresh_run_command(base, source, tmp_path / 'runs' / 'new',
                                {'prefix': str(tmp_path / 'conda')}, ['Custom'])
    assert planned['commands'][0]['argv'][-1] == 'custom.0.01'
    assert '--dataset custom.0.01' in planned['display']
    assert (source / 'run.py').read_text(encoding='utf-8') == entry


def test_gpu_static_check_rejects_cpu_bound_code(tmp_path: Path):
    (tmp_path / "run.py").write_text("model = Model()\n", encoding="utf-8")
    (tmp_path / "model.py").write_text("x = value.to('cpu')\n", encoding="utf-8")
    issues = _gpu_preview_issues(tmp_path, {})
    assert any("训练入口" in item for item in issues)
    assert any("CPU" in item for item in issues)


def test_gpu_static_check_accepts_basic_device_mapping(tmp_path: Path):
    (tmp_path / "run.py").write_text(
        "import torch\ndevice = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n"
        "model = Model().to(device)\n", encoding="utf-8")
    (tmp_path / "model.py").write_text("self.layer = nn.Linear(4, 4)\n", encoding="utf-8")
    assert _gpu_preview_issues(tmp_path, {}) == []


def test_known_gpu_patches_stay_in_preview_and_compile():
    body = ("class Model:\n"
            "    def __init__(self):\n"
            "        self.dim = 4\n"
            "        self.WW = nn.Linear(4, 4)\n"
            "    def get_neighbors_cell(self, cells):\n"
            "        entities = [cells]\n"
            "        x = torch.LongTensor(self.adj_entity[entities[h]]).to('cpu')\n"
            "    def get_neighbors_drug(self, drugs):\n"
            "        entities = [drugs]\n"
            "        x = torch.LongTensor(self.adj_relation[entities[h]]).to('cpu')\n"
            "    def sum_aggregator(self, embs):\n"
            "        pass\n"
            "    def GATMessagePass(self, embs):\n"
            "        Wx = nn.Linear(self.dim, self.dim)\n"
            "        emb_i = Wx(emb_i.sum(dim=1))\n")
    updated, changes = _complete_known_gpu_patches(body, "model.py")
    assert len(changes) == 4
    assert body.count(".to('cpu')") == 2
    assert ".to('cpu')" not in updated
    assert "self.message_projection" in updated
    compile(updated, "model.py", "exec")


def test_gpu_preview_replaces_unregistered_layer_even_if_moved_to_cuda():
    body = ("class Model:\n"
            "    def __init__(self):\n"
            "        self.dim = 4\n"
            "        self.message_projection = nn.Linear(self.dim, self.dim)\n"
            "    def GATMessagePass(self, embs):\n"
            "        Wx = nn.Linear(self.dim, self.dim).to(emb_i.device)\n"
            "        emb_i = Wx(emb_i.sum(dim=1))\n")
    updated, changes = _complete_known_gpu_patches(body, "model.py")
    assert len(changes) == 1
    assert "Wx = nn.Linear" not in updated
    assert "emb_i = self.message_projection" in updated
