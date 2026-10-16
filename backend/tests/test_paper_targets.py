from app.api.reproductions import _paper_metric_excerpt, _table_paper_targets, _verified_paper_targets


def test_paper_targets_require_numeric_evidence_and_known_dataset():
    text = "Table 2 Oncology-Screen: KGANSynergy AUC 0.897, AUPR 0.904. Baseline 0.85."
    result = _verified_paper_targets({
        "Oncology-Screen": {"auc": 0.897, "aupr": 0.904, "f1": 0.99},
        "Unknown": {"auc": 0.85},
    }, text, ["Oncology-Screen"])
    assert result == {"Oncology-Screen": {"auc": 0.897, "aupr": 0.904}}


def test_paper_excerpt_includes_nearby_table_rows():
    text = "Introduction\nBackground\nTable 3 Performance\nOncology-Screen\nAUC 89.7\nDiscussion"
    excerpt = _paper_metric_excerpt(text, ["Oncology-Screen"])
    assert "AUC 89.7" in excerpt


def test_table_parser_uses_author_row_not_baselines():
    text = """Table 3. Performance comparison of KGANSynergy and baselines
DrugCombDB Oncology-Screen
Model AUC-ROC AUC-PR ACC AUC-ROC AUC-PR ACC
GCN 0.6961 0.6797 0.6531 0.6748 0.7076 0.6534
KGANSynergy 0.8951 0.8921 0.8174 0.8911 0.8983 0.8221
"""
    assert _table_paper_targets(text, "KGANSynergy", ["DrugCombDB", "Oncology-Screen"]) == {
        "DrugCombDB": {"auc": 0.8951, "aupr": 0.8921, "accuracy": 0.8174},
        "Oncology-Screen": {"auc": 0.8911, "aupr": 0.8983, "accuracy": 0.8221},
    }
