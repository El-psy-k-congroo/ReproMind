import json

from app.api.copilot import _bounded_prompt
from app.services.paper_question_evidence import paper_question_passages, whole_paper_passages


def test_method_retrieval_reaches_body_and_preserves_exact_ligatures():
    intro = '\n'.join(['Our framework improves few-shot learning.'] * 60)
    body = ('3 Method\nThe classiﬁcation head uses a logistic regression model.\n'
            'We optimize the cosine-similarity loss with positive and negative pairs.\n'
            'Samples per class are fixed for training.\n')
    paper = (intro + '\n' + body + '\nReferences\nA different classifier uses a neural head.\n'
             'Appendix A Implementation\nThe learning rate is fixed for one epoch.')
    rows = paper_question_passages(paper, '解释句对对比学习、分类头、损失和few-shot采样', 'paper.pdf')
    combined = '\n'.join(row['text'] for row in rows)
    assert 'classiﬁcation head uses a logistic regression' in combined
    assert 'cosine-similarity loss' in combined
    assert 'different classifier' not in combined
    details = paper_question_passages(paper, '附录的学习率和轮数是什么？')
    assert any('learning rate is fixed' in row['text'] for row in details)
    for row in rows:
        assert row['text'] == '\n'.join(paper.splitlines()[row['line_start']-1:row['line_end']])


def test_code_pressure_does_not_evict_all_paper_evidence_or_cut_passages():
    row = {'source': 'paper.pdf', 'line_start': 10, 'line_end': 12,
           'text': 'The classifier is logistic regression. ' * 20}
    context = {'job': {'run_id': 'current', 'status': 'failed'}, 'paper_relevant_evidence': [row] * 8,
               'source_task_evidence': {'named_implementations': [{'file': 'model.py', 'code': 'x' * 1100}] * 30},
               'source_loss_definitions': [{'excerpt': 'y' * 8000}] * 4,
               'paper_excerpt': 'Abstract ' * 500}
    result = _bounded_prompt('论文分类头与当前实现是否相同？', context, [], '', 15000)
    assert len(result) <= 15000
    packet = json.loads(result)['selected_paper_evidence']
    assert packet['passages'] and packet['omitted_due_to_budget'] > 0
    assert all(value == row for value in packet['passages'])
    assert whole_paper_passages([row], 100)['passages'] == []
