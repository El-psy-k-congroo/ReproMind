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


def test_current_method_parameters_outrank_baseline_and_keep_section_provenance():
    paper=('4.1 Baselines\nGeneric model uses a learning rate of 2e-5 and batch size of 4.\n'
           + '\n'.join(['Discussion of baseline details.']*20)
           + '\n4.2 Experimental Setup\nWe fine-tune TargetModel with cosine-similarity loss.\n'
             'Its learning rate is 1e-3, with batch size 16, for one epoch.\n')
    rows=paper_question_passages(paper,'当前方法的学习率是什么？','paper.pdf',limit=1,method_name='TargetModel')
    assert '1e-3' in rows[0]['text'] and '2e-5' not in rows[0]['text']
    assert rows[0]['section_heading']=='4.2 Experimental Setup'
    assert paper.splitlines()[rows[0]['section_line']-1]==rows[0]['section_heading']
    adjacent=paper.replace('\n'.join(['Discussion of baseline details.']*20), '')
    row=paper_question_passages(adjacent,'学习率是什么？',limit=1,method_name='TargetModel')[0]
    assert '1e-3' in row['text'] and '2e-5' not in row['text']
    comparison=paper_question_passages(paper,'比较当前方法与基线的学习率',limit=2,method_name='TargetModel')
    assert len(comparison)==2
    assert {'4.1 Baselines','4.2 Experimental Setup'}=={row['section_heading'] for row in comparison}
    assert '2e-5' in '\n'.join(row['text'] for row in comparison)
