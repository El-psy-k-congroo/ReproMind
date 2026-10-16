from app.services.telemetry import parse_experiment_log
from app.services.tuning import validation_score
import asyncio
from types import SimpleNamespace
from app.api import copilot
from app.models import ReproductionStatus


def test_epoch_header_binds_partition_lines_but_not_final_evaluation(tmp_path):
    path=tmp_path/'example-original.log'
    path.write_text('''>>> Epoch 1 | 00:00:00
[train] loss = 0.4
[val  ] score = 0.80 | acc = 0.80
[test ] score = 0.79 | acc = 0.79
>>> Epoch 2 | 00:01:00
[train] loss = 0.3
[val  ] score = 0.75 | acc = 0.75
[test ] score = 0.76 | acc = 0.76
Final evaluation
[test ] acc = 0.81
''')
    points=parse_experiment_log(path)['points']
    assert points==[{'epoch':1,'train_loss':.4,'val_acc':.8,'test_acc':.79},
                    {'epoch':2,'train_loss':.3,'val_acc':.75,'test_acc':.76}]
    assert validation_score(path.read_text()+'Loading 1st epoch\n','accuracy')==.8


def test_bracketed_epoch_summaries_keep_training_and_test_metrics(tmp_path):
    path=tmp_path/'example-original.log'
    path.write_text('* Epoch: [0/2] Top 1-err 79.0 Top 5-err 55.0 Train Loss 4.0\n'
                    '* Epoch: [0/2] Top 1-err 70.0 Top 5-err 50.0 Test Loss 3.0\n')
    point=parse_experiment_log(path)['points'][0]
    assert point['epoch']==0 and point['train_loss']==4 and point['test_loss']==3
    assert point['train_top1error']==79 and point['test_top1error']==70


def test_loss_only_without_explicit_epoch_is_not_assigned_a_fabricated_axis(tmp_path):
    path=tmp_path/'example-original.log';path.write_text('loss: 0.5\nloss: 0.4\n')
    assert parse_experiment_log(path)['points']==[]


def test_status_log_request_shows_selected_run_tail_without_claiming_success(tmp_path,monkeypatch):
    log=tmp_path/'runs'/'selected'/'telemetry'/'example-original.log'
    log.parent.mkdir(parents=True);log.write_text('>>> Epoch 3\n[train] loss = 0.3\nlast raw progress marker\n')
    job=SimpleNamespace(id='selected',status=ReproductionStatus.executing,current_stage='executing',
        message='Execution Agent 正在运行 Example（1/1）',result_json={},
        execution_command={'commands':[{'dataset':'Example'}]})
    class Session:
        async def scalar(self,*args):return job
    monkeypatch.setattr(copilot,'_safe_workspace',lambda _:tmp_path)
    answer=asyncio.run(copilot._status_answer(Session(),SimpleNamespace(id='p'),'selected','当前进度和最近日志，是否卡住？'))
    assert 'last raw progress marker' in answer and 'example-original.log' in answer
    assert 'epoch 3' in answer and '不能宣布运行成功' in answer
