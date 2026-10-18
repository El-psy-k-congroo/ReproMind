import json

from app.api.copilot import _bounded_prompt
from app.services.runtime_evidence import execution_logs
from app.services.runtime_evidence import device_log_evidence
from app.services.runtime_log_records import printed_json_values, select_printed_values
from app.services.performance_evidence import performance_question


def test_large_author_json_at_log_end_survives_short_tail_without_exposing_secrets(tmp_path):
    directory=tmp_path/'runs/run/telemetry';directory.mkdir(parents=True)
    record={'cli_args':{'learning_rate':0.2}, 'training_effective':{'embedding_learning_rate':0.001},
            'test_label_counts':{'0':912,'1':909}, 'device':'cuda:0',
            'api_key':'DO_NOT_EXPOSE', 'nested':{'secret_device':'ALSO_PRIVATE'}, 'loss':float('inf')}
    line='Diagnostics: '+json.dumps(record)+' '*2200
    (directory/'reviews-original.log').write_text('noise\n'*15000+line+'\n',encoding='utf-8')
    logs=execution_logs(tmp_path,'run')
    assert 'embedding_learning_rate' not in logs[0]['tail']
    evidence=select_printed_values(logs,'实际编码器学习率、测试标签数量和设备是什么？')
    rendered=json.dumps(evidence)
    assert 'DO_NOT_EXPOSE' not in rendered and 'ALSO_PRIVATE' not in rendered
    assert any(row['path']=='training_effective.embedding_learning_rate' and row['value']==.001 for row in evidence['values'])
    assert any(row['path']=='test_label_counts.0' and row['value']==912 for row in evidence['values'])
    assert not any(row['path']=='loss' for row in evidence['values'])


def test_latest_value_is_kept_and_prompt_reservation_remains_bounded():
    rows=printed_json_values('{"actual_learning_rate":0.1}\n{"actual_learning_rate":0.001}','run.log','recent_tail')
    packet=select_printed_values([{'printed_json_values':rows}],'实际学习率')
    assert len(packet['values'])==1 and packet['values'][0]['value']==.001
    context={'job':{'run_id':'run','status':'completed'},'runtime_printed_values':{'values':packet['values']*500},
             'source_task_evidence':{'named_implementations':[{'file':'model.py','code':'x'*8000}]*10}}
    prompt=_bounded_prompt('实际学习率',context,[],'',15000)
    assert len(prompt)<=15000
    assert json.loads(prompt)['runtime_printed_values']['values']


def test_requested_device_is_not_training_proof_but_observed_model_device_is_retained():
    requested=printed_json_values('{"requested":{"model_device":"cuda:0"},"cuda_available":true}','run.log','recent_tail')
    assert device_log_evidence([{'log':'run.log','printed_json_values':requested}])==[]
    observed=printed_json_values('{"diagnostics":{"model_param_device":"cuda:0"}}','run.log','recent_tail')
    proof=device_log_evidence([{'log':'run.log','printed_json_values':observed}])
    assert proof[0]['kind']=='training_device_gpu'
    assert proof[0]['source']=='author_printed_json_not_operator_trace'


def test_current_data_audit_is_not_forced_into_hypotheses_only_template():
    question='当前准确率约49.64%，请检查每类8样本的数据抽样和标签、权重来源与实际训练参数，区分事实和原因。'
    assert not performance_question(question)
    assert performance_question('当前准确率为什么这么低？')
    assert performance_question('请比较上次和这次准确率差异，检查数据标签和实际训练参数。')
