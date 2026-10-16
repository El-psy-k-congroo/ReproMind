from app.api.copilot import _tuning_results_answer, _tuning_results_question


def test_actual_search_records_do_not_route_to_unexecuted_plan_or_actions():
    assert _tuning_results_question('请核对本轮6组已执行调参的实际记录，验证分数、测试分数、哪组被采用？')
    assert not _tuning_results_question('请建议调参的搜索空间和范围')
    assert not _tuning_results_question('请重新运行调参结果对应的实验')


def test_selected_candidate_is_not_reported_as_a_replaced_baseline():
    context={'job': {'run_id':'chosen', 'tuning_attempts_count':2,
        'tuning_sessions':[{'id':'session','status':'completed'}],
        'tuning_attempts':[{'session_id':'session','trial':1,'dataset':'Example','parameters':{'lr':.01},
                           'validation_metric':'accuracy','validation_score':.8,'test_metrics':{'accuracy':.6},'return_code':0,'selected':True},
                          {'session_id':'session','trial':2,'dataset':'Example','parameters':{'lr':.1},
                           'validation_metric':'accuracy','validation_score':.7,'test_metrics':{'accuracy':.9},'return_code':0,'selected':False}],
        'dataset_results':[{'dataset':'Example','metrics':{'accuracy':.75},'baseline_metrics':{'accuracy':.75},
                            'verdict':'matched','tuning_selected':False}]}}
    text=_tuning_results_answer(context)
    assert 'accuracy=0.8' in text and 'accuracy=0.9' in text
    assert '没有以调参候选覆盖' in text and 'accuracy=0.75' in text
    assert '测试分数仅展示，不参与选参' in text
