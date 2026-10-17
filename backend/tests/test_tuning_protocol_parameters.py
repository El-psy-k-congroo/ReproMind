import sys
from types import SimpleNamespace
from app.services.tuning import build_tuning_plans


def test_seed_and_fewshot_sample_count_remain_protocol_not_hpo_dimensions(tmp_path):
    source=tmp_path/'source';source.mkdir()
    (source/'train.py').write_text("import argparse\np=argparse.ArgumentParser()\n"
        "p.add_argument('--lr',type=float,default=0.01,choices=[0.001,0.01,0.1])\n"
        "p.add_argument('--seed',type=int,default=42,choices=[1,20,40,42])\n"
        "p.add_argument('--sample_sizes',type=int,default=8,choices=[8,16,32])\n")
    job=SimpleNamespace(workspace_path=str(tmp_path),code_spec={},execution_command={
        'pythonpath':str(source),'entrypoint_candidates':['train.py'],
        'commands':[{'dataset':'Reviews','argv':[sys.executable,'train.py','--seed','42','--sample_sizes','8']}]})
    plan=build_tuning_plans(job)[0]
    assert plan['trials'] and all(set(trial)=={'lr'} for trial in plan['trials'])
    assert {row['parameter']:row['baseline'] for row in plan['fixed_protocol_parameters']}=={'seed':42,'sample_sizes':8}
    assert plan['declared_parameters']['seed']==42
