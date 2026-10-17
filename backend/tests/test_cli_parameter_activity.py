import sys
from types import SimpleNamespace

from app.services.cli_parameter_activity import parameter_activity
from app.services.tuning import build_tuning_plans
from app.api.copilot import _tuning_plan_answer

BODY = '''import argparse
p=argparse.ArgumentParser()
p.add_argument('--head',default='linear')
p.add_argument('--lr',type=float,default=0.01)
p.add_argument('--batch_size',type=int,default=16)
args=p.parse_args()
if args.head == 'neural':
    train(learning_rate=args.lr)
else:
    train(batch_size=args.batch_size)
'''


def test_only_proven_inactive_cli_reads_are_excluded():
    inactive = parameter_activity(BODY, ['python', 'train.py'])
    assert inactive['lr']['inactive']
    assert not inactive['batch_size']['inactive']
    active = parameter_activity(BODY, ['python', 'train.py', '--head=neural'])
    assert not active['lr']['inactive']
    assert active['batch_size']['inactive']
    unknown = BODY.replace("args.head == 'neural'", 'runtime_condition()')
    assert not parameter_activity(unknown, [])['lr']['inactive']


def test_opaque_namespace_use_or_mutation_disables_inactive_proof():
    assert parameter_activity(BODY + '\nother_training(args)\n', []) == {}
    assert parameter_activity(BODY + '\nargs.head = select_head()\n', []) == {}
    assert parameter_activity(BODY + '\nargs = other_options\n', []) == {}
    assert parameter_activity(BODY + '\ndef export(args):\n return args\n', []) == {}
    assert parameter_activity(BODY + '\nforward(args)\ndef forward(options):\n train(options.lr)', []) == {}
    # A same-name local helper is visible to the analysis and remains safe.
    local = BODY + '\nforward(args)\ndef forward(args):\n print(args.batch_size)\n'
    assert parameter_activity(local, [])['lr']['inactive']


def test_plan_does_not_spend_trials_on_inactive_parameter(tmp_path):
    source = tmp_path/'source'; source.mkdir()
    (source/'train.py').write_text(BODY, encoding='utf-8')
    job = SimpleNamespace(workspace_path=str(tmp_path), code_spec={}, execution_command={
        'pythonpath':str(source), 'entrypoint_candidates':['train.py'],
        'commands':[{'dataset':'reviews', 'argv':[sys.executable, 'train.py']}]})
    plan = build_tuning_plans(job)[0]
    assert plan['trials'] and all('lr' not in trial for trial in plan['trials'])
    assert any(row['parameter']=='lr' for row in plan['inactive_cli_parameters'])
    assert any(row['parameter']=='batch_size' for row in plan['space'])
    answer = _tuning_plan_answer({'proposed_tuning_plans_not_executed':[plan]}, '搜索方案')
    assert '已排除本次未启用分支' in answer and '`lr`' in answer
    assert '| `lr` |' not in answer
