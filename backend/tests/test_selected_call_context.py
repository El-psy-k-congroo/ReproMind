import ast

from app.services.source_call_context import selected_call_packets, current_path_only, inactive_excerpt
from app.services.source_class_context import class_bindings, factory_bindings
from app.services.source_quotes import checked_source_quotes


def sources():
    bodies = {'model.py': '''class NeuralHead: pass
class LinearHead: pass
class Model:
    @classmethod
    def _from_pretrained(cls, name, neural_head=False):
        if neural_head:
            return NeuralHead()
        else:
            return LinearHead()
class ParentTrainer:
    def train(self, learning_rate=None):
        self.train_embeddings()
        self.train_classifier()
class CompatTrainer(ParentTrainer):
    def __init__(self, model, learning_rate=2e-5):
        self.args = Arguments(body_learning_rate=learning_rate)
''', 'train.py': '''p.add_argument('--head',default='linear')
p.add_argument('--lr',type=float,default=0.001)
model=Model.from_pretrained('encoder')
trainer=CompatTrainer(model=model)
if args.head == 'neural':
    trainer.train(learning_rate=args.lr)
else:
    trainer.train()
'''}
    return {name:ast.parse(body) for name,body in bodies.items()}, bodies


def test_inherited_call_and_constructor_defaults_outrank_disabled_call():
    trees,bodies = sources()
    packets,inactive = selected_call_packets(trees,bodies,['train.py'],[{'argv':['python','train.py']}])
    constructor = next(row for row in packets if row['resolved_symbol']=='CompatTrainer.__init__')
    assert constructor['literal_parameters']['learning_rate']==2e-5
    assert constructor['unknown_parameters']['model']=='model'
    train = next(row for row in packets if row['resolved_symbol']=='ParentTrainer.train')
    assert train['call']=='trainer.train()' and train['guards'][-1]['value'] is False
    assert train['definition']['file']=='model.py'
    assert len([row for row in packets if row['resolved_symbol']=='ParentTrainer.train'])==1
    assert inactive_excerpt({'file':'train.py','code':'L6:     trainer.train(learning_rate=args.lr)'},inactive)
    assert not inactive_excerpt({'file':'train.py','code':'L8:     trainer.train()'},inactive)
    answer='```python\n' + train['definition']['code'] + '\n```'
    assert checked_source_quotes(answer,{'calls':packets})[1]['rejected_blocks']==0


def test_factory_default_prunes_disabled_head_but_keeps_inherited_trainer():
    trees,bodies = sources()
    factory=factory_bindings(trees,['train.py'],[{}])
    binding=class_bindings(trees,['train.py'],resolved_factories=factory)
    assert 'LinearHead' in binding['reachable'] and 'NeuralHead' not in binding['reachable']
    assert 'ParentTrainer' in binding['reachable'] and binding['complete_definitions']


def test_unknown_overrides_and_multiple_dataset_branches_are_not_excluded():
    trees,bodies=sources()
    bodies['train.py'] += '\ntrainer.train(learning_rate=resolve_lr())\n'
    trees['train.py']=ast.parse(bodies['train.py'])
    commands=[{'argv':['python','train.py']},{'argv':['python','train.py','--head','neural']}]
    packets,inactive=selected_call_packets(trees,bodies,['train.py'],commands)
    unknown=next(row for row in packets if 'resolve_lr' in row['call'])
    assert 'learning_rate' not in unknown['literal_parameters']
    assert unknown['unknown_parameters']['learning_rate']=='resolve_lr()'
    assert not inactive['train.py']


def test_explicit_comparisons_keep_optional_source_evidence():
    assert current_path_only('解释当前分支。不要展开未启用的可微分类头。')
    assert not current_path_only('如果切换分类头，与当前分支有什么区别？')


def test_mutated_cli_conditions_and_starred_arguments_do_not_claim_defaults():
    trees,bodies=sources()
    bodies['train.py'] += '\nargs.head = choose_head()\ntrainer=CompatTrainer(*provided_args)\n'
    trees['train.py']=ast.parse(bodies['train.py'])
    packets,inactive=selected_call_packets(trees,bodies,['train.py'],[{}])
    assert not inactive['train.py']
    expanded=next(row for row in packets if '*provided_args' in row['call'])
    assert not expanded['literal_parameters'] and '*args' in expanded['unknown_parameters']
