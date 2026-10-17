import ast
from app.services.source_class_context import class_bindings, enclosing_class, factory_bindings


def test_shared_layer_does_not_make_another_models_optional_residual_active():
    network = ast.parse('''class Layer:
    def __init__(self, residual=False): pass
class Selected:
    def __init__(self):
        self.layer=Layer()
class Alternative:
    def __init__(self):
        self.layer=Layer(residual=True)
''')
    binding = class_bindings({'train.py': ast.parse('from network import Selected\nmodel=Selected()'), 'network.py': network}, ['train.py'])
    assert binding['unambiguous'] and binding['roots'] == ['Selected']
    assert set(binding['reachable']) == {'Selected', 'Layer'}
    calls = [node for node in ast.walk(network) if isinstance(node, ast.Call)]
    assert [(enclosing_class(network,node.lineno), ast.unparse(node)) for node in calls] == [('Selected','Layer()'),('Alternative','Layer(residual=True)')]


def test_dynamic_or_ambiguous_construction_is_not_silently_bound_to_one_model():
    tree=ast.parse('class A: pass\nclass B: pass\nmodel=A() if choose else B()')
    assert not class_bindings({'train.py':tree},['train.py'])['unambiguous']
    assert class_bindings({'train.py':tree},['train.py'],['b'])['roots'] == ['B']


def test_factory_root_and_cli_branch_do_not_activate_an_optional_head():
    trees={'model.py':ast.parse('''class Model:
    @classmethod
    def _from_pretrained(cls, name, neural_head=False): pass
'''),'train.py':ast.parse('''p.add_argument('--head',default='linear')
if args.head == 'neural':
    model=Model.from_pretrained(args.model,neural_head=True)
else:
    model=Model.from_pretrained(args.model)
''')}
    assert class_bindings(trees,['train.py'])['roots']==['Model']
    default=factory_bindings(trees,['train.py'],[{'argv':['python','train.py']}])
    assert len(default)==1 and default[0]['literal_options']['neural_head'] is False
    assert default[0]['guards'][0]['value'] is False and default[0]['guards'][0]['branch']=='else'
    override=factory_bindings(trees,['train.py'],[{'argv':['python','train.py','--head','neural']}])
    assert len(override)==1 and override[0]['literal_options']['neural_head'] is True
    trees['train.py']=ast.parse('if select_head():\n    model=Model.from_pretrained(args.model,neural_head=True)\nelse:\n    model=Model.from_pretrained(args.model)')
    unresolved=factory_bindings(trees,['train.py'],[{}])
    assert len(unresolved)==2 and all(row['guards'][0]['value']=='unknown' for row in unresolved)
    trees['train.py']=ast.parse('model=Model.from_pretrained(args.model,neural_head=dynamic())')
    unknown=factory_bindings(trees,['train.py'],[{}])[0]
    assert 'neural_head' not in unknown['literal_options'] and unknown['unknown_options']['neural_head']=='dynamic()'
