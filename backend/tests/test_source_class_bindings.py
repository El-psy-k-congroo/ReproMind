import ast
from app.services.source_class_context import class_bindings, enclosing_class


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
