from app.services.source_quotes import checked_source_quotes


def test_plural_excerpts_and_adjacent_packets_from_one_file_are_checked():
    context={'source_relevant_excerpts':[{'file':'train.py','excerpts':['L10: optimizer = make_optimizer()']}],
             'loss':[{'file':'train.py','code':'L11: loss = criterion(output)'}]}
    answer='```python\nL10: optimizer = make_optimizer()\nL11: loss = criterion(output)\n```'
    assert checked_source_quotes(answer,context)[1]['rejected_blocks']==0
    plain='```python\noptimizer = make_optimizer()\nloss = criterion(output)\n```'
    assert checked_source_quotes(plain,context)[1]['rejected_blocks']==0


def test_different_files_and_gapped_lines_are_not_combined_into_fake_code():
    context=[{'file':'one.py','code':'L10: value = 1'},{'file':'two.py','code':'L11: print(value)'}]
    answer='```python\nL10: value = 1\nL11: print(value)\n```'
    assert checked_source_quotes(answer,context)[1]['rejected_blocks']==1
    context[1]['file']='one.py';context[1]['code']='L12: print(value)'
    plain='```python\nvalue = 1\nprint(value)\n```'
    assert checked_source_quotes(plain,context)[1]['rejected_blocks']==1
    assert checked_source_quotes(plain,[{'file':'one.py','code':'L10: value = 1\nL12: print(value)'}])[1]['rejected_blocks']==1


def test_contradictory_line_packets_and_reordered_line_numbers_fail_closed():
    context=[{'file':'one.py','code':'L1: x = 1\nL2: print(x)'},
             {'file':'one.py','code':'L1: x = 2'}]
    assert checked_source_quotes('```python\nL1: x = 1\n```',context)[1]['rejected_blocks']==1
    assert checked_source_quotes('```python\nx = 1\n```',context)[1]['rejected_blocks']==1
    assert checked_source_quotes('```python\nL2: print(x)\nL1: x = 2\n```',context)[1]['rejected_blocks']==1
def test_uniform_display_dedent_is_allowed_but_conditional_scope_change_is_not():
    from app.services.source_quotes import checked_source_quotes
    context = {'packet': {'file': 'network.py', 'code': 'L10:         result = weight * branch + (1-weight) * anchor\nL11:         if self.residual:\nL12:             result = result + previous'}}
    good = '```python\nresult = weight * branch + (1-weight) * anchor\nif self.residual:\n    result = result + previous\n```'
    assert checked_source_quotes(good, context)[1]['rejected_blocks'] == 0
    changed = good.replace('    result = result + previous', 'result = result + previous')
    assert checked_source_quotes(changed, context)[1]['rejected_blocks'] == 1
    swapped = good.replace('weight * branch + (1-weight) * anchor', '(1-weight) * branch + weight * anchor')
    assert checked_source_quotes(swapped, context)[1]['rejected_blocks'] == 1
