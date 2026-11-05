from app.services import reproduction_engine as engine


def complete_group(directory, variant):
    directory.mkdir(parents=True,exist_ok=True)
    for part in engine.PLANETOID_PARTS:
        (directory/f'ind.{variant}.{part}').write_bytes(b'author data')


def test_empty_branch_cache_cannot_shadow_complete_source_data(tmp_path):
    root=tmp_path/'branches'/'run'/'source'
    cwd=root/'gcn';data=cwd/'data';empty=tmp_path/'branches'/'run'/'data'
    complete_group(data,'nell.0.001');empty.mkdir()
    argv=['python','-m','gcn.train','--dataset','nell.0.001','--early_stopping','10']
    item={'cwd':str(cwd),'argv':argv}
    command={'pythonpath':str(root),'data_dir':str(empty)}
    assert engine._runtime_data_directory(tmp_path,command,item)==data
    assert item['argv']==argv and '--early_stopping' in argv
    assert len(list(data.iterdir()))==8 and not list(empty.iterdir())


def test_matching_configured_data_remains_authoritative(tmp_path):
    source=tmp_path/'source';cwd=source/'gcn'
    configured=tmp_path/'data'
    complete_group(configured,'nell.0.001');complete_group(cwd/'data','nell.0.001')
    item={'cwd':str(cwd),'argv':['python','--dataset=nell.0.001']}
    assert engine._runtime_data_directory(tmp_path,{'pythonpath':str(source),'data_dir':str(configured)},item)==configured


def test_runtime_resolution_does_not_switch_dataset_variants(tmp_path):
    source=tmp_path/'source';cwd=source/'gcn';configured=tmp_path/'empty'
    complete_group(cwd/'data','nell.0.01')
    item={'cwd':str(cwd),'argv':['python','--dataset','nell.0.001']}
    assert engine._runtime_data_directory(tmp_path,{'pythonpath':str(source),'data_dir':str(configured)},item)==configured


def test_partial_file_group_is_not_treated_as_a_complete_dataset(tmp_path):
    source=tmp_path/'source';cwd=source/'gcn';data=cwd/'data'
    data.mkdir(parents=True);(data/'ind.nell.0.001.x').write_bytes(b'partial')
    configured=tmp_path/'empty'
    item={'cwd':str(cwd),'argv':['python','--dataset','nell.0.001']}
    assert engine._runtime_data_directory(tmp_path,{'pythonpath':str(source),'data_dir':str(configured)},item)==configured


def test_identical_dataset_name_at_wrong_path_is_directory_failure(tmp_path):
    complete_group(tmp_path/'data','nell.0.001');complete_group(tmp_path/'data','nell.0.01')
    output="FileNotFoundError: [Errno 2] No such file or directory: 'wrong/data/ind.nell.0.01.x'"
    result=engine._diagnose_failure(output,tmp_path,'NELL')
    assert result['type']=='dataset_directory_mismatch'
    assert '--dataset nell.0.01' in result['suggestion']
    assert '--dataset nell.0.001' not in result['suggestion']
