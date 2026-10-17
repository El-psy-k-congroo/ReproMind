from app.services import reproduction_engine as engine
from app.services.task_context import rank_entries


def test_documented_dataset_beats_incidental_registry_mentions(tmp_path):
    (tmp_path/'extended.py').write_text("REGISTRY = ['review_en', 'review_de']\n")
    (tmp_path/'train.py').write_text("import argparse\np=argparse.ArgumentParser()\np.add_argument('--datasets',nargs='+')\np.add_argument('--shots',type=int,default=16)\n")
    (tmp_path/'README.md').write_text('python train.py --datasets=review_en --shots=8\npython extended.py --datasets review_de\n')
    manifest=engine._repository_manifest(tmp_path)
    manifest['entrypoints']=['extended.py','train.py']
    manifest['entrypoints']=rank_entries(tmp_path,manifest,'review_en 作者训练')
    assert manifest['entrypoints'][0]=='train.py'
    planned=engine._execution_command(tmp_path,manifest,'full',{'prefix':str(tmp_path/'env')},
        {'datasets':['review_en']},{'available_datasets':['review_en']})
    assert planned['commands'][0]['argv'][1:]==['train.py','--datasets=review_en','--shots=8']


def test_multivalue_author_example_is_scoped_per_selected_dataset(tmp_path):
    (tmp_path/'train.py').write_text("p.add_argument('--datasets',nargs='+')\n")
    (tmp_path/'README.md').write_text('python train.py --datasets first second --seed 17\n')
    planned=engine._execution_command(tmp_path,engine._repository_manifest(tmp_path),'full',{'prefix':str(tmp_path/'env')},
        {'datasets':['first','second']},{'available_datasets':['first','second']})
    assert [row['argv'][2:] for row in planned['commands']]==[
        ['--datasets','first','--seed','17'],['--datasets','second','--seed','17']]
