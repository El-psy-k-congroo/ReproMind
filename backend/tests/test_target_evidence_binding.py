"""No numeric-only or baseline-derived paper target may pass as verified."""
from app.services.paper_target_evidence import bind_targets,explicit_table_targets


def evidence():
    return {'dataset':'DatasetA','metric':'Accuracy','reported':'82.3 ±0.6%',
        'method_label':'MethodX','columns':['DatasetA','DatasetB','DatasetC'],
        'header_quote':'Available data Method DatasetA DatasetB DatasetC',
        'row_quote':'X,A MethodX (ours) 82.3 ±0.6% 71.8 ±0.7% 76.8 ±0.6%'}


def test_mean_std_cell_is_bound_to_current_method_and_column():
    row=evidence();paper=row['header_quote']+'\n'+row['row_quote']
    spec={'model_name':'MethodX','datasets':['DatasetA'],'metrics':['Accuracy'],
        'target_results':{'DatasetA':'81.7%'},'target_evidence':[row]}
    result=bind_targets(spec,paper)
    assert result['target_results']=={'DatasetA':{'accuracy':.823}}
    assert result['target_results_raw']=={'DatasetA':{'accuracy':'82.3 ±0.6%'}}
    assert result['target_validation']['status']=='verified'
    for bad in [{**row,'reported':'81.7%'},{**row,'dataset':'DatasetB'},
                {**row,'columns':['DatasetC','DatasetB','DatasetA']},
                {**row,'row_quote':row['row_quote'].replace('MethodX','Baseline')}]:
        assert not bind_targets({**spec,'target_evidence':[bad]},paper)['target_results']


def test_legacy_numbers_without_table_evidence_do_not_block_author_execution():
    result=bind_targets({'model_name':'MethodX','datasets':['DatasetA'],'metrics':['Accuracy'],
        'target_results':{'DatasetA':{'accuracy':.817}}},'DatasetA MethodX 82.3 ±0.6%')
    assert not result['target_results'] and result['datasets']==['DatasetA']
    assert result['unverified_target_results'] and result['target_validation']['status']=='unverified'


def test_explicit_header_handles_context_prefix_and_rejects_ambiguous_variants():
    row=evidence();paper=row['header_quote']+'\nBaseline 81.7 70.0 72.0\n'+row['row_quote']
    spec={'model_name':'MethodX','datasets':['DatasetA'],'metrics':['Accuracy']}
    assert explicit_table_targets(spec,paper)['target_results']=={'DatasetA':{'accuracy':.823}}
    assert not explicit_table_targets(spec,paper+'\n'+row['row_quote'])['target_results']


def test_parenthetical_layer_counts_need_caption_evidence_and_variant_is_distinct():
    spec={'model_name':'MethodX','datasets':['DatasetA'],'metrics':['Accuracy']}
    table='Method DatasetA DatasetB\nMethodX 85.5 ±0.5 (64) 73.4 ±0.6 (32)\nMethodX* 85.3 ±0.2 (64) 73.2 ±0.8 (32)'
    assert not explicit_table_targets(spec,table)['target_results']
    paper='The number in parentheses corresponds to the number of layers of the model.\n'+table
    assert explicit_table_targets(spec,paper)['target_results']=={'DatasetA':{'accuracy':.855}}


def test_caption_is_replaced_only_by_unique_literal_header_in_same_table():
    row=evidence();caption='Table 2: Summary of classification accuracy.'
    row={**row,'header_quote':caption}
    paper=caption+'\nMethod DatasetA DatasetB DatasetC\n'+row['row_quote']
    spec={'model_name':'MethodX','datasets':['DatasetA'],'metrics':['Accuracy'],'target_evidence':[row]}
    result=bind_targets(spec,paper)
    assert result['target_results']=={'DatasetA':{'accuracy':.823}}
    assert result['target_validation']['evidence'][0]['header_quote']=='Method DatasetA DatasetB DatasetC'
    assert not bind_targets(spec,paper.replace('DatasetA DatasetB DatasetC','DatasetC DatasetB DatasetA'))['target_results']
    assert not bind_targets(spec,paper.replace(row['row_quote'],'unrelated row'))['target_results']
