import pytest
from app.services.host_environment import check_release, python_requirement_matches


@pytest.mark.parametrize('requirement,python,expected',[
    ('>=3.7.1','3.7',True),('>=3.7.1','3.7.0',False),('>=3.7.1','3.7.16',True),
    ('>=3.8','3.7',False),('>=3.7.1,<3.7.2','3.7',True),
    ('>=3.7.1,!=3.7.*','3.7',False),('>=3.7.1,<3.7.1','3.7',False),
])
def test_minor_candidate_and_actual_interpreter_are_distinct(requirement,python,expected):
    assert python_requirement_matches(requirement,python)==expected


def test_patch_requirement_does_not_hide_a_matching_host_wheel():
    release={'urls':[{'filename':'example-1.0-cp37-cp37m-win_amd64.whl','requires_python':'>=3.7.1'}]}
    planned=check_release('example==1.0',release,'3.7','Windows','AMD64')
    assert planned['compatible'] and '3.7' in planned['compatible_python']
    assert planned['python_check_scope']=='minor_version_candidate'
    assert not check_release('example==1.0',release,'3.7.0','Windows','AMD64')['compatible']
    actual=check_release('example==1.0',release,'3.7.16','Windows','AMD64')
    assert actual['compatible'] and actual['python_check_scope']=='actual_patch_version'
