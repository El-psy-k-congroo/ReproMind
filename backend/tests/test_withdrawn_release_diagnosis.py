from app.services.host_environment import check_release, release_issue


def test_withdrawn_universal_release_is_not_reported_as_linux_requirement():
    row=check_release('example==0.2.1', {'urls':[{'filename':'example-0.2.1-py2.py3-none-any.whl',
        'yanked':True,'yanked_reason':'Do not accidentally install on Python 3.11'}]}, '3.7','Windows','AMD64')
    assert row['release_state']=='withdrawn' and not row['compatible']
    issue=release_issue(row,'3.7','Windows','AMD64')
    assert '撤回' in issue['reason'] and '不能据此认定必须使用 Linux' in issue['reason']
    assert all('WSL' not in option for option in issue['options'])


def test_available_runtime_is_offered_before_suggesting_another_os():
    row=check_release('example==1.0',{'urls':[{'filename':'example-1.0-cp37-cp37m-win_amd64.whl'}]},'3.11','Windows','AMD64')
    issue=release_issue(row,'3.11','Windows','AMD64')
    assert any('本机' in option and '3.7' in option for option in issue['options'])
    assert all('WSL' not in option for option in issue['options'])
    empty=check_release('example==1.0',{'urls':[]},'3.11','Windows','AMD64')
    assert '没有列出' in release_issue(empty,'3.11')['reason']
