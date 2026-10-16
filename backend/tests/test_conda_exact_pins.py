import pytest
from app.services.gpu_preparation import protected_conda_specs

def test_existing_packages_are_exactly_pinned_except_approved_cuda_changes():
    inventory = ('# platform: win-64\n@EXPLICIT\n'
                 'https://repo.anaconda.com/pkgs/main/win-64/python-3.7.16-h6244533_0.conda\n'
                 'https://repo.anaconda.com/pkgs/main/win-64/ca-certificates-2025.1-h_0.tar.bz2\n'
                 'https://conda.anaconda.org/conda-forge/win-64/cudnn-8.0.0-h_0.conda\n')
    assert protected_conda_specs(inventory) == ['python=3.7.16=h6244533_0', 'ca-certificates=2025.1=h_0']

@pytest.mark.parametrize('inventory', ['', 'unknown inventory', '@EXPLICIT\nhttps://example.com/pip-1.0-build.conda'])
def test_incomplete_snapshot_cannot_guess_or_change_interpreter(inventory):
    with pytest.raises(ValueError): protected_conda_specs(inventory)
def test_snapshot_resume_accepts_only_approved_cuda_changes():
    from app.services.gpu_preparation import migration_snapshot_matches
    previous = {'conda_explicit': '@EXPLICIT\nhttps://repo.anaconda.com/pkgs/main/win-64/python-3.7.16-h6244533_0.conda', 'pip_freeze': 'tensorflow==1.15.5'}
    current = {**previous, 'conda_explicit': previous['conda_explicit'] + '\nhttps://conda.anaconda.org/conda-forge/win-64/cudnn-8.1.0.77-h3e0f4f4_0.tar.bz2'}
    assert migration_snapshot_matches(previous, current)
    assert not migration_snapshot_matches(previous, {**current, 'pip_freeze': 'tensorflow==2.10.0'})
    assert not migration_snapshot_matches(previous, {**current, 'conda_explicit': current['conda_explicit'].replace('3.7.16', '3.7.1')})


def test_installed_cuda_versions_allow_resuming_without_repeating_solver():
    from app.services.gpu_preparation import cuda_dependencies_ready
    explicit = '\n'.join('https://conda.anaconda.org/conda-forge/win-64/' + name for name in [
        'cudatoolkit-11.2.2-h7d7167e_13.tar.bz2', 'cudnn-8.1.0.77-h3e0f4f4_0.tar.bz2'])
    assert cuda_dependencies_ready(explicit)
    assert not cuda_dependencies_ready(explicit.replace('11.2.2', '12.2.0'))
    assert not cuda_dependencies_ready(explicit.splitlines()[0])
