$ErrorActionPreference = 'Stop'
$taskRoot = $PSScriptRoot
if (Get-Command py -ErrorAction SilentlyContinue) {
    foreach ($version in @('-3.11', '-3.12')) {
        & py $version -c 'import sys' 2>$null
        if ($LASTEXITCODE -eq 0) {
            & py $version (Join-Path $taskRoot 'start.py')
            exit $LASTEXITCODE
        }
    }
}
if (Get-Command python -ErrorAction SilentlyContinue) {
    & python (Join-Path $taskRoot 'start.py')
    exit $LASTEXITCODE
}
Write-Error 'Please install Python 3.11 or 3.12, then rerun this script.'
