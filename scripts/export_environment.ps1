param(
    [string]$CondaExe = "E:\anaconda3\Scripts\conda.exe",
    [string]$EnvironmentName = "base",
    [string]$PythonExe = "E:\anaconda3\python.exe"
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path $CondaExe)) {
    throw "Conda executable not found: $CondaExe"
}
if (-not (Test-Path $PythonExe)) {
    throw "Python executable not found: $PythonExe"
}

New-Item -ItemType Directory -Force -Path metadata | Out-Null

# Portable environment specification, reproducibility export, and explicit package listing.
& $CondaExe env export --name $EnvironmentName --from-history |
    Set-Content -Encoding utf8 environment.yml

& $CondaExe env export --name $EnvironmentName --no-builds |
    Set-Content -Encoding utf8 environment.lock.yml

& $CondaExe list --name $EnvironmentName --explicit |
    Set-Content -Encoding utf8 environment.explicit.txt

& $PythonExe -m pip freeze |
    Set-Content -Encoding utf8 metadata\pip_freeze.txt

# Historical Gurobi setting: no explicit Threads parameter; archived logs observed up to 16 threads.
& $PythonExe .\scripts\capture_runtime.py `
    --output-dir .\metadata `
    --historical-thread-parameter NOT_EXPLICITLY_SET `
    --observed-auto-thread-limit 16

Write-Host "Generated environment.yml, environment.lock.yml, environment.explicit.txt,"
Write-Host "metadata/pip_freeze.txt, and metadata/runtime_manifest.json."
