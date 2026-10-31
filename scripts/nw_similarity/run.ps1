$ErrorActionPreference = 'Stop'
$inputPath = Join-Path $PSScriptRoot 'unique_sequences.tsv'
if (-not (Test-Path -LiteralPath $inputPath)) { throw 'Missing unique_sequences.tsv' }
Add-Type -Path (Join-Path $PSScriptRoot 'NwEngine.cs')
[NwEngine]::LoadMatrix((Join-Path $PSScriptRoot 'BLOSUM62.txt'))
[NwEngine]::Run($inputPath, $PSScriptRoot)
python (Join-Path $PSScriptRoot 'plot_ab.py')
if ($LASTEXITCODE -ne 0) { throw 'Plotting failed' }
