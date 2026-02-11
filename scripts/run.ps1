<#
scripts/run.ps1

One-command entrypoints for the Fleet Maintenance Triage Agent project.

Design goals
- Always run from repo root.
- Prefer local .venv if it exists.
- Fail fast on non-zero native exit codes.
- "report" should be turnkey for a fresh clone (auto-generate synthetic data if missing).
- Deck PDF export is optional (PowerPoint COM preferred, LibreOffice fallback).

Usage:
  .\scripts\run.ps1 <task>

Tasks:
  install   Create .venv and install deps
  data      Generate synthetic data
  validate  Run data quality validation (Great Expectations)
  train     Train models and export artifacts
  deck      Build report\auto_deck.pptx from existing artifacts (+ optional PDF)
  report    Full pipeline -> data(if missing) -> validate -> EDA -> train -> triage -> aliases -> deck
  app       Run Streamlit GUI
  test      Run tests
  lint      Ruff lint
  format    Black format
#>

[CmdletBinding()]
param(
  [Parameter(Position = 0)]
  [ValidateSet("help","install","data","validate","train","deck","report","app","test","lint","format")]
  [string]$task = "help",

  [Parameter()]
  [string]$config = "config.yaml"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Write-Usage {
  Write-Host ""
  Write-Host "Usage:"
  Write-Host "  .\scripts\run.ps1 <task>"
  Write-Host ""
  Write-Host "Tasks:"
  Write-Host "  install   Create .venv and install deps"
  Write-Host "  data      Generate synthetic data"
  Write-Host "  validate  Run data quality validation (Great Expectations)"
  Write-Host "  train     Train models and export artifacts"
  Write-Host "  deck      Build report\auto_deck.pptx from existing artifacts (+ optional PDF)"
  Write-Host "  report    Full pipeline -> data(if missing) -> validate -> EDA -> train -> triage -> aliases -> deck"
  Write-Host "  app       Run Streamlit GUI"
  Write-Host "  test      Run tests"
  Write-Host "  lint      Ruff lint"
  Write-Host "  format    Black format"
  Write-Host ""
}

function Get-RepoRoot {
  return (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
}

function Get-PythonExe([string]$Root) {
  $venvPy = Join-Path $Root ".venv\Scripts\python.exe"
  if (Test-Path $venvPy) { return $venvPy }
  return "python"
}

function Ensure-Venv([string]$Root) {
  $venvDir = Join-Path $Root ".venv"
  $venvPy  = Join-Path $Root ".venv\Scripts\python.exe"

  if (Test-Path $venvPy) { return }

  Write-Host "Creating virtual environment (.venv)..."
  & python -m venv $venvDir

  if (-not (Test-Path $venvPy)) {
    throw "Failed to create .venv. Ensure Python is installed and on PATH."
  }
}

function Resolve-ConfigPath([string]$Root, [string]$ConfigArg) {
  $cfg = $ConfigArg
  if ([string]::IsNullOrWhiteSpace($cfg)) { $cfg = "config.yaml" }

  $cfgPath = $cfg
  if (-not [System.IO.Path]::IsPathRooted($cfgPath)) {
    $cfgPath = Join-Path $Root $cfgPath
  }

  if (-not (Test-Path $cfgPath)) {
    throw "Config file not found: $cfgPath"
  }

  return (Resolve-Path $cfgPath).Path
}

function Get-PipTrustArgs {
  # For corporate SSL interception issues:
  #   $env:FLEET_PIP_TRUSTED=1
  $v = $env:FLEET_PIP_TRUSTED
  if ([string]::IsNullOrWhiteSpace($v)) { return @() }
  $t = $v.Trim().ToLower()
  if ($t -in @("1","true","yes","y","on")) {
    return @("--trusted-host","pypi.org","--trusted-host","files.pythonhosted.org","--trusted-host","pypi.python.org")
  }
  return @()
}

function Invoke-Step([string]$Label, [scriptblock]$Action) {
  Write-Host ""
  Write-Host ">>> $Label"
  & $Action

  $exitCode = $LASTEXITCODE
  if ($exitCode -ne $null -and $exitCode -ne 0) {
    throw "FAILED ($exitCode): $Label"
  }
}

function Invoke-Step-Soft([string]$Label, [scriptblock]$Action) {
  Write-Host ""
  Write-Host ">>> $Label"
  try {
    & $Action
  } catch {
    Write-Host ("Note: step failed (continuing): {0}" -f $_.Exception.Message)
    return
  }

  $exitCode = $LASTEXITCODE
  if ($exitCode -ne $null -and $exitCode -ne 0) {
    Write-Host "Note: step returned exit code $exitCode (continuing)."
  }
}

function Ensure-RawData([string]$Root, [string]$PythonExe, [string]$ConfigPath) {
  $rawDir = Join-Path $Root "data\raw"
  $req = @(
    (Join-Path $rawDir "vehicle_day.csv"),
    (Join-Path $rawDir "dtc_event.csv"),
    (Join-Path $rawDir "work_order.csv")
  )

  $missing = @()
  foreach ($p in $req) {
    if (-not (Test-Path $p)) { $missing += $p }
  }

  if ($missing.Count -eq 0) { return }

  Write-Host ""
  Write-Host "Raw data missing (fresh clone is expected). Generating synthetic data..."
  foreach ($m in $missing) { Write-Host ("  - missing: {0}" -f $m) }

  Invoke-Step "0) Generate synthetic data (data/raw/*.csv)" {
    & $PythonExe -m src.data.simulate --config $ConfigPath
  }
}

function Export-DeckToPdf-PowerPoint([string]$PptxPath, [string]$PdfPath) {
  $ppt = $null
  $pres = $null
  try {
    $ppt = New-Object -ComObject PowerPoint.Application
    try { $ppt.Visible = 1 } catch { }

    $pres = $ppt.Presentations.Open($PptxPath, $true, $false, $true)
    $pres.ExportAsFixedFormat($PdfPath, 2)  # 2 = PDF
  } finally {
    if ($pres -ne $null) {
      try { $pres.Close() } catch { }
      try { [void][System.Runtime.InteropServices.Marshal]::ReleaseComObject($pres) } catch { }
    }
    if ($ppt -ne $null) {
      try { $ppt.Quit() } catch { }
      try { [void][System.Runtime.InteropServices.Marshal]::ReleaseComObject($ppt) } catch { }
    }
    [gc]::Collect()
    [gc]::WaitForPendingFinalizers()
  }
}

function Export-DeckToPdf-LibreOffice([string]$PptxPath, [string]$PdfOutDir) {
  $sofficeExe = $null

  $candidates = @(
    (Join-Path $env:ProgramFiles "LibreOffice\program\soffice.exe"),
    (Join-Path $env:ProgramFiles "LibreOffice\program\soffice.com")
  )
  foreach ($c in $candidates) {
    if (Test-Path $c) { $sofficeExe = $c; break }
  }

  if ($null -eq $sofficeExe) {
    $cmd = Get-Command soffice -ErrorAction SilentlyContinue
    if ($cmd) { $sofficeExe = $cmd.Source }
  }

  if ($null -eq $sofficeExe) { throw "LibreOffice not found." }

  & $sofficeExe --headless --nologo --nofirststartwizard --norestore `
    --convert-to pdf --outdir $PdfOutDir $PptxPath | Out-Null

  $exitCode = $LASTEXITCODE
  if ($exitCode -ne $null -and $exitCode -ne 0) {
    throw "LibreOffice export failed (exit code $exitCode)."
  }
}

function Try-Export-DeckToPdf([string]$PptxPath, [string]$PdfPath) {
  $noPdf = $env:FLEET_DECK_NO_PDF
  if (-not [string]::IsNullOrWhiteSpace($noPdf)) {
    $t = $noPdf.Trim().ToLower()
    if ($t -in @("1","true","yes","y","on")) {
      Write-Host "PDF export disabled via FLEET_DECK_NO_PDF=1"
      return
    }
  }

  try {
    $ppt = New-Object -ComObject PowerPoint.Application
    $ppt.Quit()
    [void][System.Runtime.InteropServices.Marshal]::ReleaseComObject($ppt)
    [gc]::Collect(); [gc]::WaitForPendingFinalizers()

    Export-DeckToPdf-PowerPoint -PptxPath $PptxPath -PdfPath $PdfPath
    Write-Host "PDF generated (PowerPoint): $PdfPath"
    return
  } catch {
    Write-Host ("PowerPoint PDF export failed (will try LibreOffice): {0}" -f $_.Exception.Message)
  }

  try {
    $outDir = Split-Path -Parent $PdfPath
    Export-DeckToPdf-LibreOffice -PptxPath $PptxPath -PdfOutDir $outDir

    if (Test-Path $PdfPath) {
      Write-Host "PDF generated (LibreOffice): $PdfPath"
      return
    }

    $base = [System.IO.Path]::GetFileNameWithoutExtension($PptxPath)
    $alt = Join-Path (Split-Path -Parent $PdfPath) ($base + ".pdf")
    if (Test-Path $alt) {
      Move-Item -Force $alt $PdfPath
      Write-Host "PDF generated (LibreOffice): $PdfPath"
      return
    }

    throw "LibreOffice ran but PDF not found."
  } catch {
    Write-Host ("No PDF exporter found (LibreOffice/PowerPoint). PPTX generated only: {0}" -f $PptxPath)
  }
}

$root = Get-RepoRoot
Push-Location $root
try {
  $configPath = Resolve-ConfigPath -Root $root -ConfigArg $config

  switch ($task) {

    "help" { Write-Usage; break }

    "install" {
      Ensure-Venv -Root $root
      $python = Get-PythonExe -Root $root
      $trust = Get-PipTrustArgs

      Invoke-Step-Soft "Upgrading pip (best-effort)" {
        & $python -m pip install --upgrade pip setuptools wheel @trust
      }

      Invoke-Step "Installing project + dependencies" {
        & $python -m pip install ".[dev]" @trust
      }

      Write-Host "Done."
      break
    }

    "data" {
      $python = Get-PythonExe -Root $root
      Invoke-Step "Generate synthetic data" {
        & $python -m src.data.simulate --config $configPath
      }
      break
    }

    "validate" {
      $python = Get-PythonExe -Root $root
      Invoke-Step "Validate data (Great Expectations)" {
        & $python -m src.data.validate --config $configPath
      }
      break
    }

    "train" {
      $python = Get-PythonExe -Root $root
      Invoke-Step "Train models + export evaluation artifacts" {
        & $python -m src.models.train --config $configPath
      }
      break
    }

    "deck" {
      $python = Get-PythonExe -Root $root
      $reportDir = Join-Path $root "report"
      if (-not (Test-Path $reportDir)) { New-Item -ItemType Directory -Force $reportDir | Out-Null }

      $pptxPath = Join-Path $reportDir "auto_deck.pptx"
      $pdfPath  = Join-Path $reportDir "auto_deck.pdf"

      Invoke-Step "Generate report\auto_deck.pptx" {
        & $python -m src.reporting.make_auto_deck --config $configPath --out $pptxPath
      }

      Try-Export-DeckToPdf -PptxPath $pptxPath -PdfPath $pdfPath
      break
    }

    "report" {
      $python = Get-PythonExe -Root $root

      Write-Host "=== Report pipeline: data(if missing) -> validate -> EDA -> train -> triage -> aliases -> deck ==="

      Ensure-RawData -Root $root -PythonExe $python -ConfigPath $configPath

      Invoke-Step "1) Data validation" {
        & $python -m src.data.validate --config $configPath
      }

      Invoke-Step "2) EDA + stats (tables + KM + Cox)" {
        & $python -m src.reporting.eda_stats --config $configPath
      }

      Invoke-Step "3) Train models (tables + calibration/PR + models)" {
        & $python -m src.models.train --config $configPath
      }

      Invoke-Step "4) Export triage snapshot (tables/figures/snippet)" {
        & $python -m src.reporting.triage_snapshot `
          --config $configPath `
          --horizon 30 `
          --model hgb `
          --k 10 `
          --ranking cost `
          --evidence 5 `
          --evidence_topn 5
      }

      Invoke-Step "5) Create *_latest aliases (stable filenames)" {
        & $python -m src.reporting.make_latest_aliases --config $configPath --top_similar 5
      }

      # IMPORTANT: generate the deck directly (do NOT call the PS script via python)
      $reportDir = Join-Path $root "report"
      if (-not (Test-Path $reportDir)) { New-Item -ItemType Directory -Force $reportDir | Out-Null }

      $pptxPath = Join-Path $reportDir "auto_deck.pptx"
      $pdfPath  = Join-Path $reportDir "auto_deck.pdf"

      Invoke-Step "6) Generate report\auto_deck.pptx" {
        & $python -m src.reporting.make_auto_deck --config $configPath --out $pptxPath
      }

      Try-Export-DeckToPdf -PptxPath $pptxPath -PdfPath $pdfPath
      break
    }

    "app" {
      $python = Get-PythonExe -Root $root
      $appPath = Join-Path $root "app\streamlit_app.py"
      if (-not (Test-Path $appPath)) { throw "Streamlit app not found: $appPath" }

      Invoke-Step "Run Streamlit app" {
        & $python -m streamlit run $appPath -- --config $configPath
      }
      break
    }

    "test" {
      $python = Get-PythonExe -Root $root
      Invoke-Step "Run pytest" {
        & $python -m pytest -q
      }
      break
    }

    "lint" {
      $python = Get-PythonExe -Root $root
      Invoke-Step "Ruff lint" {
        & $python -m ruff check .
      }
      break
    }

    "format" {
      $python = Get-PythonExe -Root $root
      Invoke-Step "Black format" {
        & $python -m black .
      }
      break
    }
  }

} finally {
  Pop-Location
}