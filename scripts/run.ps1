<#
scripts/run.ps1

Deck-first reporting:
- Generates report/auto_deck.pptx
- Exports report/auto_deck.pdf (PowerPoint first, LibreOffice fallback)

Notes:
- PowerPoint COM can refuse "hidden" mode in some environments. We do NOT force Visible.
- Windows PowerShell 5.1 does NOT support Split-Path -LeafBase; we use .NET Path methods instead.
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
  Write-Host "  deck      Build report\auto_deck.pptx from existing artifacts (+ PDF)"
  Write-Host "  report    Full pipeline -> validate -> EDA -> train -> triage -> aliases -> deck (+ PDF)"
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

function Try-Install-OptionalExtra([string]$PythonExe, [string]$ExtraSpec) {
  try {
    & $PythonExe -m pip install -e $ExtraSpec | Out-Host
  } catch {
    Write-Host "Note: optional install failed for '$ExtraSpec' (non-fatal)."
  }
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

function Export-DeckToPdf-PowerPoint([string]$DeckPath, [string]$OutPdfPath) {
  $ppt = $null
  try {
    $deckFull = (Resolve-Path $DeckPath).Path
    $outPdfFull = [System.IO.Path]::GetFullPath($OutPdfPath)
    $outDir = Split-Path -Parent $outPdfFull
    if (-not (Test-Path $outDir)) { New-Item -ItemType Directory -Force -Path $outDir | Out-Null }

    $ppt = New-Object -ComObject PowerPoint.Application
    # Do NOT force Visible (some environments disallow hiding/showing programmatically).
    try { $ppt.DisplayAlerts = 0 } catch {}

    # Open(FileName, ReadOnly, Untitled, WithWindow)
    $pres = $ppt.Presentations.Open($deckFull, $true, $true, $false)

    # 32 = ppSaveAsPDF
    $pres.SaveAs($outPdfFull, 32)

    $pres.Close()
    $ppt.Quit()

    return (Test-Path $outPdfFull)
  } catch {
    try { if ($ppt) { $ppt.Quit() } } catch {}
    Write-Host "PowerPoint PDF export failed: $($_.Exception.Message)"
    return $false
  }
}

function Export-DeckToPdf-LibreOffice([string]$DeckPath, [string]$OutPdfPath) {
  $cmd = Get-Command soffice.com -ErrorAction SilentlyContinue
  if ($null -eq $cmd) { $cmd = Get-Command soffice.exe -ErrorAction SilentlyContinue }
  if ($null -eq $cmd) { $cmd = Get-Command soffice -ErrorAction SilentlyContinue }
  if ($null -eq $cmd) { return $false }

  $soffice = $cmd.Path
  $deckFull = (Resolve-Path $DeckPath).Path

  $outPdfFull = [System.IO.Path]::GetFullPath($OutPdfPath)
  $outDir = Split-Path -Parent $outPdfFull
  if (-not (Test-Path $outDir)) { New-Item -ItemType Directory -Force -Path $outDir | Out-Null }

  # Run via cmd.exe and redirect to nul so stderr doesn't terminate the script
  $cmdLine = "`"$soffice`" --headless --nologo --nofirststartwizard --norestore --convert-to pdf --outdir `"$outDir`" `"$deckFull`" 1>nul 2>nul"
  cmd.exe /c $cmdLine | Out-Null

  $base = [System.IO.Path]::GetFileNameWithoutExtension($deckFull)
  $generated = Join-Path $outDir ($base + ".pdf")

  # If LibreOffice generated to the same name, great. If not, copy it to requested path.
  if ((Test-Path $generated) -and ($generated -ne $outPdfFull)) {
    Copy-Item -Force $generated $outPdfFull
  }

  return (Test-Path $outPdfFull)
}

function Try-Export-DeckToPdf([string]$DeckPath, [string]$OutPdfPath) {
  if (Export-DeckToPdf-PowerPoint -DeckPath $DeckPath -OutPdfPath $OutPdfPath) {
    Write-Host "PDF generated (PowerPoint): $OutPdfPath"
    return
  }
  if (Export-DeckToPdf-LibreOffice -DeckPath $DeckPath -OutPdfPath $OutPdfPath) {
    Write-Host "PDF generated (LibreOffice): $OutPdfPath"
    return
  }
  Write-Host "No PDF exporter available. PPTX generated only: $DeckPath"
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
      Invoke-Step "Upgrading pip" { & $python -m pip install --upgrade pip }
      Invoke-Step "Installing project (editable)" { & $python -m pip install -e . }
      Try-Install-OptionalExtra -PythonExe $python -ExtraSpec ".[dev]"
      Write-Host "Done."
      break
    }

    "data" {
      $python = Get-PythonExe -Root $root
      Invoke-Step "Generate synthetic data" { & $python -m src.data.simulate --config $configPath }
      break
    }

    "validate" {
      $python = Get-PythonExe -Root $root
      Invoke-Step "Validate data (Great Expectations)" { & $python -m src.data.validate --config $configPath }
      break
    }

    "train" {
      $python = Get-PythonExe -Root $root
      Invoke-Step "Train models + export evaluation artifacts" { & $python -m src.models.train --config $configPath }
      break
    }

    "deck" {
      $python = Get-PythonExe -Root $root
      Invoke-Step "Generate report\auto_deck.pptx" {
        & $python -m src.reporting.make_auto_deck --config $configPath --out ".\report\auto_deck.pptx"
      }
      Try-Export-DeckToPdf -DeckPath ".\report\auto_deck.pptx" -OutPdfPath ".\report\auto_deck.pdf"
      break
    }

    "report" {
      $python = Get-PythonExe -Root $root
      Write-Host "=== Report pipeline: validate -> EDA -> train -> triage -> aliases -> deck ==="

      Invoke-Step "1) Data validation" { & $python -m src.data.validate --config $configPath }
      Invoke-Step "2) EDA + stats (tables + KM + Cox)" { & $python -m src.reporting.eda_stats --config $configPath }
      Invoke-Step "3) Train models (tables + calibration/PR + models)" { & $python -m src.models.train --config $configPath }

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

      Invoke-Step "6) Generate report\auto_deck.pptx" {
        & $python -m src.reporting.make_auto_deck --config $configPath --out ".\report\auto_deck.pptx"
      }

      Try-Export-DeckToPdf -DeckPath ".\report\auto_deck.pptx" -OutPdfPath ".\report\auto_deck.pdf"
      break
    }

    "app" {
      $python = Get-PythonExe -Root $root
      $appPath = Join-Path $root "app\streamlit_app.py"
      if (-not (Test-Path $appPath)) { throw "Streamlit app not found: $appPath" }
      Invoke-Step "Run Streamlit app" { & $python -m streamlit run $appPath -- --config $configPath }
      break
    }

    "test" {
      $python = Get-PythonExe -Root $root
      Invoke-Step "Run pytest" { & $python -m pytest -q }
      break
    }

    "lint" {
      $python = Get-PythonExe -Root $root
      Invoke-Step "Ruff lint" { & $python -m ruff check . }
      break
    }

    "format" {
      $python = Get-PythonExe -Root $root
      Invoke-Step "Black format" { & $python -m black . }
      break
    }
  }

} finally {
  Pop-Location
}