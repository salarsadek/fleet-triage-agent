<#
Fleet Triage Agent task runner (Windows / PowerShell)

Usage:
  .\scripts\run.ps1 <task> [config.yaml]

Tasks:
  help      Show this help
  install   Create .venv and install dependencies
  data      Generate synthetic data (data/raw/*.csv)
  validate  Run Great Expectations validation
  eda       EDA + survival stats (KM + Cox)
  train     Train models + export artifacts
  triage    Export triage snapshot artifacts
  aliases   Create *_latest stable aliases
  deck      Build report\auto_deck.pptx (+ PDF export)
  report    Full pipeline: data(if missing) -> validate -> eda -> train -> triage -> aliases -> deck
  app       Run Streamlit GUI
  test      Run pytest
  lint      Ruff lint
  format    Black format

Install in corp SSL environments:
  - Quick (less secure): $env:FLEET_PIP_TRUSTED=1
  - Better:            $env:FLEET_PIP_CERT="C:\path\to\corp-ca.pem"
#>

param(
  [Parameter(Position=0)]
  [ValidateSet("help","install","data","validate","eda","train","triage","aliases","deck","report","app","test","lint","format")]
  [string]$task = "help",

  [Parameter(Position=1)]
  [string]$config = "config.yaml"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$root = Resolve-Path (Join-Path $PSScriptRoot "..")
$configPath = Join-Path $root $config

function Write-Section([string]$msg) {
  Write-Host ""
  Write-Host $msg -ForegroundColor Cyan
}

function Invoke-Step([string]$Label, [scriptblock]$Block) {
  Write-Section ">>> $Label"
  & $Block
  $code = $LASTEXITCODE
  if ($code -ne 0) {
    throw "FAILED ($code): $Label"
  }
}

function Ensure-Dir([string]$Path) {
  if (-not (Test-Path $Path)) { New-Item -ItemType Directory -Force -Path $Path | Out-Null }
}

function Ensure-GitKeep([string]$Dir) {
  Ensure-Dir $Dir
  $p = Join-Path $Dir ".gitkeep"
  if (-not (Test-Path $p)) { New-Item -ItemType File -Force -Path $p | Out-Null }
}

function Get-VenvPython([string]$Root) {
  $py = Join-Path $Root ".venv\Scripts\python.exe"
  if (Test-Path $py) { return $py }
  return $null
}

function Ensure-Venv([string]$Root) {
  $venvPy = Get-VenvPython -Root $Root
  if ($venvPy) { return $venvPy }

  $sysPy = (Get-Command python -ErrorAction Stop).Source
  Write-Section "Creating virtual environment (.venv)..."
  & $sysPy -m venv (Join-Path $Root ".venv")
  $venvPy = Get-VenvPython -Root $Root
  if (-not $venvPy) { throw "Failed to create venv at: $Root\.venv" }
  return $venvPy
}

function Get-PythonExe([string]$Root) {
  $py = Get-VenvPython -Root $Root
  if (-not $py) { throw ".venv not found. Run: .\scripts\run.ps1 install" }
  return $py
}

function Get-PipNetArgs {
  $args = @("--disable-pip-version-check")
  if ($env:FLEET_PIP_INDEX_URL)      { $args += @("-i", $env:FLEET_PIP_INDEX_URL) }
  if ($env:FLEET_PIP_EXTRA_INDEX_URL){ $args += @("--extra-index-url", $env:FLEET_PIP_EXTRA_INDEX_URL) }
  if ($env:FLEET_PIP_CERT)           { $args += @("--cert", $env:FLEET_PIP_CERT) }

  if ($env:FLEET_PIP_TRUSTED -eq "1") {
    $args += @("--trusted-host","pypi.org","--trusted-host","files.pythonhosted.org","--trusted-host","pypi.pythonhosted.org")
    $args += @("--trusted-host","pypi.python.org")
  }
  return ,$args
}

function Invoke-Pip {
  param(
    [string]$Label,
    [string]$PythonExe,
    [string[]]$Args,
    [switch]$RetryOnSsl
  )

  Write-Section ">>> $Label"
  $net = Get-PipNetArgs
  $cmd = @("-m","pip") + $Args + $net

  $out = & $PythonExe @cmd 2>&1
  $code = $LASTEXITCODE
  $out | ForEach-Object { Write-Host $_ }

  if ($code -ne 0 -and $RetryOnSsl -and ($env:FLEET_PIP_TRUSTED -ne "1")) {
    if ($out -match "CERTIFICATE_VERIFY_FAILED|certificate verify failed|SSLError|TLS") {
      Write-Warning "pip failed due to SSL verification. Retrying with trusted-hosts (set FLEET_PIP_TRUSTED=1 to always do this)."
      $trusted = @("--trusted-host","pypi.org","--trusted-host","files.pythonhosted.org","--trusted-host","pypi.pythonhosted.org","--trusted-host","pypi.python.org")
      $cmd2 = @("-m","pip") + $Args + $trusted + $net
      $out2 = & $PythonExe @cmd2 2>&1
      $code2 = $LASTEXITCODE
      $out2 | ForEach-Object { Write-Host $_ }
      $code = $code2
    }
  }

  if ($code -ne 0) { throw "FAILED ($code): $Label" }
}

function Assert-CoreDeps([string]$PythonExe) {
  $code = "import numpy,pandas,sklearn,scipy,matplotlib,joblib,tqdm,yaml,lifelines,great_expectations,mlflow,streamlit,pptx,PIL; print('deps ok')"
  $out = & $PythonExe -c $code 2>&1
  if ($LASTEXITCODE -ne 0) {
    $out | ForEach-Object { Write-Host $_ }
    throw "Python deps missing in .venv. Run: .\scripts\run.ps1 install"
  }
}

function Ensure-RawDataOrGenerate([string]$PythonExe, [string]$ConfigPath) {
  $p1 = Join-Path $root "data\raw\vehicle_day.csv"
  $p2 = Join-Path $root "data\raw\dtc_event.csv"
  $p3 = Join-Path $root "data\raw\work_order.csv"
  $missing = @()
  foreach ($p in @($p1,$p2,$p3)) { if (-not (Test-Path $p)) { $missing += $p } }

  if ($missing.Count -gt 0) {
    Write-Section "Raw data missing (fresh clone is expected). Generating synthetic data..."
    foreach ($m in $missing) { Write-Host "  - missing: $m" }
    Invoke-Step "0) Generate synthetic data (data/raw/*.csv)" {
      & $PythonExe -m src.data.simulate --config $ConfigPath
    }
  }
}

function Export-DeckPdf([string]$PptxPath, [string]$PdfPath) {
  # 1) Try PowerPoint COM (best fidelity)
  try {
    $ppt = New-Object -ComObject PowerPoint.Application
    # Some PowerPoint versions do not allow hiding; keep it non-intrusive with WithWindow=$false below
    $ppt.Visible = 1

    $fullPptx = (Resolve-Path $PptxPath).Path
    $fullPdf  = (Resolve-Path (Split-Path $PdfPath -Parent)).Path + "\" + (Split-Path $PdfPath -Leaf)

    $pres = $ppt.Presentations.Open($fullPptx, $true, $true, $false)
    # SaveAs PDF (32 = ppSaveAsPDF)
    $pres.SaveAs($fullPdf, 32)
    $pres.Close()
    $ppt.Quit()

    [System.Runtime.InteropServices.Marshal]::ReleaseComObject($pres) | Out-Null
    [System.Runtime.InteropServices.Marshal]::ReleaseComObject($ppt)  | Out-Null
    [GC]::Collect()
    [GC]::WaitForPendingFinalizers()

    Write-Host "PDF generated (PowerPoint): $PdfPath"
    return
  } catch {
    Write-Warning "PowerPoint PDF export failed (will try LibreOffice): $($_.Exception.Message)"
    try { if ($ppt) { $ppt.Quit() | Out-Null } } catch {}
  }

  # 2) Fallback: LibreOffice (headless)
  $soffice = (Get-Command soffice.com -ErrorAction SilentlyContinue)
  if (-not $soffice) { $soffice = (Get-Command soffice.exe -ErrorAction SilentlyContinue) }

  if (-not $soffice) {
    Write-Warning "No PDF exporter found (PowerPoint/LibreOffice). PPTX generated only: $PptxPath"
    return
  }

  $outDir = Split-Path $PdfPath -Parent
  Ensure-Dir $outDir

  Write-Section ">>> Export deck to PDF (LibreOffice)"
  & $soffice.Source --headless --nologo --nofirststartwizard --norestore --convert-to pdf --outdir $outDir $PptxPath | Out-Null

  if (Test-Path $PdfPath) {
    Write-Host "PDF generated (LibreOffice): $PdfPath"
  } else {
    Write-Warning "LibreOffice conversion ran but PDF not found at: $PdfPath"
  }
}

Push-Location $root
try {
  switch ($task) {

    "help" {
      Write-Host (Get-Content $PSCommandPath -TotalCount 40 | Out-String)
      break
    }

    "install" {
      Ensure-GitKeep (Join-Path $root "data\raw")
      Ensure-GitKeep (Join-Path $root "data\processed")

      $python = Ensure-Venv -Root $root

      $req = Join-Path $root "requirements.txt"
      $reqDev = Join-Path $root "requirements-dev.txt"
      if (-not (Test-Path $req))    { throw "Missing requirements.txt (expected at repo root)." }
      if (-not (Test-Path $reqDev)) { throw "Missing requirements-dev.txt (expected at repo root)." }

      # Best-effort upgrade (don’t block install on corp SSL)
      try {
        Invoke-Pip -Label "Upgrading pip/setuptools/wheel (best-effort)" -PythonExe $python -Args @("install","-U","pip","setuptools","wheel") -RetryOnSsl
      } catch {
        Write-Warning "pip upgrade failed; continuing. ($($_.Exception.Message))"
      }

      Invoke-Pip -Label "Installing project dependencies" -PythonExe $python -Args @("install","-r",$req,"-r",$reqDev) -RetryOnSsl

      Invoke-Step "Verify core imports" {
        Assert-CoreDeps -PythonExe $python
      }

      Write-Host "Install complete."
      break
    }

    default {
      $python = Get-PythonExe -Root $root
      Assert-CoreDeps -PythonExe $python

      if (-not (Test-Path $configPath)) { throw "Config not found: $configPath" }

      switch ($task) {

        "data" {
          Invoke-Step "Generate synthetic data (data/raw/*.csv)" {
            & $python -m src.data.simulate --config $configPath
          }
          break
        }

        "validate" {
          Invoke-Step "Data validation" {
            & $python -m src.data.validate --config $configPath
          }
          break
        }

        "eda" {
          Invoke-Step "EDA + stats (KM + Cox)" {
            & $python -m src.reporting.eda_stats --config $configPath
          }
          break
        }

        "train" {
          Invoke-Step "Train models + export artifacts" {
            & $python -m src.models.train --config $configPath
          }
          break
        }

        "triage" {
          Invoke-Step "Export triage snapshot" {
            & $python -m src.reporting.triage_snapshot --config $configPath
          }
          break
        }

        "aliases" {
          Invoke-Step "Create *_latest aliases" {
            & $python -m src.reporting.make_latest_aliases --config $configPath
          }
          break
        }

        "deck" {
          Ensure-Dir (Join-Path $root "report")

          $pptxPath = Join-Path $root "report\auto_deck.pptx"
          $pdfPath  = Join-Path $root "report\auto_deck.pdf"

          Invoke-Step "Generate report\auto_deck.pptx" {
            & $python -m src.reporting.make_auto_deck --config $configPath --out $pptxPath
          }

          # Optional: print slide count if available
          try {
            $cnt = & $python -c "from pptx import Presentation; p=Presentation(r'report/auto_deck.pptx'); print('Slides:', len(p.slides))"
            Write-Host $cnt
          } catch {}

          Export-DeckPdf -PptxPath $pptxPath -PdfPath $pdfPath
          break
        }

        "report" {
          Write-Section "=== Report pipeline: data(if missing) -> validate -> EDA -> train -> triage -> aliases -> deck ==="

          Ensure-RawDataOrGenerate -PythonExe $python -ConfigPath $configPath

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
            & $python -m src.reporting.triage_snapshot --config $configPath
          }

          Invoke-Step "5) Create *_latest aliases (stable filenames)" {
            & $python -m src.reporting.make_latest_aliases --config $configPath
          }

          Invoke-Step "6) Deck" {
            & $PSCommandPath deck $config
          }
          break
        }

        "app" {
          $appPath = Join-Path $root "app\streamlit_app.py"
          if (-not (Test-Path $appPath)) { throw "Streamlit app not found: $appPath" }

          Invoke-Step "Run Streamlit app" {
            & $python -m streamlit run $appPath -- --config $configPath
          }
          break
        }

        "test" {
          Invoke-Step "Run pytest" {
            & $python -m pytest -q
          }
          break
        }

        "lint" {
          Invoke-Step "Ruff lint" {
            & $python -m ruff check .
          }
          break
        }

        "format" {
          Invoke-Step "Black format" {
            & $python -m black .
          }
          break
        }
      }
    }
  }

} finally {
  Pop-Location
}