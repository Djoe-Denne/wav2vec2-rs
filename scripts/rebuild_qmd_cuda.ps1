param(
    [string]$CudaPath = $env:CUDA_PATH,
    [string]$CudaArch = "86",
    [switch]$SkipDownload,
    [switch]$SkipBuild,
    [switch]$SkipEmbed
)

$ErrorActionPreference = "Stop"

function Resolve-RequiredPath {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$Description
    )

    if (-not (Test-Path -LiteralPath $Path)) {
        throw "$Description not found: $Path"
    }
    (Resolve-Path -LiteralPath $Path).Path
}

function Find-SystemNode {
    $candidate = "C:\Program Files\nodejs\node.exe"
    if (Test-Path -LiteralPath $candidate) {
        return (Resolve-Path -LiteralPath $candidate).Path
    }

    $node = Get-Command node -ErrorAction Stop
    return $node.Source
}

function Find-VcVars64 {
    # If cl.exe is already visible, prefer the current environment. This avoids
    # choosing a mismatched Visual Studio installation from vswhere.
    if (Get-Command cl.exe -ErrorAction SilentlyContinue) {
        return $null
    }

    $vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
    if (Test-Path -LiteralPath $vswhere) {
        $installPath = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
        if ($installPath) {
            $vcvars = Join-Path $installPath "VC\Auxiliary\Build\vcvars64.bat"
            if (Test-Path -LiteralPath $vcvars) {
                return (Resolve-Path -LiteralPath $vcvars).Path
            }
        }
    }

    $candidates = @(
        "${env:ProgramFiles}\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat",
        "${env:ProgramFiles}\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat",
        "${env:ProgramFiles}\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat",
        "${env:ProgramFiles}\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvars64.bat"
    )

    foreach ($candidate in $candidates) {
        if (Test-Path -LiteralPath $candidate) {
            return (Resolve-Path -LiteralPath $candidate).Path
        }
    }

    return $null
}

function Invoke-Nlc {
    param(
        [Parameter(Mandatory = $true)][string[]]$Arguments
    )

    $argText = ($Arguments | ForEach-Object { '"' + ($_ -replace '"', '\"') + '"' }) -join " "

    if ($script:VcVars64) {
        $command = "call `"$script:VcVars64`" >nul && `"$script:NodeExe`" `"$script:NlcCli`" $argText"
        & cmd.exe /d /s /c $command
    } else {
        & $script:NodeExe $script:NlcCli @Arguments
    }

    if ($LASTEXITCODE -ne 0) {
        throw "node-llama-cpp command failed with exit code ${LASTEXITCODE}: $($Arguments -join ' ')"
    }
}

function Invoke-Qmd {
    param(
        [Parameter(Mandatory = $true)][string[]]$Arguments
    )

    & $script:NodeExe $script:QmdCli @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "qmd command failed with exit code ${LASTEXITCODE}: $($Arguments -join ' ')"
    }
}

$script:NodeExe = Find-SystemNode
$npmGlobal = Join-Path $env:APPDATA "npm"
$script:QmdRoot = Resolve-RequiredPath (Join-Path $npmGlobal "node_modules\@tobilu\qmd") "QMD global package"
$script:QmdCli = Resolve-RequiredPath (Join-Path $script:QmdRoot "dist\cli\qmd.js") "QMD CLI"
$script:NlcCli = Resolve-RequiredPath (Join-Path $script:QmdRoot "node_modules\node-llama-cpp\dist\cli\cli.js") "node-llama-cpp CLI"

if ([string]::IsNullOrWhiteSpace($CudaPath)) {
    $CudaPath = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2"
}
$CudaPath = Resolve-RequiredPath $CudaPath "CUDA toolkit"
$nvcc = Resolve-RequiredPath (Join-Path $CudaPath "bin\nvcc.exe") "NVCC"

$script:VcVars64 = Find-VcVars64

$env:CUDA_PATH = $CudaPath
$env:CUDACXX = $nvcc
$env:NODE_LLAMA_CPP_CMAKE_OPTION_CMAKE_CUDA_ARCHITECTURES = $CudaArch
$env:NODE_LLAMA_CPP_CMAKE_OPTION_GGML_CUDA = "ON"
$env:NODE_LLAMA_CPP_CMAKE_OPTION_GGML_CUDA_NO_VMM = "ON"
$env:GYP_MSVS_VERSION = "2022"
$env:npm_config_msvs_version = "2022"

Write-Host "Node:      $script:NodeExe"
Write-Host "QMD root:  $script:QmdRoot"
Write-Host "CUDA:      $env:CUDA_PATH"
Write-Host "NVCC:      $env:CUDACXX"
Write-Host "CUDA arch: sm_$CudaArch"
Write-Host "VMM:       disabled via GGML_CUDA_NO_VMM=ON"
if ($script:VcVars64) {
    Write-Host "VC env:    $script:VcVars64"
} else {
    Write-Warning "Visual C++ vcvars64.bat was not found. Build may fail if cl.exe is not already on PATH."
}

Push-Location $script:QmdRoot
try {
    Write-Host ""
    Write-Host "Preflight: node-llama-cpp GPU inspection"
    Invoke-Nlc @("inspect", "gpu")

    if (-not $SkipDownload) {
        Write-Host ""
        Write-Host "Downloading llama.cpp source for local CUDA build..."
        Invoke-Nlc @("source", "download", "--gpu", "cuda", "--skipBuild")
    }

    if (-not $SkipBuild) {
        Write-Host ""
        Write-Host "Building node-llama-cpp with CUDA and GGML_CUDA_NO_VMM=ON..."
        Invoke-Nlc @("source", "build", "--gpu", "cuda")
    }

    Write-Host ""
    Write-Host "Post-build: node-llama-cpp GPU inspection"
    Invoke-Nlc @("inspect", "gpu")

    Write-Host ""
    Write-Host "QMD status"
    Invoke-Qmd @("status")

    if (-not $SkipEmbed) {
        Write-Host ""
        Write-Host "Running qmd embed..."
        Invoke-Qmd @("embed")
    }
} finally {
    Pop-Location
}
