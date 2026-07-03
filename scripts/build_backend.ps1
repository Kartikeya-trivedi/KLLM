# Builds the CUDA backend shared library (toyengine_backend.dll).
# nvcc on Windows requires MSVC's cl.exe as host compiler; we locate it via
# vswhere so this works without running from a "Developer PowerShell".
#
# Usage:  .\scripts\build_backend.ps1 [-Arch sm_75] [-Configuration Release]
#   sm_75 = GTX 1650 (this lab box), sm_86 = A6000 (engine box)

param(
    [string]$Arch = "sm_75",
    [ValidateSet("Release", "Debug")]
    [string]$Configuration = "Release"
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot

# --- Locate MSVC cl.exe (x64 host, x64 target) via vswhere ---
$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
if (-not (Test-Path $vswhere)) {
    throw "vswhere.exe not found - install Visual Studio with the C++ workload."
}
$vsPath = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (-not $vsPath) {
    throw "No Visual Studio installation with the C++ toolchain found."
}
$msvcVersion = Get-ChildItem "$vsPath\VC\Tools\MSVC" | Sort-Object Name -Descending | Select-Object -First 1
$ccbin = "$($msvcVersion.FullName)\bin\Hostx64\x64"
if (-not (Test-Path "$ccbin\cl.exe")) {
    throw "cl.exe not found under $ccbin"
}

# --- Build ---
$buildDir = Join-Path $repoRoot "build"
New-Item -ItemType Directory -Force $buildDir | Out-Null

$out = Join-Path $buildDir "toyengine_backend.dll"
$sources = @(Get-ChildItem (Join-Path $repoRoot "backend") -Filter *.cu | ForEach-Object { $_.FullName })

$flags = @("-shared", "-arch=$Arch", "-ccbin", $ccbin, "-lcublas", "-o", $out)
if ($Configuration -eq "Debug") {
    $flags += @("-g", "-G", "-lineinfo")
} else {
    $flags += @("-O2", "-lineinfo")
}

Write-Host "nvcc $($flags -join ' ') $($sources -join ' ')"
& nvcc @flags @sources
if ($LASTEXITCODE -ne 0) {
    throw "nvcc failed with exit code $LASTEXITCODE"
}
Write-Host "Built $out ($Arch, $Configuration)"
