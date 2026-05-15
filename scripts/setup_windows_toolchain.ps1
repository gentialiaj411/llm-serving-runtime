param(
  [string]$VcpkgRoot = 'C:\tools\vcpkg',
  [string]$ProtocBin = "$env:LOCALAPPDATA\Microsoft\WinGet\Packages\Google.Protobuf_Microsoft.Winget.Source_8wekyb3d8bbwe\bin",
  [string]$CudaBin = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\bin'
)

if (Test-Path $ProtocBin) {
  $env:Path = "$ProtocBin;$env:Path"
}
if (Test-Path $CudaBin) {
  $env:Path = "$CudaBin;$env:Path"
}
if (Test-Path "$VcpkgRoot\scripts\buildsystems\vcpkg.cmake") {
  $env:CMAKE_TOOLCHAIN_FILE = "$VcpkgRoot\scripts\buildsystems\vcpkg.cmake"
}

Write-Host "CMAKE_TOOLCHAIN_FILE=$env:CMAKE_TOOLCHAIN_FILE"
Write-Host "protoc path check:"; Get-Command protoc -ErrorAction SilentlyContinue | Format-List -Property Source
Write-Host "nvcc path check:"; Get-Command nvcc -ErrorAction SilentlyContinue | Format-List -Property Source
