# Imports the MSVC x64 build environment into the current PowerShell session.
# This is a helper used by build/test commands; dot-source it in the same
# pwsh invocation (a fresh tools.pwsh call does not inherit shell state).
$vcvars = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
if (Test-Path $vcvars) {
  cmd /c "`"$vcvars`" && set" | ForEach-Object {
    if ($_ -match '^(.*?)=(.*)$') { Set-Item -Path "env:$($matches[1])" -Value $matches[2] }
  }
} else {
  throw "vcvars64.bat not found at $vcvars"
}
Write-Host ("[vsenv] cl={0} nvcc={1}" -f [bool](Get-Command cl -ErrorAction SilentlyContinue), [bool](Get-Command nvcc -ErrorAction SilentlyContinue))
