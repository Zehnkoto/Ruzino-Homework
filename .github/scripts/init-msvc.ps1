# Initialize MSVC Build Tools environment
# This script finds and initializes vcvars64.bat for Visual Studio 2022

$vsVersions = @(
    "Community",
    "Professional", 
    "Enterprise"
)

foreach ($version in $vsVersions) {
    $vcvarsPath = "C:\Program Files\Microsoft Visual Studio\2022\$version\VC\Auxiliary\Build\vcvars64.bat"
    if (Test-Path $vcvarsPath) {
        Write-Output "Initializing MSVC from $version edition"
        & cmd /c "call `"$vcvarsPath`" && set" | Where-Object { $_ -match '=' } | ForEach-Object {
            $name, $value = $_.split('=', 2)
            if ($name -and $value) {
                [Environment]::SetEnvironmentVariable($name, $value)
            }
        }
        return
    }
}

Write-Error "Visual Studio 2022 not found"
exit 1
