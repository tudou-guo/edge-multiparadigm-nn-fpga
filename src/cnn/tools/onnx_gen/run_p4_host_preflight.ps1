<#
Read-only Windows tool/resource gate; never starts a build or writes a gate.
Formal execution MUST use the default full check (no -AuditOnly):
  powershell -NoProfile -ExecutionPolicy Bypass -File tools\onnx_gen\run_p4_host_preflight.ps1
  powershell ... -EvidenceLabel p7_review   # NEW probe; existing files refused
-AuditOnly is for read-only inspection/repair: exit 0 means tools reachable,
NOT permission/resource readiness to execute stages. The result distinguishes
toolchain_ready from execution_ready. No evidence file is written by default.
Exit: 0 ready (or audit tools ready), 1 missing inputs/evidence collision,
2 toolchain unavailable/non-zero tool probe, 3 insufficient/unknown resources.
Disk >=40 GiB free (2026-10-21 resource-gate revision) and RAM >=12 GiB.
40 GiB is measured: the P2 full run (implementation + replay + regressions)
occupied 3.55 GiB; P4 (22 replays + two attempts + release) is estimated at
3-4x that, so 40 GiB leaves more than 2x headroom for Vivado temp.  The old
100 GiB line had no measurement behind it and is superseded here.
#>
param(
    [string]$Run = "build\onnx_e1\e1p4_20261021_110123",
    [string]$Vivado = "E:\Vitis\Vivado\2021.1\bin\vivado.bat",
    [string]$Python = "E:\miniconda\python.exe",
    [ValidateRange(10, 1000)][int]$MinFreeGiB = 40,
    [ValidatePattern('^[A-Za-z0-9][A-Za-z0-9_-]*$')][string]$EvidenceLabel = "",
    [switch]$AuditOnly
)

$ErrorActionPreference = "Stop"
$root = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\.."))
$runPath = [IO.Path]::GetFullPath($(if ([IO.Path]::IsPathRooted($Run)) { $Run } else { Join-Path $root $Run }))
if (-not $runPath.StartsWith($root.TrimEnd('\') + '\', [StringComparison]::OrdinalIgnoreCase)) {
    Write-Error "Run must be inside this workspace" -ErrorAction Continue
    exit 1
}
$probes = @()

function Add-Probe {
    param([string]$Name, [string]$Target, [bool]$Ok, [string]$Detail)
    $script:probes += [ordered]@{ name = $Name; target = $Target; ok = $Ok; detail = $Detail }
}

function Invoke-P4ToolProbe {
    param([string]$Path, [string[]]$Arguments, [string]$ExpectedPattern)
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        return @{ ok = $false; exit_code = $null; detail = "not found" }
    }
    try {
        # A real short invocation is required; a .bat existing is insufficient.
        # Native stderr can be an ErrorRecord in Windows PowerShell 5.1.
        $savedPreference = $ErrorActionPreference
        $ErrorActionPreference = "Continue"
        try {
            $global:LASTEXITCODE = $null
            $output = (& $Path @Arguments 2>&1 | Out-String).Trim()
            $code = $global:LASTEXITCODE
        } finally { $ErrorActionPreference = $savedPreference }
        return @{ ok = ($null -ne $code -and $code -eq 0 -and $output -match $ExpectedPattern)
                  exit_code = $code; detail = ("exit=" + $code + [Environment]::NewLine + $output) }
    } catch {
        return @{ ok = $false; exit_code = $null; detail = ("invoke failed: " + $_.Exception.Message) }
    }
}

function Get-P4PreflightDecision {
    param([bool]$ToolchainReady, [bool]$InputsReady, [bool]$ResourcesReady, [bool]$IsAudit)
    $executionReady = $ToolchainReady -and $InputsReady -and $ResourcesReady
    $code = 0
    if (-not $ToolchainReady) { $code = 2 }
    elseif (-not $IsAudit -and -not $InputsReady) { $code = 1 }
    elseif (-not $IsAudit -and -not $ResourcesReady) { $code = 3 }
    $conclusion = if (-not $ToolchainReady) { "host_tools_unreachable" }
                  elseif ($IsAudit) { "audit_tools_ready_not_execution_authorization" }
                  elseif (-not $InputsReady) { "host_inputs_missing" }
                  elseif (-not $ResourcesReady) { "host_resource_blocked" }
                  else { "host_execution_ready" }
    return @{ exit_code = $code; conclusion = $conclusion; execution_ready = $executionReady }
}

Add-Probe "platform" "PowerShell / Windows" ($env:OS -eq "Windows_NT") ("OS=" + $env:OS + " PSVersion=" + $PSVersionTable.PSVersion)
$v = Invoke-P4ToolProbe $Vivado @("-version") 'Vivado\s+v?2021\.1'
Add-Probe "vivado" $Vivado $v.ok $v.detail
$xsim = Join-Path (Split-Path $Vivado -Parent) "xsim.bat"
$x = Invoke-P4ToolProbe $xsim @("-help") '(?i)xsim|usage'
Add-Probe "xsim" $xsim $x.ok $x.detail
$p = Invoke-P4ToolProbe $Python @("-V") '^Python\s+3\.'
Add-Probe "python" $Python $p.ok $p.detail

$runOk = Test-Path -LiteralPath $runPath -PathType Container
$frozen = @("protected_before.json", "protected_after.json", "inheritance_table.json", "p4_final_manifest.json")
$missing = @($frozen | Where-Object { -not (Test-Path -LiteralPath (Join-Path $runPath $_) -PathType Leaf) })
$inputsReady = $runOk -and $missing.Count -eq 0
Add-Probe "run" $runPath $inputsReady $(if (-not $runOk) { "run dir missing" } elseif ($missing.Count) { "missing: " + ($missing -join ",") } else { "identity/protection files present" })
$hostStart = Join-Path $runPath "identity_host_start.json"
Add-Probe "host_identity" $hostStart $true $(if (Test-Path -LiteralPath $hostStart) { "already written; freshness separately checked by driver" } else { "not written; this probe does not create host identity" })

try {
    $drive = New-Object IO.DriveInfo ([IO.Path]::GetPathRoot($runPath))
    $freeBytes = $drive.AvailableFreeSpace
    $requiredBytes = [double]$MinFreeGiB * 1GB
    Add-Probe "disk" $drive.Name ($freeBytes -ge $requiredBytes) (([math]::Round($freeBytes / 1GB, 1)).ToString() + " GiB free; requires >= " + $MinFreeGiB + " GiB")
} catch { Add-Probe "disk" $runPath $false ("unable to measure disk: " + $_.Exception.Message) }
try {
    $memoryBytes = (Get-CimInstance Win32_ComputerSystem -ErrorAction Stop).TotalPhysicalMemory
    Add-Probe "memory" "physical RAM" ($memoryBytes -ge 12GB) (([math]::Round($memoryBytes / 1GB, 1)).ToString() + " GiB total; requires >=12 GiB; process peaks must still be scheduled")
} catch { Add-Probe "memory" "physical RAM" $false ("unable to measure RAM: " + $_.Exception.Message) }

$toolchainReady = @($probes | Where-Object { $_.name -in @("platform", "vivado", "xsim", "python") -and -not $_.ok }).Count -eq 0
$resourcesReady = @($probes | Where-Object { $_.name -in @("disk", "memory") -and -not $_.ok }).Count -eq 0
$decision = Get-P4PreflightDecision $toolchainReady $inputsReady $resourcesReady ([bool]$AuditOnly)
$result = [ordered]@{
    schema = "e1-p4-host-env-probe-2"; run = $Run; stage = "E1-P4-STAGE1"
    date = (Get-Date).ToString("yyyy-MM-ddTHH:mm:sszzz"); timezone = "Asia/Shanghai"
    audit_only = [bool]$AuditOnly; min_free_gib = $MinFreeGiB; toolchain_ready = $toolchainReady
    inputs_ready = $inputsReady; resources_ready = $resourcesReady
    execution_ready = $decision.execution_ready; exit_code = $decision.exit_code
    conclusion = $decision.conclusion; probes = $probes
    next = $(if ($AuditOnly) { "READ-ONLY ONLY: default full preflight must return 0 before execution" }
             elseif ($decision.execution_ready) { "identity -> plan -> replay -Pilot -> impl (max two) -> final identity/plan -RefreshPlan -> full replay" }
             else { "STOP execution: resolve failed probes; never lower thresholds or reuse old readiness" })
}
$json = $result | ConvertTo-Json -Depth 6
$json

if ($EvidenceLabel) {
    if (-not $runOk) { Write-Error "cannot write evidence: run directory missing" -ErrorAction Continue; exit 1 }
    $out = Join-Path $runPath ("identity_host_env_probe_" + $EvidenceLabel + ".json")
    try {
        # CreateNew is atomic and refuses replacement of any historical probe.
        $stream = [IO.File]::Open($out, [IO.FileMode]::CreateNew, [IO.FileAccess]::Write)
        try {
            $bytes = (New-Object Text.UTF8Encoding($false)).GetBytes($json + [Environment]::NewLine)
            $stream.Write($bytes, 0, $bytes.Length)
        } finally { $stream.Dispose() }
        Write-Host ("wrote NEW evidence " + $out)
    } catch { Write-Error ("evidence not written (existing files are protected): " + $_.Exception.Message) -ErrorAction Continue; exit 1 }
}
exit $decision.exit_code
