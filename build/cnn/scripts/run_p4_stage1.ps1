<#
Windows-only P4 stage-one driver. Execution calls first enforce the FULL host
preflight, including >=40 GiB free disk (2026-10-21 resource-gate revision);
AuditOnly is never used here.
status is read-only; finalize may record failed/incomplete gates with only a
Python executable, even while execution resources are blocked. Finalize never
authorizes heavy execution or bypasses the separate release resource gate.
Formal order:
  -Stage identity                 # current host + fresh protected-tree scan
  -Stage plan                     # freeze complete 20+2 plan
  -Stage replay -Pilot            # ONLY s00_first_fit, not all 22
  -Stage impl -Attempt attempt1   # requires a current complete pilot
  # at most one targeted repair + new plan/pilot + attempt2 after a failed a1
  -Stage identity                 # final source/tool freeze after physical fix
  -Stage plan -RefreshPlan        # archive prior plan, bind final source
  -Stage replay -Parallel 2       # all 22, stale prior results cannot be skipped
  -Stage regress; -Stage finalize; -Stage release
-Stage replay -Jobs s00_first_fit selects explicit jobs; it is not a full gate.
-Force can force selected replay jobs, NEVER overwrite an implementation try.
-DryRun prints commands only; it creates no log or execution/readiness evidence.
#>
param(
    [Parameter(Mandatory=$true)]
    [ValidateSet("identity","plan","replay","impl","regress","finalize","release","recover","status")]
    [string]$Stage,
    [string]$Run = "build\onnx_e1\e1p4_20261021_110123",
    [ValidateSet("attempt1","attempt2")][string]$Attempt = "attempt1",
    [ValidateRange(1,3)][int]$Parallel = 1,
    [ValidateSet(20)][int]$FirstFitCount = 20,
    [string]$NoReuseClasses = "0,1",
    [Alias("Job")][string[]]$Jobs = @(),
    [switch]$Pilot,
    [switch]$RefreshPlan,
    [string]$Groups = "",
    [string]$Reason = "",
    [string]$Note = "",
    [string]$Dcp = "",
    [string]$Python = "E:\miniconda\python.exe",
    [switch]$Force,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$root = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\.."))
$runPath = [IO.Path]::GetFullPath($(if ([IO.Path]::IsPathRooted($Run)) { $Run } else { Join-Path $root $Run }))
if (-not $runPath.StartsWith($root.TrimEnd('\') + '\', [StringComparison]::OrdinalIgnoreCase)) {
    throw "Run must be inside this workspace"
}
$logDir = Join-Path $runPath ("logs\" + $Stage)
$stamp = Get-Date -Format "yyyyMMdd_HHmmss_fffffff"

function ConvertTo-P4NativeArgument {
    param([AllowEmptyString()][string]$Argument)
    # Windows CommandLineToArgvW/C-runtime quoting, including trailing slashes.
    $escaped = [regex]::Replace($Argument, '(\\*)"', '$1$1\"')
    $escaped = [regex]::Replace($escaped, '(\\+)$', '$1$1')
    return '"' + $escaped + '"'
}

function Get-P4CandidateIp {
    param([string]$RunPath)
    $config = Join-Path $RunPath "optimization_manifest.json"
    if (-not (Test-Path -LiteralPath $config)) {
        return [pscustomobject]@{ Name = "e1p4npu"; Version = "1.0" }
    }
    $candidate = Get-Content -LiteralPath $config -Raw | ConvertFrom-Json
    $ipName = [string]$candidate.planned_ip.name
    $ipVersion = [string]$candidate.planned_ip.version
    if ($ipName -notmatch '^[A-Za-z_][A-Za-z0-9_]*$' -or $ipVersion -notmatch '^[0-9]+\.[0-9]+$') {
        throw "invalid optimization IP name/version"
    }
    return [pscustomobject]@{ Name = $ipName; Version = $ipVersion }
}

function Get-P4PackagedIpDirectory {
    param([string]$RunPath, $PlatformManifest)
    $active = ([string]$PlatformManifest.active_attempt_dir).Replace("/", "\")
    if ($active -notmatch '^platform\\attempt[12]$') { throw "invalid active attempt pointer" }
    $attemptRoot = [IO.Path]::GetFullPath((Join-Path $RunPath $active))
    $ip = $PlatformManifest.stages.ip
    $vlnv = ([string]$ip.vlnv).Split(":")
    if ($vlnv.Count -ne 4 -or $vlnv[2] -notmatch '^[A-Za-z_][A-Za-z0-9_]*$' -or
        $vlnv[3] -notmatch '^[0-9]+\.[0-9]+$' -or -not [string]$ip.repo) {
        throw "invalid packaged IP repository/VLNV"
    }
    $repo = [IO.Path]::GetFullPath($(if ([IO.Path]::IsPathRooted([string]$ip.repo)) {
        [string]$ip.repo
    } else { Join-Path $root ([string]$ip.repo) }))
    if (-not $repo.StartsWith($attemptRoot + '\', [StringComparison]::OrdinalIgnoreCase)) {
        throw "IP repository does not belong to the active implementation attempt"
    }
    if (Test-Path -LiteralPath (Join-Path $RunPath "optimization_manifest.json")) {
        $planned = Get-P4CandidateIp $RunPath
        if ($vlnv[2] -ne $planned.Name -or $vlnv[3] -ne $planned.Version) {
            throw "packaged IP version differs from the optimization candidate"
        }
    }
    return (Join-Path $repo ($vlnv[2] + "_" + $vlnv[3]))
}

function Start-PyWorker {
    param([string[]]$PyArgs, [string]$LogName)
    $cmd = @("-B") + $PyArgs
    Write-Host ("[py] " + $Python + " " + (($cmd | ForEach-Object { ConvertTo-P4NativeArgument $_ }) -join " "))
    if ($DryRun) { return $null }
    New-Item -ItemType Directory -Force -Path $logDir | Out-Null
    $log = Join-Path $logDir ($LogName + "_" + $stamp + ".log")
    $info = New-Object System.Diagnostics.ProcessStartInfo
    $info.FileName = $Python
    $info.Arguments = ($cmd | ForEach-Object { ConvertTo-P4NativeArgument $_ }) -join " "
    $info.WorkingDirectory = $root
    $info.UseShellExecute = $false
    $info.CreateNoWindow = $true
    $info.RedirectStandardOutput = $true
    $info.RedirectStandardError = $true
    $proc = New-Object System.Diagnostics.Process
    $proc.StartInfo = $info
    $null = $proc.Start()
    # Start draining BOTH pipes immediately. Sequential ReadToEnd can deadlock
    # if stderr fills while waiting for stdout, including across other workers.
    $stdoutTask = $proc.StandardOutput.ReadToEndAsync()
    $stderrTask = $proc.StandardError.ReadToEndAsync()
    return [pscustomobject]@{ Process = $proc; Log = $log
                            StdoutTask = $stdoutTask; StderrTask = $stderrTask }
}

function Complete-PyWorker {
    param($Worker)
    if ($DryRun) { return 0 }
    $Worker.Process.WaitForExit()
    $out = $Worker.StdoutTask.GetAwaiter().GetResult()
    $err = $Worker.StderrTask.GetAwaiter().GetResult()
    $rc = $Worker.Process.ExitCode
    # Separate channels are explicit; this is not a claimed interleaved timeline.
    $nl = [Environment]::NewLine
    [IO.File]::WriteAllText($Worker.Log, "[stdout]" + $nl + $out + $nl + "[stderr]" + $nl + $err,
                           (New-Object Text.UTF8Encoding($false)))
    if ($out) { Write-Host $out }
    if ($err) { Write-Host $err }
    Write-Host ("[worker done] rc=" + $rc + " log=" + $Worker.Log)
    $Worker.Process.Dispose()
    return $rc
}

function Invoke-PyLogged {
    param([string[]]$PyArgs, [string]$LogName)
    $worker = Start-PyWorker $PyArgs $LogName
    return (Complete-PyWorker $worker)
}

function Select-P4ReplayJobs {
    param($Plan, [string[]]$RequestedJobs, [bool]$IsPilot)
    $requested = @($RequestedJobs | ForEach-Object { $_ -split "," } |
                   ForEach-Object { $_.Trim() } | Where-Object { $_ })
    if ($IsPilot -and $requested.Count) { throw "Use -Pilot or -Jobs, not both" }
    if ($IsPilot) { $requested = @("s00_first_fit") }
    $available = @($Plan.jobs | ForEach-Object { $_.job_id })
    if (-not $available.Count) { throw "Replay plan has no jobs" }
    if (-not $requested.Count) { return $available }
    $unknown = @($requested | Where-Object { $_ -notin $available })
    if ($unknown.Count) { throw ("Unknown replay jobs: " + ($unknown -join ",")) }
    return @($requested | Select-Object -Unique)
}

function Assert-P4AttemptAvailable {
    param([string]$RunPath, [string]$AttemptName)
    $target = Join-Path $RunPath ("platform\" + $AttemptName)
    if (Test-Path -LiteralPath $target) {
        throw ("Refusing to overwrite started/complete " + $AttemptName +
               "; -Force cannot bypass. Keep evidence; use report recovery or the next allowed attempt.")
    }
    if ($AttemptName -eq "attempt2") {
        $a1Dir = Join-Path $RunPath "platform\attempt1"
        $prior = Join-Path $a1Dir "platform_manifest.json"
        if (-not (Test-Path -LiteralPath $prior -PathType Leaf)) { throw "attempt2 requires attempt1 evidence" }
        $a1 = Get-Content -LiteralPath $prior -Raw | ConvertFrom-Json
        # Qualification, not the raw Vivado exit code, decides.  A zero exit code
        # with a bitstream but negative slack used to be recorded as status=pass,
        # which silently blocked the one legitimate second build; conversely a
        # true physical pass must refuse a rebuild.  This is the fast guard;
        # p4_platform.py re-checks with the full decision table (attempt budget,
        # running process, unknown state) immediately before launching anything.
        $impl = $a1.stages.implementation
        if (-not $impl) { throw "attempt2 refused: attempt1 has no implementation record" }
        $physical = [string]$impl.physical_status
        $flow = [string]$impl.flow_status
        if ($physical -eq "pass") {
            throw "attempt2 refused: attempt1 already qualifies physically at 5 ns; a rebuild is unnecessary"
        }
        if ($physical -eq "fail") { return }
        # No recorded qualification: allow only a confirmed, process-ended
        # interruption that has been registered as an auditable terminal state.
        $recoveryPath = Join-Path $a1Dir "attempt_recovery.json"
        if (Test-Path -LiteralPath $recoveryPath -PathType Leaf) {
            $recovery = Get-Content -LiteralPath $recoveryPath -Raw | ConvertFrom-Json
            if ($recovery.terminal_state -eq "interrupted") { return }
            if ($recovery.terminal_state -eq "physical_fail") { return }
            if ($recovery.terminal_state -eq "physical_pass") {
                throw "attempt2 refused: recovery records a physical pass for attempt1"
            }
        }
        if ($flow -eq "running") {
            throw "attempt2 refused: attempt1 process is still running; do not start a concurrent rebuild"
        }
        if ($flow -eq "failed" -or $flow -eq "timeout") {
            # The tool ran and its process ended without a design.  The attempt
            # is spent and one directed rebuild remains.
            return
        }
        throw ("attempt2 refused: attempt1 qualification is unknown (flow=" + $flow +
               "); register a terminal state first: run_p4_stage1.ps1 -Stage recover -Attempt attempt1 -Reason interrupted")
    }
}

function Get-P4PreflightMode {
    param([string]$StageName)
    if ($StageName -eq "status") { return "none" }
    # finalize and recover only read/write evidence records; they must work
    # while heavy execution is resource-blocked, or an interrupted attempt
    # could never be given its auditable terminal state.
    if ($StageName -in @("finalize", "recover")) { return "python_only" }
    return "full"
}

try {
    if (($Jobs.Count -or $Pilot) -and $Stage -ne "replay") { throw "-Jobs/-Pilot only apply to replay" }
    if ($RefreshPlan -and $Stage -ne "plan") { throw "-RefreshPlan only applies to -Stage plan" }
    if (($Reason -or $Note) -and $Stage -ne "recover") { throw "-Reason/-Note only apply to -Stage recover" }
    if ($Stage -eq "recover" -and -not $Reason) { throw "-Stage recover requires an explicit -Reason" }
    $preflightMode = Get-P4PreflightMode $Stage
    if ($preflightMode -eq "python_only") {
        Write-Host ("[" + $Stage + "] evidence-only stage; resource blocks remain in force for execution/release")
        if (-not $DryRun -and -not (Test-Path -LiteralPath $Python -PathType Leaf)) {
            throw ($Stage + " requires the configured Python executable")
        }
    } elseif ($preflightMode -eq "full") {
        $preflight = Join-Path $PSScriptRoot "run_p4_host_preflight.ps1"
        Write-Host "[preflight] full execution check; resource failures stop this stage"
        if (-not $DryRun) {
            & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $preflight -Run $runPath -Python $Python
            $preflightRc = $LASTEXITCODE
            if ($preflightRc -ne 0) { Write-Host ("preflight blocked rc=" + $preflightRc); exit $preflightRc }
        }
    }

    switch ($Stage) {
        "status" {
            # Read-only: do not run the finalizer, which writes gate manifests.
            $rc = Invoke-PyLogged @("tools\onnx_gen\p4_model_rtl.py","status","--run",$runPath) "status"
            if ($rc -ne 0) { exit $rc }
        }
        "identity" {
            $rc = Invoke-PyLogged @("tools\onnx_gen\p4_identity.py","--run",$runPath,"--stage","after","--label","host_start","--assemble") "identity"
            if ($rc -ne 0) { throw "identity assembly failed" }
            $rc = Invoke-PyLogged @("tools\onnx_gen\p4_identity.py","--run",$runPath,"--stage","verify","--fresh") "identity_verify_fresh"
            if ($rc -ne 0) { throw "fresh protected-tree verification failed" }
        }
        "plan" {
            $classes = @($NoReuseClasses -split "," | ForEach-Object { [int]$_.Trim() })
            if ($classes.Count -ne 2 -or @($classes | Select-Object -Unique).Count -ne 2) {
                throw "Final plan requires two distinct no_reuse classes"
            }
            $pyArgs = @("tools\onnx_gen\p4_model_rtl.py","plan","--run",$runPath,
                        "--first-fit-count","20","--no-reuse-classes",($classes -join ","))
            if ($RefreshPlan) { $pyArgs += "--refresh" }
            $rc = Invoke-PyLogged $pyArgs "replay_plan"
            if ($rc -ne 0) { throw "plan freeze/refresh failed; preserve prior plan and inspect identity" }
        }
        "impl" {
            Assert-P4AttemptAvailable $runPath $Attempt
            $rc = Invoke-PyLogged @("tools\onnx_gen\p4_model_rtl.py","check-job","--run",$runPath,"--job","s00_first_fit") "pilot_current"
            if ($rc -ne 0) { throw "implementation requires a complete current-source s00_first_fit pilot" }
            $generated = Join-Path $runPath "model\rtl\s00\first_fit\compile\generated"
            if (-not $DryRun -and -not (Test-Path -LiteralPath $generated)) { throw "current pilot generated RTL missing" }
            $candidateIp = Get-P4CandidateIp $runPath
            $rc = Invoke-PyLogged @("tools\onnx_gen\p4_platform.py","--run",$runPath,"--generated",$generated,
                                    "--ip-name",$candidateIp.Name,"--ip-version",$candidateIp.Version,
                                    "--attempt",$Attempt,"--stage","all","--synth-jobs","8","--impl-jobs","8") "impl"
            if ($rc -ne 0) {
                # Not every non-zero exit is a physical verdict.  Point the host
                # at the qualification record rather than guessing.
                throw ("implementation did not qualify; inspect platform\" + $Attempt +
                       "\implementation_attempt.json and export\ for physical_status/flow_status. " +
                       "Keep this attempt; do not delete it or hand-edit the JSON.")
            }
        }
        "regress" {
            $pyArgs = @("tools\onnx_gen\p4_regressions.py","--run",$runPath,"--merge")
            if ($Groups) { $pyArgs += @("--groups", $Groups) }
            $rc = Invoke-PyLogged $pyArgs "regress"
            if ($rc -ne 0) { throw "regressions failed" }
        }
        "replay" {
            $planPath = Join-Path $runPath "replay\replay_plan.json"
            if (-not (Test-Path -LiteralPath $planPath)) { throw "Run -Stage plan before replay; use -Pilot for s00 only" }
            $rc = Invoke-PyLogged @("tools\onnx_gen\p4_model_rtl.py","check-plan","--run",$runPath) "plan_current"
            if ($rc -ne 0) { throw "stale replay plan; freeze final source and use -Stage plan -RefreshPlan" }
            $planJson = Get-Content -LiteralPath $planPath -Raw | ConvertFrom-Json
            $selected = @(Select-P4ReplayJobs $planJson $Jobs ([bool]$Pilot))
            $isSubset = [bool]$Pilot -or $Jobs.Count -gt 0
            Write-Host ("replay " + $(if ($isSubset) { "SELECTED ONLY" } else { "FULL PLAN" }) +
                        ": " + ($selected -join ",") + " across " + $Parallel + " worker(s)")
            $shards = @()
            for ($w = 0; $w -lt $Parallel; $w++) { $shards += ,@() }
            for ($i = 0; $i -lt $selected.Count; $i++) { $shards[$i % $Parallel] += $selected[$i] }
            $workers = @()
            $workerFailed = $false
            for ($w = 0; $w -lt $Parallel; $w++) {
                if ($shards[$w].Count -eq 0) { continue }
                $pyArgs = @("tools\onnx_gen\p4_model_rtl.py","run","--run",$runPath,
                            "--resume-jobs",($shards[$w] -join ","))
                if ($Force) { $pyArgs += "--force" }
                $workers += Start-PyWorker $pyArgs ("replay_w" + $w)
            }
            foreach ($worker in $workers) {
                if ((Complete-PyWorker $worker) -ne 0) { $workerFailed = $true }
            }
            if ($workerFailed) { throw "one or more replay workers failed; full PASS cannot be published" }
            if ($isSubset) {
                # check-job is authoritative for the selected jobs. A pilot's
                # success cannot turn an incomplete 22-job summary into PASS.
                foreach ($id in $selected) {
                    $rc = Invoke-PyLogged @("tools\onnx_gen\p4_model_rtl.py","check-job","--run",$runPath,"--job",$id) ("selected_" + $id)
                    if ($rc -ne 0) { throw ("selected replay is not complete/current: " + $id) }
                }
                Write-Host "Selected replay passed; full 20+2 qualification remains separate."
            } else {
                $rc = Invoke-PyLogged @("tools\onnx_gen\p4_model_rtl.py","summarize","--run",$runPath) "replay_summary"
                if ($rc -ne 0) { throw "full replay summary is incomplete/failed; see rtl_replay_summary.json" }
            }
        }
        "finalize" {
            $rc = Invoke-PyLogged @("tools\onnx_gen\p4_finalize.py","--run",$runPath) "finalize"
            if ($rc -ne 0) { throw "finalize refused: required local gates are not all PASS" }
        }
        "release" {
            $finalPath = Join-Path $runPath "p4_final_manifest.json"
            if (-not (Test-Path -LiteralPath $finalPath)) { throw "run finalize before release" }
            $final = Get-Content -LiteralPath $finalPath -Raw | ConvertFrom-Json
            if ($final.status -ne "local_pass_board_not_run") { throw ("release refused: " + $final.status) }
            if ($final.gates.physical_qualification -ne "pass") {
                throw "release refused: physical qualification is not pass"
            }
            $releaseName = "e1p4_" + (Split-Path $runPath -Leaf).Substring(5)
            $releasePath = "releases\" + $releaseName
            $compile = Join-Path $runPath "model\rtl\s00\first_fit\compile"
            $platformManifest = Join-Path $runPath "platform\platform_manifest.json"
            if (-not (Test-Path -LiteralPath $platformManifest)) { throw "platform manifest missing" }
            $pm = Get-Content -LiteralPath $platformManifest -Raw | ConvertFrom-Json
            $active = [string]$pm.active_attempt_dir
            # Strict: only an explicit platform/attemptN value names the
            # authoritative build.  Anything else is refused, never defaulted.
            $active = $active.Replace("/", "\")
            if ($active -notmatch '^platform\\attempt[12]$') {
                throw ("release refused: invalid active attempt pointer '" + $pm.active_attempt_dir + "'")
            }
            $activeName = Split-Path $active -Leaf
            $ipientry = Join-Path $runPath ("platform\" + $activeName)
            if (-not (Test-Path -LiteralPath $ipientry -PathType Container)) {
                throw "release refused: active attempt directory is missing"
            }
            if ($pm.stages.implementation.physical_status -ne "pass") {
                throw "release refused: active attempt is not physically qualified"
            }
            $ipDir = Get-P4PackagedIpDirectory $runPath $pm
            if (-not (Test-Path -LiteralPath $ipDir)) { throw "missing packaged IP dir" }
            $rc = Invoke-PyLogged @("tools\onnx_gen\p4_package_release.py","--run",$runPath,
                                    "--release",$releasePath,"--compile",$compile,"--ip-src",$ipDir) "release"
            if ($rc -ne 0) { throw "release packager refused or failed" }
        }
        "recover" {
            # Registers an auditable terminal state for an interrupted attempt.
            # Never deletes or rewrites attempt evidence and never resets budget.
            # -Dcp <routed.dcp> re-qualifies from that checkpoint ONLY after the
            # Tcl proves it opened those exact bytes; it never rebuilds.
            $pyArgs = @("tools\onnx_gen\p4_qualification.py","recover","--run",$runPath,
                        "--attempt",$Attempt,"--reason",$Reason)
            if ($Note) { $pyArgs += @("--note", $Note) }
            if ($Dcp) { $pyArgs += @("--dcp", $Dcp) }
            $rc = Invoke-PyLogged $pyArgs "recover"
            if ($rc -ne 0) { throw "recovery registration failed" }
        }
    }
    Write-Host ("stage " + $Stage + $(if ($DryRun) { " dry-run only" } else { " finished" }))
    exit 0
} catch {
    Write-Error $_ -ErrorAction Continue
    exit 1
}
