param(
    [Parameter(Mandatory=$true)][string]$CompileDir,
    [Parameter(Mandatory=$true)][string]$ResultDir,
    [string]$ProjectRoot = "D:\dl_v0.1",
    [string]$SourceRoot = "",
    # Optional replacement testbench (v0.2 second batch: fast functional profile and
    # source-identity experiments).  Empty keeps the frozen graph shell testbench.
    [string]$Testbench = "",
    [string]$TbParamsFile = "",
    # Simulation defines; comma or semicolon separated values become repeated -d.
    [string]$SimDefine = "",
    [int]$Seed = 20261020,
    [string]$Xvlog = "E:\Vitis\Vivado\2021.1\bin\xvlog.bat",
    [string]$Xelab = "E:\Vitis\Vivado\2021.1\bin\xelab.bat",
    [string]$Xsim  = "E:\Vitis\Vivado\2021.1\bin\xsim.bat",
    # v0.2 batch 2: functional-replay backend.  "xsim" (default) is the evidence
    # engine of the first batch.  "modelsim" is a faster RTL-only functional
    # replay used for the very large fusion graphs; it requires a twin
    # (TB_MAC_IMPL_2) elaboration because the Xilinx DSP48 functional netlist is
    # not available to the ModelSim simulator.  OOC/IP never use this path.
    [ValidateSet("xsim", "modelsim")][string]$Simulator = "xsim",
    [string]$Vlog = "D:\modelsim\win64\vlog.exe",
    [string]$Vsim = "D:\modelsim\win64\vsim.exe",
    [string]$Vlib = "D:\modelsim\win64\vlib.exe"
)
$ErrorActionPreference = "Stop"
function Get-P4SourceHash([string]$Path) {
    # Use the CLR directly: child Windows PowerShell can inherit a PS7 module
    # path in which the optional Get-FileHash cmdlet is not discoverable.
    $stream = [IO.File]::OpenRead($Path)
    $digest = [Security.Cryptography.SHA256]::Create()
    try { return [BitConverter]::ToString($digest.ComputeHash($stream)).Replace("-", "").ToLowerInvariant() }
    finally { $digest.Dispose(); $stream.Dispose() }
}
$root = [IO.Path]::GetFullPath($ProjectRoot)
$rtlRoot = if ($SourceRoot) { [IO.Path]::GetFullPath($SourceRoot) } else { $root }
if ($rtlRoot -ne $root -and -not $rtlRoot.StartsWith($root + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) { throw "source root escaped project" }
$compile = [IO.Path]::GetFullPath($CompileDir)
$res = [IO.Path]::GetFullPath($ResultDir)
if (-not $compile.StartsWith($root + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) { throw "compile dir escaped project" }
if (-not $res.StartsWith($root + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) { throw "result dir escaped project" }
New-Item -ItemType Directory -Force -Path $res | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $res "snapshots") | Out-Null
$work = Join-Path $res "work"
New-Item -ItemType Directory -Force -Path $work | Out-Null
$manifestPath = Join-Path $compile "compile_manifest.json"
if (-not (Test-Path -LiteralPath $manifestPath)) { throw "missing compile manifest" }
$manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
if ($manifest.status -ne "pass") { throw "compile manifest is not pass" }
$sources = Join-Path $compile "generated/sources.f"
$rmPath = Join-Path $compile "generated/rtl_manifest.json"
if (-not (Test-Path -LiteralPath $sources)) { throw "missing sources.f" }
$rm = Get-Content -LiteralPath $rmPath -Raw | ConvertFrom-Json
# New compilations carry an explicit root. The runner follows that binding
# and rejects an override rather than silently selecting a different tree.
if ($rm.source_root) {
    $boundRoot = [IO.Path]::GetFullPath([string]$rm.source_root)
    if ($SourceRoot -and -not [string]::Equals($rtlRoot, $boundRoot, [StringComparison]::OrdinalIgnoreCase)) { throw "SourceRoot differs from compiled source_root" }
    $rtlRoot = $boundRoot
}
$boundRows = @()
if ($rm.source_files) {
    $boundRows += @($rm.source_files)
    if ($rm.template) { $boundRows += $rm.template }
    if ($rm.vendor_ip) { $boundRows += @($rm.vendor_ip) }
    foreach ($row in $boundRows) {
        $boundFile = [IO.Path]::GetFullPath((Join-Path $rtlRoot ([string]$row.path)))
        if (-not $boundFile.StartsWith($rtlRoot + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) { throw "bound source escaped root" }
        if (-not (Test-Path -LiteralPath $boundFile) -or (Get-P4SourceHash $boundFile) -ne $row.sha256) { throw "compiled source changed: $boundFile" }
    }
}
if ($rm.files_sha256) {
    foreach ($property in $rm.files_sha256.PSObject.Properties) {
        if (-not (Test-Path -LiteralPath $property.Name) -or (Get-P4SourceHash $property.Name) -ne $property.Value) { throw "compiled generated artifact changed: $($property.Name)" }
    }
}
$memName = [string]$rm.mem_file
$mem = Join-Path $compile ("generated/" + $memName)
Copy-Item -LiteralPath $mem -Destination (Join-Path $work $memName) -Force

# ---------------------------------------------------------------------------
# v0.2 simulation capacity parameters (tb_params.vh contract).
#
# Resolution order:
#   1. -TbParamsFile / memory_limits.tb_params / generated/tb_params.vh in
#      the compile directory (written by the v0.2 spatial compiler);
#   2. a runner-synthesized tb_params.vh derived from compile_manifest
#      memory_limits (sim_ddr_limit_bytes / mem_words / tb_* fields);
#   3. legacy defaults 8192 words (128KiB), 128 snapshots, 16KiB chunks
#      and a 20M-cycle watchdog -- identical to the pre-v0.2 hard-coded TB.
# The selected file is always passed with xvlog -i so the TB include
# resolves; -d TB_SIM_PARAMS_V02 identifies the parameterized path.
# ---------------------------------------------------------------------------
$bt = [string][char]96
$legacyMemBytes = 8192 * 16
$paramInfo = [ordered]@{
    schema = "e1-sim-tb-params-1"
    mem_bytes = $legacyMemBytes
    mem_words = 8192
    max_snaps = 128
    snap_chunk = 16384
    watchdog_cycles = 20000000
    source = "legacy_default"
    file = $null
}
$memLimits = $null
if ($manifest.PSObject.Properties.Match("memory_limits").Count -gt 0) { $memLimits = $manifest.memory_limits }
$tbParamsRel = "generated/tb_params.vh"
if ($memLimits -and $memLimits.PSObject.Properties.Match("tb_params").Count -gt 0 -and "$($memLimits.tb_params)" -ne "") {
    $tbParamsRel = [string]$memLimits.tb_params
}
$tbParamsPath = if ($TbParamsFile) { [IO.Path]::GetFullPath($TbParamsFile) } else { [IO.Path]::GetFullPath((Join-Path $compile $tbParamsRel)) }
if ($TbParamsFile) {
    if (-not $tbParamsPath.StartsWith($root + $sep, [StringComparison]::OrdinalIgnoreCase)) { throw "TbParamsFile escaped project: $tbParamsPath" }
    if (-not (Test-Path -LiteralPath $tbParamsPath)) { throw "TbParamsFile not found: $tbParamsPath" }
} elseif (-not $tbParamsPath.StartsWith($compile + $sep, [StringComparison]::OrdinalIgnoreCase)) { throw "tb_params escaped compile dir: $tbParamsPath" }
if (Test-Path -LiteralPath $tbParamsPath) {
    $paramInfo.source = "compile_generated"
} else {
    if ($memLimits) {
        # Prefer the actual modelled window (tb_mem_bytes / mem_bytes) over the
        # sim_ddr_limit_bytes budget: the budget is the planner limit (e.g. 32MiB)
        # while the real image window is much smaller (e.g. 640KiB for A224).
        if ($memLimits.PSObject.Properties.Match("tb_mem_bytes").Count -gt 0 -and "$($memLimits.tb_mem_bytes)" -ne "") { $paramInfo.mem_bytes = [int64]$memLimits.tb_mem_bytes }
        elseif ($memLimits.PSObject.Properties.Match("mem_bytes").Count -gt 0 -and "$($memLimits.mem_bytes)" -ne "") { $paramInfo.mem_bytes = [int64]$memLimits.mem_bytes }
        elseif ($memLimits.PSObject.Properties.Match("sim_ddr_limit_bytes").Count -gt 0 -and "$($memLimits.sim_ddr_limit_bytes)" -ne "") { $paramInfo.mem_bytes = [int64]$memLimits.sim_ddr_limit_bytes }
        if ($memLimits.PSObject.Properties.Match("mem_words").Count -gt 0 -and "$($memLimits.mem_words)" -ne "") { $paramInfo.mem_words = [int64]$memLimits.mem_words }
        if ($memLimits.PSObject.Properties.Match("tb_mem_words").Count -gt 0 -and "$($memLimits.tb_mem_words)" -ne "") { $paramInfo.mem_words = [int64]$memLimits.tb_mem_words }
        if ($memLimits.PSObject.Properties.Match("tb_max_snaps").Count -gt 0 -and "$($memLimits.tb_max_snaps)" -ne "") { $paramInfo.max_snaps = [int64]$memLimits.tb_max_snaps }
        if ($memLimits.PSObject.Properties.Match("tb_snap_chunk").Count -gt 0 -and "$($memLimits.tb_snap_chunk)" -ne "") { $paramInfo.snap_chunk = [int64]$memLimits.tb_snap_chunk }
        if ($memLimits.PSObject.Properties.Match("tb_watchdog_cycles").Count -gt 0 -and "$($memLimits.tb_watchdog_cycles)" -ne "") { $paramInfo.watchdog_cycles = [int64]$memLimits.tb_watchdog_cycles }
    }
    if ($paramInfo.mem_bytes -le 0 -or ($paramInfo.mem_bytes % 16) -ne 0) { throw "memory_limits sim_ddr_limit_bytes must be a positive multiple of 16" }
    $impliedWords = [int64]($paramInfo.mem_bytes / 16)
    if ($paramInfo.mem_words -lt $impliedWords) { $paramInfo.mem_words = $impliedWords }
    $paramDir = Join-Path $work "tb_params"
    New-Item -ItemType Directory -Force -Path $paramDir | Out-Null
    $tbParamsPath = Join-Path $paramDir "tb_params.vh"
    $paramLines = @(
        "// tb_params.vh -- synthesized by verification/sim/run_p3_graph_shell.ps1"
        "// Contract: e1-sim-tb-params-1; mirrors compile_manifest memory_limits or legacy defaults."
        ($bt + "ifndef TB_PARAMS_VH")
        ($bt + "define TB_PARAMS_VH 1")
        ($bt + ("define TB_MEM_BYTES {0}" -f $paramInfo.mem_bytes))
        ($bt + ("define TB_MEM_WORDS {0}" -f $paramInfo.mem_words))
        ($bt + ("define TB_MAX_SNAPS {0}" -f $paramInfo.max_snaps))
        ($bt + ("define TB_SNAP_CHUNK {0}" -f $paramInfo.snap_chunk))
        ($bt + ("define TB_WATCHDOG_CYCLES {0}" -f $paramInfo.watchdog_cycles))
        ($bt + "endif")
    )
    Set-Content -LiteralPath $tbParamsPath -Value ($paramLines -join ([string][char]10)) -Encoding ascii
    $paramInfo.source = "runner_synthesized"
}
$tbParamsText = Get-Content -LiteralPath $tbParamsPath
$tbPrefix = $bt + "define "
foreach ($tbLine in $tbParamsText) {
    $tbTrim = $tbLine.Trim()
    if (-not $tbTrim.StartsWith($tbPrefix)) { continue }
    $tbBody = $tbTrim.Substring($tbPrefix.Length).Trim()
    $tbParts = $tbBody -split '\s+', 2
    if ($tbParts.Count -ne 2) { continue }
    if ($tbParts[1] -notmatch '^[0-9]+$') { continue }
    $macroName = $tbParts[0]
    $macroValue = [int64]$tbParts[1]
    if ($macroName -eq "TB_MEM_BYTES") { $paramInfo.mem_bytes = $macroValue }
    elseif ($macroName -eq "TB_MEM_WORDS") { $paramInfo.mem_words = $macroValue }
    elseif ($macroName -eq "TB_MAX_SNAPS") { $paramInfo.max_snaps = $macroValue }
    elseif ($macroName -eq "TB_SNAP_CHUNK") { $paramInfo.snap_chunk = $macroValue }
    elseif ($macroName -eq "TB_WATCHDOG_CYCLES") { $paramInfo.watchdog_cycles = $macroValue }
}
if ($paramInfo.mem_bytes -le 0 -or ($paramInfo.mem_bytes % 16) -ne 0) { throw "tb_params TB_MEM_BYTES must be a positive multiple of 16: $($paramInfo.mem_bytes)" }
if (($paramInfo.mem_words * 16) -lt $paramInfo.mem_bytes) { throw "tb_params TB_MEM_WORDS does not cover TB_MEM_BYTES" }
if ($paramInfo.max_snaps -le 0 -or $paramInfo.snap_chunk -le 0 -or $paramInfo.watchdog_cycles -le 0) { throw "tb_params has a non-positive budget macro" }
$paramIncludeDir = Split-Path -Parent $tbParamsPath
$paramInfo.file = ($tbParamsPath -replace "\\", "/")
$paramDefineArgs = @("-d", "TB_SIM_PARAMS_V02")
$paramIncludeArg = ($paramIncludeDir -replace "\\", "/")
$ddrInitPath = Join-Path $compile "ddr_init.bin"
if (-not (Test-Path -LiteralPath $ddrInitPath)) { throw "missing ddr_init.bin" }
$ddrBytes = [int64]((Get-Item -LiteralPath $ddrInitPath).Length)
if ($ddrBytes -gt $paramInfo.mem_bytes) { throw "ddr_init.bin ($ddrBytes bytes) exceeds simulation DDR model ($($paramInfo.mem_bytes) bytes)" }
Write-Output ("SIM_PARAMS: source={0} mem_bytes={1} mem_words={2} max_snaps={3} snap_chunk={4} watchdog_cycles={5} ddr_init_bytes={6}" -f $paramInfo.source, $paramInfo.mem_bytes, $paramInfo.mem_words, $paramInfo.max_snaps, $paramInfo.snap_chunk, $paramInfo.watchdog_cycles, $ddrBytes)

$absLines = @()
$sourcePaths = @()
foreach ($raw in Get-Content -LiteralPath $sources) {
    $t = $raw.Trim()
    if ($t -eq "" -or $t.StartsWith("#")) { continue }
    if ($t.StartsWith("-i ")) {
        $incText = $t.Substring(3).Trim()
        $incRoot = if ($incText -match '^rtl[/\\]') { $rtlRoot } else { $root }
        $inc = [IO.Path]::GetFullPath((Join-Path $incRoot $incText))
        if (-not $inc.StartsWith($root + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) { throw "include escaped project" }
        $absLines += "-i " + ($inc -replace "\\", "/")
    } else {
        $fileRoot = if ($t -match '^rtl[/\\]') { $rtlRoot } else { $root }
        $p = if ([IO.Path]::IsPathRooted($t)) { [IO.Path]::GetFullPath($t) } else { [IO.Path]::GetFullPath((Join-Path $fileRoot $t)) }
        if (-not $p.StartsWith($root + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) { throw "source escaped project" }
        $absLines += ($p -replace "\\", "/")
        $sourcePaths += $p
    }
}
$tb = if ($Testbench) { [IO.Path]::GetFullPath($Testbench) } else { [IO.Path]::GetFullPath((Join-Path $root "verification/sim/tb_p3_graph_shell.sv")) }
if (-not $tb.StartsWith($root + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) { throw "testbench escaped project" }
$absLines += ($tb -replace "\\", "/")
$sourcePaths += $tb
$sourcePaths += $mem
$sourcePaths += $sources
$sourcePaths += $PSCommandPath
foreach ($row in $boundRows) { $sourcePaths += Join-Path $rtlRoot ([string]$row.path) }
$sourcePaths = @($sourcePaths | Sort-Object -Unique)
$sourceRows = @($sourcePaths | ForEach-Object { [ordered]@{path=$_; sha256=(Get-P4SourceHash $_)} })
$sourceIdentity = [ordered]@{schema="e1-p4-simulation-sources-1"; source_root=$rtlRoot; source_digest=$rm.source_digest; testbench=$tb; seed=$Seed; tb_params=$paramInfo; sim_define=$SimDefine; files=$sourceRows; source_unchanged=$null}
($sourceIdentity | ConvertTo-Json -Depth 8) | Set-Content -LiteralPath (Join-Path $res "simulation_identity.json") -Encoding utf8
$absSources = Join-Path $work "abs_sources.f"
Set-Content -LiteralPath $absSources -Value ($absLines -join "`n") -Encoding ascii
$xvlogLog = Join-Path $res "xvlog.log"
$xelabLog = Join-Path $res "xelab.log"
$xsimLog = Join-Path $res "xsim.log"
$snapshot = "p3_graph_snap"
$paramIncludeArg = ($paramIncludeDir -replace "\\", "/")
Push-Location $work
try {

    # ------------------------------------------------------------------
    # ModelSim functional-replay backend (v0.2 batch 2, large fusion graphs)
    #
    # The first-batch evidence engine is XSim.  XSim runs this design at only
    # a few hundred cycles/s on this host, which makes a B640 fusion graph a
    # >10h run.  ModelSim SE-64 runs the SAME RTL/testbench ~10x faster and was
    # cross-checked to produce the same snapshots and cycle counts on the A224
    # first-batch graph.  It is a FUNCTIONAL replay only:
    #   * twin backend (TB_MAC_IMPL_2) is required because the Xilinx DSP48
    #     functional netlist is not available to ModelSim;
    #   * the Xilinx-only sources are excluded from the compile and recorded;
    #   * OOC/IP always use the compiled MAC_IMPL=3 Vivado build.
    # ------------------------------------------------------------------
    if ($Simulator -eq "modelsim") {
        if ($SimDefine -notmatch "TB_MAC_IMPL_2") {
            throw "modelsim backend is functional-replay only and requires -SimDefine TB_MAC_IMPL_2 (twin backend)"
        }
        foreach ($tool in @($Vlib, $Vlog, $Vsim)) {
            if (-not (Test-Path -LiteralPath $tool)) { throw "ModelSim tool not found: $tool" }
        }
        $msimWork = Join-Path $work "msim_work"
        if (Test-Path -LiteralPath $msimWork) { Remove-Item -Recurse -Force -LiteralPath $msimWork }
        # ModelSim tool arguments use forward slashes: the Vivado/MSVC-style
        # batch wrappers on this host treat backslash paths inconsistently.
        $msimWorkArg = $msimWork -replace "\\", "/"
        $vlibOut = & $Vlib $msimWorkArg 2>&1
        if ($LASTEXITCODE -ne 0) { throw "vlib failed: $vlibOut" }
        $xilinxOnly = @(
            "rtl/ops/conv/dsp_mul_pipe4_sim_netlist.v",
            "rtl/ops/conv/conv3x3_calc_dsp.v",
            "rtl/ops/conv/signed_int8_dsp2_mac.v",
            "rtl/ops/conv/conv_mul_dsp.v")
        $msimFiles = @()
        $excluded = @()
        $incArgs = @()
        foreach ($line in (Get-Content -LiteralPath $absSources)) {
            $t = ([string]$line).Trim()
            if ($t -eq "") { continue }
            if ($t.StartsWith("-i ")) {
                $incArgs += ("+incdir+" + (($t.Substring(3).Trim()) -replace "\\", "/"))
                continue
            }
            $norm = $t -replace "\\", "/"
            $skip = $false
            foreach ($rel in $xilinxOnly) {
                if ($norm.EndsWith("/" + $rel)) { $skip = $true }
            }
            if ($skip) { $excluded += $norm } else { $msimFiles += $norm }
        }
        $incArgs += ("+incdir+" + ($paramIncludeDir -replace "\\", "/"))
        $defineArgs = @("+define+TB_SIM_PARAMS_V02")
        foreach ($defineName in ($SimDefine -split '[,;]')) {
            $trimmed = $defineName.Trim()
            if ($trimmed -ne "") { $defineArgs += ("+define+" + $trimmed) }
        }
        $vlogList = Join-Path $msimWork "vlog.f"
        Set-Content -LiteralPath $vlogList -Value ($msimFiles -join ([string][char]10)) -Encoding ascii
        $vlogLog = Join-Path $res "vlog.log"
        $vsimLog = Join-Path $res "simulator.log"
        Push-Location $work
        try {
            $vlogArgs = @("-sv", "-work", $msimWorkArg) + $incArgs + $defineArgs + @("-f", ($vlogList -replace "\\", "/"))
            $out = & $Vlog @vlogArgs 2>&1
            $rc1 = $LASTEXITCODE
            $out | Set-Content -LiteralPath $vlogLog -Encoding utf8
            if ($rc1 -ne 0) { throw "vlog failed rc=$rc1; see $vlogLog" }
            $version = ((& $Vsim -version 2>&1) | Out-String).Trim()
            $ddr = (Join-Path $compile "ddr_init.bin") -replace "\\", "/"
            $snap = (Join-Path $compile "snapshot_desc.txt") -replace "\\", "/"
            $snapRoot = ($compile -replace "\\", "/")
            $resultForSim = ($res -replace "\\", "/")
            $vsimArgs = @(
                "-c", "-lib", $msimWorkArg, "tb_p3_graph_shell",
                ("+DDR_INIT=" + $ddr),
                ("+SNAP_DESC=" + $snap),
                ("+SNAP_ROOT=" + $snapRoot),
                ("+RESULT_DIR=" + $resultForSim),
                ("+PERF_SEED=" + $Seed),
                "-do", "run -all; quit -f")
            Set-Content -LiteralPath $vsimLog -Value ("# modelsim streaming log seed={0} define={1}" -f $Seed, $SimDefine) -Encoding utf8
            & $Vsim @vsimArgs 2>&1 | ForEach-Object {
                if ($null -ne $_) { Add-Content -LiteralPath $vsimLog -Value ([string]$_) -Encoding utf8 }
            }
            $rc3 = $LASTEXITCODE
            $simText = Get-Content -LiteralPath $vsimLog -Raw
            $pass = ($null -ne $simText) -and ($simText -match "P3_RESULT: PASS") -and
                    ($simText -notmatch "\*\* (Error|Fatal)")
            $status = if ($rc3 -eq 0 -and $pass) { "pass" } else { "fail" }
            $changed = @($sourceRows | Where-Object { (Get-P4SourceHash $_.path) -ne $_.sha256 })
            $sourceIdentity.source_unchanged = ($changed.Count -eq 0)
            $sourceIdentity.simulator = "modelsim"
            $sourceIdentity.simulator_version = $version
            $sourceIdentity.excluded_xilinx_sources = $excluded
            ($sourceIdentity | ConvertTo-Json -Depth 8) | Set-Content -LiteralPath (Join-Path $res "simulation_identity.json") -Encoding utf8
            if (-not $sourceIdentity.source_unchanged) { $status = "fail" }
            $summary = [ordered]@{
                schema = "e1-p3-graph-sim-2"
                simulator = "modelsim"
                simulator_version = $version
                functional_replay = $true
                status = $status
                vlog_rc = $rc1
                sim_rc = $rc3
                pass_marker = $pass
                source_unchanged = $sourceIdentity.source_unchanged
                sim_define = $SimDefine
                seed = $Seed
                board_status = "not_run"
                compile_dir = $snapRoot
                result_dir = $resultForSim
                tb_params = $paramInfo
                ddr_init_bytes = $ddrBytes
                excluded_xilinx_sources = $excluded
                logs = [ordered]@{ vlog = ($vlogLog -replace "\\", "/");
                                   simulator = ($vsimLog -replace "\\", "/") }
            }
            ($summary | ConvertTo-Json -Depth 6) | Set-Content -LiteralPath (Join-Path $res "summary.json") -Encoding utf8
            if ($status -ne "pass") { throw "ModelSim functional replay failed; see $vsimLog" }
            Write-Output (@{status="pass"; simulator="modelsim"; summary=(Join-Path $res "summary.json")} | ConvertTo-Json -Compress)
            return
        } finally { Pop-Location }
    }
    # Optional functional-replay backend macro.  The backend is encoded in the
    # macro NAME (TB_MAC_IMPL_2) so no '=' travels through the Vivado batch
    # loader.  Empty keeps the legacy xvlog command byte-identical.
    $xvlogDefineArgs = @($paramDefineArgs)
    if ($SimDefine -ne "") {
        foreach ($defineName in ($SimDefine -split '[,;]')) {
            $trimmed = $defineName.Trim()
            if ($trimmed -ne "") { $xvlogDefineArgs += @("-d", $trimmed) }
        }
    }
    $out = & $Xvlog -sv -f $absSources -i $paramIncludeArg @xvlogDefineArgs 2>&1
    $rc1 = $LASTEXITCODE
    $out | Set-Content -LiteralPath $xvlogLog -Encoding utf8
    if ($rc1 -ne 0) { throw "xvlog failed rc=$rc1; see $xvlogLog" }

    $elabArgs = @("-debug", "typical", "tb_p3_graph_shell", "glbl", "-s", $snapshot,
                  "-L", "unisims_ver", "-L", "unimacro_ver", "-L", "secureip")
    $out = & $Xelab @elabArgs 2>&1
    $rc2 = $LASTEXITCODE
    $out | Set-Content -LiteralPath $xelabLog -Encoding utf8
    if ($rc2 -ne 0) { throw "xelab failed rc=$rc2; see $xelabLog" }

    $ddr = (Join-Path $compile "ddr_init.bin") -replace "\\", "/"
    $snap = (Join-Path $compile "snapshot_desc.txt") -replace "\\", "/"
    $snapRoot = ($compile -replace "\\", "/")
    $resultForXsim = ($res -replace "\\", "/")
    # Stream XSim output to $xsimLog line by line instead of buffering all of it
    # in memory: long spatial graphs (e.g. 640 backbone, millions of cycles) need
    # P3_HEARTBEAT/P3_SNAP progress visible on disk while the run is in flight, and
    # a killed/crashed run must leave the partial log behind for diagnosis.
    $xsimArgs = @(
        $snapshot, "-runall", "-sv_seed", $Seed,
        "-testplusarg", ('"DDR_INIT={0}"' -f $ddr),
        "-testplusarg", ('"SNAP_DESC={0}"' -f $snap),
        "-testplusarg", ('"SNAP_ROOT={0}"' -f $snapRoot),
        "-testplusarg", ('"RESULT_DIR={0}"' -f $resultForXsim),
        "-testplusarg", ('"PERF_SEED={0}"' -f $Seed))
    Set-Content -LiteralPath $xsimLog -Value ("# xsim streaming log snapshot={0} seed={1}" -f $snapshot, $Seed) -Encoding utf8
    & $Xsim @xsimArgs 2>&1 | ForEach-Object {
        if ($null -ne $_) { Add-Content -LiteralPath $xsimLog -Value ([string]$_) -Encoding utf8 }
    }
    $rc3 = $LASTEXITCODE
    $xsimText = Get-Content -LiteralPath $xsimLog -Raw
    $pass = ($null -ne $xsimText) -and ($xsimText -match "P3_RESULT: PASS")
    $status = if ($rc3 -eq 0 -and $pass) { "pass" } else { "fail" }
    $changed = @($sourceRows | Where-Object { (Get-P4SourceHash $_.path) -ne $_.sha256 })
    $sourceIdentity.source_unchanged = ($changed.Count -eq 0)
    ($sourceIdentity | ConvertTo-Json -Depth 8) | Set-Content -LiteralPath (Join-Path $res "simulation_identity.json") -Encoding utf8
    if (-not $sourceIdentity.source_unchanged) { $status = "fail" }
    $summary = [ordered]@{
        schema = "e1-p3-graph-xsim-1"
        status = $status
        xvlog_rc = $rc1
        xelab_rc = $rc2
        xsim_rc = $rc3
        pass_marker = $pass
        source_unchanged = $sourceIdentity.source_unchanged
        sim_define = $SimDefine
        seed = $Seed
        board_status = "not_run"
        compile_dir = $snapRoot
        result_dir = $resultForXsim
        tb_params = $paramInfo
        ddr_init_bytes = $ddrBytes
        logs = [ordered]@{ xvlog = ($xvlogLog -replace "\\", "/");
                            xelab = ($xelabLog -replace "\\", "/");
                            xsim = ($xsimLog -replace "\\", "/") }
    }
    ($summary | ConvertTo-Json -Depth 6) | Set-Content -LiteralPath (Join-Path $res "summary.json") -Encoding utf8
    if ($status -ne "pass") { throw "XSim graph failed; see $xsimLog" }
    Write-Output (@{status="pass"; summary=(Join-Path $res "summary.json")} | ConvertTo-Json -Compress)
}
finally { Pop-Location }
