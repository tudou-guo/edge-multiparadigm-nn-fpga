param(
    [string]$RunRoot = "D:/dl_v1/build/onnx_e1/compute_profile_20261023_180522_7a3c"
)
# S0 bootstrap for the static-gather optimization round (2026-10-23).
# Native PowerShell is the exclusive creator of this run.  It records identity,
# before/ snapshots and a fresh protection scan; it does NOT build anything.
$ErrorActionPreference = "Stop"
$root = "D:/dl_v1"
$py = "E:/miniconda/python.exe"
$vivado = "E:/Vitis/Vivado/2021.1/bin/vivado.bat"
$xvlog = "E:/Vitis/Vivado/2021.1/bin/xvlog.bat"
$parentRun = "D:/dl_v1/build/onnx_e1/compute_profile_20261023_121224_5554"
$nl = [Environment]::NewLine

function Get-Hash([string]$Path) {
    $stream = [IO.File]::OpenRead($Path)
    $digest = [Security.Cryptography.SHA256]::Create()
    try { return [BitConverter]::ToString($digest.ComputeHash($stream)).Replace("-", "").ToLowerInvariant() }
    finally { $digest.Dispose(); $stream.Dispose() }
}
function Get-Rel([string]$Path) { return ($Path.Substring($root.Length + 1) -replace "\\", "/") }

# --- hard gates -------------------------------------------------------------
if (Test-Path -LiteralPath $RunRoot) { throw "run root already exists; refusing to reuse: $RunRoot" }
if (-not (Test-Path -LiteralPath $parentRun -PathType Container)) { throw "parent run missing: $parentRun" }
foreach ($tool in @($py, $vivado, $xvlog)) { if (-not (Test-Path -LiteralPath $tool)) { throw "tool missing: $tool" } }
$free = [IO.DriveInfo]::new("D:/").AvailableFreeSpace
$phys = (Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory
if ($free -lt 40GB) { throw "resource gate: free < 40GiB" }
if ($phys -lt 12GB) { throw "resource gate: physical memory < 12GiB" }

# --- exclusive creation -----------------------------------------------------
New-Item -ItemType Directory -Path $RunRoot | Out-Null
$dirs = @("identity", "before", "candidates", "evidence", "selector", "exports",
          "sim", "ooc", "platform", "comparison", "release", "fixtures", "logs",
          "artifacts", "tmp_tests", "tmp_lead")
foreach ($d in $dirs) { New-Item -ItemType Directory -Path (Join-Path $RunRoot $d) | Out-Null }

# Scanner: a byte copy of the parent-run scanner (read-only source) is the
# proof that this round reuses the same manifest semantics.
$scanSrc = Join-Path $parentRun "identity/bootstrap_scan_v2.py"
$scanDst = Join-Path $RunRoot "identity/bootstrap_scan_v2.py"
Copy-Item -LiteralPath $scanSrc -Destination $scanDst
Copy-Item -LiteralPath $PSCommandPath -Destination (Join-Path $RunRoot "identity/bootstrap.ps1") -Force

# --- this round's editable files (before/ snapshots; promotion only at S5) ---
$editable = @(
    "rtl/ops/conv/conv3x3_calc_balanced.sv",
    "rtl/ops/conv/conv3x3_core.v",
    "tools/onnx_gen/compute_profile_unit_vectors.py",
    "tools/onnx_gen/compute_profile_platform.py",
    "tools/onnx_gen/compute_profile_verify.py",
    "verification/sim/run_compute_profile_conv.ps1"
)
# --- files this round is expected to ADD (excluded from the before scan) ----
$newFiles = @(
    "tools/onnx_gen/bootstrap_compute_profile_opt_20261023.ps1",
    "tools/onnx_gen/compute_profile_opt.py",
    "tools/onnx_gen/compute_profile_opt_verify.py",
    "tools/onnx_gen/compute_profile_selector_vectors.py",
    "tools/npu_gen/run_compute_profile_selector_ooc.tcl",
    "verification/sim/tb_compute_profile_selector.sv",
    "verification/sim/run_compute_profile_selector_sim.ps1",
    "docs/E1_COMPUTE_PROFILE_STATIC_GATHER_DELIVERY_20261023.md",
    "docs/E1_COMPUTE_PROFILE_PHASE3_TASK_20261023.md",
    "docs/NEW_CHAT_OPENING_E1_COMPUTE_PROFILE_PHASE3_20261023.txt"
)
$beforeRows = @()
foreach ($rel in $editable) {
    $src = Join-Path $root $rel
    if (-not (Test-Path -LiteralPath $src -PathType Leaf)) { throw "editable source missing: $rel" }
    $snap = Join-Path $RunRoot ("before/" + $rel)
    $snapDir = Split-Path -Parent $snap
    if (-not (Test-Path -LiteralPath $snapDir)) { New-Item -ItemType Directory -Path $snapDir -Force | Out-Null }
    Copy-Item -LiteralPath $src -Destination $snap -Force
    $beforeRows += [ordered]@{ path = $rel; snapshot = (Get-Rel $snap);
        snapshot_sha256 = (Get-Hash $snap); snapshot_bytes = (Get-Item -LiteralPath $snap).Length;
        before_sha256 = (Get-Hash $src) }
}
$newRows = @()
foreach ($rel in $newFiles) {
    $live = Join-Path $root $rel
    if (Test-Path -LiteralPath $live) {
        $newRows += [ordered]@{ path = $rel; present = $true; sha256 = (Get-Hash $live); bytes = (Get-Item -LiteralPath $live).Length }
    } else {
        $newRows += [ordered]@{ path = $rel; present = $false; sha256 = $null; bytes = $null }
    }
}

# --- frozen inputs this round must not change ------------------------------
$frozen = @(
    "rtl/common/npu_backend_contract.vh",
    "rtl/common/op_abi_defs.vh",
    "rtl/ops/conv/spram.v",
    "rtl/ops/conv/signed_int8_dsp2_mac.v",
    "rtl/ops/conv/conv3x3_weight_faces.v",
    "tools/npu_gen/pack_npu_ip.tcl",
    "tools/npu_gen/verify_npu_ip_ooc.tcl",
    "tools/onnx_gen/compute_profiles.py",
    "tools/onnx_gen/p2_compile.py",
    "tools/onnx_gen/p3_compile.py",
    "tools/onnx_gen/verilog_codegen.py",
    "tools/onnx_gen/export_model.py",
    "tools/onnx_gen/compute_profile_check.py",
    "tools/onnx_gen/compute_profile_vectors.py",
    "tools/onnx_gen/profiles/tn4.json",
    "tools/onnx_gen/profiles/tn8.json",
    "tools/onnx_gen/profiles/tn16.json",
    "tools/onnx_gen/profiles/internal/tn4_tm8.json",
    "tools/onnx_gen/profiles/internal/tn8_tm4.json",
    "tools/onnx_gen/profiles/internal/tn4_tm16.json",
    "tools/onnx_gen/profiles/internal/tn16_tm4.json",
    "tools/onnx_gen/profiles/internal/tn8_tm16.json",
    "tools/onnx_gen/profiles/internal/tn16_tm8.json",
    "verification/sim/tb_compute_profile_conv.sv",
    "verification/sim/tb_p3_graph_shell.sv",
    "verification/sim/run_p3_graph_shell.ps1",
    "docs/E1_COMPUTE_PROFILE_STATIC_GATHER_OPTIMIZATION_PLAN_20261023.md",
    "docs/E1_COMPUTE_PROFILE_DELIVERY_20261023.md"
)
$frozenRows = @()
foreach ($rel in $frozen) {
    $src = Join-Path $root $rel
    if (-not (Test-Path -LiteralPath $src)) { throw "frozen input missing: $rel" }
    $frozenRows += [ordered]@{ path = $rel; bytes = (Get-Item -LiteralPath $src).Length; sha256 = (Get-Hash $src) }
}
$rtlFiles = @(Get-ChildItem -LiteralPath (Join-Path $root "rtl") -Recurse -File -Force | Sort-Object FullName)
$rtlRows = @()
foreach ($f in $rtlFiles) { $rtlRows += [ordered]@{ path = (Get-Rel $f.FullName); bytes = $f.Length; sha256 = (Get-Hash $f.FullName) } }

# --- parent run immutable reference (read-only inheritance candidates) -----
function Ref-Hash([string]$Path) {
    if (Test-Path -LiteralPath $Path -PathType Leaf) { return (Get-Hash $Path) } else { return $null }
}
$parentRef = [ordered]@{
    run_root = ($parentRun -replace "\\", "/")
    delivery = "docs/E1_COMPUTE_PROFILE_DELIVERY_20261023.md"
    ooc = [ordered]@{}
    qualification = [ordered]@{}
    release = [ordered]@{}
    exports = [ordered]@{}
    sim = [ordered]@{}
}
foreach ($tier in @("tn4", "tn8", "tn16")) {
    $parentRef.ooc[$tier] = [ordered]@{
        utilization = Ref-Hash (Join-Path $parentRun ("ooc/cp16/" + $tier + "/ooc_utilization.rpt"))
        timing = Ref-Hash (Join-Path $parentRun ("ooc/cp16/" + $tier + "/ooc_timing_summary.rpt"))
        drc = Ref-Hash (Join-Path $parentRun ("ooc/cp16/" + $tier + "/ooc_drc.rpt"))
        dcp = Ref-Hash (Join-Path $parentRun ("ooc/cp16/" + $tier + "/cp16_" + $tier + "_ooc.dcp"))
    }
    $parentRef.qualification[$tier] = Ref-Hash (Join-Path $parentRun ("qualification/cp16/" + $tier + "/functional.json"))
    $parentRef.release[$tier] = Ref-Hash (Join-Path $parentRun ("release/cp16/" + $tier + "/evidence_index.json"))
    $parentRef.exports[$tier] = Ref-Hash (Join-Path $parentRun ("exports/cp16/" + $tier + "/first_fit/compile/compile_manifest.json"))
    $parentRef.sim[$tier] = Ref-Hash (Join-Path $parentRun ("sim/unit/" + $tier + "/unit_summary.json"))
}
$parentRef.tilesCheck = Ref-Hash (Join-Path $parentRun "qualification/internal_tiles_check.json")
$parentRef.fixtures = Ref-Hash (Join-Path $parentRun "fixtures/fixture_manifest.json")

# Reference (read-only) YOLO fact source for the borrowed fixed-wiring idea.
$yoloRows = @()
foreach ($rel in @("D:/yolov8/src/cnn/conv/conv_calc.v", "D:/yolov8/src/cnn/conv/conv_weight.v")) {
    if (Test-Path -LiteralPath $rel) { $yoloRows += [ordered]@{ path = $rel; bytes = (Get-Item -LiteralPath $rel).Length; sha256 = (Get-Hash $rel) } }
}

# --- protection scan (threaded; run root + this round's new files excluded) -
$scanOut = Join-Path $RunRoot "identity/protected_before.json"
$scanArgs = @((Join-Path $RunRoot "identity/bootstrap_scan_v2.py"), "--out", $scanOut,
              "--root", $root, "--exclude-prefix", ($RunRoot -replace "\\", "/"),
              "--threads", "8")
foreach ($rel in $newFiles) { $scanArgs += @("--exclude-file", ($root + "/" + $rel)) }
$previousEap = $ErrorActionPreference
$ErrorActionPreference = "Continue"
$scanText = & $py @scanArgs
$scanRc = $LASTEXITCODE
$ErrorActionPreference = $previousEap
if ($scanRc -ne 0) { throw "protection scan failed rc=$scanRc" }
$scan = (($scanText | Out-String).Trim() | ConvertFrom-Json)
if ($scan.status -ne "pass") { throw "protection scan status not pass" }

$os = Get-CimInstance Win32_OperatingSystem
$now = (Get-Date)
$bootstrap = [ordered]@{
    schema = "e1-compute-profile-opt-bootstrap-1"
    round = "static-gather optimization 2026-10-23 (S0-S5)"
    plan = "docs/E1_COMPUTE_PROFILE_STATIC_GATHER_OPTIMIZATION_PLAN_20261023.md"
    run_root = ($RunRoot -replace "\\", "/")
    creator = "lead-agent (deepseek-v4.1-flash, DSH session)"
    created_local = $now.ToString("o")
    created_timezone = "Asia/Shanghai"
    bootstrap_completed_local = (Get-Date).ToString("o")
    parent_run = [ordered]@{ path = ($parentRun -replace "\\", "/"); reference = $parentRef
        status = "read_only_reference; no parent file is written by this round" }
    host = [ordered]@{ name = $env:COMPUTERNAME; os = $os.Caption; os_version = $os.Version;
                       total_phys_bytes = $phys; free_bytes_d = $free; powershell = $PSVersionTable.PSVersion.ToString() }
    tools = [ordered]@{ python = $py; python_version = (& $py --version 2>&1 | Out-String).Trim();
                        vivado = $vivado; vivado_version = "2021.1"; xvlog = $xvlog;
                        part = "xczu15eg-ffvb1156-2-i"; clock_period_ns = 5.0 }
    resource_gate = [ordered]@{ free_gib = [math]::Round($free/1GB, 3); total_phys_gib = [math]::Round($phys/1GB, 3);
                                free_gate_gib = 40; phys_gate_gib = 12; pass = $true }
    editable_sources = $beforeRows
    new_files_expected_absent = $newRows
    frozen_inputs = $frozenRows
    rtl_files = $rtlRows
    reference_yolo = $yoloRows
    protection_scan = [ordered]@{ file = "identity/protected_before.json"; scanner = "bootstrap_scan_v2.py";
                                  threads = 8; file_count = $scan.files; total_bytes = $scan.bytes;
                                  error_count = $scan.errors; elapsed_seconds = $scan.elapsed_seconds;
                                  manifest_sha256 = $scan.manifest_sha256 }
    scripts = [ordered]@{ bootstrap_sha256 = (Get-Hash (Join-Path $RunRoot "identity/bootstrap.ps1"));
                          scan_sha256 = (Get-Hash $scanDst) }
}
$bootstrapText = ($bootstrap | ConvertTo-Json -Depth 10) + $nl
Set-Content -LiteralPath (Join-Path $RunRoot "identity/bootstrap.json") -Value $bootstrapText -Encoding utf8
$bootstrapSha = (Get-Hash (Join-Path $RunRoot "identity/bootstrap.json"))
$ownership = [ordered]@{
    schema = "e1-compute-profile-opt-run-ownership-1"
    run_root = ($RunRoot -replace "\\", "/")
    owner = "lead-agent (deepseek-v4.1-flash, DSH session)"
    created_local = $now.ToString("o")
    bootstrap_sha256 = $bootstrapSha
    exclusive = $true
    note = "Created exclusively by the S0 bootstrap; every later tool must verify this record before writing."
    locks = @()
}
Set-Content -LiteralPath (Join-Path $RunRoot "identity/ownership.json") -Value (($ownership | ConvertTo-Json -Depth 6) + $nl) -Encoding utf8
Write-Output ([ordered]@{ status = "pass"; run_root = $bootstrap.run_root; bootstrap_sha256 = $bootstrapSha;
    protected_files = $scan.files; protected_bytes = $scan.bytes; scan_seconds = $scan.elapsed_seconds;
    editable_sources = $beforeRows.Count; frozen_inputs = $frozenRows.Count; rtl_files = $rtlRows.Count } | ConvertTo-Json -Compress)
