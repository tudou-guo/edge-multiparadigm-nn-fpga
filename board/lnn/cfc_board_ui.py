#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Notebook display/plot helpers for the LNN CfC PYNQ demo.

Used by board/lnn_cfc_demo.ipynb (and by the offline notebook validation).
Falls back to plain text printing when IPython is not active, so the same
notebook code runs in the offline check harness (matplotlib Agg).
"""

from __future__ import annotations

import html
import json
import re
import traceback
from pathlib import Path

Q_SCALE = 4096
SOURCE_HARDWARE = "hardware"
SOURCE_MODEL = "simulation-model"


# --------------------------------------------------------------- IPython glue
def _ipython_active():
    try:
        from IPython import get_ipython
        return get_ipython() is not None
    except Exception:
        return False


def _ipy_display():
    if not _ipython_active():
        return None
    try:
        from IPython.display import display, Markdown, HTML
        return display, Markdown, HTML
    except Exception:
        return None


def display_md(text):
    glue = _ipy_display()
    if glue:
        glue[0](glue[1](text))
    else:
        print(_strip_md(text))


def display_html(markup):
    glue = _ipy_display()
    if glue:
        glue[0](glue[2](markup))
    else:
        print(_strip_html(markup))


def _strip_md(text):
    return re.sub(r"[*_`#>|]", "", str(text))


def _strip_html(markup):
    text = re.sub(r"<br\s*/?>", "", str(markup))
    text = re.sub(r"</(tr|p|div|h[1-6])>", "", text)
    text = re.sub(r"</t[dh]>", " | ", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = html.unescape(text)
    return re.sub(r"[ \t]+", " ", text)


def _esc(value):
    return html.escape(str(value))


def _num(value, ndigits=6):
    if value is None:
        return "n/a"
    try:
        return ("%." + str(ndigits) + "g") % float(value)
    except Exception:
        return str(value)


def table_html(headers, rows):
    out = ['<table style="border-collapse:collapse;font-size:13px;margin:4px 0">']
    out.append("<tr>")
    for h in headers:
        out.append('<th style="border:1px solid #bbb;padding:4px 8px;'
                   'background:#f2f2f2;text-align:left">%s</th>' % _esc(h))
    out.append("</tr>")
    for row in rows:
        out.append("<tr>")
        for cell in row:
            out.append('<td style="border:1px solid #ddd;padding:3px 8px">%s</td>' % cell)
        out.append("</tr>")
    out.append("</table>")
    return "".join(out)


def badge(source):
    if source == SOURCE_HARDWARE:
        return ('<div style="background:#e6f6e6;border-left:5px solid #2e8b57;'
                'padding:8px 12px;margin:6px 0;font-size:14px">'
                '<b>数据来源：真实板卡（overlay + MMIO）· 本次实测</b></div>')
    return ('<div style="background:#fff4e0;border-left:5px solid #d2691e;'
            'padding:8px 12px;margin:6px 0;font-size:14px">'
            '<b>⚠ 数据来源：离线寄存器模型（simulation-model）—— 未在真实板卡上运行，'
            '不能作为板端实测证据</b></div>')


def _verdict_banner(ok, ok_text, fail_text):
    color = "#2e8b57" if ok else "#b22222"
    bg = "#e6f6e6" if ok else "#fdecea"
    text = ok_text if ok else fail_text
    return ('<div style="background:%s;border-left:5px solid %s;padding:8px 12px;'
            'margin:6px 0;font-size:15px"><b>%s</b></div>' % (bg, color, _esc(text)))


# --------------------------------------------------------------------- figures
def _get_pyplot():
    try:
        import matplotlib
        if not hasattr(matplotlib, "get_backend") or matplotlib.get_backend() == "":
            pass
        import matplotlib.pyplot as plt
        try:
            matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei",
                                                      "DejaVu Sans", "Arial"]
            matplotlib.rcParams["axes.unicode_minus"] = False
        except Exception:
            pass
        return plt
    except Exception:
        return None


def _finish_fig(fig, save_path=None):
    if save_path:
        try:
            Path(str(save_path)).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(str(save_path), dpi=130, bbox_inches="tight")
            print("figure saved:", save_path)
        except Exception as exc:
            print("figure save failed:", exc)
    if _ipython_active():
        try:
            import matplotlib.pyplot as plt
            plt.show()
        except Exception:
            pass
    return fig


def _series(sample, result):
    n = result.get("n_steps") or sample.get("n_steps")
    x_int = sample.get("x_int") or []
    x_float = sample.get("float_x")
    if x_float is None and x_int:
        x_float = [v / float(Q_SCALE) for v in x_int]
    target = sample.get("target_float")
    float_ref = sample.get("float_y")
    out_int = result.get("outputs_int") or []
    return {
        "steps": list(range(n)),
        "input": (x_float[:n] if x_float else None),
        "target": (target[:n] if target else None),
        "float_ref": (float_ref[:n] if float_ref else None),
        "fpga": [v / float(Q_SCALE) for v in out_int[:n]],
        "golden_int": (sample.get("y_int_expected") or [])[:n],
    }


def plot_sample(sample, save_path=None, title=None):
    plt = _get_pyplot()
    if plt is None:
        print("matplotlib unavailable - skip plot")
        return None
    s = _series(sample, {"n_steps": sample.get("n_steps"), "outputs_int": []})
    fig, ax = plt.subplots(figsize=(10, 3.6))
    if s["input"]:
        ax.plot(s["steps"], s["input"], label="input x[t]", color="#1f77b4")
    if s["target"]:
        ax.plot(s["steps"], s["target"], label="task target x[t-4]", color="#2ca02c",
                linestyle="--")
    if s["float_ref"]:
        ax.plot(s["steps"], s["float_ref"], label="float reference y[t]", color="#9467bd")
    if s["golden_int"]:
        ax.plot(s["steps"], [v / float(Q_SCALE) for v in s["golden_int"]],
                label="integer golden y[t]", color="#ff7f0e", linestyle=":")
    ax.set_xlabel("step t")
    ax.set_ylabel("value (real)")
    ax.set_title(title or ("sample %s: input / target / float reference / integer golden"
                           % sample.get("sample_id")))
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    return _finish_fig(fig, save_path)


def plot_four_curves(sample, result, save_path=None, title=None):
    """The four required curves: input, task target, float reference, FPGA out."""
    plt = _get_pyplot()
    if plt is None:
        print("matplotlib unavailable - skip plot")
        return None
    s = _series(sample, result)
    fig, ax = plt.subplots(figsize=(10.5, 4.2))
    if s["input"]:
        ax.plot(s["steps"], s["input"], label="input x[t]", color="#1f77b4", alpha=0.85)
    if s["target"]:
        ax.plot(s["steps"], s["target"], label="task target x[t-4]", color="#2ca02c",
                linestyle="--", linewidth=2)
    if s["float_ref"]:
        ax.plot(s["steps"], s["float_ref"], label="float reference y[t]", color="#9467bd")
    if s["fpga"]:
        ax.plot(s["steps"], s["fpga"], label="FPGA output y[t]", color="#d62728",
                marker="o", markersize=2.5)
    cmp = result.get("compare") or {}
    ax.set_xlabel("step t")
    ax.set_ylabel("value (real)")
    ax.set_title(title or ("%s | mode=%s | matched %s/%s | task MSE=%s | %s"
                           % (sample.get("sample_id"), result.get("mode"),
                              cmp.get("output_match_count"), cmp.get("output_total"),
                              _num(cmp.get("task_mse"), 8),
                              result.get("source") or "")))
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    return _finish_fig(fig, save_path)


def plot_mode_compare(sample, normal_result, clear_result, save_path=None):
    plt = _get_pyplot()
    if plt is None:
        print("matplotlib unavailable - skip plot")
        return None
    ns = _series(sample, normal_result)
    cs = _series(sample, clear_result)
    fig, ax = plt.subplots(figsize=(10.5, 4.0))
    if ns["target"]:
        ax.plot(ns["steps"], ns["target"], label="task target x[t-4]", color="#2ca02c",
                linestyle="--", linewidth=2)
    if ns["fpga"]:
        ax.plot(ns["steps"], ns["fpga"], label="FPGA output (state kept)", color="#d62728",
                marker="o", markersize=2.5)
    if cs["fpga"]:
        ax.plot(cs["steps"], cs["fpga"], label="FPGA output (clear state each step)",
                color="#ff7f0e", marker="s", markersize=2.5, linestyle="--")
    cmp_n = normal_result.get("compare") or {}
    cmp_c = clear_result.get("compare") or {}
    ax.set_xlabel("step t")
    ax.set_ylabel("value (real)")
    ax.set_title("state memory effect | task MSE kept=%s vs clear-each-step=%s"
                 % (_num(cmp_n.get("task_mse"), 8), _num(cmp_c.get("task_mse"), 8)))
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    return _finish_fig(fig, save_path)


# ----------------------------------------------------------------- step views
def show_bringup(report):
    source = report.get("source")
    display_html(badge(source))
    dev = report.get("device") or {}
    ov = report.get("overlay") or {}
    mf = report.get("manifest") or {}
    spec = report.get("spec") or {}
    rc = report.get("reference_selfcheck") or {}
    rows = [
        ["backend", _esc(dev.get("backend"))],
        ["CSR base", _esc(dev.get("csr_base")) + " (64-bit=%s)" % dev.get("csr_base_is_64bit")],
        ["ID", _esc(dev.get("id")) + (" ✓" if dev.get("id_ok") else " ✗")],
        ["VERSION", _esc(dev.get("version")) + (" ✓" if dev.get("version_ok") else " ✗")],
        ["CLOCK_HZ", _esc(dev.get("clock_hz")) + (" ✓" if dev.get("clock_hz_ok") else " ✗")],
        ["bit", _esc(ov.get("bit") or "(not found)")],
        ["hwh", _esc(ov.get("hwh") or "(not found)") + " (same stem=%s)" % ov.get("hwh_same_stem")],
        ["xsa", _esc(ov.get("xsa") or "(not found)")],
        ["bit_sha256", _esc(ov.get("bit_sha256") or "n/a")],
        ["hwh_sha256", _esc(ov.get("hwh_sha256") or "n/a")],
        ["manifest", _esc(mf.get("path") or "(not found)")],
        ["manifest_sha256", _esc(mf.get("sha256") or "n/a")],
        ["manifest files", "total=%s verified=%s mismatch=%s" % (
            (mf.get("files") or {}).get("total"), (mf.get("files") or {}).get("verified"),
            len((mf.get("files") or {}).get("mismatch") or []))],
        ["cfc_spec check", "found=%s ok=%s" % (spec.get("found"), spec.get("ok"))],
        ["reference selfcheck", "ok=%s y_mismatch=%s h_mismatch=%s" % (
            rc.get("ok"), rc.get("y_mismatch"), rc.get("h_mismatch"))],
        ["fixture export tree", _esc(report.get("fixture"))],
    ]
    display_md("### 步骤 1 · Overlay 加载与版本核对")
    display_html(table_html(["item", "value"], rows))
    warnings = report.get("warnings") or []
    if warnings:
        display_md("**警告：** " + "; ".join(warnings))
    if report.get("errors"):
        display_html(_verdict_banner(False, "", "br:".join(["bring-up 校验失败"] + list(report["errors"]))))


def show_sample(sample, save_path=None):
    display_md("### 步骤 2 · 固定测试样本（输入 / 任务目标 / 参考值）")
    rows = [
        ["sample", _esc(sample.get("sample_id"))],
        ["n_steps", _esc(sample.get("n_steps"))],
        ["delta_t_int (Q12)", _esc(sample.get("delta_t_int"))],
        ["delta_t_float", _esc(sample.get("delta_t_float"))],
        ["h_init", _esc(sample.get("h_init"))],
        ["phase / seed", "%s / %s" % (_esc(sample.get("phase")), _esc(sample.get("seed")))],
        ["warmup skip", _esc(sample.get("warmup_skip"))],
        ["golden final state", _esc(sample.get("h_expected_final"))],
        ["fixture sample", _esc(sample.get("fixture"))],
        ["source", _esc(", ".join(sample.get("source_files") or []) or "(gen hex)")],
    ]
    display_html(table_html(["item", "value"], rows))
    cons = sample.get("consistency") or {}
    if cons:
        display_md("一致性检查: " + json.dumps(cons, ensure_ascii=False))
    n = sample.get("n_steps") or 0
    xs = (sample.get("x_int") or [])[:8]
    ts = (sample.get("target_float") or [])[:8]
    rows = [[i, v] for i, v in enumerate(xs)]
    display_md("输入前 8 点（int16 Q12，实值 = int/4096）")
    display_html(table_html(["i", "x_int", "real"], [[i, v, _num(v / float(Q_SCALE), 6)]
                                                     for i, v in rows]))
    plot_sample(sample, save_path=save_path)


def show_run(result):
    display_md("### 步骤 3 · 下发与运行")
    status = result.get("status")
    ok = status == "done"
    display_html(_verdict_banner(ok, "运行完成：DONE", "运行异常：%s" % status))
    rows = [
        ["sample / mode", "%s / %s" % (_esc(result.get("sample_id")), _esc(result.get("mode")))],
        ["n_steps / delta_t_int", "%s / %s" % (result.get("n_steps"), result.get("delta_t_int"))],
        ["status / status_word", "%s / %s" % (_esc(status), _esc(result.get("status_word")))],
        ["error_code", "%s (%s)" % (_esc(result.get("error_code")), _esc(result.get("error_name")))],
        ["step_count", _esc(result.get("step_count"))],
        ["run_cycles", _esc(result.get("run_cycles"))],
        ["poll count / busy_seen", "%s / %s" % (result.get("polls"), result.get("busy_seen"))],
        ["input write-back ok", _esc(result.get("input_writeback_ok"))],
        ["write / wait / readback (s)", "%s / %s / %s" % (
            _num(result.get("write_time_s"), 6), _num(result.get("wait_time_s"), 6),
            _num(result.get("readback_time_s"), 6))],
        ["e2e (s)", _num(result.get("e2e_time_s"), 6)],
    ]
    display_html(table_html(["item", "value"], rows))
    if result.get("status") != "done":
        diag = result.get("diagnostics")
        if diag:
            display_md("诊断信息:")
            display_html("<pre>" + _esc(json.dumps(diag, indent=2, ensure_ascii=False)) + "</pre>")


def show_compare(result):
    display_md("### 步骤 4 · 整数逐点比对（先比对，后画图）")
    cmp = result.get("compare") or {}
    ok = bool(cmp.get("outputs_bit_exact")) and bool(cmp.get("final_state_match")) \
        and bool(cmp.get("step_count_ok"))
    display_html(_verdict_banner(ok, "整数比对 PASS：%s/%s 点逐位一致，末状态一致"
                                     % (cmp.get("output_match_count"), cmp.get("output_total")),
                                 "整数比对 FAIL：%s/%s 点一致，mismatch=%s"
                                 % (cmp.get("output_match_count"), cmp.get("output_total"),
                                    cmp.get("output_mismatch_count"))))
    rows = [
        ["matched", "%s / %s" % (cmp.get("output_match_count"), cmp.get("output_total"))],
        ["mismatch indices", _esc(cmp.get("mismatch_indices"))],
        ["final state (FPGA)", _esc(cmp.get("final_state"))],
        ["final state (expected)", _esc(cmp.get("final_state_expected"))],
        ["final_state_match", _esc(cmp.get("final_state_match"))],
        ["step_count ok", _esc(cmp.get("step_count_ok"))],
        ["expected source", _esc(cmp.get("expected_source"))],
    ]
    display_html(table_html(["item", "value"], rows))


def show_metrics(result):
    cmp = result.get("compare") or {}
    display_md("**实测/模型指标**")
    rows = [
        ["任务 MSE（FPGA vs 目标 x[t-4]，跳过预热 %s 步）" % cmp.get("warmup_skip"),
         _num(cmp.get("task_mse"), 8)],
        ["定点误差 MSE（FPGA 整数 vs 定点 golden，real 单位）",
         _num(cmp.get("fixed_point_mse_real"), 10)],
        ["浮点参考任务 MSE（float ref vs 目标）", _num(cmp.get("float_reference_mse"), 8)],
        ["run_cycles", _esc(result.get("run_cycles"))],
        ["clock_hz", _esc(result.get("clock_hz"))],
        ["硬件耗时 = run_cycles / clock_hz (s)", _num(result.get("hardware_time_s"), 9)],
        ["端到端耗时（板端 monotonic，含写/等/读, s)", _num(result.get("e2e_time_s"), 6)],
        ["source", _esc(result.get("source") or "")],
    ]
    display_html(table_html(["metric", "value"], rows))
    display_html(badge(result.get("source") or SOURCE_MODEL))


def show_mode_compare(normal_result, clear_result, sample, save_path=None):
    display_md("### 步骤 5 · 每步清状态模式对比（状态记忆作用）")
    cmp_n = normal_result.get("compare") or {}
    cmp_c = clear_result.get("compare") or {}
    rows = [
        ["保留隐状态（normal）", _esc(cmp_n.get("output_match_count")) + "/"
         + _esc(cmp_n.get("output_total")),
         _num(cmp_n.get("task_mse"), 8), _esc(cmp_n.get("final_state_match"))],
        ["每步清状态（clear-each-step）", _esc(cmp_c.get("output_match_count")) + "/"
         + _esc(cmp_c.get("output_total")),
         _num(cmp_c.get("task_mse"), 8), _esc(cmp_c.get("final_state_match"))],
    ]
    display_html(table_html(["模式", "整数比对", "任务 MSE", "末状态一致"], rows))
    display_md("每步清状态时输出只依赖当前输入 x[t]（无历史信息），任务 MSE 应当明显变差；"
               "保留状态时硬件连续维护 4 维隐状态，才能重建延迟 4 步的波形。")
    plot_mode_compare(sample, normal_result, clear_result, save_path=save_path)


def show_repeat_and_segments(checks, save_path=None):
    display_md("### 步骤 6 · 可重复性与状态保持验证")
    rows = []
    rep = (checks.get("checks") or {}).get("repeat")
    if rep:
        rows.append(["清零后重复运行", "PASS" if rep.get("pass") else "FAIL",
                     "outputs_equal=%s state_equal=%s"
                     % (rep.get("outputs_equal"), rep.get("final_state_equal"))])
    seg = (checks.get("checks") or {}).get("segments")
    if seg:
        rows.append(["64 步 = 32 + 32（中途保留状态）", "PASS" if seg.get("pass") else "FAIL",
                     "combined_equals_full=%s state_equal=%s"
                     % (seg.get("combined_equals_full"), seg.get("final_state_equals_full"))])
    ce = (checks.get("checks") or {}).get("clear_each_step")
    if ce:
        rows.append(["每步清状态 vs 整数参考", "PASS" if ce.get("pass") else "FAIL",
                     "task_mse=%s" % _num(ce.get("task_mse"), 8)])
    display_html(table_html(["验证项", "结果", "细节"], rows))
    display_html(_verdict_banner(bool(checks.get("pass")),
                                 "全部状态验证 PASS", "存在状态验证 FAIL"))
    if checks.get("checks", {}).get("exception"):
        display_md("异常: " + _esc(checks["checks"]["exception"].get("message")))


def show_saved(paths):
    display_md("### 结果保存")
    rows = [[_esc(k), _esc(v)] for k, v in sorted((paths or {}).items())]
    display_html(table_html(["artefact", "path"], rows))
    saved = checks_note = ""
    if isinstance(paths, dict) and paths.get("summary"):
        try:
            with open(paths["summary"], "r", encoding="utf-8") as fh:
                doc = json.load(fh)
            saved = ("summary.json: source=%s hardware_measurement=%s pass=%s"
                     % (doc.get("source"), doc.get("is_hardware_measurement"), doc.get("pass")))
        except Exception:
            saved = "summary.json written"
    if saved:
        display_md(saved)


def show_error(exc, session=None):
    display_md("### ⚠ 错误 / 诊断")
    display_html(_verdict_banner(False, "", "运行失败: " + str(exc)))
    diag = getattr(exc, "diagnostics", None)
    if diag:
        display_html("<pre style=\"max-height:360px;overflow:auto\">"
                     + _esc(json.dumps(diag, indent=2, ensure_ascii=False)) + "</pre>")
    tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    display_html("<pre style=\"max-height:240px;overflow:auto\">" + _esc(tb) + "</pre>")


def show_checks(checks):
    rows = []
    for name, val in (checks.get("checks") or {}).items():
        if name == "exception":
            rows.append(["exception", "FAIL", _esc(val.get("message"))])
        else:
            rows.append([name, "PASS" if val.get("pass") else "FAIL", ""])
    display_html(table_html(["check", "result", "detail"], rows))


def show_environment(paths):
    display_md("### 0 · 环境与路径解析")
    rows = [[_esc(k), _esc(v)] for k, v in sorted((paths or {}).items())]
    display_html(table_html(["item", "path"], rows))
