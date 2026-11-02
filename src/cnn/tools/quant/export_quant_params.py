#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ABI 1.0 量化参数导出工具（conv / pool 共用）。

本模块是 project_2 / project_3 量化链路的**唯一 ABI 打包入口**：把 PyTorch 侧
训练/校准得到的浮点 scale / zero-point 转成 ABI 1.0 冻结格式，供
``verification/golden/conv3x3_ref.py``、XSim 与 PYNQ 驱动共用。

输出契约（ABI 1.0.0，见 docs/op_abi.md 第 7 节）
------------------------------------------------
QUANT_CFG[31:0]::

    [7:0]   X_ZERO_POINT   signed INT8，输入 activation
    [15:8]  W_ZERO_POINT   冻结为 0（ABI 1.0 不支持非零 weight zero point）
    [17:16] ROUND_MODE     0 = nearest, ties away from zero
    [31:18] reserved       必须写 0

每输出通道 8 byte 记录（``<IBbh``，little-endian）::

    offset 0: M0[31:0]        unsigned multiplier
    offset 4: SHIFT[7:0]      unsigned, 0..62
    offset 5: out_zero_point  signed INT8
    offset 6: reserved[15:0]  MUST be 0

重定标（与 RTL / golden 逐位一致）::

    M_real = s_x * s_w[oc] / s_y
    M_real ~= M0 / 2**SHIFT
    y      = round_ties_away(acc32 * M0 / 2**SHIFT) + out_zero_point[oc]
    yq     = clamp(y, -128, 127)

``quant.bin`` 按 ``Cout_alloc = align_up(Cout, 8)`` 条记录写出，通道补齐项
全零；这与 ``conv3x3_ref.generate_golden()`` 的向量校验完全一致。

本模块不依赖 torch / numpy，可离线复算，便于 CI 与板端交叉核对。

CLI::

    python tools/quant/export_quant_params.py --self-test
    python tools/quant/export_quant_params.py \
        --layer-json layer_quant.json --out-dir out/conv1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import struct
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

__all__ = [
    "ABI_VERSION",
    "ABI_VERSION_REG",
    "QUANT_RECORD_BYTES",
    "M0_MAX",
    "SHIFT_MAX",
    "INT8_MIN",
    "INT8_MAX",
    "CHANNEL_BLOCK",
    "ALIGN_BYTES",
    "QuantParamError",
    "align_up",
    "round_half_away",
    "decompose_multiplier",
    "quant_cfg_snapshot",
    "pack_quant_record",
    "build_quant_blob",
    "build_bias_blob",
    "quant_record_from_scales",
    "quantize_bias_int32",
    "requantize_acc",
    "symmetric_scale",
    "export_layer",
    "load_layer_spec",
    "self_test",
    "main",
]

ABI_VERSION = "1.0.0"
ABI_VERSION_REG = 0x00010000

#: ABI 1.0 QUANT 记录布局常量
QUANT_RECORD_BYTES = 8
M0_MAX = 0xFFFFFFFF
SHIFT_MAX = 62
INT8_MIN = -128
INT8_MAX = 127
INT32_MIN = -(2 ** 31)
INT32_MAX = 2 ** 31 - 1

#: ABI 1.0 channel block / 对齐
CHANNEL_BLOCK = 8
ALIGN_BYTES = 16

#: QUANT_CFG round mode
ROUND_MODE_NEAREST_AWAY = 0


class QuantParamError(ValueError):
    """量化参数越界，或违反 ABI 1.0 冻结约定。"""


# ---------------------------------------------------------------------------
# 基础工具
# ---------------------------------------------------------------------------

def align_up(value: int, alignment: int = ALIGN_BYTES) -> int:
    """向上对齐到 ``alignment`` 字节。"""
    value = int(value)
    alignment = int(alignment)
    if alignment <= 0:
        raise QuantParamError("alignment must be positive")
    return ((value + alignment - 1) // alignment) * alignment


def round_half_away(value: float) -> int:
    """nearest, ties away from zero（ABI 1.0 冻结的舍入模式）。"""
    value = float(value)
    if not math.isfinite(value):
        raise QuantParamError("cannot round non-finite value: %r" % value)
    if value >= 0.0:
        return int(math.floor(value + 0.5))
    return -int(math.floor((-value) + 0.5))


def symmetric_scale(max_abs: float, levels: int = INT8_MAX) -> float:
    """对称 INT8 量化 scale = max_abs / levels（levels 默认 127）。"""
    max_abs = float(max_abs)
    levels = int(levels)
    if levels <= 0:
        raise QuantParamError("levels must be positive")
    if max_abs <= 0.0:
        return 1.0
    return max_abs / float(levels)


def decompose_multiplier(m_real: float, shift_max: int = SHIFT_MAX) -> Tuple[int, int]:
    """把正实数乘数分解成 ABI 的 ``(M0, SHIFT)``，取精度最高的可行解。

    在 ``0 <= SHIFT <= shift_max`` 且 ``1 <= M0 <= 0xFFFFFFFF`` 的约束下，
    选择使 ``M0`` 最大（即定点精度最高）的那个 SHIFT。
    """
    m_real = float(m_real)
    shift_max = int(shift_max)
    if not math.isfinite(m_real) or m_real < 0.0:
        raise QuantParamError("multiplier must be finite and non-negative: %r" % m_real)
    if shift_max < 0 or shift_max > SHIFT_MAX:
        raise QuantParamError("shift_max must be in [0, %d]" % SHIFT_MAX)
    if m_real == 0.0:
        return 0, 0

    best: Optional[Tuple[int, int]] = None
    for shift in range(0, shift_max + 1):
        m0 = round_half_away(m_real * float(1 << shift))
        if m0 > M0_MAX:
            break
        if m0 >= 1:
            best = (m0, shift)
    if best is None:
        raise QuantParamError(
            "multiplier %r cannot be represented with SHIFT<=%d" % (m_real, shift_max)
        )
    return best


def quant_cfg_snapshot(x_zero_point: int, w_zero_point: int = 0,
                       round_mode: int = ROUND_MODE_NEAREST_AWAY) -> int:
    """组装 32-bit ``QUANT_CFG`` 字（ABI 1.0 第 7.4 节）。"""
    x_zp = int(x_zero_point)
    w_zp = int(w_zero_point)
    mode = int(round_mode)
    if not (INT8_MIN <= x_zp <= INT8_MAX):
        raise QuantParamError("x_zero_point out of INT8 range: %d" % x_zp)
    if w_zp != 0:
        raise QuantParamError("ABI 1.0 requires w_zero_point == 0 (got %d)" % w_zp)
    if mode != ROUND_MODE_NEAREST_AWAY:
        raise QuantParamError("ABI 1.0 supports only nearest-ties-away rounding")
    return (x_zp & 0xFF) | ((w_zp & 0xFF) << 8) | ((mode & 0x3) << 16)


# ---------------------------------------------------------------------------
# 记录打包
# ---------------------------------------------------------------------------

def pack_quant_record(m0: int, shift: int, out_zero_point: int,
                      reserved: int = 0) -> bytes:
    """打包单条 8-byte ABI 量化记录（little-endian ``<IBbh``）。"""
    m0 = int(m0)
    shift = int(shift)
    out_zp = int(out_zero_point)
    reserved = int(reserved)
    if not (0 <= m0 <= M0_MAX):
        raise QuantParamError("M0 out of range: %d" % m0)
    if not (0 <= shift <= SHIFT_MAX):
        raise QuantParamError("SHIFT out of range: %d" % shift)
    if not (INT8_MIN <= out_zp <= INT8_MAX):
        raise QuantParamError("out_zero_point out of range: %d" % out_zp)
    if reserved != 0:
        raise QuantParamError("reserved must be 0")
    blob = struct.pack("<IBbh", m0, shift, out_zp, reserved)
    if len(blob) != QUANT_RECORD_BYTES:  # pragma: no cover - 防御性检查
        raise QuantParamError("internal error: quant record is not 8 bytes")
    return blob


def _record_fields(record: Any) -> Tuple[int, int, int, int]:
    """从 dict / tuple 中取出 (M0, SHIFT, out_zero_point, reserved)。"""
    if isinstance(record, dict):
        return (
            int(record["M0"]),
            int(record["SHIFT"]),
            int(record["out_zero_point"]),
            int(record.get("reserved", 0)),
        )
    return (int(record[0]), int(record[1]), int(record[2]),
            int(record[3]) if len(record) > 3 else 0)


def build_quant_blob(records: Sequence[Any], cout_alloc: int) -> bytes:
    """按 Cout_alloc 条记录打包 ``quant.bin``（通道补齐项全零）。"""
    cout_alloc = int(cout_alloc)
    if cout_alloc <= 0:
        raise QuantParamError("cout_alloc must be positive")
    if len(records) > cout_alloc:
        raise QuantParamError("records(%d) > cout_alloc(%d)" % (len(records), cout_alloc))
    blob = bytearray(align_up(cout_alloc * QUANT_RECORD_BYTES))
    for index, record in enumerate(records):
        packed = pack_quant_record(*_record_fields(record))
        start = index * QUANT_RECORD_BYTES
        blob[start:start + QUANT_RECORD_BYTES] = packed
    return bytes(blob)


def build_bias_blob(bias_int32: Sequence[int], cout_alloc: int) -> bytes:
    """按 Cout_alloc 条 INT32 打包 ``bias.bin``（通道补齐项为零）。"""
    cout_alloc = int(cout_alloc)
    if len(bias_int32) > cout_alloc:
        raise QuantParamError("bias(%d) > cout_alloc(%d)" % (len(bias_int32), cout_alloc))
    blob = bytearray(align_up(cout_alloc * 4))
    for index, value in enumerate(bias_int32):
        value = int(value)
        if not (INT32_MIN <= value <= INT32_MAX):
            raise QuantParamError("bias32 out of INT32 range: %d" % value)
        struct.pack_into("<i", blob, index * 4, value)
    return bytes(blob)


# ---------------------------------------------------------------------------
# PyTorch scale/zp -> ABI 记录
# ---------------------------------------------------------------------------

def quant_record_from_scales(x_scale: float, w_scale: float, y_scale: float,
                             out_zero_point: int = 0) -> Dict[str, Any]:
    """由浮点 scale 计算一条 per-output-channel ABI 量化记录。

    ``M_real = x_scale * w_scale / y_scale``，再分解为 ``M0 / 2**SHIFT``。
    返回值带 ``m_real`` / ``m_quantized`` / ``rel_error`` 诊断字段，方便
    在校准日志里核对定点精度。
    """
    x_scale = float(x_scale)
    w_scale = float(w_scale)
    y_scale = float(y_scale)
    if x_scale <= 0.0 or w_scale <= 0.0 or y_scale <= 0.0:
        raise QuantParamError(
            "scales must be positive: x=%r w=%r y=%r" % (x_scale, w_scale, y_scale)
        )
    m_real = (x_scale * w_scale) / y_scale
    m0, shift = decompose_multiplier(m_real)
    m_quantized = float(m0) / float(1 << shift)
    rel_error = 0.0 if m_real == 0.0 else abs(m_quantized - m_real) / m_real
    record = {
        "M0": int(m0),
        "SHIFT": int(shift),
        "out_zero_point": int(out_zero_point),
        "reserved": 0,
        "m_real": m_real,
        "m_quantized": m_quantized,
        "rel_error": rel_error,
    }
    pack_quant_record(m0, shift, out_zero_point, 0)  # 早失败
    return record


def quantize_bias_int32(bias_float: float, x_scale: float, w_scale: float) -> int:
    """``bias32 = round(bias / (x_scale * w_scale))``（accumulator domain）。"""
    x_scale = float(x_scale)
    w_scale = float(w_scale)
    if x_scale <= 0.0 or w_scale <= 0.0:
        raise QuantParamError("scales must be positive")
    value = round_half_away(float(bias_float) / (x_scale * w_scale))
    if not (INT32_MIN <= value <= INT32_MAX):
        raise QuantParamError("bias32 out of INT32 range: %d" % value)
    return value


def requantize_acc(acc: int, m0: int, shift: int, out_zero_point: int = 0) -> int:
    """ABI 1.0 重定标（Python 参考实现，用于自检 / 对拍）。"""
    pack_quant_record(m0, shift, out_zero_point, 0)
    acc = int(acc)
    if not (INT32_MIN <= acc <= INT32_MAX):
        raise QuantParamError("accumulator outside INT32: %d" % acc)
    p = acc * int(m0)
    shift = int(shift)
    if shift == 0:
        rounded = p
    elif p >= 0:
        rounded = (p + (1 << (shift - 1))) >> shift
    else:
        rounded = -(((-p) + (1 << (shift - 1))) >> shift)
    y = rounded + int(out_zero_point)
    return max(INT8_MIN, min(INT8_MAX, y))


# ---------------------------------------------------------------------------
# 层导出
# ---------------------------------------------------------------------------

def load_layer_spec(path: str) -> Dict[str, Any]:
    """读取 layer quant spec JSON（见模块 docstring）。"""
    with open(str(path), "r", encoding="utf-8") as handle:
        spec = json.load(handle)
    if not isinstance(spec, dict):
        raise QuantParamError("layer spec must be a JSON object")
    return spec


def export_layer(spec: Dict[str, Any]) -> Dict[str, Any]:
    """按 layer spec 生成 quant.bin / bias.bin 及其 meta 字段。

    spec 必填: ``name, cout, x_scale, y_scale, weight_scales``；
    可选: ``cin, x_zero_point, y_zero_point, bias_float``。
    """
    name = str(spec.get("name", "layer"))
    cout = int(spec["cout"])
    if cout <= 0:
        raise QuantParamError("cout must be positive")
    x_scale = float(spec["x_scale"])
    y_scale = float(spec["y_scale"])
    x_zp = int(spec.get("x_zero_point", 0))
    y_zp = int(spec.get("y_zero_point", 0))
    weight_scales = [float(s) for s in spec["weight_scales"]]
    if len(weight_scales) != cout:
        raise QuantParamError(
            "weight_scales length %d != cout %d" % (len(weight_scales), cout)
        )

    records = [quant_record_from_scales(x_scale, w_scale, y_scale, y_zp)
               for w_scale in weight_scales]
    cout_alloc = align_up(cout, CHANNEL_BLOCK)
    quant_blob = build_quant_blob(records, cout_alloc)

    result: Dict[str, Any] = {
        "name": name,
        "abi_version": ABI_VERSION,
        "abi_version_reg": "0x%08X" % ABI_VERSION_REG,
        "cin": int(spec.get("cin", 0)),
        "cout": cout,
        "cout_alloc": cout_alloc,
        "x_scale": x_scale,
        "x_zero_point": x_zp,
        "y_scale": y_scale,
        "y_zero_point": y_zp,
        "quant_cfg_snapshot": "0x%08X" % quant_cfg_snapshot(x_zp, 0),
        "quant_bytes": len(quant_blob),
        "quant_sha256": hashlib.sha256(quant_blob).hexdigest(),
        "records": [
            {
                "oc": index,
                "M0": int(record["M0"]),
                "SHIFT": int(record["SHIFT"]),
                "out_zero_point": int(record["out_zero_point"]),
                "reserved": int(record["reserved"]),
            }
            for index, record in enumerate(records)
        ],
        "record_diagnostics": [
            {
                "oc": index,
                "m_real": float(record["m_real"]),
                "m_quantized": float(record["m_quantized"]),
                "rel_error": float(record["rel_error"]),
            }
            for index, record in enumerate(records)
        ],
        "max_rel_error": max([float(r["rel_error"]) for r in records] or [0.0]),
        "_quant_blob": quant_blob,
    }

    if "bias_float" in spec and spec["bias_float"] is not None:
        bias_float = [float(b) for b in spec["bias_float"]]
        if len(bias_float) != cout:
            raise QuantParamError(
                "bias_float length %d != cout %d" % (len(bias_float), cout)
            )
        bias_int32 = [
            quantize_bias_int32(bias, x_scale, w_scale)
            for bias, w_scale in zip(bias_float, weight_scales)
        ]
        bias_blob = build_bias_blob(bias_int32, cout_alloc)
        result["bias_int32"] = bias_int32
        result["bias_bytes"] = len(bias_blob)
        result["bias_sha256"] = hashlib.sha256(bias_blob).hexdigest()
        result["_bias_blob"] = bias_blob

    return result


def _public_result(result: Dict[str, Any]) -> Dict[str, Any]:
    """去掉内部 bytes 字段，得到可 JSON 序列化的 meta。"""
    return {key: value for key, value in result.items() if not key.startswith("_")}


# ---------------------------------------------------------------------------
# 自检 / CLI
# ---------------------------------------------------------------------------

def self_test() -> None:
    """无需 torch 的纯函数自检。"""
    assert align_up(0) == 0 and align_up(1) == 16 and align_up(16) == 16
    assert round_half_away(0.5) == 1 and round_half_away(-0.5) == -1
    assert round_half_away(2.5) == 3 and round_half_away(-2.5) == -3

    for m_real, tol in ((1.0, 0.0), (0.5, 1e-9), (3.7e-3, 1e-9), (1.0e-7, 1e-6)):
        m0, shift = decompose_multiplier(m_real)
        assert 1 <= m0 <= M0_MAX, (m_real, m0, shift)
        assert 0 <= shift <= SHIFT_MAX, (m_real, m0, shift)
        approx = m0 / float(1 << shift)
        assert abs(approx - m_real) <= tol * max(1.0, m_real), (m_real, approx)

    m0, shift = decompose_multiplier(0.5)
    assert requantize_acc(8, m0, shift, 0) == 4
    assert requantize_acc(-8, m0, shift, 0) == -4

    # ties away from zero：SHIFT=1, M0=1 -> 0.5 应进位到 1，-0.5 应进到 -1
    assert requantize_acc(1, 1, 1, 0) == 1
    assert requantize_acc(-1, 1, 1, 0) == -1
    assert requantize_acc(0, 1, 1, 0) == 0

    # 饱和在加 zero_point 之后
    assert requantize_acc(300, 1, 0, 0) == 127
    assert requantize_acc(-300, 1, 0, 0) == -128
    assert requantize_acc(200, 1, 0, -128) == 72
    assert requantize_acc(-1000, 1, 0, -128) == -128

    assert quant_cfg_snapshot(0) == 0x00000000
    assert quant_cfg_snapshot(-128) == 0x00000080
    assert quant_cfg_snapshot(5) == 0x00000005

    blob = build_quant_blob([{"M0": 1, "SHIFT": 0,
                              "out_zero_point": 7, "reserved": 0}], 16)
    assert len(blob) == 128
    assert blob[:8] == struct.pack("<IBbh", 1, 0, 7, 0)
    assert blob[8:] == bytes(len(blob) - 8)

    bias = build_bias_blob([-1, 2], 8)
    assert len(bias) == 32
    assert struct.unpack_from("<i", bias, 0)[0] == -1
    assert struct.unpack_from("<i", bias, 4)[0] == 2

    record = quant_record_from_scales(0.02, 0.003, 0.05)
    assert record["M0"] > 0 and record["rel_error"] < 1e-6, record

    spec = {
        "name": "selftest", "cin": 1, "cout": 2,
        "x_scale": 0.02, "x_zero_point": 0,
        "y_scale": 0.05, "y_zero_point": 0,
        "weight_scales": [0.003, 0.004],
        "bias_float": [0.01, -0.02],
    }
    result = export_layer(spec)
    assert result["cout_alloc"] == 8, result["cout_alloc"]
    assert result["quant_bytes"] == 64, result["quant_bytes"]
    assert result["bias_bytes"] == 32, result["bias_bytes"]
    json.dumps(_public_result(result))
    print("export_quant_params: self-test passed")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="ABI 1.0 quant parameter exporter (quant.bin + quant_meta.json)")
    parser.add_argument("--layer-json", help="layer quant spec JSON (see docstring)")
    parser.add_argument("--out-dir", help="output directory for quant.bin / bias.bin")
    parser.add_argument("--name", default=None, help="override layer name")
    parser.add_argument("--self-test", action="store_true", help="run pure-python self test")
    parser.add_argument("--print-json", action="store_true", help="print meta to stdout")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.self_test:
        self_test()
        if not args.layer_json:
            return 0
    if not args.layer_json:
        _build_parser().print_help()
        return 2
    try:
        spec = load_layer_spec(args.layer_json)
        if args.name:
            spec["name"] = args.name
        result = export_layer(spec)
    except QuantParamError as exc:
        print("E_QUANT: %s" % exc, file=sys.stderr)
        return 6

    meta = _public_result(result)
    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)
        with open(os.path.join(args.out_dir, "quant.bin"), "wb") as handle:
            handle.write(result["_quant_blob"])
        if "_bias_blob" in result:
            with open(os.path.join(args.out_dir, "bias.bin"), "wb") as handle:
                handle.write(result["_bias_blob"])
        with open(os.path.join(args.out_dir, "quant_meta.json"), "w",
                  encoding="utf-8") as handle:
            json.dump(meta, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        meta["out_dir"] = args.out_dir
    if args.print_json or not args.out_dir:
        print(json.dumps(meta, ensure_ascii=False, indent=2))
    print("EXPORT_QUANT_OK %s records=%d quant_bytes=%d max_rel_error=%.3e"
          % (meta["name"], len(meta["records"]), meta["quant_bytes"],
             meta["max_rel_error"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())