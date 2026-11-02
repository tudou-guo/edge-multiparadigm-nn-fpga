#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Frozen shared constants for the CfC PYNQ demo (docs/CFC_CONTRACT.md).

Importable by the Python driver (board), the integer reference and test scripts.
No third-party dependencies.
"""

CFC_ID = 0x43464331              # "CFC1"
CFC_VERSION = 0x00010000         # V1.0
SPEC_VERSION = "1.0.0"
CSR_BASE = 0x0480000000
CSR_SIZE = 0x1000
CLOCK_HZ = 300000000
TIMEOUT_CYCLES_DEFAULT = 1000000
N_STEPS_DEFAULT = 64
N_STEPS_MAX = 64
DELTA_T_INT = 655
DELTA_T_FLOAT_TRAIN = 0.16
N_INPUT_MAX = 64
N_OUTPUT_MAX = 64

REG_ID = 0x000
REG_VERSION = 0x004
REG_CONTROL = 0x008
REG_STATUS = 0x00C
REG_ERROR_CODE = 0x010
REG_N_STEPS = 0x014
REG_DELTA_T = 0x018
REG_MODE = 0x01C
REG_RUN_CYCLES = 0x020
REG_STEP_COUNT = 0x024
REG_TIMEOUT_CYCLES = 0x028
REG_CLOCK_HZ = 0x02C
REG_H_FINAL = 0x080
REG_INPUT = 0x100
REG_OUTPUT = 0x200

CONTROL_START = 1 << 0
CONTROL_CLEAR_STATE = 1 << 1
CONTROL_CLEAR_STATUS = 1 << 2
STATUS_BUSY = 1 << 0
STATUS_DONE = 1 << 1
STATUS_ERROR = 1 << 2
MODE_CLEAR_BEFORE_RUN = 1 << 0
MODE_CLEAR_EACH_STEP = 1 << 1
ERROR_NONE = 0
ERROR_CONFIG = 1
ERROR_BUSY_START = 2
ERROR_TIMEOUT = 3

N_STATE = 4
N_IN = 1
DATA_FRAC_BITS = 12
DATA_SCALE = 1 << DATA_FRAC_BITS
DATA_DTYPE = "int16"
BIAS_FRAC_BITS = 24
BIAS_SCALE = 1 << BIAS_FRAC_BITS
BIAS_DTYPE = "int32"
ACC_BITS = 48
ACC_MIN_LOGICAL_BITS = 40
DELTA_T_INT_MIN = 1
DELTA_T_INT_MAX = 32767

LUT_SIZE = 1025
LUT_INDEX_MIN = 0
LUT_INDEX_MAX = 1024
LUT_MIN_REAL = -8.0
LUT_MAX_REAL = 8.0
LUT_STEP_REAL = 1.0 / 64.0
LUT_INDEX_BIAS = 32768
LUT_INDEX_ROUND = 32
LUT_INDEX_SHIFT = 6
LUT_OUT_FRAC_BITS = 12

N_WEIGHTS = 84
N_BIASES = 17
W1_BASE = 0
W2_BASE = 20
WA_BASE = 40
WB_BASE = 60
WO_BASE = 80
B1_BASE = 0
B2_BASE = 4
BA_BASE = 8
BB_BASE = 12
BO_INDEX = 16
BRANCH_BASES = (W1_BASE, W2_BASE, WA_BASE, WB_BASE)
BRANCH_BIAS_BASES = (B1_BASE, B2_BASE, BA_BASE, BB_BASE)
BRANCH_NAMES = ("W1", "W2", "Wa", "Wb")


def weight_index(branch, o, i):
    return BRANCH_BASES[branch] + o * (N_IN + N_STATE) + i


def bias_index(branch, o):
    return BRANCH_BIAS_BASES[branch] + o


def round_q12_sat16(v):
    if v >= 0:
        q = (v + 2048) >> 12
    else:
        q = -(((-v) + 2048) >> 12)
    return max(-32768, min(32767, q))


def lut_index(v):
    idx = (v + 32768 + 32) >> 6
    return max(0, min(1024, idx))


def sat16(v):
    return max(-32768, min(32767, v))


def sat32(v):
    return max(-2147483648, min(2147483647, v))


def pack_i16(v):
    return int(v) & 0xFFFF


def pack_i32(v):
    return int(v) & 0xFFFFFFFF


def unpack_i16(word):
    word = int(word) & 0xFFFF
    return word - 0x10000 if word & 0x8000 else word


def unpack_i32(word):
    word = int(word) & 0xFFFFFFFF
    return word - 0x100000000 if word & 0x80000000 else word


DELTA_T = DELTA_T_INT
N_STEPS = N_STEPS_DEFAULT
CFC_CSR_BASE = CSR_BASE
