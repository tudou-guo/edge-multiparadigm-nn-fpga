#!/bin/sh
# Offline sanity check for the P4 tools; no Vivado/XSim/torch/onnx needed.
# Run from anywhere:  sh tools/onnx_gen/run_p4_offline_tests.sh
set -e
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../.." && pwd)
cd "$root"
python3 -I -B verification/restart/test_e1_p4.py
