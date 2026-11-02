"""E1 ONNX subset frontend and canonical graph IR."""

from .graph_ir import CompileError, Graph, Node, Tensor

__all__ = ["CompileError", "Graph", "Node", "Tensor"]
__version__ = "0.1.0"