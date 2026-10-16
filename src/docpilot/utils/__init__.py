"""工具模块：执行轨迹收集、JSON 容错解析、图结构可视化。"""

from docpilot.utils.jsonx import parse_json_loose, safe_float
from docpilot.utils.tracing import render_trace_text, trace_event
from docpilot.utils.viz import describe_nodes, print_graph_ascii, to_mermaid

__all__ = [
    "trace_event",
    "render_trace_text",
    "parse_json_loose",
    "safe_float",
    "print_graph_ascii",
    "to_mermaid",
    "describe_nodes",
]
