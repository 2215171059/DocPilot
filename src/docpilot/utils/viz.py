"""
viz.py —— 把编译好的图"画"出来

LangGraph 自带两个很好用的方法：
  - graph.get_graph().print_ascii()  → 终端里画 ASCII 图
  - graph.get_graph().draw_mermaid() → 生成 Mermaid 源码，可贴到 Markdown / 在线编辑器

强烈建议：
第一次把图画出来看一眼，对"节点和边"的理解会立刻具体起来。
不画图的时候，图的拓扑很容易记错。
"""

from __future__ import annotations

from typing import Any


def print_graph_ascii(graph: Any) -> None:
    """在终端打印图的 ASCII 结构。"""
    try:
        graph.get_graph().print_ascii()
    except Exception as exc:  # pragma: no cover - 仅用于调试
        print(f"[viz] 无法打印 ASCII 图：{exc}")


def to_mermaid(graph: Any) -> str:
    """导出 Mermaid 流程图源码。"""
    try:
        return graph.get_graph().draw_mermaid()
    except Exception as exc:  # pragma: no cover
        return f"%% 生成 Mermaid 失败：{exc}"


def describe_nodes(graph: Any) -> list[str]:
    """列出图里所有节点名，方便快速核对注册是否遗漏。"""
    try:
        return sorted(graph.get_graph().nodes.keys())
    except Exception:  # pragma: no cover
        return []
