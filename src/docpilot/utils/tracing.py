"""
tracing.py —— 执行轨迹收集

它解决什么问题？
LangGraph 跑起来之后，你只看到最后一句答案，
"中间到底走了哪些节点、每步花了多久、状态怎么变的"全靠猜。

做法很朴素：每个节点执行完，往 state["trace"] 里追加一条记录
（trace 字段的 reducer 是 operator.add，所以会自动累加）。
最后界面就能把整条思考链路画出来。
"""

from __future__ import annotations

import time
from typing import Any

# 节点名 → 中文标签 + 图标，界面上看得更直观
NODE_LABELS: dict[str, str] = {
    "route_question": "① 路由判断",
    "retrieve": "② 检索资料",
    "grade_documents": "③ 证据评分",
    "rewrite_query": "④ 重写查询",
    "generate": "⑤ 生成答案",
    "reflect": "⑥ 反思自查",
    "human_review": "⑦ 人工审核",
    "finalize": "⑧ 定稿输出",
}


def trace_event(
    node: str,
    *,
    detail: str = "",
    status: str = "ok",
    **extra: Any,
) -> dict[str, Any]:
    """
    生成一条轨迹记录。

    参数：
        node   —— 节点名，必须和 add_node 时注册的名字一致
        detail —— 一句话说明这一步做了什么
        status —— ok / retry / warn / paused / error
        extra  —— 想额外记录的任意字段（比如打分、迭代次数）
    """
    return {
        "node": node,
        "label": NODE_LABELS.get(node, node),
        "detail": detail,
        "status": status,
        "ts": time.time(),
        **extra,
    }


def with_trace(state: dict[str, Any], *events: dict[str, Any]) -> list[dict[str, Any]]:
    """
    把新事件拼到已有轨迹后面，返回一份新列表。

    节点里这样用：
        return {"trace": with_trace(state, trace_event("retrieve", detail="..."))}

    为什么不能像 messages 那样用 reducer 自动累加？
    见 state.py 里那段解释 —— 简单说就是：带检查点时 reducer 会跨轮累积，
    而轨迹我们希望每轮重置。
    """
    return [*(state.get("trace") or []), *events]


def render_trace_text(trace: list[dict[str, Any]]) -> str:
    """把轨迹渲染成纯文本，命令行下用。"""
    if not trace:
        return "（没有轨迹记录）"

    icons = {
        "ok": "✓",
        "retry": "↻",
        "warn": "!",
        "paused": "⏸",
        "error": "✗",
    }
    lines = []
    for i, event in enumerate(trace, start=1):
        icon = icons.get(event.get("status", "ok"), "·")
        label = event.get("label", event.get("node", "?"))
        detail = event.get("detail", "")
        lines.append(f"  {icon} {i:02d}. {label:<12} {detail}")
    return "\n".join(lines)
