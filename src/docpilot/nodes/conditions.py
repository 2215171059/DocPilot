"""
conditions.py —— 条件边用的「路由器函数」

条件边是什么？
普通边是"执行完 A 一定去 B"。
条件边是"执行完 A，调用一个函数看看状态，根据返回值决定去 B 还是 C"。

这个函数就是"路口的红绿灯"。它只做判断，不修改状态，也不调用模型 ——
保持轻量很重要，因为它会被频繁调用。

LangGraph 里的写法：
    builder.add_conditional_edges(
        "grade_documents",        # 从哪个节点出发
        after_grade,              # 判断函数
        {"generate": "generate",  # 函数返回值 → 目标节点名
         "rewrite": "rewrite_query"},
    )
"""

from __future__ import annotations

from docpilot.config import get_settings
from docpilot.state import DocPilotState


def after_route(state: DocPilotState) -> str:
    """路由节点之后：需要资料就去检索，不需要就直接生成。"""
    return "retrieve" if state.get("route") == "retrieve" else "generate"


def after_grade(state: DocPilotState) -> str:
    """
    评分节点之后：证据够用就生成，不够用就重写查询再来一轮。

    注意最后那道保险：重写次数用尽时必须放行到 generate，
    否则图会永远转圈（虽然在 LangGraph 里最终会被 recursion_limit 拦住，
    但那是兜底，不该当作正常控制流来用）。
    """
    settings = get_settings()

    score = float(state.get("retrieval_score", 0.0) or 0.0)
    used = int(state.get("rewrite_count", 0) or 0)

    if score >= settings.score_threshold:
        return "generate"

    if used >= settings.max_rewrites:
        # 次数用尽 —— 带着现有证据去生成，答案里会说明资料不足
        return "generate"

    return "rewrite"


def after_reflect(state: DocPilotState) -> str:
    """反思节点之后：需要人工确认就停下来等人，否则直接定稿。"""
    return "human_review" if state.get("need_human") else "finalize"
