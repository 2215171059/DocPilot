"""
nodes —— 图里的每一个节点

约定：一个节点 = 一个纯函数，签名统一为

    def 节点名(state: DocPilotState) -> dict:
        ...做一件事...
        return {"要更新的字段": 新值, "trace": [trace_event(...)]}

返回值是「增量更新」，不是完整状态。
LangGraph 会拿这个 dict 去合并状态 —— 合并规则由 state.py 里的 reducer 决定。

为什么坚持"一个节点只做一件事"？
  1. 条件边需要明确的判断点，粒度太粗就没法分叉
  2. 出错时能精确定位是哪一步坏了
  3. 每个节点可以单独测试、单独替换（比如把评分节点从"用大模型"换成"用规则"）
"""

from docpilot.nodes.conditions import after_grade, after_reflect, after_route
from docpilot.nodes.finalize import finalize
from docpilot.nodes.generate import generate
from docpilot.nodes.grade import grade_documents
from docpilot.nodes.reflect import reflect
from docpilot.nodes.retrieve import retrieve
from docpilot.nodes.review import human_review
from docpilot.nodes.rewrite import rewrite_query
from docpilot.nodes.router import route_question

__all__ = [
    "route_question",
    "retrieve",
    "grade_documents",
    "rewrite_query",
    "generate",
    "reflect",
    "human_review",
    "finalize",
    "after_route",
    "after_grade",
    "after_reflect",
]
