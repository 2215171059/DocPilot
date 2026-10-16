"""
router.py —— 节点 ①：路由判断

【它在干什么】
先判断用户这句话"要不要查资料"。
- 要查 → 走检索分支
- 不用查（比如"你好""你是谁"）→ 直接回答，省一次检索开销

【为什么需要它？】
不做路由的话，每句话都要走一遍完整检索流程。
用户说句"谢谢"，你还去知识库里翻半天，既慢又费钱。

【面试考点】
这叫「意图路由 / 分诊」。可以延伸聊：
- 工业界常见做法是先用小模型或规则做粗筛，只有复杂的才交给大模型
- 路由错了会有什么后果？（该检索的没检索 → 幻觉；不该检索的检索了 → 延迟和成本上升）
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from docpilot.config import get_llm
from docpilot.prompts import ROUTER_HUMAN, ROUTER_SYSTEM
from docpilot.state import DocPilotState
from docpilot.utils.tracing import trace_event, with_trace


def route_question(state: DocPilotState) -> dict:
    """判断问题该走「检索」还是「直答」。"""
    question = state.get("question") or state.get("original_question", "")

    llm = get_llm()
    response = llm.invoke(
        [
            SystemMessage(content=ROUTER_SYSTEM),
            HumanMessage(content=ROUTER_HUMAN.format(question=question)),
        ]
    )

    # 模型可能返回 "retrieve"、"'retrieve'"、或者带一堆解释，所以做宽容判断
    raw = str(response.content).strip().lower()
    route = "retrieve" if "retrieve" in raw else "direct"

    return {
        "route": route,
        "trace": with_trace(
            state, trace_event("route_question", detail=f"判定为「{route}」")
        ),
    }
