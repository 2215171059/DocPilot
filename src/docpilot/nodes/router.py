"""
router.py —— 节点 ①：路由判断

职责：
先判断用户这句话"要不要查资料"。
- 要查 → 走检索分支
- 不用查（比如"你好""你是谁"）→ 直接回答，省一次检索开销

为什么需要它？
不做路由的话，每句话都要走一遍完整检索流程。
用户说句"谢谢"，你还去知识库里翻半天，既慢又费钱。
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
