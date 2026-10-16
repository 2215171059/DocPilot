"""
review.py —— 节点 ⑦：人工审核（Human-in-the-Loop）

【它在干什么】
这个节点执行到 interrupt() 那一行会**停住**，把整张图的状态存进检查点数据库，
然后把"待审核的内容"交给外面的人看。等人回复之后，图从断点继续跑。

【这是 LangGraph 最"杀手级"的能力之一】
普通程序没法"暂停三分钟等人类回复"。
HTTP 请求有超时，进程重启状态就没了。
LangGraph 靠检查点（checkpointer）把状态持久化，
所以理论上可以停几天再恢复 —— 状态一直在，内存里什么都不占。

【三个必须记住的前提】
1. 必须先配 checkpointer，否则 interrupt 无从保存状态，会直接报错
2. 必须有固定的 thread_id，恢复时靠它找回是哪一次对话
3. 恢复要用 Command(resume=值)，这个值就是 interrupt() 的返回值

【一个容易踩的坑】
节点被恢复时是**从头重新执行**的，不是从 interrupt() 那一行接着执行。
所以 interrupt() 之前的代码会被再跑一遍。
如果前面有写数据库、发消息这类副作用操作，会重复执行 ——
正确做法是把副作用挪到 interrupt() 之后，或者做成幂等。

【面试考点】
"哪些场景需要人机协同？怎么落地的？"
场景：高风险操作（转账、删数据、发公告）、合规要求留痕、
低置信度分流、内容对外发布前的终审。
落地要点：检查点选型（生产用 PostgresSaver）、审核界面的超时与
催办策略、审核结果的留痕，以及刚才说的副作用幂等问题。
"""

from __future__ import annotations

from typing import Any

from langgraph.types import interrupt

from docpilot.prompts import HUMAN_REVIEW_PAYLOAD, format_issues
from docpilot.state import DocPilotState
from docpilot.utils.tracing import trace_event, with_trace

#: 哪些回复算"通过"
_APPROVE_WORDS = {"approve", "approved", "yes", "y", "ok", "pass", "通过", "同意", "可以", "确认"}


def human_review(state: DocPilotState) -> dict:
    """暂停执行，等待人工确认。"""
    answer = state.get("answer", "")
    reflection: dict[str, Any] = state.get("reflection") or {}
    issues = [
        *(reflection.get("unsupported_claims") or []),
    ]
    if not reflection.get("grounded", True) and not issues:
        issues.append("答案未被参考资料完全支撑")

    payload = HUMAN_REVIEW_PAYLOAD.format(
        answer=answer, issues=format_issues([str(i) for i in issues])
    )

    # ── 执行到这里会「停住」──
    # 恢复时，decision 就是外部通过 Command(resume=...) 传回来的值
    decision = interrupt(payload)

    approved = str(decision).strip().lower() in _APPROVE_WORDS
    final_answer = answer
    if not approved:
        final_answer = (
            f"{answer}\n\n---\n"
            "⚠️ 人工审核未通过，以上内容已被拦截，未对外发送。\n"
            f"审核意见：{decision}"
        )

    return {
        "review_decision": "approve" if approved else "reject",
        "answer": final_answer,
        "trace": with_trace(
            state,
            trace_event(
                "human_review",
                detail="人工审核通过" if approved else f"人工审核未通过：{decision}",
                status="ok" if approved else "warn",
                decision="approve" if approved else "reject",
            ),
        ),
    }
