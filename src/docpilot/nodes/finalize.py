"""
finalize.py —— 节点 ⑧：定稿输出

职责：
一个"收口"节点。主要做三件事：
  1. 如果前面出了错，把错误信息整理成一句人话
  2. 如果一路都没检索到资料，在答案上追加一个明确的提示
  3. 记一条轨迹，让执行时间线有个完整的结尾

为什么要有这么一个"什么都不做"的节点？
它是图上的**汇聚点（join）**。
human_review 和 reflect 两条分支最后都要流到这里，
有了它，加边的时候就不用给每条分支单独连一个终点，
以后想加"统一打日志""统一写审计表"也只需要改这一处。
这类节点在复杂图里非常常见。
"""

from __future__ import annotations

from docpilot.state import DocPilotState
from docpilot.utils.tracing import trace_event, with_trace


def finalize(state: DocPilotState) -> dict:
    """收口：整理输出，记录结束事件。"""
    answer = state.get("answer", "").strip()
    error = state.get("error", "")

    if error:
        answer = f"处理过程中出现错误：{error}\n\n请稍后重试，或联系管理员查看日志。"
    elif not answer:
        answer = "抱歉，本次没有生成有效回答。"

    documents = state.get("documents") or []
    if not documents and not error:
        answer += (
            "\n\n---\n提示：本次回答未引用任何知识库资料。"
            "如需基于内部文档的准确答复，请补充更具体的描述。"
        )

    return {
        "answer": answer,
        "trace": with_trace(
            state,
            trace_event(
                "finalize",
                detail=f"输出最终答案（{len(answer)} 字），引用 {len(documents)} 条资料",
                citation_count=len(documents),
            ),
        ),
    }
