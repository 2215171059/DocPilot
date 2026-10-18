"""
retrieve.py —— 节点 ②：检索资料

职责：
拿当前的问题（可能是原问题，也可能是被重写过的版本）去检索器里召回证据。

一个重要的设计选择：
检索为什么做成"节点"，而不是做成"工具"？
两者都行，但取舍不同：

  做法 A：做成工具（node + ToolNode）
    优点：由模型自己决定何时检索、用什么 query，灵活
    缺点：模型可能忘记调用，或者把 query 写得乱七八糟；流程不可控

  做法 B：做成节点（本项目主图采用）
    优点：流程确定，每一步都在你的掌控里；
          检索结果能直接落进 state，方便评分、引用、调试
    缺点：灵活性低，不能"按需多次检索"

经验：**流程确定的部分用节点，需要模型自主决策的部分用工具。**
本项目的 S2/S3 阶段会演示工具做法，S4 之后切回节点做法。
"""

from __future__ import annotations

from docpilot.config import get_settings
from docpilot.retrievers.factory import get_retriever
from docpilot.state import DocPilotState
from docpilot.utils.tracing import trace_event, with_trace


def retrieve(state: DocPilotState) -> dict:
    """用当前问题检索知识库，把命中的证据写进 state。"""
    settings = get_settings()
    retriever = get_retriever()

    query = state.get("question") or state.get("original_question", "")
    chunks = retriever.retrieve(query, top_k=settings.top_k)

    # 转成普通 dict —— 状态里的东西要能被序列化进检查点数据库
    documents = [dict(chunk) for chunk in chunks]

    preview = query if len(query) <= 60 else query[:60] + "…"

    return {
        # 注意：documents 没有 reducer，所以这里是「覆盖」而不是「追加」。
        # 这正确反映了语义：每次检索是一次全新的召回，不该和历史结果混在一起。
        "documents": documents,
        "trace": with_trace(
            state,
            trace_event(
                "retrieve",
                detail=f"用「{preview}」召回 {len(documents)} 条证据",
                hit_count=len(documents),
                query=preview,
            ),
        ),
    }
