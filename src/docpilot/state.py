"""
state.py —— 整张图的「状态」定义

【一句话理解 State】
State 就是一张在整个流程里传来传去的共享便签纸。
每个节点都能读上面的内容，也能往上面写新内容。

【为什么不能只用一个普通 dict？】
因为 LangGraph 需要知道：当两个节点同时往同一个字段写东西时，该以谁为准？
这个"合并规则"就叫 reducer（归约函数）。

  - 不写 Annotated → 默认是「覆盖」：新值直接盖掉旧值
  - 写 Annotated[list, add] → 变成「追加」：新值拼接到旧值后面

这就是新手最容易踩的坑：消息列表如果忘了加 reducer，
第二轮对话时历史消息会凭空消失。

【面试考点】
"LangGraph 的 State 是什么？reducer 有什么用？"
参考答法：State 是贯穿图的共享数据结构，用 TypedDict 定义；
reducer 决定字段的合并语义，默认覆盖，
用 Annotated[x, add] 可以改成累加，典型场景是多轮对话的消息列表。
"""

from __future__ import annotations

from typing import Annotated, Any, TypedDict

from langchain_core.messages import AnyMessage

# add_messages 是 LangGraph 内置的消息专用 reducer。
# 它比 operator.add 聪明一点：会根据消息 id 做去重/更新，而不是无脑追加。
from langgraph.graph.message import add_messages


class DocPilotState(TypedDict, total=False):
    """
    total=False 表示所有字段都是可选的 —— 初始调用时不必把所有字段都填上。

    字段分成四组，看注释就能明白每个节点在读写什么。
    """

    # ── 组 1：对话消息 ───────────────────────────────
    # AnyMessage = HumanMessage / AIMessage / ToolMessage 的联合类型
    # reducer = add_messages → 新消息追加，历史自动保留（多轮记忆的基础）
    messages: Annotated[list[AnyMessage], add_messages]

    # ── 组 2：问题与路由 ─────────────────────────────
    original_question: str      # 用户最初的问题（重写查询时保留原版）
    question: str               # 当前使用的问题（可能已被重写）
    route: str                  # 路由结果："retrieve" 或 "direct"

    # ── 组 3：检索与自纠正 ───────────────────────────
    # 注意：这里存的是普通 dict 而不是 LangChain 的 Document 对象。
    # 原因：状态需要被检查点序列化后存进数据库，
    #      简单的 dict 最不容易出岔子。这是工程上的务实选择。
    documents: list[dict[str, Any]]   # [{"content":..., "source":..., "score":...}]
    retrieval_score: float            # 证据评分（0~10）
    grade_reason: str                 # 评分理由
    rewrite_count: int                # 已经重写了几次（防死循环的计数器）

    # ── 组 4：输出与审核 ─────────────────────────────
    answer: str                       # 最终答案
    reflection: dict[str, Any]        # 反思节点返回的结构化结果
    need_human: bool                  # 是否需要人工确认
    review_decision: str              # 人工审核结果："approve" / "reject" / ""

    # ── 组 5：可观测性 ───────────────────────────────
    # 执行轨迹。这里**故意没有用 add 型 reducer**，后面会解释原因。
    # 每个节点通过 utils.tracing.with_trace() 显式拼接（读取旧值 + 追加新值），
    # 语义上是「覆盖」，所以每一轮提问都只保留本轮自己的轨迹。
    trace: list[dict[str, Any]]

    # 出错了也不让图崩掉，把错误信息留在状态里给界面展示
    error: str


# ============================================================
# 为什么 trace 不用 Annotated[list, operator.add]？
# ------------------------------------------------------------
# 上面 messages 用了 add_messages，一用就上瘾，很容易顺手给所有列表都加上 add。
# 但 trace 加 add 会踩到一个坑：
#
#   一旦挂上 checkpointer，同一个 thread_id 的状态会跨轮保留。
#   add 型 reducer 的语义是"旧值 + 新值"，
#   于是第二轮提问时，第一轮的轨迹**还在**，第三轮又叠上去……
#   几轮之后界面就变成一团乱麻。
#
# 所以这里做了个取舍：
#   问自己一句 —— "这个列表是希望跨轮累积，还是每轮重来？"
#   历史消息要累积 → 用 reducer
#   执行轨迹要每轮重来 → 用覆盖，让节点自己拼
#
# 代价是：节点里要写成 with_trace(state, event)，
# 而且**必须保证节点之间是串行执行**的 ——
# 如果有并行的两条分支同时写 trace，后写的会盖掉先写的。
# 真遇到并行场景，就得老老实实改回 reducer 了。
# ============================================================


def initial_state(question: str) -> DocPilotState:
    """构造一次新调用的初始状态。只填必要的字段，其余走默认值。"""
    from langchain_core.messages import HumanMessage

    return DocPilotState(
        messages=[HumanMessage(content=question)],
        original_question=question,
        question=question,
        route="",
        documents=[],
        retrieval_score=0.0,
        grade_reason="",
        rewrite_count=0,
        answer="",
        reflection={},
        need_human=False,
        review_decision="",
        trace=[],
        error="",
    )
