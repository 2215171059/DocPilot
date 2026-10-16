"""
main.py —— 阶段 S6：完整的主图（记忆 + 人机协同）

这是整个项目最终对外提供的图。和 S5 相比，它多了三样东西：

  1. reflect 节点      —— 答案生成后再审一遍，查幻觉、判敏感度
  2. human_review 节点 —— 需要人拍板时用 interrupt() 把图暂停
  3. finalize 节点     —— 统一收口，顺便给两条分支提供汇聚点

  4. checkpointer      —— 让整张图具备"记忆"和"可暂停"能力

完整拓扑：

    START
      │
      ▼
  route_question ──(不需要资料)──────────────► generate
      │(需要资料)
      ▼
   retrieve ◄────────────────┐
      │                      │
      ▼                      │
  grade_documents ──(不够好)─► rewrite_query
      │(够好)
      ▼
   generate
      │
      ▼
   reflect ──(需人工确认)──► human_review ──┐
      │(不需要)                            │
      └──────────────► finalize ◄──────────┘
                          │
                          ▼
                         END

【checkpointer 到底是什么？】
它像一个"自动存档"装置。图每往前走一步，就把当前完整状态写一份到存储里，
用 thread_id 作为存档槽位编号。

带来三个能力：
  1. 多轮记忆：同一个 thread_id 再调用，会自动把历史状态读回来
  2. 断点续跑：程序崩了、人走开了，状态都还在
  3. 时间旅行：get_state_history() 能列出所有历史存档，回到任意一步重跑

存储选型（面试会问）：
  InMemorySaver   —— 存内存，重启就没，适合开发和测试
  SqliteSaver     —— 存本地文件，单机演示够用
  PostgresSaver   —— 生产环境标配，支持并发和高可用

【为什么 messages 字段能实现"多轮记忆"？】
因为它的 reducer 是 add_messages。
同一个 thread_id 第二次调用时，LangGraph 把上次的消息列表读回来，
新消息通过 reducer 追加在后面 —— 于是模型就"记得"之前聊过什么。

注意：这只解决了"记住对话"。
如果要做"跨会话记住用户偏好"（比如这个用户偏好简短回答），
那要用另一个东西叫 Store（长期记忆），和 checkpointer 是两回事。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from langgraph.graph import END, START, StateGraph

from docpilot.nodes.conditions import after_grade, after_reflect, after_route
from docpilot.nodes.finalize import finalize
from docpilot.nodes.generate import generate
from docpilot.nodes.grade import grade_documents
from docpilot.nodes.reflect import reflect
from docpilot.nodes.retrieve import retrieve
from docpilot.nodes.review import human_review
from docpilot.nodes.rewrite import rewrite_query
from docpilot.nodes.router import route_question
from docpilot.state import DocPilotState


def make_checkpointer(kind: str = "memory", *, sqlite_path: str | Path | None = None):
    """
    创建检查点存储器。

    参数：
        kind        —— "memory"（内存）或 "sqlite"（本地文件）
        sqlite_path —— sqlite 模式下的数据库路径
    """
    if kind == "sqlite":
        import sqlite3

        from langgraph.checkpoint.sqlite import SqliteSaver

        path = Path(sqlite_path or "docpilot_checkpoints.db")
        conn = sqlite3.connect(str(path), check_same_thread=False)
        return SqliteSaver(conn)

    from langgraph.checkpoint.memory import InMemorySaver

    return InMemorySaver()


def build_graph(*, checkpointer: Any | None = None, checkpointer_kind: str = "memory"):
    """
    构建主图。

    参数：
        checkpointer      —— 直接传入一个检查点实例（优先级最高）
        checkpointer_kind —— 不传实例时，按这个值自动创建（"memory" / "sqlite" / "none"）

    返回：
        编译好的图对象，支持 invoke / stream / get_state / get_state_history
    """
    builder = StateGraph(DocPilotState)

    # ── 注册节点 ────────────────────────────────────────
    builder.add_node("route_question", route_question)
    builder.add_node("retrieve", retrieve)
    builder.add_node("grade_documents", grade_documents)
    builder.add_node("rewrite_query", rewrite_query)
    builder.add_node("generate", generate)
    builder.add_node("reflect", reflect)
    builder.add_node("human_review", human_review)
    builder.add_node("finalize", finalize)

    # ── 连接边 ─────────────────────────────────────────
    builder.add_edge(START, "route_question")

    builder.add_conditional_edges(
        "route_question",
        after_route,
        {"retrieve": "retrieve", "generate": "generate"},
    )

    builder.add_edge("retrieve", "grade_documents")

    builder.add_conditional_edges(
        "grade_documents",
        after_grade,
        {"generate": "generate", "rewrite": "rewrite_query"},
    )

    builder.add_edge("rewrite_query", "retrieve")
    builder.add_edge("generate", "reflect")

    builder.add_conditional_edges(
        "reflect",
        after_reflect,
        {"human_review": "human_review", "finalize": "finalize"},
    )

    builder.add_edge("human_review", "finalize")
    builder.add_edge("finalize", END)

    # ── 编译 ───────────────────────────────────────────
    if checkpointer is not None:
        cp = checkpointer
    elif checkpointer_kind == "none":
        cp = None
    else:
        cp = make_checkpointer(checkpointer_kind)

    return builder.compile(checkpointer=cp)


def new_thread_config(thread_id: str, *, recursion_limit: int = 25) -> dict[str, Any]:
    """
    构造一次调用的配置。

    thread_id 就像"会话窗口"的编号：
      - 同一 thread_id 反复调用 → 接着上次聊（多轮记忆）
      - 换一个 thread_id        → 全新会话，互不影响

    recursion_limit 是保险丝：LangGraph 里的"一步"= 访问一个节点，
    默认 25 步对于本项目足够，但显式写出来能让读代码的人知道这里有保护。
    """
    return {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": recursion_limit,
    }
