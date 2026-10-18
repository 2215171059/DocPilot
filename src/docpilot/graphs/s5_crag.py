"""
s5_crag.py —— 阶段 S5：自纠正 RAG（本项目最核心的亮点）

本阶段目标：
给朴素 RAG 加上"反馈回路"，让它能自己发现"我找的资料不对"，然后换个姿势再找一次。

图长这样：

                    ┌─────────────────────────────┐
                    │                             │
    START           │                             │
      │             ▼                             │
      ▼        ┌──────────┐  分数不够   ┌──────────────┐
  ┌───────┐    │ 证据评分 │───────────►│  重写查询     │
  │ 路由  │    └────┬─────┘            └──────┬───────┘
  └───┬───┘         │ 分数够                  │
      │             ▼                         │
      │        ┌──────────┐                   │
      │        │  生成答案 │                  │
      │        └──────────┘                   │
      │                                        │
      ▼ (不需要查资料)                          │
     END                                       │
                    ┌──────────┐               │
                    │  检索资料 │◄──────────────┘
                    └──────────┘

（实际连线见 add_conditional_edges 那几行，别被 ASCII 图绕晕）

这个循环的价值：
朴素 RAG 是"单向链"，自纠正 RAG 是"带反馈的环"。
前者只能被动接受检索结果，后者能主动修正自己的错误。

只多了两个节点（评分、重写），但系统的行为档次完全不一样：
从"检索到什么就答什么"变成"先判断资料够不够，不够就再找"。

防死循环的三道保险：
  1. 计数器 rewrite_count：重写次数上限（本配置默认 2 次）
  2. 及格线判定：分数够了立刻离开循环
  3. recursion_limit：LangGraph 层面的兜底，超过步数直接抛异常

任何"带环的图"都必须想清楚怎么退出。这是设计图时第一个要回答的问题。
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

import sys
from pathlib import Path

# 让「python src/docpilot/graphs/s5_crag.py」这条命令能直接跑：
# 把项目根的 src 加进模块搜索路径，否则会报 ModuleNotFoundError: No module named 'docpilot'。
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from docpilot.nodes.conditions import after_grade, after_route
from docpilot.nodes.generate import generate
from docpilot.nodes.grade import grade_documents
from docpilot.nodes.retrieve import retrieve
from docpilot.nodes.rewrite import rewrite_query
from docpilot.nodes.router import route_question
from docpilot.state import DocPilotState, initial_state


def build_crag_graph():
    """自纠正 RAG：路由 → 检索 → 评分 →（重写 → 检索）→ 生成。"""
    builder = StateGraph(DocPilotState)

    builder.add_node("route_question", route_question)
    builder.add_node("retrieve", retrieve)
    builder.add_node("grade_documents", grade_documents)
    builder.add_node("rewrite_query", rewrite_query)
    builder.add_node("generate", generate)

    builder.add_edge(START, "route_question")

    # 分支 1：要不要查资料
    builder.add_conditional_edges(
        "route_question",
        after_route,
        {"retrieve": "retrieve", "generate": "generate"},
    )

    builder.add_edge("retrieve", "grade_documents")

    # 分支 2（核心）：证据够不够用
    builder.add_conditional_edges(
        "grade_documents",
        after_grade,
        {"generate": "generate", "rewrite": "rewrite_query"},
    )

    # 回边：重写完再检索一次，形成循环
    builder.add_edge("rewrite_query", "retrieve")

    builder.add_edge("generate", END)

    return builder.compile()


if __name__ == "__main__":
    from docpilot.utils.render import print_trace
    from docpilot.utils.viz import print_graph_ascii

    graph = build_crag_graph()

    print("=" * 66)
    print("S5 自纠正 RAG —— 图的形状：")
    print("=" * 66)
    print_graph_ascii(graph)

    # 这三个问题刻意做了区分：
    #   第 1 个：直接命中，评分高，一次通过
    #   第 2 个：口语化提问，第一次检索分数低，触发重写后通过
    #   第 3 个：闲聊，走 direct 分支，不做检索
    questions = [
        "连续输错密码 5 次会怎么样？",
        "账号被锁了多久才能再登",
        "你好，你是谁？",
    ]

    for question in questions:
        print("=" * 66)
        print(f"提问：{question}")
        print("=" * 66)
        result = graph.invoke(initial_state(question))

        print(f"路由结果：{result.get('route')}")
        print(f"证据条数：{len(result.get('documents') or [])}")
        print(f"证据评分：{result.get('retrieval_score')}（理由：{result.get('grade_reason')}）")
        print(f"重写次数：{result.get('rewrite_count')}")
        print()
        print_trace(result.get("trace") or [])
        print()
        print("回答：")
        print(result.get("answer", ""))
        print()
