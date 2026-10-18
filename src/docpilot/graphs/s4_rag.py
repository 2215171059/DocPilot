"""
s4_rag.py —— 阶段 S4：接入 RAG（朴素版）

本阶段目标：
把知识库接进来，让回答有据可依。

图非常简单，就是一条直线：

    START → 检索 → 生成 → END

为什么还要单独提供一个简化版？
因为它是基准线（baseline）。
很多人的 RAG 项目到此为止，然后就会发现两个甩不掉的问题：

  问题 1：检索不到就不管了
    用户问的东西知识库里没有，检索返回空列表，
    模型还是硬答了一段 —— 这就是幻觉的来源。

  问题 2：检索到垃圾也照用
    召回了几段毫不相关的文字，模型一本正经地总结了一通，
    用户看起来像是"有依据"，其实全是错的。

S5 就是来修这两个问题的。
**先把基准线跑出来，才能说清楚"优化了什么、提升了多少"。**
做工程尤其要养成这个习惯：没有 baseline 的优化，都是自嗨。
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

import sys
from pathlib import Path

# 让「python src/docpilot/graphs/s4_rag.py」这条命令能直接跑：
# 把项目根的 src 加进模块搜索路径，否则会报 ModuleNotFoundError: No module named 'docpilot'。
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from docpilot.nodes.generate import generate
from docpilot.nodes.retrieve import retrieve
from docpilot.state import DocPilotState, initial_state


def build_rag_graph():
    """
    朴素 RAG：检索 → 生成。

    注意这里直接复用了 nodes/ 目录下的 retrieve 和 generate 函数 ——
    和 S5、S6 用的是**同一批节点**，只是连法不同。

    这是个值得记住的结论：**节点是可复用的积木，图是搭法。**
    先想清楚有哪些"原子步骤"，再想怎么把它们连起来，
    比一上来就纠结拓扑结构要高效得多。
    """
    builder = StateGraph(DocPilotState)

    builder.add_node("retrieve", retrieve)
    builder.add_node("generate", generate)

    builder.add_edge(START, "retrieve")
    builder.add_edge("retrieve", "generate")
    builder.add_edge("generate", END)

    return builder.compile()


if __name__ == "__main__":
    from docpilot.utils.viz import print_graph_ascii

    graph = build_rag_graph()

    print("=" * 62)
    print("S4 朴素 RAG —— 图的形状：")
    print("=" * 62)
    print_graph_ascii(graph)

    for question in (
        "连续输错密码会怎么样？",
        "你们支持哪些接入渠道？",
    ):
        print("=" * 62)
        print(f"提问：{question}")
        print("=" * 62)
        result = graph.invoke(initial_state(question))
        print(f"命中证据：{len(result.get('documents') or [])} 条")
        print("\n回答：")
        print(result.get("answer", ""))
        print()
