"""
s1_basic.py —— 阶段 S1：最小可运行的状态图

【这一阶段的目标】
先别管大模型聪不聪明、检索准不准。
只搞明白一件事：**状态是怎么在节点之间流动的**。

整张图只有三个节点，一条直线：

    START → 分类 → 回答 → 润色 → END

【为什么分类节点不用大模型？】
为了让这一阶段的注意力完全集中在"图"上。
分类用关键词规则就够，跑起来快、结果稳定、没有任何不确定性。
等 S4 之后再考虑把这种小判断换成模型 —— 那时你已经知道图怎么跑了。

【这一阶段要重点观察的三件事】
1. 每个节点函数的返回值，都只是**一部分**字段，不是完整状态
2. 返回的字段会被合并进全局状态，下一个节点读到的就是合并后的结果
3. trace 字段用了 add 类型的 reducer，所以是累加；
   而 answer 被覆盖了两次（先取值，再被润色节点改写）

跑一下看输出，比读十遍文档都管用。
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

import sys
from pathlib import Path

# 让「python src/docpilot/graphs/s1_basic.py」这条命令能直接跑：
# 把项目根的 src 加进模块搜索路径，否则会报 ModuleNotFoundError: No module named 'docpilot'。
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from docpilot.config import get_llm


# ============================================================
# 状态定义
# ============================================================
class BasicState(TypedDict, total=False):
    question: str
    category: str
    answer: str
    # 用 add 做 reducer，多个节点各自追加的记录会拼在一起，而不是互相覆盖
    trace: Annotated[list[dict[str, Any]], operator.add]


# ============================================================
# 节点 1：分类（纯规则，不调模型）
# ============================================================
_TECH_WORDS = ("接口", "报错", "部署", "配置", "参数", "代码", "错误码", "超时", "限流", "数据库")
_BIZ_WORDS = ("价格", "合同", "退款", "发票", "工单", "投诉", "续费", "许可", "坐席")


def classify(state: BasicState) -> dict:
    """给问题打个粗分类标签。"""
    question = state.get("question", "")

    if any(word in question for word in _TECH_WORDS):
        category = "技术问题"
    elif any(word in question for word in _BIZ_WORDS):
        category = "业务问题"
    else:
        category = "其他问题"

    return {
        "category": category,
        "trace": [{"step": "① 分类", "detail": f"判定为「{category}」"}],
    }


# ============================================================
# 节点 2：回答（这里才调模型）
# ============================================================
def answer(state: BasicState) -> dict:
    """根据分类结果组织回答。"""
    question = state.get("question", "")
    category = state.get("category", "其他问题")

    llm = get_llm()
    response = llm.invoke(
        [
            SystemMessage(
                content=f"你是一个简洁的助手。用户的问题是「{category}」，请用两三句话回答。"
            ),
            HumanMessage(content=question),
        ]
    )

    return {
        "answer": str(response.content).strip(),
        "trace": [{"step": "② 回答", "detail": "已调用模型生成原始回答"}],
    }


# ============================================================
# 节点 3：润色（演示"覆盖"语义）
# ============================================================
def polish(state: BasicState) -> dict:
    """给答案加上分类前缀。注意：answer 字段被覆盖了。"""
    category = state.get("category", "")
    original = state.get("answer", "")

    return {
        "answer": f"【{category}】{original}",
        "trace": [{"step": "③ 润色", "detail": "已拼接分类前缀（answer 被覆盖）"}],
    }


# ============================================================
# 组装成图
# ============================================================
def build_basic_graph():
    """
    最朴素的用法：三个节点 + 三条普通边。

    记住四个固定动作：
      1. StateGraph(状态类型)      → 声明这张图用什么状态
      2. add_node(名字, 函数)      → 注册节点
      3. add_edge(A, B)            → 连边
      4. compile()                 → 编译成可执行对象
    """
    builder = StateGraph(BasicState)

    builder.add_node("classify", classify)
    builder.add_node("answer", answer)
    builder.add_node("polish", polish)

    builder.add_edge(START, "classify")
    builder.add_edge("classify", "answer")
    builder.add_edge("answer", "polish")
    builder.add_edge("polish", END)

    return builder.compile()


# ============================================================
# 直接跑这个文件时的小演示
# ============================================================
if __name__ == "__main__":
    from docpilot.utils.viz import print_graph_ascii

    graph = build_basic_graph()

    print("=" * 62)
    print("S1 最小状态图 —— 图的形状：")
    print("=" * 62)
    print_graph_ascii(graph)

    question = "接口调用报错 RATE_LIMITED 是什么意思？"
    print("=" * 62)
    print(f"提问：{question}")
    print("=" * 62)

    result = graph.invoke({"question": question, "trace": []})

    print("\n── 执行轨迹 ──")
    for item in result["trace"]:
        print(f"  {item['step']}：{item['detail']}")

    print("\n── 最终状态里的字段 ──")
    print(f"  question = {result.get('question')!r}")
    print(f"  category = {result.get('category')!r}")
    print(f"  answer   = {result.get('answer')!r}")
