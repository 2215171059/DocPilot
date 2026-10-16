"""
s3_react.py —— 阶段 S3：把 S2 的图拆开，一行行手写出来

【这一阶段的目标】
搞清楚 `create_react_agent` 内部到底画了什么。

它其实就是这张图：

        ┌──────────┐
   ┌───►│  agent   │  调用模型，模型决定要不要用工具
   │    └────┬─────┘
   │         │ 条件边：有 tool_calls 吗？
   │    ┌────┴────┐
   │    │         │
   │  有工具     没工具
   │    │         │
   │    ▼         ▼
   │ ┌───────┐   END
   └─│ tools │   执行工具，结果塞回消息列表
     └───────┘

就是两个节点 + 一条条件边 + 一条回边。就这么简单。

【关键的三个知识点】

1. 条件边的返回值可以是 END 本身
   {"tools": "tools", END: END} 表示：返回 END 就直接结束。

2. 循环靠的是"回边"
   tools → agent 这条边让流程又回到起点。
   所以必须设 recursion_limit（默认 25 步）——
   万一模型一直调工具不罢休，图会自己停住并抛 GraphRecursionError。
   LangGraph 里"一步"= 访问一个节点，一次工具往返算两步。

3. ToolNode 负责真执行
   模型输出的只是一句"我想调用 xxx"，不会真的执行任何东西。
   真正去跑函数、把结果包装成 ToolMessage 的，是 ToolNode。

【面试考点】
"ReAct 循环可能出什么问题？"
- 死循环：模型反复调同一个工具 → 用 recursion_limit 兜底
- 工具异常：LangGraph 不像老 AgentExecutor 那样吞异常，会抛出来；
  需要重试策略或 handle_tool_errors
- 上下文爆炸：每轮工具结果都进消息列表，聊久了 token 会超
  → 需要上下文裁剪 / 摘要压缩
- 参数幻觉：模型编造不存在的参数 → 用严格 schema + 校验
"""

from __future__ import annotations

from typing import Annotated, Any, TypedDict

from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

import sys
from pathlib import Path

# 让「python src/docpilot/graphs/s3_react.py」这条命令能直接跑：
# 把项目根的 src 加进模块搜索路径，否则会报 ModuleNotFoundError: No module named 'docpilot'。
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from docpilot.config import get_llm
from docpilot.tools import ALL_TOOLS

REACT_SYSTEM = """你是一个能使用工具的助手。

规则：
1. 需要查内部资料时，调用 search_knowledge_base 工具
2. 需要算数时，调用 calculator 工具
3. 工具返回结果后，用中文简洁地总结给用户，不要复述工具的原始输出格式
4. 不需要工具就能回答的问题，直接回答
"""


class ReactState(TypedDict, total=False):
    """
    只有消息列表就够了 —— 这是 ReAct 模式最精简的状态。

    注意 messages 必须带 add_messages reducer，
    否则第二轮工具结果会把第一轮的消息覆盖掉，循环直接断掉。
    """
    messages: Annotated[list[AnyMessage], add_messages]


def call_model(state: ReactState) -> dict:
    """调用模型，让它决定下一步：回答，还是调工具。"""
    llm = get_llm()
    model_with_tools = llm.bind_tools(ALL_TOOLS)

    messages = [SystemMessage(content=REACT_SYSTEM), *state["messages"]]
    response = model_with_tools.invoke(messages)

    return {"messages": [response]}


def should_continue(state: ReactState) -> str:
    """
    条件边的判断函数：看一眼模型最后说了什么。

    如果它输出了 tool_calls → 去执行工具
    否则 → 说明它给出了最终答案，结束
    """
    last_message = state["messages"][-1]
    tool_calls = getattr(last_message, "tool_calls", None)

    if tool_calls:
        return "tools"
    return END


def build_react_graph():
    """手写版的 ReAct Agent。"""
    builder = StateGraph(ReactState)

    # 节点 1：模型。负责"想"
    builder.add_node("agent", call_model)
    # 节点 2：工具执行器。负责"做"
    # ToolNode 会自动读取最后一条消息里的 tool_calls，逐个执行并把结果包成 ToolMessage
    builder.add_node("tools", ToolNode(ALL_TOOLS))

    builder.add_edge(START, "agent")

    # 条件边：模型说完话之后，路由到 tools 或直接结束
    builder.add_conditional_edges(
        "agent",
        should_continue,
        {"tools": "tools", END: END},
    )

    # 回边：工具执行完，回到模型继续想
    builder.add_edge("tools", "agent")

    return builder.compile()


if __name__ == "__main__":
    graph = build_react_graph()

    question = "连续输错密码 5 次会怎么样？顺便帮我算一下 5 次机会用掉 3 次还剩百分之多少。"
    print(f"提问：{question}\n")

    # recursion_limit 是"最多走几步"的保险丝，防止无限循环
    result = graph.invoke(
        {"messages": [HumanMessage(content=question)]},
        config={"recursion_limit": 12},
    )

    print("── 消息流转过程 ──")
    for i, msg in enumerate(result["messages"], start=1):
        kind = type(msg).__name__
        tool_calls = getattr(msg, "tool_calls", None)
        if tool_calls:
            print(f"  {i}. [{kind}] → 决定调用 {[tc['name'] for tc in tool_calls]}")
        elif kind == "ToolMessage":
            print(f"  {i}. [{kind}] → 返回 {len(str(msg.content))} 字结果")
        else:
            print(f"  {i}. [{kind}] → {str(msg.content)[:80]}")

    print("\n── 最终答案 ──")
    print(result["messages"][-1].content)
