"""
s2_tools.py —— 阶段 S2：用高层 API 一行搭出「会调工具的 Agent」

【这一阶段的目标】
上一阶段的图是"我指定怎么走"。这一阶段反过来：
**让模型自己决定要不要调工具、调哪个、传什么参数。**

我们只给它两个工具：
  - search_knowledge_base：查知识库
  - calculator：算数

它该用哪个、用几次，全由模型自己判断。这叫 ReAct
（Reasoning + Acting，边推理边行动）。

【为什么先讲高层 API？】
因为生产上大部分场景，`create_react_agent` 就够用了。
它已经被打磨得很稳，自己手搓反而容易出 bug。
**能一行解决的事，不要写一百行。**

代价是：你只能看到结果，看不到内部结构。
所以 S3 我们会把它拆开，一行行重写一遍 —— 那时你既有"快速交付"的手段，
又有"出了事能拆开修"的底气。

【面试考点】
"LangGraph 的 API 分几层？"
答：三层，自顶向下是 ——
  1. 预构建图（create_react_agent / create_agent）：一行搞定标准模式
  2. Agent Node 中层 API：保留循环控制的同时省去手写细节
  3. 构建图底层 API（StateGraph）：完全手动，多分支、循环、人工介入都能显式表达
经验法则是：能用高层就别用底层，只有默认行为确实不满足时才下沉。
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage

import sys
from pathlib import Path

# 让「python src/docpilot/graphs/s2_tools.py」这条命令能直接跑：
# 把项目根的 src 加进模块搜索路径，否则会报 ModuleNotFoundError: No module named 'docpilot'。
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from docpilot.config import get_llm
from docpilot.tools import ALL_TOOLS


def _create_prebuilt_agent(model, tools):
    """
    兼容不同版本的预构建 Agent 工厂。

    ⚠️ 这里踩过一个真实的坑，很值得记住：
    `create_react_agent` 在 LangGraph 1.0 起被标记为**已废弃**（deprecated），
    官方要求改用 `langchain.agents.create_agent`，并计划在 2.0 移除。

    如果你照抄网上 2024 年的教程，会看到一堆 `from langgraph.prebuilt import
    create_react_agent`，现在跑起来会弹 DeprecationWarning。

    **面试时能主动说出这件事，是很有说服力的加分项** ——
    它说明你真的跟着版本在维护代码，而不是背了一篇博客。

    下面的写法叫「版本兼容垫片」：
    优先试新 API，没有就退回旧 API，两者都没有才报错。
    代价是要多写十几行，收益是别人换环境跑不起来的时候你还能跑。
    """
    # 首选：LangChain 1.x 之后的新入口
    try:
        from langchain.agents import create_agent

        return create_agent(model, tools)
    except ImportError:
        pass

    # 次选：旧版入口（会有 DeprecationWarning，但功能正常）
    try:
        from langgraph.prebuilt import create_react_agent

        return create_react_agent(model, tools)
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "没有找到可用的预构建 Agent 工厂。"
            "请确认已安装 langgraph 或 langchain。"
        ) from exc


def build_tool_graph():
    """返回一个「会自己挑工具用」的 Agent。"""
    llm = get_llm()
    return _create_prebuilt_agent(llm, ALL_TOOLS)


if __name__ == "__main__":
    agent = build_tool_graph()
    question = "接口返回 429 是什么意思？该怎么处理？"
    print(f"提问：{question}\n")

    result = agent.invoke({"messages": [HumanMessage(content=question)]})

    print("── 完整消息链路 ──")
    for msg in result["messages"]:
        kind = type(msg).__name__
        tool_calls = getattr(msg, "tool_calls", None)
        if tool_calls:
            names = [tc["name"] for tc in tool_calls]
            print(f"  [{kind}] 想调用工具：{names}")
        elif kind == "ToolMessage":
            preview = str(msg.content)[:80].replace("\n", " ")
            print(f"  [{kind}] 工具返回：{preview}...")
        else:
            content = str(msg.content)
            preview = content[:120].replace("\n", " ")
            print(f"  [{kind}] {preview}{'...' if len(content) > 120 else ''}")

    print("\n── 最终回答 ──")
    print(result["messages"][-1].content)
