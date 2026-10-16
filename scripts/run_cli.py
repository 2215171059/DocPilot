"""
run_cli.py —— 命令行版 DocPilot（不依赖界面也能完整演示）

跑法：
    python scripts/run_cli.py

它演示了三件事，正好对应主图最关键的三个能力：
  1. 多轮记忆 —— 同一个 thread_id 连续提问，它能记得上一轮聊了什么
  2. 人工审核 —— 遇到敏感内容会暂停，等你在终端里输入 approve / reject
  3. 时间旅行 —— 输入 :history 可以看到这条会话的所有历史存档

面试现场如果对方说"能不能演示一下"，这个脚本是最稳的选择：
不用打开浏览器，一行命令，输出全在终端里。
"""

from __future__ import annotations

import sys
import uuid
from pathlib import Path

# 把 src 加进模块搜索路径，这样直接 python scripts/run_cli.py 也能跑
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from langgraph.types import Command  # noqa: E402

from docpilot.config import get_settings  # noqa: E402
from docpilot.graphs.main import build_graph, new_thread_config  # noqa: E402
from docpilot.state import initial_state  # noqa: E402
from docpilot.utils.render import print_result  # noqa: E402
from docpilot.utils.viz import print_graph_ascii  # noqa: E402

HELP_TEXT = """
可用命令：
  :help      显示这份帮助
  :history   查看本会话的历史存档（时间旅行）
  :state     查看当前状态
  :quit      退出

直接输入问题即可提问。如果触发了人工审核，
程序会停下来问你要 approve（通过）还是 reject（打回）。
"""


def _handle_interrupt(graph, config, state) -> dict:
    """
    处理人工审核中断。

    LangGraph 在 interrupt() 处暂停时，返回结果的 __interrupt__ 字段里
    会带上等待人工决策的载荷。我们把它打印出来，
    等用户输入后用 Command(resume=...) 让图继续跑。
    """
    while True:
        interrupts = state.get("__interrupt__")
        if not interrupts:
            return state

        payload = interrupts[0].value
        print("\n" + "─" * 66)
        print(payload)
        print("─" * 66)

        decision = input("你的决定（approve 通过 / reject 打回）：").strip() or "approve"

        # 关键：Command(resume=值) 的"值"会变成 interrupt() 的返回值
        state = graph.invoke(Command(resume=decision), config=config)


def main() -> None:
    settings = get_settings()
    graph = build_graph(checkpointer_kind="memory")

    thread_id = f"cli-{uuid.uuid4().hex[:8]}"
    config = new_thread_config(thread_id)

    print("=" * 66)
    print("  DocPilot —— 基于 LangGraph 的自纠正 RAG 文档助手")
    print("=" * 66)
    print(f"  {settings.describe()}")
    print(f"  检索器 = {settings.retriever}    召回 {settings.top_k} 条")
    print(f"  及格线 = {settings.score_threshold}    最多重写 {settings.max_rewrites} 次")
    print(f"  会话 ID = {thread_id}")
    print(HELP_TEXT)

    print("图的形状：")
    print_graph_ascii(graph)
    print()

    while True:
        try:
            question = input("你：").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见。")
            break

        if not question:
            continue

        if question in {":quit", ":q", "exit", "quit"}:
            print("再见。")
            break

        if question == ":help":
            print(HELP_TEXT)
            continue

        if question == ":history":
            print("\n── 历史存档（每次节点执行都会存一份）──")
            for snapshot in graph.get_state_history(config):
                step = snapshot.metadata.get("step", "?")
                next_nodes = snapshot.next or ("(结束)",)
                preview = (snapshot.values.get("answer") or snapshot.values.get("question") or "")[:36]
                print(f"  第 {step} 步 → 下一步：{next_nodes}    状态摘要：{preview}")
            print()
            continue

        if question == ":state":
            snapshot = graph.get_state(config)
            print("\n── 当前状态字段 ──")
            for key, value in snapshot.values.items():
                text = str(value)
                print(f"  {key}: {text[:90]}{'…' if len(text) > 90 else ''}")
            print()
            continue

        print()
        state = graph.invoke(initial_state(question), config=config)
        state = _handle_interrupt(graph, config, state)
        print_result(state)


if __name__ == "__main__":
    main()
