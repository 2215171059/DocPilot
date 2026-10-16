"""
test_graph.py —— 关键路径测试

【为什么这个文件对面试很重要？】
面试官问"你的项目测过吗"，如果你只能说"我手动点了点"，说服力很弱。
这个文件用最少的代码覆盖了最关键的几条路径：

  1. 检索器能召回正确文档
  2. 图能完整跑通并产出答案
  3. 自纠正循环真的会在低分时触发重写
  4. 人工审核真的会中断，且能正确恢复
  5. 多轮对话真的记住了历史

跑法：
    pytest tests/ -v
"""

from __future__ import annotations

import pytest
from langgraph.types import Command

from docpilot.graphs.main import build_graph, new_thread_config
from docpilot.graphs.s5_crag import build_crag_graph
from docpilot.retrievers.bm25 import BM25Retriever, tokenize
from docpilot.retrievers.factory import build_retriever
from docpilot.state import initial_state
from docpilot.utils.jsonx import parse_json_loose

# ============================================================
# 检索层
# ============================================================


def test_tokenize_handles_chinese_and_english():
    """中英混合文本要能正确切成 token。"""
    tokens = tokenize("限流 RATE_LIMITED 429")
    assert "rate_limited" in tokens
    assert "429" in tokens
    assert any(len(t) == 2 and "\u4e00" <= t[0] <= "\u9fff" for t in tokens)


def test_retriever_returns_relevant_document():
    """问密码锁定，应该召回登录相关的那篇文档。"""
    retriever = build_retriever()
    assert len(retriever) > 0, "知识库为空，请检查 data/docs 目录"

    hits = retriever.retrieve("连续输错密码 锁定", top_k=3)
    assert hits, "检索没有命中任何内容"
    assert any("登录" in hit["source"] for hit in hits)
    # 分数应当从高到低排序
    scores = [hit["score"] for hit in hits]
    assert scores == sorted(scores, reverse=True)


def test_empty_retriever_is_safe():
    """空知识库不应该崩，应该安静地返回空列表。"""
    assert BM25Retriever([]).retrieve("随便问点什么") == []


# ============================================================
# 自纠正循环
# ============================================================


def test_self_correction_triggers_rewrite():
    """
    口语化提问第一次检索分数偏低，应当触发重写，并且重写后通过。

    这条测试是在验证整个项目最核心的设计，别删。
    """
    graph = build_crag_graph()
    result = graph.invoke(initial_state("账号被锁了多久才能再登"))

    assert result["route"] == "retrieve"
    assert result["rewrite_count"] >= 1, "低分场景没有触发查询重写"
    assert result["retrieval_score"] >= 6.5, "重写之后仍然没有达到及格线"


def test_small_talk_skips_retrieval():
    """闲聊不应该走检索分支。"""
    graph = build_crag_graph()
    result = graph.invoke(initial_state("你好"))
    assert result["route"] == "direct"
    assert not result["documents"]


# ============================================================
# 完整主图：记忆 + 人机协同
# ============================================================


def test_full_graph_produces_answer():
    graph = build_graph(checkpointer_kind="memory")
    config = new_thread_config("t-full")
    result = graph.invoke(initial_state("连续输错密码 5 次会怎么样？"), config=config)

    assert result["answer"]
    assert result["trace"], "没有采集到执行轨迹"


def test_human_review_interrupts_and_resumes():
    """敏感内容应当触发 interrupt，并且能用 Command(resume=) 恢复。"""
    graph = build_graph(checkpointer_kind="memory")
    config = new_thread_config("t-hitl")

    state = graph.invoke(initial_state("连续输错密码 5 次会怎么样？"), config=config)
    assert state.get("__interrupt__"), "没有按预期中断，人工审核环节可能被跳过了"

    resumed = graph.invoke(Command(resume="approve"), config=config)
    assert resumed["review_decision"] == "approve"

    # 再来一次，这次打回
    config2 = new_thread_config("t-hitl-reject")
    graph.invoke(initial_state("连续输错密码 5 次会怎么样？"), config=config2)
    rejected = graph.invoke(Command(resume="reject 理由不合适"), config=config2)
    assert rejected["review_decision"] == "reject"
    assert "未对外发送" in rejected["answer"]


def test_multi_turn_memory_accumulates_messages():
    """同一个 thread_id 连问两轮，消息历史应当累积，而不是被覆盖。"""
    from langchain_core.messages import HumanMessage

    graph = build_graph(checkpointer_kind="memory")
    config = new_thread_config("t-memory")

    state = graph.invoke(initial_state("连续输错密码 5 次会怎么样？"), config=config)
    if state.get("__interrupt__"):
        graph.invoke(Command(resume="approve"), config=config)

    # 第二轮：换一个问题，同一个 thread_id
    state2 = graph.invoke(initial_state("那锁定期间能扫码登录吗？"), config=config)
    if state2.get("__interrupt__"):
        state2 = graph.invoke(Command(resume="approve"), config=config)

    snapshot = graph.get_state(config)
    human_messages = [m for m in snapshot.values["messages"] if isinstance(m, HumanMessage)]

    assert len(human_messages) == 2, (
        f"历史消息没有累积，期望 2 条用户消息，实际 {len(human_messages)} 条 —— "
        "检查 messages 字段的 reducer 是否配置正确"
    )


def test_different_threads_are_isolated():
    """不同 thread_id 之间状态必须隔离，不能串味。"""
    graph = build_graph(checkpointer_kind="memory")

    graph.invoke(initial_state("连续输错密码 5 次会怎么样？"), config=new_thread_config("t-a"))
    graph.invoke(initial_state("你好"), config=new_thread_config("t-b"))

    snap_a = graph.get_state(new_thread_config("t-a"))
    snap_b = graph.get_state(new_thread_config("t-b"))

    assert len(snap_a.values["messages"]) != len(snap_b.values["messages"]) or (
        snap_a.values["question"] != snap_b.values["question"]
    )


# ============================================================
# 工具函数
# ============================================================


@pytest.mark.parametrize(
    "raw,expected_score",
    [
        ('{"score": 7.5, "reason": "还行"}', 7.5),
        ('```json\n{"score": 8}\n```', 8.0),
        ('好的，结果是：{"score": 5} 请查收', 5.0),
        ("完全不是 JSON", None),
    ],
)
def test_parse_json_loose(raw, expected_score):
    payload = parse_json_loose(raw)
    if expected_score is None:
        assert payload == {}
    else:
        assert payload.get("score") == expected_score
