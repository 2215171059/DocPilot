"""
grade.py —— 节点 ③：证据评分（自纠正循环的"裁判"）

职责：
让模型给刚检索到的证据打分：这些资料到底能不能回答问题？
分数会决定下一步走向 —— 够好就生成答案，不够好就去重写查询。

最有价值的设计点：
朴素 RAG 的致命问题：**检索到什么就用什么**。
就算检索回来一堆毫不相干的段落，模型也会硬着头皮编一段答案出来。

加了这个"裁判"之后，系统第一次有了"自知之明"：
我能判断我手上的材料够不够用，不够就换个姿势再找。
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from docpilot.config import get_llm, get_settings
from docpilot.prompts import GRADER_HUMAN, GRADER_SYSTEM, format_documents
from docpilot.state import DocPilotState
from docpilot.utils.jsonx import parse_json_loose, safe_float
from docpilot.utils.tracing import trace_event, with_trace


def grade_documents(state: DocPilotState) -> dict:
    """给检索到的证据打分，写入 retrieval_score 与 grade_reason。"""
    settings = get_settings()
    documents = state.get("documents") or []

    # 用「当前 query」打分，而不是最初的原问题。
    #
    # 原因：每一轮重写之后，question 已经变成新的检索意图。
    # 评分节点要回答的问题是"这一轮找到的资料，够不够支撑这一轮的检索意图"，
    # 只有这样才能反映"重写到底有没有起作用"。
    #
    # 注意别搞混：最终**生成答案**时用的永远是 original_question（见 generate.py），
    # 因为用户想知道的是他最初问的那件事，不是我们改写后的检索词。
    question = state.get("question") or state.get("original_question", "")

    # 一条都没召回到，不用浪费一次模型调用，直接给 0 分
    if not documents:
        return {
            "retrieval_score": 0.0,
            "grade_reason": "没有检索到任何证据",
            "trace": with_trace(
                state,
                trace_event(
                    "grade_documents",
                    detail="证据为空，直接判 0 分",
                    status="warn",
                    score=0.0,
                ),
            ),
        }

    llm = get_llm()
    response = llm.invoke(
        [
            SystemMessage(content=GRADER_SYSTEM),
            HumanMessage(
                content=GRADER_HUMAN.format(
                    question=question, documents=format_documents(documents)
                )
            ),
        ]
    )

    payload = parse_json_loose(response.content)
    score = safe_float(payload.get("score"), 0.0)
    reason = str(payload.get("reason", "")).strip() or "模型未给出理由"

    passed = score >= settings.score_threshold
    used = int(state.get("rewrite_count", 0))
    left = settings.max_rewrites - used

    if passed:
        detail = f"{score} 分 ≥ 及格线 {settings.score_threshold}，证据可用"
        status = "ok"
    elif left > 0:
        detail = f"{score} 分 < 及格线 {settings.score_threshold}，还剩 {left} 次重写机会"
        status = "warn"
    else:
        detail = f"{score} 分 < 及格线，且已用尽重写次数，带着现有证据生成"
        status = "warn"

    return {
        "retrieval_score": score,
        "grade_reason": reason,
        "trace": with_trace(
            state,
            trace_event(
                "grade_documents",
                detail=detail,
                status=status,
                score=score,
                reason=reason,
                passed=passed,
            ),
        ),
    }
