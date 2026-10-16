"""
generate.py —— 节点 ⑤：生成答案

【它在干什么】
拿着【原始问题】+【检索到的证据】去让模型作答，强制要求标注引用来源。

【两个刻意的设计】
1. 用的是 original_question，不是被重写过的 question。
   重写只是为了"更好地检索"，不是为了"换个问题回答"。
   用错会导致答案跑偏 —— 这是个很隐蔽的坑。

2. 要求每条结论都标 [1] [2] 这样的来源编号。
   这是把"可验证性"做进产品里：用户能自己点回去看原文，
   而不是只能选择相信模型。企业场景里这一点几乎是硬要求。

【面试考点】
"怎么缓解大模型的幻觉？"
从这四层答，层次感就出来了：
  1. 输入端：给足上下文（RAG），提示词明确"只能用给定资料"
  2. 生成端：要求标注引用、温度调低、必要时限制输出格式
  3. 输出端：加一个独立的反思节点做事实性核查（本项目 S5）
  4. 兜底：拿不准就明确说"不知道"，或者转人工（本项目 S6 的 HITL）
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from docpilot.config import get_llm
from docpilot.prompts import (
    ANSWER_HUMAN,
    ANSWER_HUMAN_NO_DOCS,
    ANSWER_SYSTEM,
    format_documents,
)
from docpilot.state import DocPilotState
from docpilot.utils.tracing import trace_event, with_trace


def generate(state: DocPilotState) -> dict:
    """基于证据生成答案。"""
    llm = get_llm()

    question = state.get("original_question") or state.get("question", "")
    documents = state.get("documents") or []

    if documents:
        human_content = ANSWER_HUMAN.format(
            question=question, documents=format_documents(documents)
        )
    else:
        human_content = ANSWER_HUMAN_NO_DOCS.format(question=question)

    response = llm.invoke(
        [
            SystemMessage(content=ANSWER_SYSTEM),
            HumanMessage(content=human_content),
        ]
    )

    answer = str(response.content).strip()

    return {
        "answer": answer,
        # 同时往 messages 里追加一条 AI 消息。
        # 因为 messages 的 reducer 是 add_messages（追加语义），
        # 这样历史对话才会一轮轮累积起来 —— 这就是"多轮记忆"的原材料。
        "messages": [AIMessage(content=answer)],
        "trace": with_trace(
            state,
            trace_event(
                "generate",
                detail=f"基于 {len(documents)} 条证据生成答案（{len(answer)} 字）",
                evidence_count=len(documents),
            ),
        ),
    }
