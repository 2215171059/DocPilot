"""
reflect.py —— 节点 ⑥：反思自查（幻觉检查 + 敏感度判断）

职责：
答案刚生成，先别急着发出去。让一个"审稿人"再看一遍：
  1. 答案里的每一句话，在证据里找得到依据吗？（grounded）
  2. 有没有哪些句子是凭空编的？（unsupported_claims）
  3. 这个答案需不需要人来拍板？（need_human）

为什么不直接让生成节点"注意别编"？
因为同一个模型、同一段上下文，"自己检查自己"效果有限。
换一个**独立的调用**、给一套**不同的提示词**、让它的任务是挑错，
准确率会明显高一截。这是"生成与校验分离"的基本思路 ——
和编译器的道理一样，别让写代码的人自己批改卷子。

need_human 的判定逻辑：
只要 grounded 为 false（答案有编造成分），就强制人工确认。
另外答案里出现"建议""必须""金额""期限"这类表述时也要人工确认 ——
这类内容一旦答错，后果比"答不上来"严重得多。
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from docpilot.config import get_llm
from docpilot.prompts import REFLECTOR_HUMAN, REFLECTOR_SYSTEM, format_documents
from docpilot.state import DocPilotState
from docpilot.utils.jsonx import parse_json_loose
from docpilot.utils.tracing import trace_event, with_trace


def reflect(state: DocPilotState) -> dict:
    """检查答案是否有据可依，并决定是否需要人工确认。"""
    llm = get_llm()

    question = state.get("original_question") or state.get("question", "")
    documents = state.get("documents") or []
    answer = state.get("answer", "")

    response = llm.invoke(
        [
            SystemMessage(content=REFLECTOR_SYSTEM),
            HumanMessage(
                content=REFLECTOR_HUMAN.format(
                    question=question,
                    documents=format_documents(documents),
                    answer=answer,
                )
            ),
        ]
    )

    payload = parse_json_loose(response.content)
    grounded = bool(payload.get("grounded", True))
    unsupported = payload.get("unsupported_claims") or []
    if not isinstance(unsupported, list):
        unsupported = [str(unsupported)]

    need_human = bool(payload.get("need_human", False))

    # 保险丝：只要发现有编造内容，一律人工确认，不给模型"自我豁免"的机会
    if not grounded:
        need_human = True

    reflection = {
        "grounded": grounded,
        "unsupported_claims": [str(item) for item in unsupported],
        "need_human": need_human,
    }

    if not grounded:
        detail = f"发现 {len(unsupported)} 处无依据表述，转人工确认"
        status = "warn"
    elif need_human:
        detail = "内容敏感（含建议/结论性判断），转人工确认"
        status = "warn"
    else:
        detail = "自查通过，答案有据可依"
        status = "ok"

    return {
        "reflection": reflection,
        "need_human": need_human,
        "trace": with_trace(
            state, trace_event("reflect", detail=detail, status=status, **reflection)
        ),
    }
