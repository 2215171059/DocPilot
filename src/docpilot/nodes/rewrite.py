"""
rewrite.py —— 节点 ④：查询重写（自纠正循环的"补救动作"）

职责：
证据评分不达标，说明"检索用的那句话"没问对。
于是让模型把问题改写成更适合检索的表述，然后拿新 query 再去检索一次。

为什么重写真的有用？
用户说话的方式和文档写作的方式往往对不上：
  用户：这个功能咋用的来着？
  文档：功能配置与使用方法说明
字面上一个字都不重合，关键词检索自然一条都召不回来。
重写就是在这道鸿沟上架桥。

常用的改写策略有这么几种：
  1. 去噪：删掉"请问""一下"这类没有信息量的词
  2. 同义扩展：把口语换成文档里的正式说法（打不开 → 无法加载）
  3. 拆解：一个复杂问题拆成几个子查询，分别检索
  4. 假设性答案（HyDE）：先让模型编一段"可能的答案"，用答案去检索
  5. 多查询（Multi-Query）：生成几个不同角度的问法，多路召回后合并
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from docpilot.config import get_llm, get_settings
from docpilot.prompts import REWRITER_HUMAN, REWRITER_SYSTEM
from docpilot.state import DocPilotState
from docpilot.utils.tracing import trace_event, with_trace


def rewrite_query(state: DocPilotState) -> dict:
    """改写查询语句，并把重写计数器 +1。"""
    settings = get_settings()
    llm = get_llm()

    question = state.get("original_question") or state.get("question", "")
    reason = state.get("grade_reason", "")

    response = llm.invoke(
        [
            SystemMessage(content=REWRITER_SYSTEM),
            HumanMessage(
                content=REWRITER_HUMAN.format(question=question, reason=reason)
            ),
        ]
    )

    # 模型有时会给 query 套上引号，去掉
    new_query = str(response.content).strip().strip("\"'“”‘’")
    if not new_query:
        new_query = question  # 兜底：改写失败就用原问题

    count = int(state.get("rewrite_count", 0)) + 1

    # 轨迹里只留前 80 个字，避免模型啰嗦时把界面刷爆
    preview = new_query if len(new_query) <= 80 else new_query[:80] + "…"

    return {
        # question 会被覆盖（没有 reducer），下一次 retrieve 就用新 query 了
        "question": new_query,
        "rewrite_count": count,
        "trace": with_trace(
            state,
            trace_event(
                "rewrite_query",
                detail=f"第 {count}/{settings.max_rewrites} 次重写：{preview}",
                status="retry",
                attempt=count,
                new_query=preview,
            ),
        ),
    }
