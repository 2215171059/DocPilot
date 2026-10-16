"""
render.py —— 终端里的漂亮输出

命令行演示时，光有数据不够，还要看得清楚。
这里提供几个打印辅助函数，让终端输出也有"界面感"。
"""

from __future__ import annotations

from typing import Any

from docpilot.utils.tracing import render_trace_text

_STATUS_ICON = {
    "ok": "✓",
    "retry": "↻",
    "warn": "!",
    "paused": "⏸",
    "error": "✗",
}


def print_trace(trace: list[dict[str, Any]]) -> None:
    """打印执行轨迹（思考链路）。"""
    print("── 执行轨迹 ──")
    print(render_trace_text(trace))


def print_answer(answer: str, *, title: str = "回答") -> None:
    """打印最终答案，带一个简单的边框。"""
    print(f"── {title} ──")
    for line in (answer or "").splitlines():
        print(f"  {line}")
    print()


def print_result(result: dict[str, Any]) -> None:
    """把一次运行的完整结果打出来：路由、证据、评分、轨迹、答案。"""
    print(f"路由结果：{result.get('route', '-')}")

    documents = result.get("documents") or []
    print(f"命中证据：{len(documents)} 条")
    for i, doc in enumerate(documents, start=1):
        preview = str(doc.get("content", ""))[:50].replace("\n", " ")
        print(f"  [{i}] {doc.get('source')}  {preview}...")

    score = result.get("retrieval_score")
    if score is not None:
        print(f"证据评分：{score}（{result.get('grade_reason', '')}）")
    print(f"重写次数：{result.get('rewrite_count', 0)}")

    if result.get("reflection"):
        reflection = result["reflection"]
        grounded = "是" if reflection.get("grounded") else "否"
        need_human = "是" if reflection.get("need_human") else "否"
        print(f"有据可依：{grounded}    需人工确认：{need_human}")

    if result.get("review_decision"):
        print(f"人工审核：{result['review_decision']}")

    print()
    print_trace(result.get("trace") or [])
    print()
    print_answer(result.get("answer", ""))
