"""
app.py —— DocPilot 的界面（Streamlit）

跑法：
    streamlit run app.py

【这个界面存在的意义】
面试时对方最想看的不是代码，是"这东西真的能跑吗、长什么样"。
这个界面把图的执行过程摊开给人看：
左边是对话，右边是"刚才系统到底走了哪几步、每步拿到了什么证据"。

**能看见中间过程**，是 LangGraph 类项目最值得展示的地方。
很多人的 Demo 只有一问一答，看不出和套壳 ChatGPT 有什么区别。

【为什么选 Streamlit？】
因为在"快速做出可演示界面"这件事上，它几乎没有对手：
不用写 HTML/CSS/JS，不用起前后端两个进程，一个脚本搞定。
把省下来的时间花在图的设计上，性价比更高。
"""

from __future__ import annotations

import sys
import uuid
from pathlib import Path

import streamlit as st
from langgraph.types import Command

# 让 streamlit run app.py 也能找到 src 下的包
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from docpilot.config import get_settings  # noqa: E402
from docpilot.graphs.main import build_graph, new_thread_config  # noqa: E402
from docpilot.state import initial_state  # noqa: E402
from docpilot.utils.viz import to_mermaid  # noqa: E402

st.set_page_config(page_title="DocPilot · LangGraph 实践", page_icon="🔎", layout="wide")


# ============================================================
# 资源缓存
# ============================================================
@st.cache_resource(show_spinner="正在编译图并建立索引……")
def get_graph():
    """
    @st.cache_resource 保证整个会话只构建一次图。

    这一点很关键：如果每次交互都重新 compile，
    InMemorySaver 里的历史存档就全丢了，多轮记忆直接失效。
    """
    return build_graph(checkpointer_kind="memory")


def init_session() -> None:
    """初始化会话状态。Streamlit 每次交互都会重跑整个脚本，所以状态必须存起来。"""
    defaults = {
        "thread_id": f"ui-{uuid.uuid4().hex[:8]}",
        "chat": [],            # [{"role": "user"/"assistant", "content": str}]
        "trace": [],
        "evidence": [],
        "interrupt": None,     # 待人工审核的载荷
        "running": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_thread() -> None:
    """开一个新会话（换一个 thread_id，等价于另开一个窗口）。"""
    st.session_state.thread_id = f"ui-{uuid.uuid4().hex[:8]}"
    st.session_state.chat = []
    st.session_state.trace = []
    st.session_state.evidence = []
    st.session_state.interrupt = None


# ============================================================
# 执行图
# ============================================================
def run_graph(payload, *, on_node=None):
    """
    跑一次图，把过程记录下来。

    用 stream_mode="updates" 而不是 invoke，好处是能**逐个节点**拿到进度：
    每走完一个节点就吐一块，界面上就能实时显示"现在走到哪一步了"。

    invoke 只能给你最终结果 —— 中间发生了什么完全看不到，
    调试时会很痛苦。
    """
    graph = get_graph()
    config = new_thread_config(st.session_state.thread_id)

    collected_trace: list[dict] = []
    for chunk in graph.stream(payload, config=config, stream_mode="updates"):
        for node_name, update in chunk.items():
            if on_node:
                on_node(node_name)
            if isinstance(update, dict) and update.get("trace"):
                collected_trace = update["trace"]

    st.session_state.trace = collected_trace

    # 读回完整状态
    snapshot = graph.get_state(config)
    st.session_state.evidence = snapshot.values.get("documents") or []

    # 判断有没有卡在人工审核上
    st.session_state.interrupt = None
    if snapshot.next:
        for task in snapshot.tasks:
            for item in task.interrupts or []:
                st.session_state.interrupt = item.value
                break

    return snapshot.values


# ============================================================
# 右侧面板
# ============================================================
_STATUS_ICON = {"ok": "✅", "retry": "🔁", "warn": "⚠️", "paused": "⏸️", "error": "❌"}


def render_trace() -> None:
    st.markdown("#### 🧭 执行轨迹")
    trace = st.session_state.trace
    if not trace:
        st.caption("还没有执行记录。提一个问题试试。")
        return

    for item in trace:
        icon = _STATUS_ICON.get(item.get("status", "ok"), "•")
        label = item.get("label", item.get("node", "?"))
        detail = item.get("detail", "")
        st.markdown(f"{icon} **{label}**  \n<small>{detail}</small>", unsafe_allow_html=True)


def render_evidence() -> None:
    st.markdown("#### 📚 召回的证据")
    evidence = st.session_state.evidence
    if not evidence:
        st.caption("本轮没有检索到证据。")
        return

    for i, doc in enumerate(evidence, start=1):
        score = doc.get("score", 0)
        with st.expander(f"[{i}] {doc.get('source', '未知来源')} · 相关度 {score}"):
            st.text(str(doc.get("content", ""))[:1200])


def render_graph_shape() -> None:
    with st.expander("🗺️ 图的形状（面试可以指着这里讲）"):
        graph = get_graph()
        try:
            graph.get_graph().print_ascii()
        except Exception:
            pass
        # Streamlit 把 stdout 吞掉了，这里直接画一个 ASCII 版本
        st.code(_ASCII_GRAPH, language="text")
        st.caption("完整 Mermaid 源码（可粘到支持 Mermaid 的编辑器里看）")
        st.code(to_mermaid(graph), language="text")


_ASCII_GRAPH = """\
        START
          |
          v
   +----------------+   direct    +----------+
   | route_question |------------>| generate |
   +--------+-------+             +----+-----+
            | retrieve                 ^
            v                          |
      +-----------+                    |
      | retrieve  |<-----------+       |
      +-----+-----+            |       |
            v                  |       |
   +-----------------+  low    |       |
   | grade_documents |---------+       |
   +--------+--------+  (rewrite_query)|
            | good                     |
            +--------------------------+
            |
            v
      +----------+     +--------------+     +----------+
      | generate |---->|   reflect    |---->| finalize |--> END
      +----------+     +------+-------+     +----^-----+
                              | need_human       |
                              v                  |
                       +--------------+          |
                       | human_review |----------+
                       +--------------+
"""


# ============================================================
# 页面主体
# ============================================================
def main() -> None:
    init_session()
    settings = get_settings()

    st.title("🔎 DocPilot")
    st.caption(
        "基于 **LangGraph** 的自纠正 RAG 文档助手 —— "
        "路由 → 检索 → 评分 → 重写 → 生成 → 反思 → 人工审核"
    )

    with st.sidebar:
        st.markdown("### ⚙️ 当前配置")
        st.write(settings.describe())
        st.write(f"检索器：`{settings.retriever}`　召回 `{settings.top_k}` 条")
        st.write(
            f"及格线：`{settings.score_threshold}`　最多重写 `{settings.max_rewrites}` 次"
        )
        st.divider()
        st.write(f"会话 ID：`{st.session_state.thread_id}`")
        st.caption("同一个会话 ID 下，系统会记住之前的对话。")

        if st.button("🆕 开一个新会话", use_container_width=True):
            reset_thread()
            st.rerun()

        st.divider()
        st.markdown("### 💡 试试这些问题")
        st.caption("直接复制到输入框即可")
        st.code(
            "连续输错密码 5 次会怎么样？\n"
            "账号被锁了多久才能再登\n"
            "接口返回 429 该怎么处理？\n"
            "客户反馈页面打不开怎么排查\n"
            "你好，你是谁？",
            language="text",
        )
        st.caption("第 2 条会触发查询重写，第 5 条会跳过检索。")

    col_chat, col_side = st.columns([1.6, 1], gap="large")

    with col_chat:
        st.markdown("#### 💬 对话")

        for message in st.session_state.chat:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # ── 人工审核环节 ─────────────────────────────
        if st.session_state.interrupt:
            with st.chat_message("assistant"):
                st.warning(st.session_state.interrupt)
                col_a, col_b = st.columns(2)
                approve = col_a.button("✅ 通过并发送", use_container_width=True, type="primary")
                reject = col_b.button("🚫 打回", use_container_width=True)

            if approve or reject:
                decision = "approve" if approve else "reject 内容依据不足，需要人工重写"
                st.session_state.chat.append(
                    {"role": "user", "content": f"（人工审核）{decision}"}
                )
                with st.spinner("正在恢复图执行……"):
                    run_graph(Command(resume=decision))
                snapshot_result = get_graph().get_state(
                    new_thread_config(st.session_state.thread_id)
                ).values
                final = snapshot_result.get("answer", "")
                st.session_state.chat.append({"role": "assistant", "content": final})
                st.rerun()

        # ── 提问输入框 ───────────────────────────────
        question = st.chat_input("问点什么，比如：连续输错密码 5 次会怎么样？")

        if question:
            st.session_state.chat.append({"role": "user", "content": question})
            st.session_state.interrupt = None

            with st.status("正在执行图……", expanded=True) as status:

                def on_node(node_name: str) -> None:
                    status.write(f"▸ 执行节点 `{node_name}`")

                result = run_graph(initial_state(question), on_node=on_node)
                status.update(label="执行完成", state="complete", expanded=False)

            if not st.session_state.interrupt:
                st.session_state.chat.append(
                    {"role": "assistant", "content": result.get("answer", "")}
                )

            st.rerun()

    with col_side:
        render_trace()
        st.divider()
        render_evidence()
        st.divider()
        render_graph_shape()

        st.divider()
        with st.expander("🕰️ 时间旅行：查看历史存档"):
            st.caption(
                "检查点会保存每一步的状态。用它可以把图倒回任意一步重跑 —— "
                "这就是 LangGraph Studio 里「回退到某一步」的原理。"
            )
            graph = get_graph()
            history = list(graph.get_state_history(new_thread_config(st.session_state.thread_id)))
            if not history:
                st.caption("暂无存档。")
            for snapshot in history[:10]:
                step = snapshot.metadata.get("step", "?")
                next_nodes = "、".join(snapshot.next) if snapshot.next else "（结束）"
                st.write(f"第 `{step}` 步 → 下一步：`{next_nodes}`")


if __name__ == "__main__":
    main()
