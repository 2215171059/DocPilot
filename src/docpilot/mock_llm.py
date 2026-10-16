"""
mock_llm.py —— 离线模拟模型（不需要 API Key、不需要网络）

【为什么要有这个文件？】
面试现场最怕什么？网络抽风、Key 突然失效。
有了它，只要把 .env 里的 DOCPILOT_PROVIDER 改成 mock，
整条链路（路由 → 检索 → 评分 → 重写 → 生成 → 反思 → 人工审核）照样跑得通，
演示永远不会翻车。

【它是怎么"假装"成大模型的？】
它继承 LangChain 的 BaseChatModel，实现了最核心的 _generate 方法。
不同节点的提示词里带了不同的角色标记（【角色:评分】等），
它就读这个标记来决定自己该扮演谁、该返回什么格式。

注意：它是**规则驱动**的，不是真的在"理解"。
比如评分是用"问题关键词和资料的重合度"算出来的——这是个粗糙但可解释的近似。
真实场景请一定用真模型。
"""

from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

from docpilot import prompts as P

# ============================================================
# 辅助工具
# ============================================================

# 匹配「整行只有一个【xxx】」的分节标题
_SECTION_RE = re.compile(r"^【(.+?)】\s*$", re.MULTILINE)

# 口语噪声词：重写查询时把它们删掉，剩下的就是检索关键词
_QUERY_NOISE = (
    "请问", "帮我", "麻烦", "我想知道", "我想问", "一下", "怎么", "怎样", "如何",
    "是什么", "是啥", "哪些", "哪个", "为什么", "能不能", "可不可以", "可以",
    "来着", "这个", "那个", "吗", "呢", "啊", "吧", "的", "了",
)

# 同义词扩展表：口语 → 文档里的正式说法
# 这是真实工程里常用的查询改写手段之一（query expansion），
# 目的是弥合"用户怎么说话"和"文档怎么写"之间的用词鸿沟。
_SYNONYMS: dict[str, str] = {
    "登不上": "登录失败",
    "登不了": "登录失败",
    "进不去": "无法登录",
    "打不开": "无法加载",
    "上不去": "无法访问",
    "白屏": "页面空白",
    "锁": "锁定",
    "报错": "错误码",
    "卡顿": "响应时间",
    "很慢": "响应时间",
    "限流": "限流 429",
}

# 出现这些词就当作闲聊，走 direct 分支不去检索
_GREETINGS = (
    "你好", "您好", "hello", "hi", "hey", "在吗", "谢谢", "谢谢你",
    "你是谁", "介绍一下你自己", "哈喽", "早上好", "晚上好",
)

# 需要从文本里剔除的标点与空白。
# 注意这里用普通字符串 + 集合判断，而不是写正则 —— 
# 因为中文全角标点和引号混在一起时，正则里的转义非常容易写错。
_PUNCTUATION = set(
    " \t\r\n"
    "，。？！、；：·…—～"
    "\"'“”‘’（）〔〕【】《》〈〉「」『』"
    ",.?!;:()[]{}<>-_/\\|"
)


def _strip_punctuation(text: str) -> str:
    """去掉标点和空白，只留内容字符。"""
    return "".join(ch for ch in text if ch not in _PUNCTUATION)

# 答案里出现这些词就认为是"敏感内容"，需要人工确认
_SENSITIVE_WORDS = ("建议", "必须", "应该", "风险", "合同", "金额", "期限", "务必", "推荐")


def _split_sections(text: str) -> dict[str, str]:
    """
    把提示词里的【小标题】切成字典。

    输入：
        【问题】
        登录失败怎么办

        【参考资料】
        [1] 来源：xxx.md
    输出：
        {"问题": "登录失败怎么办", "参考资料": "[1] 来源：xxx.md"}
    """
    parts = _SECTION_RE.split(text)
    result: dict[str, str] = {}
    for i in range(1, len(parts) - 1, 2):
        result[parts[i].strip()] = parts[i + 1].strip()
    return result


def _relevance(question: str, evidence: str) -> float:
    """
    粗略估计"证据和问题的相关度"，取值 0~1。

    做法：把问题切成相邻两字的片段（bigram），看有多少片段在证据里出现过。
    中文双字片段是很朴素的"伪分词"，但对关键词检索够用了。
    """
    cleaned = _strip_punctuation(question)
    if not cleaned:
        return 0.0
    grams = {cleaned[i : i + 2] for i in range(len(cleaned) - 1)} or {cleaned}
    hits = sum(1 for gram in grams if gram in evidence)
    return hits / len(grams)


def _needs_retrieval(question: str) -> bool:
    """判断一句话是不是闲聊。只有明确是闲聊才走 direct 分支。"""
    q = (question or "").strip().lower()
    if not q:
        return False
    # 完全等于寒暄语，或者是「很短的」寒暄语，才当作闲聊
    if q in _GREETINGS:
        return False
    if len(q) <= 10 and any(word in q for word in _GREETINGS):
        return False
    return True


# ============================================================
# 模拟模型本体
# ============================================================
class MockChatModel(BaseChatModel):
    """离线模拟模型。接口与真实模型完全一致，可以直接替换。"""

    bound_tool_names: list[str] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "docpilot-mock"

    # --------------------------------------------------------
    # bind_tools：真实模型叫它"我给你几个工具，你按需调用"
    # 这里我们只把工具名字记下来，用于后面判断能不能发起工具调用。
    # --------------------------------------------------------
    def bind_tools(self, tools, **kwargs):  # type: ignore[override]
        names = [getattr(t, "name", str(t)) for t in tools]
        return self.model_copy(update={"bound_tool_names": names})

    # --------------------------------------------------------
    # _generate：LangChain 规定子类要实现的核心方法
    # --------------------------------------------------------
    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        message = self._reply(messages)
        return ChatResult(generations=[ChatGeneration(message=message)])

    # --------------------------------------------------------
    # 分派：根据系统提示词里的角色标记决定演谁
    # --------------------------------------------------------
    def _reply(self, messages: list[BaseMessage]) -> AIMessage:
        system_text = "\n".join(
            str(m.content) for m in messages if isinstance(m, SystemMessage)
        )
        human_text = "\n".join(
            str(m.content) for m in messages if isinstance(m, HumanMessage)
        )
        sections = _split_sections(human_text)

        if P.TAG_GRADER in system_text:
            return AIMessage(content=self._act_grader(sections))
        if P.TAG_REWRITER in system_text:
            return AIMessage(content=self._act_rewriter(sections))
        if P.TAG_REFLECTOR in system_text:
            return AIMessage(content=self._act_reflector(sections))
        if P.TAG_ROUTER in system_text:
            return AIMessage(content=self._act_router(sections))
        return self._act_answer(messages, sections)

    # --------------------------------------------------------
    # 角色 1：路由
    # --------------------------------------------------------
    def _act_router(self, sections: dict[str, str]) -> str:
        question = sections.get("用户输入", "")
        return "retrieve" if _needs_retrieval(question) else "direct"

    # --------------------------------------------------------
    # 角色 2：证据评分
    # --------------------------------------------------------
    def _act_grader(self, sections: dict[str, str]) -> str:
        question = sections.get("问题", "")
        documents = sections.get("参考资料", "")

        if not documents or "没有检索到" in documents:
            payload = {"score": 0.0, "reason": "没有检索到任何资料"}
        else:
            relevance = _relevance(question, documents)
            # 系数 23 是拍出来的经验值，目的是让"口语化提问"落在及格线以下、
            # "关键词对齐的提问"落在及格线以上，从而能演示出自我纠正的过程。
            # 真实场景请用真模型来判断相关性，这个系数没有任何普适意义。
            score = min(10.0, round(relevance * 23, 1))
            if score >= 8:
                reason = "关键词高度重合，证据可以直接支撑回答"
            elif score >= 6.5:
                reason = "部分关键词命中，信息基本够用"
            else:
                reason = "关键词重合度偏低，证据与问题关联不大"
            payload = {"score": score, "reason": reason}

        return json.dumps(payload, ensure_ascii=False)

    # --------------------------------------------------------
    # 角色 3：查询重写
    # --------------------------------------------------------
    def _act_rewriter(self, sections: dict[str, str]) -> str:
        question = sections.get("原问题", "")
        cleaned = question

        # 第一步：同义词扩展。注意要避免把一个词扩成它自己的重复
        # （比如"锁定"里已经含"锁"，就不该再替换）
        for oral, formal in _SYNONYMS.items():
            if oral in cleaned and formal not in cleaned:
                cleaned = cleaned.replace(oral, formal)

        # 第二步：删掉口语噪声词
        for word in _QUERY_NOISE:
            cleaned = cleaned.replace(word, " ")

        # 第三步：清理多余标点和空白。
        # 注意要**逐个字符顺序替换**，不能写成
        #   " ".join(cleaned.replace(ch, " ") for ch in _PUNCTUATION)
        # 那种写法每次都是在原始字符串上替换，会把原串拼几十遍。
        for ch in _PUNCTUATION:
            cleaned = cleaned.replace(ch, " ")
        cleaned = " ".join(cleaned.split())

        # 万一全被删光了，就退回原问题，保证不会输出空字符串
        return cleaned or question

    # --------------------------------------------------------
    # 角色 4：反思 / 自查
    # --------------------------------------------------------
    def _act_reflector(self, sections: dict[str, str]) -> str:
        answer = sections.get("待审核的答案", "")
        documents = sections.get("参考资料", "")
        has_docs = bool(documents) and "没有检索到" not in documents

        unsupported: list[str] = []
        grounded = has_docs or "无法回答" in answer or "没有检索到" in answer
        if not grounded:
            unsupported.append("答案中的结论在参考资料里找不到依据")

        hit_words = [w for w in _SENSITIVE_WORDS if w in answer]
        # 只有"基于资料"的回答才需要人工确认；
        # 闲聊类回答不打扰用户。
        need_human = bool(hit_words) and has_docs

        payload = {
            "grounded": grounded,
            "unsupported_claims": unsupported,
            "need_human": need_human,
        }
        return json.dumps(payload, ensure_ascii=False)

    # --------------------------------------------------------
    # 角色 5：生成答案（支持发起工具调用）
    # --------------------------------------------------------
    def _act_answer(self, messages: list[BaseMessage], sections: dict[str, str]) -> AIMessage:
        question = sections.get("问题", "")
        if not question:
            human_msgs = [m for m in messages if isinstance(m, HumanMessage)]
            question = str(human_msgs[-1].content) if human_msgs else ""

        # ── 优先看提示词里有没有参考资料 ──
        # 主图里的检索是一个"节点"，结果直接写进 state 再拼进提示词，
        # 不会经过工具调用，所以这里要从【参考资料】小节里读。
        documents = sections.get("参考资料", "")
        has_docs = bool(documents) and "没有检索到" not in documents

        if has_docs:
            return AIMessage(content=self._compose_from_documents(documents))

        tool_messages = [m for m in messages if isinstance(m, ToolMessage)]

        # ── 手上有工具、还没调用过、问题确实需要查资料 → 发起工具调用 ──
        if self.bound_tool_names and not tool_messages and _needs_retrieval(question):
            tool_name = self._pick_tool()
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": tool_name,
                        "args": {"query": question},
                        "id": "call_mock_0001",
                        "type": "tool_call",
                    }
                ],
            )

        # ── 工具返回过结果 → 基于结果作答 ──
        if tool_messages:
            return AIMessage(content=self._compose_from_documents(str(tool_messages[-1].content)))

        # ── 都没有：闲聊或直答 ──
        return AIMessage(
            content=(
                "你好，我是 DocPilot 文档助手。\n\n"
                "你可以问我关于产品功能、接口用法、操作流程、故障排查、账号权限的问题。"
                "我会先在知识库里查资料，再基于资料回答，并标注引用来源。"
            )
        )

    def _compose_from_documents(self, documents: str) -> str:
        """把检索到的资料拼成一段带引用的回答。"""
        first_block = documents.split("\n\n")[0].strip()
        body = first_block if len(first_block) <= 320 else first_block[:320] + "……"
        return (
            "根据知识库中的资料，整理如下：\n\n"
            f"{body}\n\n"
            "以上内容来自上述资料 [1]。建议以文档的最新版本为准，"
            "如与实际系统行为不一致，请以线上环境为准。"
        )

    def _pick_tool(self) -> str:
        """优先挑知识库检索工具；没有就退而用第一个。"""
        for name in self.bound_tool_names:
            if "knowledge" in name or "search" in name or "retrieve" in name:
                return name
        return self.bound_tool_names[0]
