"""
config.py —— 全局配置 + 「模型工厂」

这个文件在图中的角色：
整张图里所有节点都不自己 new 模型，而是统一调用 get_llm() 拿。
好处：想换模型（DeepSeek → 通义千问 → 本地 Ollama）只改 .env，业务代码一行都不用动。

这叫「配置与逻辑分离」，也是依赖注入的雏形：
模型从哪里来、用什么实现，全部由外部配置决定，调用方只依赖 get_llm() 这个抽象。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

# ── 项目根目录：src/docpilot/config.py → 上溯三层 = makeLangGrap/ ──
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# 读取 .env（文件不存在也不会报错，静默跳过）
load_dotenv(PROJECT_ROOT / ".env")


# ============================================================
# 供应商预设表
# ------------------------------------------------------------
# 这几家的 API 都兼容 OpenAI 协议，所以可以共用同一个客户端类，
# 只是 base_url（接口地址）和模型名不同。
# ============================================================
PROVIDER_PRESETS: dict[str, dict[str, str]] = {
    "deepseek": {
        "label": "DeepSeek",
        "base_url": "https://api.deepseek.com/v1",
        "model": "deepseek-chat",
        "env_key": "DEEPSEEK_API_KEY",
    },
    "qwen": {
        "label": "通义千问",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model": "qwen-plus",
        "env_key": "DASHSCOPE_API_KEY",
    },
    "openai": {
        "label": "OpenAI",
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-4o-mini",
        "env_key": "OPENAI_API_KEY",
    },
    "ollama": {
        # 本地跑的模型，Ollama 会暴露一个 OpenAI 兼容端口，所以不需要 Key
        "label": "本地 Ollama",
        "base_url": "http://localhost:11434/v1",
        "model": "qwen2.5:7b",
        "env_key": "",
    },
    "mock": {
        # 离线模拟模型：不联网、不要 Key，用于本地演示和断网兜底
        "label": "离线模拟模型",
        "base_url": "",
        "model": "docpilot-mock",
        "env_key": "",
    },
}


def _env_str(name: str, default: str = "") -> str:
    return (os.getenv(name) or default).strip()


def _env_float(name: str, default: float) -> float:
    raw = _env_str(name)
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    raw = _env_str(name)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _detect_provider() -> str:
    """没显式配置时，自动挑一个"有 Key 的"供应商；都没有就用离线模拟。"""
    explicit = _env_str("DOCPILOT_PROVIDER")
    if explicit:
        return explicit.lower()

    for name, preset in PROVIDER_PRESETS.items():
        env_key = preset["env_key"]
        if env_key and _env_str(env_key):
            return name
    return "mock"


# ============================================================
# 配置对象
# ------------------------------------------------------------
# frozen=True 表示创建后不可修改 —— 避免某个节点偷偷改了全局配置导致诡异 bug。
# ============================================================
@dataclass(frozen=True)
class Settings:
    provider: str
    model: str
    api_key: str
    base_url: str
    temperature: float
    timeout: int
    # —— 图的行为参数 ——
    retriever: str               # 检索器类型：bm25（默认）| vector
    top_k: int                   # 检索召回几条证据
    score_threshold: float       # 证据及格线，低于它就触发自纠正
    max_rewrites: int            # 自纠正最多重写几次（防死循环）
    docs_dir: Path               # 知识库文档所在目录

    @property
    def is_mock(self) -> bool:
        return self.provider == "mock"

    @property
    def provider_label(self) -> str:
        return PROVIDER_PRESETS.get(self.provider, {}).get("label", self.provider)

    def describe(self) -> str:
        """一行话描述当前配置，方便启动时打印出来确认。"""
        if self.is_mock:
            return f"模型 = {self.provider_label}（无需 Key，离线可跑）"
        masked = (self.api_key[:6] + "..." + self.api_key[-4:]) if len(self.api_key) > 12 else "(未设置)"
        return (
            f"模型 = {self.provider_label} / {self.model}  |  "
            f"Key = {masked}  |  接口 = {self.base_url}"
        )


def load_settings() -> Settings:
    provider = _detect_provider()
    preset = PROVIDER_PRESETS.get(provider)

    if preset is None:
        raise ValueError(
            f"未知的供应商 '{provider}'。可选值：{', '.join(PROVIDER_PRESETS)}"
        )

    env_key = preset["env_key"]
    api_key = _env_str(env_key) if env_key else ""
    if env_key and not api_key:
        raise ValueError(
            f"你选择了 {preset['label']}，但没有配置 {env_key}。\n"
            f"请把 .env.example 复制成 .env 并填入 Key，"
            f"或者把 DOCPILOT_PROVIDER 改成 mock 先用离线模型跑通流程。"
        )

    return Settings(
        provider=provider,
        model=_env_str("DOCPILOT_MODEL") or preset["model"],
        api_key=api_key,
        base_url=preset["base_url"],
        temperature=_env_float("DOCPILOT_TEMPERATURE", 0.0),
        timeout=_env_int("DOCPILOT_TIMEOUT", 60),
        retriever=_env_str("DOCPILOT_RETRIEVER", "bm25").lower(),
        top_k=_env_int("DOCPILOT_TOP_K", 4),
        score_threshold=_env_float("DOCPILOT_SCORE_THRESHOLD", 6.5),
        max_rewrites=_env_int("DOCPILOT_MAX_REWRITES", 2),
        docs_dir=PROJECT_ROOT / "data" / "docs",
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """带缓存的配置读取：整个进程只解析一次 .env。"""
    return load_settings()


def get_llm(settings: Settings | None = None, *, temperature: float | None = None):
    """
    模型工厂：根据配置返回一个可用的对话模型对象。

    返回的对象支持 LangChain 的标准接口：
        - llm.invoke(messages)              → 一次性拿到结果
        - llm.stream(messages)              → 流式逐字返回
        - llm.bind_tools([...])             → 让模型知道有哪些工具可用
        - llm.with_structured_output(Schema)→ 让模型按指定结构返回

    这三种能力我们后面都会用到。
    """
    settings = settings or get_settings()

    if settings.is_mock:
        # 延迟导入，避免没装依赖时也报错
        from docpilot.mock_llm import MockChatModel

        return MockChatModel()

    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=settings.model,
        api_key=settings.api_key,
        base_url=settings.base_url,
        temperature=settings.temperature if temperature is None else temperature,
        timeout=settings.timeout,
        max_retries=2,
    )
