"""
factory.py —— 检索器工厂

上层只需要调 get_retriever()，拿到的永远是实现了 BaseRetriever 接口的对象，
至于底层是 BM25 还是向量检索，由 .env 里的一个配置项决定。

【这里还有个容易被忽略的工程细节】
检索器做了进程级缓存（lru_cache）。
因为建索引是"读文件 + 算词频"的活儿，每次提问都重建一遍纯属浪费。
但也要注意：如果你往 data/docs 里加了新文档，
进程不会自动感知 —— 需要重启，或者手动调 get_retriever.cache_clear()。
这是很典型的"缓存一致性"权衡，面试时聊到缓存可以举这个例子。
"""

from __future__ import annotations

from functools import lru_cache

from docpilot.config import Settings, get_settings
from docpilot.retrievers.base import BaseRetriever
from docpilot.retrievers.bm25 import BM25Retriever
from docpilot.retrievers.loader import load_corpus


def build_retriever(settings: Settings | None = None) -> BaseRetriever:
    """按配置构造一个检索器实例。"""
    settings = settings or get_settings()
    chunks = load_corpus(settings.docs_dir)

    if settings.retriever == "vector":
        # 延迟导入：没装可选依赖时不应该影响 bm25 模式
        from docpilot.retrievers.vector import VectorRetriever

        return VectorRetriever(chunks)

    return BM25Retriever(chunks)


@lru_cache(maxsize=1)
def get_retriever() -> BaseRetriever:
    """拿全局唯一的检索器实例（懒加载 + 缓存）。"""
    return build_retriever()
