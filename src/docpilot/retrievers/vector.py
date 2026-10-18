"""
vector.py —— （可选）向量检索器

默认不启用。想升级时：
    pip install chromadb langchain-chroma langchain-huggingface sentence-transformers
再把 .env 里的 DOCPILOT_RETRIEVER 改成 vector。

它和 BM25 的本质区别：
BM25 比的是"字面有没有相同的词"。
向量检索先把文本映射成一串数字（向量），比的是"意思像不像"。
  用户问："登录不上去"        → 向量能找到"无法完成身份认证"这段
  但 BM25 找不到，因为一个字都没重合

代价：
一来要跑 embedding 模型（几百 MB，首次下载慢），
二来对"精确串"（错误码、订单号）反而不如 BM25，
所以生产上常把两者混起来用。

这里给出一个最小可用的实现，保持接口与 BM25 完全一致，
上层代码感知不到差异 —— 这就是 base.py 那层抽象的收益。
"""

from __future__ import annotations

from docpilot.retrievers.base import BaseRetriever, Chunk

_INSTALL_HINT = (
    "向量检索需要额外依赖。请先执行：\n"
    "    pip install chromadb langchain-chroma sentence-transformers\n"
    "然后把 .env 里的 DOCPILOT_RETRIEVER 设为 vector。"
)


class VectorRetriever(BaseRetriever):
    """基于 Chroma + 句向量模型的语义检索器。"""

    name = "vector"

    def __init__(self, chunks: list[Chunk], *, persist_dir: str | None = None) -> None:
        try:
            from langchain_chroma import Chroma
            from langchain_huggingface import HuggingFaceEmbeddings
        except ImportError as exc:  # pragma: no cover - 取决于是否装了可选依赖
            raise ImportError(_INSTALL_HINT) from exc

        self._embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-zh-v1.5",
            encode_kwargs={"normalize_embeddings": True},
        )
        self._store = Chroma(
            collection_name="docpilot",
            embedding_function=self._embeddings,
            persist_directory=persist_dir,
        )

        if chunks and self._store._collection.count() == 0:
            self._store.add_texts(
                texts=[c["content"] for c in chunks],
                metadatas=[
                    {"source": c["source"], **c.get("metadata", {})} for c in chunks
                ],
            )

    def __len__(self) -> int:
        return self._store._collection.count()

    def retrieve(self, query: str, top_k: int = 4) -> list[Chunk]:
        pairs = self._store.similarity_search_with_score(query, k=top_k)
        results: list[Chunk] = []
        for doc, distance in pairs:
            results.append(
                Chunk(
                    content=doc.page_content,
                    source=str(doc.metadata.get("source", "未知来源")),
                    # Chroma 返回的是"距离"，越小越相关；转成"越大越相关"便于统一
                    score=round(1.0 / (1.0 + float(distance)), 4),
                    metadata=dict(doc.metadata),
                )
            )
        return results
