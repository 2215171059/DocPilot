"""可插拔检索层：默认 BM25 关键词检索，可选升级为向量检索。"""

from docpilot.retrievers.base import BaseRetriever, Chunk
from docpilot.retrievers.bm25 import BM25Retriever
from docpilot.retrievers.factory import build_retriever, get_retriever
from docpilot.retrievers.loader import load_corpus

__all__ = [
    "BaseRetriever",
    "Chunk",
    "BM25Retriever",
    "build_retriever",
    "get_retriever",
    "load_corpus",
]
