"""
bm25.py —— BM25 关键词检索器

【BM25 是什么？】
一句话：它是"关键词匹配"的经典算法，也是搜索引擎的基础打分公式之一。
核心直觉是两点：
  1. 一个词在这篇文档里出现得越多，越说明这篇文档和它相关（词频 TF）
  2. 但"的""是"这种到处都有的词，含金量就低（逆文档频率 IDF）
  3. 文档太长会稀释关键信息，所以要按长度做归一化

【为什么默认用它？】
- 不用下载任何模型，pip 装完就能跑，30 秒看到效果
- 不需要 GPU，不需要联网
- 对"专有名词、错误码、接口名"这类查询，关键词匹配其实比向量检索还准

【中文怎么分词？】
BM25 原本是为英文设计的（按空格切词）。
中文没有空格，这里用了一个朴素但有效的技巧：**双字片段（bigram）**。
  "登录失败" → ["登录", "录失", "失败"]
虽然会产生"录失"这种没意义的片段，但它们几乎没有区分度，
对排序结果影响很小，工程上完全够用。

【面试考点】
"BM25 和向量检索有什么区别？什么时候用哪个？"
- BM25 管"字面命中"，向量管"语义相似"
- 用户搜"ERR_502"这种精确串，BM25 更准；用户说"上不去网"，向量更准
- 生产上常见做法是混合检索（Hybrid），两路召回后用 RRF 融合排序
"""

from __future__ import annotations

import re

from rank_bm25 import BM25Okapi

from docpilot.retrievers.base import BaseRetriever, Chunk

_EN_WORD_RE = re.compile(r"[a-z0-9_]+")
_CJK_RUN_RE = re.compile(r"[\u4e00-\u9fff]+")


def tokenize(text: str) -> list[str]:
    """
    中英文混合分词。

    - 英文/数字：按连续字母数字切
    - 中文：切成相邻两字的片段
    """
    lowered = (text or "").lower()
    tokens = _EN_WORD_RE.findall(lowered)

    for run in _CJK_RUN_RE.findall(lowered):
        if len(run) == 1:
            tokens.append(run)
        else:
            tokens.extend(run[i : i + 2] for i in range(len(run) - 1))

    return tokens


class BM25Retriever(BaseRetriever):
    """基于 rank_bm25 的关键词检索器。"""

    name = "bm25"

    def __init__(self, chunks: list[Chunk]) -> None:
        self.chunks: list[Chunk] = list(chunks)
        self._tokenized: list[list[str]] = [tokenize(c["content"]) for c in self.chunks]

        # BM25Okapi 在语料为空时会报错，所以这里做一下保护
        self._bm25 = BM25Okapi(self._tokenized) if self._tokenized else None

    def __len__(self) -> int:
        return len(self.chunks)

    def retrieve(self, query: str, top_k: int = 4) -> list[Chunk]:
        if not self._bm25 or not self.chunks:
            return []

        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        scores = self._bm25.get_scores(query_tokens)

        # 取分数最高的 top_k 条；分数为 0 表示一个词都没命中，直接丢掉
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

        results: list[Chunk] = []
        for idx in ranked:
            if len(results) >= top_k:
                break
            if scores[idx] <= 0:
                break
            chunk = dict(self.chunks[idx])
            chunk["score"] = round(float(scores[idx]), 4)
            results.append(chunk)  # type: ignore[arg-type]

        return results
