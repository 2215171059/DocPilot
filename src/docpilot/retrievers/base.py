"""
base.py —— 检索器的统一接口

为什么要抽一层接口？
因为"怎么找到相关资料"有好几种做法：
  - BM25：关键词匹配，零依赖、离线可跑
  - 向量检索：语义匹配，能理解"登录不上"≈"无法认证"
  - 混合检索：两者融合
  - 交给外部服务（Elasticsearch / 向量数据库）

如果业务代码里直接写死 chroma.similarity_search(...)，
以后想换方案就得改一堆地方。

抽成 BaseRetriever 之后，上层只认 `retrieve(query, top_k)` 这一个方法，
底层换实现，上层一行都不用动。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, TypedDict


class Chunk(TypedDict):
    """一条检索结果。用 TypedDict 而不是自定义类，是为了能直接塞进状态被序列化。"""

    content: str          # 文本内容
    source: str           # 来源文件名
    score: float          # 相关度分数（越大越相关）
    metadata: dict[str, Any]


class BaseRetriever(ABC):
    """所有检索器都要实现这个接口。"""

    #: 给人看的名字，界面上展示用
    name: str = "base"

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 4) -> list[Chunk]:
        """
        根据查询召回最相关的若干条文本块。

        参数：
            query —— 查询语句（可能是被重写过的版本）
            top_k —— 最多返回几条
        返回：
            按相关度从高到低排序的 Chunk 列表；没有命中就返回空列表
        """
        raise NotImplementedError

    def __len__(self) -> int:
        """索引里一共有多少条文本块，用于自检。"""
        return 0
