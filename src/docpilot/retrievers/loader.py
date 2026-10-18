"""
loader.py —— 把知识库文件读进来、切成小块

为什么必须"切块"（chunking）？
1. 一份文档可能几万字，整段塞给模型会超长、又贵又慢
2. 检索的粒度越细，越容易精准命中"那一小段真正相关的内容"
3. 太细又会丢上下文，所以要取个平衡

这里的策略：先按空行分段，再把段落拼到目标长度（默认 400 字），
这样既不会在句子中间被切断，又能控制单块大小。
"""

from __future__ import annotations

import re
from pathlib import Path

from docpilot.retrievers.base import Chunk

# 支持的文件类型：先覆盖纯文本类，避免引入额外解析依赖
TEXT_SUFFIXES = {".md", ".txt", ".markdown"}


def split_text(text: str, chunk_size: int = 400, overlap: int = 60) -> list[str]:
    """
    把长文本切成若干段。

    步骤：
      1. 按空行切成自然段
      2. 依次把段落拼进当前块，超过 chunk_size 就收口
      3. 新块开头带上上一块的末尾若干字（overlap），避免语义被硬切断
    """
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if not paragraphs:
        return []

    chunks: list[str] = []
    current = ""

    for para in paragraphs:
        # 单个段落就超长：直接按长度硬切
        if len(para) > chunk_size * 1.5:
            if current:
                chunks.append(current)
                current = ""
            for i in range(0, len(para), chunk_size - overlap):
                piece = para[i : i + chunk_size]
                if piece.strip():
                    chunks.append(piece.strip())
            continue

        if len(current) + len(para) + 1 <= chunk_size:
            current = f"{current}\n{para}".strip()
        else:
            if current:
                chunks.append(current)
            tail = current[-overlap:] if current and overlap else ""
            current = f"{tail}\n{para}".strip() if tail else para

    if current:
        chunks.append(current)

    return chunks


def load_file(path: Path) -> list[Chunk]:
    """读取单个文件并切块。"""
    text = path.read_text(encoding="utf-8", errors="ignore")
    chunks: list[Chunk] = []
    for idx, piece in enumerate(split_text(text)):
        chunks.append(
            Chunk(
                content=piece,
                source=path.name,
                score=0.0,
                metadata={"path": str(path), "chunk_index": idx},
            )
        )
    return chunks


def load_corpus(docs_dir: Path) -> list[Chunk]:
    """
    读取整个目录（递归）下的所有文本文件。

    返回空列表也不报错 —— 首次运行时知识库为空是很正常的，
    调用方应该给出友好提示而不是崩溃。
    """
    if not docs_dir.exists():
        return []

    chunks: list[Chunk] = []
    for path in sorted(docs_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in TEXT_SUFFIXES:
            chunks.extend(load_file(path))
    return chunks
