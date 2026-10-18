"""
jsonx.py —— 宽容的 JSON 解析

为什么需要它？
我们让模型输出 JSON，但真实模型经常会这样返回：

    ```json
    {"score": 8}
    ```
    好的，评分是 8 分。

外面裹了 markdown 代码块，或者前后带了客套话。
这时候直接 json.loads 就会抛异常，整个节点崩掉。

工程上必须容错 —— 模型输出是不可信输入，永远要做防御性解析。
这也是为什么不推荐在关键路径上假设"模型一定会按格式返回"。
"""

from __future__ import annotations

import json
import re
from typing import Any

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def parse_json_loose(text: Any) -> dict[str, Any]:
    """
    尽最大努力把文本解析成 dict。解析不出来就返回空 dict，绝不抛异常。

    依次尝试：
      1. 直接解析
      2. 剥掉 markdown 代码块后再解析
      3. 截取第一个 { 到最后一个 } 之间的部分再解析
    """
    if isinstance(text, dict):
        return text

    raw = str(text or "").strip()
    if not raw:
        return {}

    # 尝试 1：直接解析
    try:
        result = json.loads(raw)
        return result if isinstance(result, dict) else {}
    except json.JSONDecodeError:
        pass

    # 尝试 2：剥掉 ```json ... ``` 包裹
    fenced = _FENCE_RE.search(raw)
    if fenced:
        try:
            result = json.loads(fenced.group(1))
            return result if isinstance(result, dict) else {}
        except json.JSONDecodeError:
            pass

    # 尝试 3：粗暴地截取最外层的花括号
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end > start:
        try:
            result = json.loads(raw[start : end + 1])
            return result if isinstance(result, dict) else {}
        except json.JSONDecodeError:
            pass

    return {}


def safe_float(value: Any, default: float = 0.0) -> float:
    """把任意值稳妥地转成 float，转不动就用默认值。"""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
