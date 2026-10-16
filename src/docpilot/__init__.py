"""
DocPilot —— 基于 LangGraph 的自纠正 RAG 智能文档助手。

便捷入口：
    from docpilot import build_graph, get_settings

这里用「惰性导入」是为了避免循环依赖：
config 和 mock_llm 会被 graphs 里的模块使用，
如果这个文件在导入时就急着加载 graphs，就会变成自己引用自己。
"""

from docpilot.config import Settings, get_llm, get_settings, load_settings

__all__ = ["Settings", "get_llm", "get_settings", "load_settings", "build_graph"]
__version__ = "0.1.0"


def build_graph(*args, **kwargs):
    """构建主图（真正的实现在 docpilot.graphs.main，用到时才导入）。"""
    from docpilot.graphs.main import build_graph as _build

    return _build(*args, **kwargs)
