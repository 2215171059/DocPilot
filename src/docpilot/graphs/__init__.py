"""分阶段的可运行示例图，对应 S1 ~ S6，每个阶段都能单独跑起来。"""

__all__ = [
    "build_basic_graph",
    "build_tool_graph",
    "build_react_graph",
    "build_rag_graph",
    "build_crag_graph",
]


def build_basic_graph(*args, **kwargs):
    from docpilot.graphs.s1_basic import build_basic_graph as _f

    return _f(*args, **kwargs)


def build_tool_graph(*args, **kwargs):
    from docpilot.graphs.s2_tools import build_tool_graph as _f

    return _f(*args, **kwargs)


def build_react_graph(*args, **kwargs):
    from docpilot.graphs.s3_react import build_react_graph as _f

    return _f(*args, **kwargs)


def build_rag_graph(*args, **kwargs):
    from docpilot.graphs.s4_rag import build_rag_graph as _f

    return _f(*args, **kwargs)


def build_crag_graph(*args, **kwargs):
    from docpilot.graphs.s5_crag import build_crag_graph as _f

    return _f(*args, **kwargs)
