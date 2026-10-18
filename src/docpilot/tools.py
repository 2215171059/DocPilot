"""
tools.py —— 给模型用的「工具」

工具是什么？
工具就是模型可以主动调用的函数。模型不会自己执行代码，
它只会输出一句"我想调用 search_knowledge_base，参数是 xxx"，
真正执行的是 LangGraph 里的 ToolNode。这个分工要分清楚：
  - 模型负责「决定调不调、调哪个、传什么参数」
  - 框架负责「真的去执行，并把结果塞回对话」

写工具的两个关键点：
1. 函数的 **docstring 就是工具说明书**，模型靠它判断什么时候该用这个工具。
   所以要写清楚"这个工具是干什么的、什么时候用"，
   而不是写"这是一个函数"这种废话。docstring 建议控制在 1024 字符内。
2. 参数类型注解要写，LangChain 会据此自动生成 JSON Schema 给模型看。
"""

from __future__ import annotations

from langchain_core.tools import tool

from docpilot.retrievers.factory import get_retriever


@tool
def search_knowledge_base(query: str, top_k: int = 4) -> str:
    """在公司内部知识库中检索资料。

    当用户的问题涉及产品功能、接口用法、操作流程、技术规范、参数配置、
    故障排查等需要依据内部文档才能准确回答的内容时，使用这个工具。
    不要用它来回答闲聊或通用常识问题。

    参数：
        query: 检索用的查询语句。建议用陈述式的关键词组合，
               比如"登录失败 排查 步骤"，而不是"登录失败怎么办？"。
        top_k: 返回多少条资料，默认 4 条。问题越宽泛可以调大一些。
    """
    retriever = get_retriever()
    chunks = retriever.retrieve(query, top_k=max(1, min(top_k, 10)))

    if not chunks:
        return "知识库中没有检索到与查询相关的内容。可以尝试换用更具体的关键词。"

    blocks = []
    for i, chunk in enumerate(chunks, start=1):
        blocks.append(
            f"[{i}] 来源：{chunk['source']}（相关度 {chunk['score']}）\n{chunk['content']}"
        )
    return "\n\n".join(blocks)


@tool
def calculator(expression: str) -> str:
    """计算一个数学表达式，支持加减乘除、括号和幂运算。

    当用户的问题需要进行数值计算（比如算比例、算差值、算总和）时使用。
    参数 expression 必须是纯数学表达式，例如 "(120-95)/120*100"。
    不要在表达式里写中文、单位或文字说明。
    """
    # 只允许数字和基本运算符，避免执行任意代码 —— 安全第一
    allowed = set("0123456789+-*/(). %")
    if not set(expression) <= allowed:
        return "表达式包含不允许的字符，只支持数字和 + - * / ( ) . % 这些符号。"

    try:
        # 这里用 eval 是有意为之，但已经严格限制了字符集。
        # 生产环境建议换成 ast.literal_eval 或专门的表达式解析库。
        result = eval(expression, {"__builtins__": {}}, {})  # noqa: S307
    except Exception as exc:
        return f"计算失败：{exc}"

    return f"{expression} = {result}"


#: 暴露给模型的工具清单
ALL_TOOLS = [search_knowledge_base, calculator]
