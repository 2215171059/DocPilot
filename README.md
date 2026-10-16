# DocPilot · 基于 LangGraph 的自纠正 RAG 智能文档助手

> 一个"能讲、能跑、能改"的 LangGraph 实践项目。
> 配套教学报告：[`docs/教学报告.html`](docs/教学报告.html) —— 从零讲到面试。

---

## 它解决什么问题

普通 RAG 的链路是单向的：**检索 → 生成**。
检索到什么就用什么，就算召回一堆垃圾，模型也会硬着头皮编一段答案。

DocPilot 在中间插了一个"裁判"，把单向链改成了**带反馈的环**：

```
START → 路由判断 → 检索资料 → 证据评分 ──┬─(够好)→ 生成答案 → 反思自查 ──┬─(普通)→ 定稿输出 → END
                       ▲                 │                              │
                       └── 重写查询 ◄────┘(不够好)          人工审核 ◄────┘(敏感)
```

于是系统第一次有了"自知之明"：**知道自己不知道**。

---

## 30 秒跑起来

```bash
# 1. 建虚拟环境（Python 3.11+）
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

# 2. 装依赖
pip install -r requirements.txt
# 国内网络慢就加上：-i https://pypi.tuna.tsinghua.edu.cn/simple

# 3. 配环境变量
copy .env.example .env          # Windows
# cp .env.example .env          # macOS / Linux

# 4. 跑命令行版（默认用离线模拟模型，不需要 API Key）
python scripts/run_cli.py

# 或者跑界面版
streamlit run app.py
```

**不需要任何 API Key 就能跑通全流程** —— 默认用的是内置的离线模拟模型。

### Windows 用户：双击即用

仓库根目录附了两个批处理脚本，省去敲命令：

| 双击 | 效果 |
|---|---|
| `run_app.bat` | 启动网页界面（http://127.0.0.1:8501） |
| `run_cli.bat` | 启动命令行对话版 |

它们优先使用项目内的 `.venv`，找不到则回退到系统 `python`。
注意：Streamlit 应用**必须**用 `streamlit run` 启动，直接 `python app.py` 是跑不起来的。

### 想换成真实大模型？

在 `.env` 里改两行就行，代码一行都不用动：

```ini
DOCPILOT_PROVIDER=deepseek
DEEPSEEK_API_KEY=sk-你的key
```

也支持 `qwen`（通义千问）、`openai`、`ollama`（本地模型）。

---

## 项目结构

```
makeLangGrap/
├── app.py                      Streamlit 界面（带执行轨迹面板 + 人工审核）
├── requirements.txt
├── .env.example                配置模板
│
├── src/docpilot/
│   ├── config.py               配置 + 模型工厂（换模型只改这里）
│   ├── state.py                ★ 状态定义（reducer 的核心示范）
│   ├── prompts.py              所有提示词集中管理
│   ├── tools.py                给模型用的工具（docstring 即说明书）
│   ├── mock_llm.py             离线模拟模型（断网也能演示）
│   │
│   ├── retrievers/             可插拔检索层
│   │   ├── base.py             统一接口
│   │   ├── bm25.py             BM25 关键词检索（默认）
│   │   ├── vector.py           向量检索（可选，需额外装包）
│   │   └── loader.py           文档加载与切块
│   │
│   ├── nodes/                  ★ 每个节点一个函数
│   │   ├── router.py           ① 路由判断
│   │   ├── retrieve.py         ② 检索资料
│   │   ├── grade.py            ③ 证据评分  ┐
│   │   ├── rewrite.py          ④ 重写查询  ┘ 自纠正循环
│   │   ├── generate.py         ⑤ 生成答案
│   │   ├── reflect.py          ⑥ 反思自查
│   │   ├── review.py           ⑦ 人工审核（interrupt）
│   │   ├── finalize.py         ⑧ 定稿输出
│   │   └── conditions.py       条件边的路由函数
│   │
│   ├── graphs/                 ★ 分阶段的可运行示例
│   │   ├── s1_basic.py         最小状态图（3 节点直线）
│   │   ├── s2_tools.py         预构建工具 Agent
│   │   ├── s3_react.py         手写 ReAct 循环
│   │   ├── s4_rag.py           朴素 RAG（基准线）
│   │   ├── s5_crag.py          自纠正 RAG（核心）
│   │   └── main.py             完整主图（记忆 + 人机协同）
│   │
│   └── utils/                  轨迹收集、JSON 容错、图可视化
│
├── data/docs/                  示例知识库（云雀智能客服系统）
├── scripts/run_cli.py          命令行版
└── tests/test_graph.py         关键路径测试
```

---

## 学习路径（建议按顺序）

每个文件都能**单独运行**，边跑边看输出理解最快：

```bash
python src/docpilot/graphs/s1_basic.py    # 先看懂状态怎么流动
python src/docpilot/graphs/s3_react.py    # 再看循环怎么写
python src/docpilot/graphs/s4_rag.py      # 朴素 RAG 的局限
python src/docpilot/graphs/s5_crag.py     # 加上自纠正后不一样了
python scripts/run_cli.py                 # 完整形态：记忆 + 人工审核
```

跑测试：

```bash
pytest tests/ -v
```

---

## 这个项目覆盖的 LangGraph 知识点

| 知识点 | 在哪看 |
|---|---|
| `StateGraph` / `State` / `TypedDict` | `state.py`、`graphs/s1_basic.py` |
| **reducer**（`add_messages` / 覆盖 / 自定义） | `state.py`（含一个真实的踩坑记录） |
| 节点与普通边 | `graphs/s1_basic.py` |
| **条件边**与分支路由 | `nodes/conditions.py` |
| **循环**与递归上限 | `graphs/s3_react.py`、`s5_crag.py` |
| 工具调用（`bind_tools` / `ToolNode`） | `tools.py`、`graphs/s3_react.py` |
| 预构建 Agent（含 API 废弃迁移） | `graphs/s2_tools.py` |
| **Checkpointer** 与 `thread_id` | `graphs/main.py` |
| **`interrupt()` + `Command(resume=)`** | `nodes/review.py` |
| 流式执行（`stream_mode="updates"`） | `app.py` |
| **时间旅行**（`get_state_history`） | `app.py`、`scripts/run_cli.py` |
| 图可视化（ASCII / Mermaid） | `utils/viz.py` |

---

## 常见问题

**Q：为什么默认检索用 BM25，不用向量检索？**
向量检索要下载几百 MB 的 embedding 模型，新手第一天就卡在环境上。
BM25 是纯 Python 的关键词检索，装完就能跑，30 秒看到效果。
等主线跑通了，改 `.env` 里一行 `DOCPILOT_RETRIEVER=vector` 就能切换。

**Q：没有 API Key 能演示吗？**
能。`DOCPILOT_PROVIDER=mock` 会启用内置的离线模拟模型，
路由、检索、评分、重写、生成、反思、人工审核整条链路都能跑通，面试现场断网也不慌。

**Q：想换成自己的文档？**
把 `.md` / `.txt` 丢进 `data/docs/` 目录，重启即可（索引做了进程级缓存）。

---

## 许可

本项目采用 MIT 协议，详见 [LICENSE](LICENSE)。代码可自由使用、修改与分发。
