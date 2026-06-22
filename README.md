# PatchHarness

> 面向自动化 Bug 修复的 Multi-Agent Harness 系统：提交任务描述与工作区路径 → 动态规划 Agent 链路 → 工具调用修复代码 → 返回补丁证据、测试结果与成本指标。

![Python](https://img.shields.io/badge/Python-3.11+-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-API-green)
![LLM](https://img.shields.io/badge/LLM-DeepSeek%20%7C%20Ark-purple)
![Benchmark](https://img.shields.io/badge/Benchmark-Reproducible-orange)

---

## 项目简介

PatchHarness 是一个面向自动化 Bug 修复场景的 Multi-Agent Harness 系统。它将一次 Bug 修复请求拆成规划、审查、修复、测试验证和结果汇总等可配置 Agent 链路，通过统一工具层访问代码搜索、文件读写、测试执行、Git diff 和 Skill 管理能力，最终返回结构化执行报告。

项目重点不是做一个“聊天式代码助手”，而是验证一个更工程化的问题：

```text
固定 4-Agent 管道是否会造成不必要 LLM 调用？
动态 Planner 能否在保持成功率的同时减少 LLM 调用和 token 消耗？
```

当前项目已经支持：

- 默认无真实 LLM 的确定性 Bug 修复闭环。
- 配置 DeepSeek 或火山 Ark 后启用真实 LLM ReAct 修复。
- 规则 Planner + LLM fallback Planner。
- ContextIsolation、Skill 渐进式披露、HITL 风险拦截、ContextCompressor。
- 估算型 Planner Benchmark、Skill token Benchmark、真实 Ark LLM Benchmark。

---

## 核心工作流

```text
用户提交 Bug 修复 / 审查请求
         │
         ▼
  FastAPI: POST /bugfix
         │
         ▼
  RulePlanner / LLMFallbackPlanner
         │
         ├── review        -> code_review
         ├── fix           -> bug_fix -> test_verify
         ├── full          -> code_review -> bug_fix -> test_verify -> summary
         └── ambiguous     -> LLM fallback / HITL
         │
         ▼
  AgentRegistry 加载 YAML Agent 配置
         │
         ▼
  Workflow 编排 Agent 执行
         │
         ├── Context fork / merge / cleanup
         ├── Skill frontmatter 披露与按需加载
         ├── MCP-style tool call
         ├── HITL 风险策略
         └── LLM JSON Action ReAct fallback
         │
         ▼
  返回结构化报告：planned_agents / actions / observations / changed_files / tests / metrics
```

---

## 系统架构

```text
┌─────────────────────────────────────────────────────────────┐
│                    FastAPI 服务层                            │
│  POST /bugfix  │  GET /health  │  OpenAPI Docs              │
└───────────────────────────┬─────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────┐
│                    动态规划层                                │
│  RulePlanner: 关键词 + mode 决策，0 LLM 调用                  │
│  LLMFallbackPlanner: 模糊请求分类 + 规划层 HITL              │
└───────────────────────────┬─────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────┐
│                    Agent 执行层                              │
│  AgentRegistry: YAML 配置注册                                │
│  BaseAgent: Thought -> Action -> Observation                 │
│  bug_fix / code_review / test_verify / summary               │
└───────────────────────────┬─────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────┐
│                    工程能力层                                │
│  ContextIsolation: fork / merge / cleanup                    │
│  SkillManager: frontmatter 渐进式披露                         │
│  HITL: .env / 密钥 / 配置 / 大范围替换风险拦截                 │
│  ContextCompressor: token 预算 + 最近 Observation 保留        │
└───────────────────────────┬─────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────┐
│                    工具与模型层                              │
│  MCP-style Tools: grep / read / edit / test / git_diff       │
│  LLMClient: DeepSeek / Volcengine Ark / Mock                 │
│  Metrics: llm_calls / tokens / timeouts / tool_calls         │
└─────────────────────────────────────────────────────────────┘
```

说明：项目提供可选 LangGraph 适配层，但默认主路径使用轻量 sequential workflow，便于本地运行、测试和 benchmark 稳定复现。

---

## 技术亮点

### 1. 动态 Agent 规划

传统固定管道会让所有请求都走完整链路：

```text
code_review -> bug_fix -> test_verify -> summary
```

PatchHarness 使用 `RulePlanner` 根据请求类型选择 Agent 链路：

```text
审查类任务       -> code_review
明确修复任务     -> bug_fix -> test_verify
完整检查任务     -> code_review -> bug_fix -> test_verify -> summary
模糊任务         -> LLM fallback / HITL
```

这让 review-only、deterministic-fix 等简单任务避免进入不必要的 LLM ReAct 链路。

### 2. 确定性修复优先，LLM ReAct fallback

当任务描述里包含明确 old/new 替换时，系统直接走确定性工具链：

```text
read_file -> edit_file -> git_diff -> run_test
```

例如：

```text
在 `calculator.py` 中将 `return a - b` 替换为 `return a + b`
```

当任务描述不包含明确替换计划时，可启用真实 LLM：

```json
{
  "enable_llm": true
}
```

此时 Agent 要求模型输出 JSON Action：

```json
{
  "thought": "string",
  "action": "read_file | edit_file | ... | null",
  "action_input": {},
  "final": "string | null"
}
```

### 3. MCP-style 工具层

工具层提供统一 schema 和调用边界，Agent 不直接依赖具体实现类。

内置工具：

| Tool | 作用 |
| --- | --- |
| `grep_search` | 搜索工作区文本 |
| `read_file` | 读取文件 |
| `edit_file` | 精确替换文件内容 |
| `run_test` | 执行测试命令 |
| `git_diff` | 输出当前修改 diff |
| `search_skill` | 搜索 Skill |
| `download_skill` | 按需加载完整 Skill |
| `create_skill` | 创建本地 Skill |
| `update_skill` | 更新本地 Skill |

### 4. ContextIsolation

每个 Agent 执行时拥有独立上下文：

```text
fork    -> 创建 Agent 独立上下文
merge   -> 只合并结构化报告
cleanup -> 丢弃临时状态
```

同时支持三态可见性：

```text
read_only  : 任务描述、系统事实
writable   : Agent 报告、中间结果
hidden     : internal_secrets、敏感配置
```

### 5. Skill 渐进式披露

系统启动时只读取 Skill frontmatter：

```text
name / description / triggers
```

Agent 需要时再通过 `download_skill` 加载完整内容。这样避免把所有 Skill 全量塞进 prompt。

当前 Skill token Benchmark：

```text
Full injection:     161 tokens
Frontmatter only:    71 tokens
Saved tokens:        90
Saved rate:        55.9%
```

说明：当前内置 Skill 较短，因此节省率是 55.9%；更长的真实 Skill 文档会更能体现渐进式披露价值。

### 6. HITL 风险拦截

高风险操作不会直接执行，而是返回：

```json
{
  "requires_human_approval": true
}
```

当前拦截范围包括：

- 修改 `.env`
- 修改密钥、证书、核心配置
- 删除文件
- 大范围或多行替换

### 7. 真实 LLM Provider

当前支持：

| Provider | 实现 |
| --- | --- |
| DeepSeek | `app/llm/deepseek_client.py` |
| 火山 Ark | `app/llm/volcengine_client.py` |
| Mock LLM | `app/llm/mock_client.py` |

DeepSeek 与火山 Ark 均使用原生 `httpx.post()` adapter，不依赖 OpenAI SDK。业务层只依赖统一 `LLMClient` 协议。

### 8. Benchmark 证据

项目包含三类可复现 benchmark：

| Benchmark | 脚本 | 说明 |
| --- | --- | --- |
| Planner Benchmark | `benchmarks/run_planner_benchmark.py` | 估算固定管道 vs 动态规划 |
| Skill Token Benchmark | `benchmarks/run_skill_token_benchmark.py` | 全量注入 vs frontmatter |
| Real LLM Benchmark | `benchmarks/run_real_llm_benchmark.py` | 真实 Ark / DeepSeek 调用 |

---

## Benchmark 结果

### Planner Benchmark

100 条模拟请求：

```text
固定 4-Agent 管道：400 次 Agent 调用，估算 800 次 LLM 调用
动态 Planner：200 次 Agent 调用，估算 400 次 LLM 调用
节省估算 LLM 调用：400 次
节省率：50.0%
规划命中率：100.0%
```

结果文件：

```text
results/planner_benchmark.csv
results/planner_benchmark_summary.md
```

### 真实 Ark LLM Benchmark

9 个自定义 case，覆盖：

```text
review_only
deterministic_fix
llm_fix
full_workflow
hitl_risk
```

结果文件：

```text
results/real_llm_benchmark/runs/20260621_214846/summary.md
results/real_llm_benchmark/runs/20260621_214846/comparison.csv
```

结果摘要：

| Metric | Fixed 4-Agent | Dynamic Planner |
| --- | ---: | ---: |
| Success rate | 66.7% | 100.0% |
| Real LLM calls | 7 | 3 |
| LLM timeouts | 2 | 0 |
| Total tokens | 4139 | 1398 |

节省结果：

```text
Saved LLM calls: 4
Saved LLM call rate: 57.1%
Saved total tokens: 2741
Token saving rate: 66.2%
```

说明：该结论限定在当前自定义 benchmark 中，不代表所有代码修复任务的全局性能。

---

## 快速开始

### 环境要求

- Python 3.11+
- Windows PowerShell / macOS / Linux shell
- 可选：DeepSeek API Key 或火山 Ark API Key

### 1. 克隆项目

```bash
git clone https://github.com/shilv520/PatchHarness.git
cd PatchHarness
```

如果你的仓库名仍是 `harness`，进入实际目录即可。

### 2. 创建虚拟环境

Windows PowerShell：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

如果 PowerShell 执行策略拦截：

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### 3. 安装依赖

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

可选依赖：

```bash
python -m pip install -r requirements-optional.txt
```

`requirements-optional.txt` 包含 LangGraph、Redis、ChromaDB。默认运行不需要它们。

### 4. 运行测试

```bash
python -m unittest discover -s tests
```

预期：

```text
OK
```

### 5. 运行 Demo

```bash
python demo/run_demo.py
```

Demo 会复制 `demo/buggy_project` 到运行时目录，并修复：

```diff
-    return a - b
+    return a + b
```

### 6. 启动 API

```bash
uvicorn app.main:app --reload
```

访问：

```text
GET http://127.0.0.1:8000/health
GET http://127.0.0.1:8000/docs
```

---

## API 接口

| Method | Path | 说明 |
| --- | --- | --- |
| `GET` | `/health` | 健康检查 |
| `POST` | `/bugfix` | 提交 Bug 修复 / 审查请求 |

### 请求示例

PowerShell：

```powershell
Invoke-RestMethod `
  -Uri "http://127.0.0.1:8000/bugfix" `
  -Method Post `
  -ContentType "application/json" `
  -Body '{
    "task_description": "修复 bug：在 `calculator.py` 中将 `return a - b` 替换为 `return a + b`",
    "workspace_path": "demo/buggy_project",
    "mode": "fix",
    "allow_edit": true,
    "run_tests": true,
    "test_command": "python -m unittest discover -s tests"
  }'
```

### 启用真实 LLM

请求中设置：

```json
{
  "enable_llm": true
}
```

配置火山 Ark：

```powershell
$env:PATCHHARNESS_LLM_PROVIDER="ark"
$env:ARK_API_KEY="your-api-key"
$env:ARK_MODEL="your-endpoint-id"
$env:ARK_BASE_URL="https://ark.cn-beijing.volces.com/api/v3"
```

配置 DeepSeek：

```powershell
$env:PATCHHARNESS_LLM_PROVIDER="deepseek"
$env:DEEPSEEK_API_KEY="your-api-key"
$env:DEEPSEEK_MODEL="deepseek-chat"
```

---

## Benchmark 命令

Planner Benchmark：

```bash
python benchmarks/run_planner_benchmark.py
```

Skill token Benchmark：

```bash
python benchmarks/run_skill_token_benchmark.py
```

Demo evidence：

```bash
python benchmarks/generate_demo_evidence.py
```

Real LLM Benchmark：

```bash
python benchmarks/run_real_llm_benchmark.py --provider mock --max-cases 9
python benchmarks/run_real_llm_benchmark.py --provider ark --max-cases 9
python benchmarks/run_real_llm_benchmark.py --provider deepseek --max-cases 9
```

按类别运行：

```bash
python benchmarks/run_real_llm_benchmark.py --provider ark --category review_only --max-cases 3
python benchmarks/run_real_llm_benchmark.py --provider ark --category llm_fix --max-cases 1
```

---

## 项目结构

```text
PatchHarness/
├── app/
│   ├── main.py                         # FastAPI 入口
│   ├── schemas.py                      # 请求、响应、指标模型
│   ├── planner/
│   │   ├── rule_planner.py             # 0 LLM 规则规划
│   │   └── llm_fallback.py             # 模糊请求 LLM fallback
│   ├── agents/
│   │   ├── base.py                     # Agent 执行逻辑
│   │   ├── registry.py                 # AgentRegistry
│   │   └── configs/                    # YAML Agent 配置
│   ├── graph/
│   │   ├── workflow.py                 # 默认 sequential workflow
│   │   └── langgraph_workflow.py       # 可选 LangGraph 适配
│   ├── tools/                          # grep/read/edit/test/git_diff
│   ├── mcp/                            # MCP-style server/client
│   ├── context/                        # ContextIsolation + Compressor
│   ├── skills/                         # SkillManager + Skill tools
│   ├── hitl/                           # 风险策略
│   ├── llm/                            # DeepSeek / Ark / Mock
│   └── metrics/                        # 指标与 benchmark 逻辑
├── benchmarks/
│   ├── run_planner_benchmark.py
│   ├── run_skill_token_benchmark.py
│   ├── run_real_llm_benchmark.py
│   └── real_cases/
├── demo/
│   ├── run_demo.py
│   └── buggy_project/
├── results/
│   ├── demo_cases/
│   ├── planner_benchmark_summary.md
│   ├── skill_token_benchmark_summary.md
│   └── real_llm_benchmark/
├── tests/
├── requirements.txt
├── requirements-optional.txt
├── .env.example
└── README.md
```

---

## 组件一览

| 组件 | 文件 | 功能 |
| --- | --- | --- |
| FastAPI API | `app/main.py` | `/bugfix` 和 `/health` |
| RulePlanner | `app/planner/rule_planner.py` | 关键词与 mode 决策 |
| LLMFallbackPlanner | `app/planner/llm_fallback.py` | 模糊任务分类 |
| AgentRegistry | `app/agents/registry.py` | YAML 配置加载 |
| BaseAgent | `app/agents/base.py` | ReAct 风格工具调用 |
| Workflow | `app/graph/workflow.py` | 默认顺序编排 |
| LangGraph Adapter | `app/graph/langgraph_workflow.py` | 可选 StateGraph 适配 |
| ToolRegistry | `app/tools/base.py` | 工具注册与调用 |
| MCPClient/MCPServer | `app/mcp/` | MCP-style 工具协议边界 |
| ContextManager | `app/context/manager.py` | fork / merge / cleanup |
| ContextCompressor | `app/context/compressor.py` | token 预算压缩 |
| SkillManager | `app/skills/manager.py` | frontmatter + 按需加载 |
| HITL Policy | `app/hitl/policy.py` | 高风险编辑拦截 |
| LLM Providers | `app/llm/` | DeepSeek / Ark / Mock |
| MetricsTracker | `app/metrics/tracker.py` | 调用、token、timeout 指标 |
| Real Benchmark | `benchmarks/run_real_llm_benchmark.py` | fixed vs dynamic 真实 LLM 对比 |

---

## 技术栈

| 层级 | 技术 |
| --- | --- |
| API | FastAPI, Uvicorn |
| Agent 配置 | YAML, AgentRegistry |
| 编排 | Sequential Workflow, optional LangGraph |
| 工具协议 | MCP-style Tool Schema |
| LLM | DeepSeek, Volcengine Ark, Mock |
| 上下文 | ContextIsolation, ContextCompressor, tiktoken |
| Skill | Markdown frontmatter, local persistence |
| 安全 | HITL risk policy |
| Benchmark | CSV, JSONL, Markdown summary |
| 测试 | unittest |

