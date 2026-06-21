# PatchHarness

面向自动化 Bug 修复的生产级 Multi-Agent Harness 系统。

## 本地运行

安装依赖：

```bash
python -m pip install -r requirements.txt
```

运行测试：

```bash
python -m unittest discover -s tests
```

运行 demo：

```bash
python demo/run_demo.py
```

生成 Benchmark 证据：

```bash
python benchmarks/run_planner_benchmark.py
python benchmarks/generate_demo_evidence.py
```

启动 FastAPI：

```bash
uvicorn app.main:app --reload
```

接口：

```text
GET /health
POST /bugfix
```

