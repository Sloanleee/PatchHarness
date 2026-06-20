from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from app.agents.registry import AgentConfig
from app.schemas import AgentReport, BugfixRequest, ToolResult
from app.tools.base import ToolRegistry


class BaseAgent:
    def __init__(self, config: AgentConfig, tools: ToolRegistry) -> None:
        self.config = config
        self.tools = tools

    def run(
        self,
        request: BugfixRequest,
        prior_reports: list[AgentReport] | None = None,
    ) -> AgentReport:
        prior_reports = prior_reports or []
        workspace = Path(request.workspace_path).resolve()

        if self.config.name == "code_review":
            return self._run_code_review(request, workspace)
        if self.config.name == "bug_fix":
            return self._run_bug_fix(request, workspace)
        if self.config.name == "test_verify":
            return self._run_test_verify(request, workspace)
        if self.config.name == "summary":
            return self._run_summary(prior_reports, workspace)
        return AgentReport(
            self.config.name,
            "skipped",
            summary=f"No deterministic runner implemented for agent: {self.config.name}",
        )

    def _run_code_review(self, request: BugfixRequest, workspace: Path) -> AgentReport:
        report = AgentReport(self.config.name, "completed")
        query = _best_query(request.task_description)
        self._record_thought(report, f"先搜索与任务最相关的关键词：{query}")
        result = self._run_tool(report, workspace, "grep_search", query=query, max_results=10)

        matches = result.data.get("matches", []) if result.ok else []
        if matches:
            first_path = str(matches[0]["path"])
            self._record_thought(report, f"发现相关文件 {first_path}，读取内容做轻量审查")
            self._run_tool(report, workspace, "read_file", path=first_path)
            report.summary = f"找到 {len(matches)} 处相关匹配，建议优先检查 {first_path}。"
        else:
            report.summary = "未找到明显相关代码，建议补充更具体的报错、函数名或文件路径。"
        return report

    def _run_bug_fix(self, request: BugfixRequest, workspace: Path) -> AgentReport:
        report = AgentReport(self.config.name, "completed")
        edit_plan = _parse_edit_plan(request.task_description)

        if edit_plan is None:
            query = _best_query(request.task_description)
            self._record_thought(report, f"未识别到明确替换指令，先搜索关键词：{query}")
            result = self._run_tool(report, workspace, "grep_search", query=query, max_results=10)
            matches = result.data.get("matches", []) if result.ok else []
            if matches:
                report.summary = (
                    f"定位到 {len(matches)} 处候选位置，但 MVP 需要明确 old/new 替换指令才会自动编辑。"
                )
            else:
                report.summary = "未定位到候选位置，未执行编辑。"
            return report

        self._record_thought(report, f"识别到明确替换计划，准备修改 {edit_plan['path']}")
        read_result = self._run_tool(report, workspace, "read_file", path=edit_plan["path"])
        if not read_result.ok:
            report.status = "failed"
            report.summary = f"读取文件失败：{read_result.error}"
            return report

        edit_result = self._run_tool(
            report,
            workspace,
            "edit_file",
            path=edit_plan["path"],
            old=edit_plan["old"],
            new=edit_plan["new"],
            allow_edit=request.allow_edit,
        )
        if edit_result.ok:
            report.changed_files.append(edit_plan["path"])
            self._run_tool(report, workspace, "git_diff", path=edit_plan["path"])
            report.summary = f"已在 {edit_plan['path']} 完成 1 处替换。"
        else:
            report.status = "failed"
            report.summary = f"未完成编辑：{edit_result.error}"
        return report

    def _run_test_verify(self, request: BugfixRequest, workspace: Path) -> AgentReport:
        report = AgentReport(self.config.name, "completed")
        if not request.run_tests:
            report.status = "skipped"
            report.summary = "请求关闭了测试执行。"
            return report

        command = request.test_command or "python -m unittest discover -s tests"
        self._record_thought(report, f"运行测试命令验证修改：{command}")
        result = self._run_tool(report, workspace, "run_test", command=command)
        report.test_result = result.data
        if result.ok:
            report.summary = "测试命令执行成功。"
        else:
            report.status = "failed"
            report.summary = "测试命令执行失败，需查看 stdout/stderr。"
        return report

    def _run_summary(self, prior_reports: list[AgentReport], workspace: Path) -> AgentReport:
        report = AgentReport(self.config.name, "completed")
        self._record_thought(report, "汇总前序 Agent 报告，并查看当前 diff")
        self._run_tool(report, workspace, "git_diff")
        completed = sum(1 for item in prior_reports if item.status == "completed")
        failed = sum(1 for item in prior_reports if item.status == "failed")
        report.summary = f"前序 Agent 完成 {completed} 个，失败 {failed} 个。"
        return report

    def _run_tool(self, report: AgentReport, workspace: Path, name: str, **kwargs: Any) -> ToolResult:
        if name not in self.config.tools:
            result = ToolResult(name, False, error=f"Tool not allowed for {self.config.name}: {name}")
        else:
            result = self.tools.run(name, workspace, **kwargs)
        report.actions.append({"tool": name, "input": _safe_action_input(kwargs)})
        report.observations.append(
            {
                "tool": result.tool,
                "ok": result.ok,
                "data": result.data,
                "error": result.error,
            }
        )
        return result

    @staticmethod
    def _record_thought(report: AgentReport, thought: str) -> None:
        report.thoughts.append(thought)


def _best_query(text: str) -> str:
    inline_code = re.findall(r"`([^`]+)`", text)
    if inline_code:
        return inline_code[0]
    words = re.findall(r"[A-Za-z_][A-Za-z0-9_]{2,}", text)
    if words:
        return words[-1]
    return text.strip()[:20] or "TODO"


def _parse_edit_plan(text: str) -> dict[str, str] | None:
    patterns = [
        r"in\s+`(?P<path>[^`]+)`\s+replace\s+`(?P<old>[^`]+)`\s+with\s+`(?P<new>[^`]+)`",
        r"replace\s+`(?P<old>[^`]+)`\s+with\s+`(?P<new>[^`]+)`\s+in\s+`(?P<path>[^`]+)`",
        r"在\s*`(?P<path>[^`]+)`\s*中?将\s*`(?P<old>[^`]+)`\s*替换为\s*`(?P<new>[^`]+)`",
        r"将\s*`(?P<path>[^`]+)`\s*中的\s*`(?P<old>[^`]+)`\s*替换为\s*`(?P<new>[^`]+)`",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.groupdict()
    return None


def _safe_action_input(payload: dict[str, Any]) -> dict[str, Any]:
    safe = dict(payload)
    if "content" in safe:
        safe["content"] = "<omitted>"
    return safe

