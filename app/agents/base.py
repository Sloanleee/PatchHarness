from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from app.agents.registry import AgentConfig
from app.context import AgentContext
from app.hitl import HitlPolicy
from app.llm import LLMClient
from app.metrics import MetricsTracker
from app.schemas import AgentReport, BugfixRequest, ToolResult
from app.skills import SkillManager
from app.tools.base import ToolRegistry


class BaseAgent:
    def __init__(
        self,
        config: AgentConfig,
        tools: ToolRegistry,
        skill_manager: SkillManager | None = None,
        hitl_policy: HitlPolicy | None = None,
        metrics: MetricsTracker | None = None,
        llm_client: LLMClient | None = None,
    ) -> None:
        self.config = config
        self.tools = tools
        self.skill_manager = skill_manager
        self.hitl_policy = hitl_policy
        self.metrics = metrics
        self.llm_client = llm_client

    def run(
        self,
        request: BugfixRequest,
        prior_reports: list[AgentReport] | None = None,
        context: AgentContext | None = None,
    ) -> AgentReport:
        prior_reports = prior_reports or []
        workspace = Path(request.workspace_path).resolve()

        if self.config.name == "code_review":
            report = self._run_code_review(request, workspace)
        elif self.config.name == "bug_fix":
            report = self._run_bug_fix(request, workspace)
        elif self.config.name == "test_verify":
            report = self._run_test_verify(request, workspace)
        elif self.config.name == "summary":
            report = self._run_summary(prior_reports, workspace)
        else:
            report = AgentReport(
                self.config.name,
                "skipped",
                summary=f"No deterministic runner implemented for agent: {self.config.name}",
            )

        if context is not None:
            report.context_events.extend(context.events)
        return report

    def _run_code_review(self, request: BugfixRequest, workspace: Path) -> AgentReport:
        report = AgentReport(self.config.name, "completed")
        self._prepare_skills(report)
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
        self._prepare_skills(report)
        edit_plan = _parse_edit_plan(request.task_description)

        if edit_plan is None:
            if self.llm_client is not None:
                return self._run_llm_react(request, workspace, report)
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

    def _run_llm_react(
        self,
        request: BugfixRequest,
        workspace: Path,
        report: AgentReport,
    ) -> AgentReport:
        self._record_thought(report, "未识别到明确替换指令，进入 LLM ReAct 决策。")
        messages = [
            {
                "role": "system",
                "content": (
                    f"You are {self.config.name}. Return one JSON object only. "
                    "Required schema: "
                    "{\"thought\": string, \"action\": string|null, "
                    "\"action_input\": object, \"final\": string|null}. "
                    "Never return boolean final. Never return string/list action_input. "
                    "When calling read_file use {\"path\": \"...\"}. "
                    "When calling edit_file use {\"path\": \"...\", \"old\": \"...\", \"new\": \"...\"}. "
                    f"Allowed actions: {', '.join(self.config.tools)}. "
                    "Use final only after the necessary tool actions are complete."
                ),
            },
            {"role": "user", "content": request.task_description},
        ]
        for _ in range(self.config.max_iterations):
            try:
                response = self.llm_client.complete_json(messages, temperature=self.config.temperature)
            except Exception as exc:
                if _is_timeout_error(exc):
                    if self.metrics is not None:
                        self.metrics.llm_timed_out()
                    report.status = "failed"
                    report.summary = f"LLM request timed out: {exc}"
                    return report
                raise
            if self.metrics is not None:
                self.metrics.llm_called(response.prompt_tokens, response.completion_tokens)
            try:
                action = response.to_action()
            except ValueError as exc:
                report.status = "failed"
                report.summary = f"LLM action parse failed: {exc}"
                return report
            if action.thought:
                self._record_thought(report, action.thought)
            if action.final:
                if self.config.name == "bug_fix" and not report.changed_files:
                    report.status = "failed"
                    report.summary = (
                        "LLM returned final before producing a code change; "
                        "bug_fix requires edit_file success or explicit changed_files."
                    )
                    return report
                report.summary = action.final
                return report
            if not action.action:
                report.summary = "LLM stopped without an action."
                return report
            action_input = dict(action.action_input)
            if action.action == "edit_file":
                action_input.setdefault("allow_edit", request.allow_edit)
            result = self._run_tool(report, workspace, action.action, **action_input)
            if result.ok and action.action == "edit_file" and "path" in result.data:
                report.changed_files.append(str(result.data["path"]))
                self._run_tool(report, workspace, "git_diff", path=str(result.data["path"]))
            messages.append({"role": "assistant", "content": response.content})
            messages.append(
                {
                    "role": "user",
                    "content": f"Observation: ok={result.ok}, data={result.data}, error={result.error}",
                }
            )
            if report.requires_human_approval:
                report.status = "failed"
                report.summary = "LLM ReAct paused for human approval."
                return report
        report.status = "failed"
        report.summary = "LLM ReAct reached max_iterations without final answer."
        return report

    def _run_test_verify(self, request: BugfixRequest, workspace: Path) -> AgentReport:
        report = AgentReport(self.config.name, "completed")
        self._prepare_skills(report)
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
        self._prepare_skills(report)
        self._record_thought(report, "汇总前序 Agent 报告，并查看当前 diff")
        self._run_tool(report, workspace, "git_diff")
        completed = sum(1 for item in prior_reports if item.status == "completed")
        failed = sum(1 for item in prior_reports if item.status == "failed")
        report.summary = f"前序 Agent 完成 {completed} 个，失败 {failed} 个。"
        return report

    def _run_tool(self, report: AgentReport, workspace: Path, name: str, **kwargs: Any) -> ToolResult:
        dynamic_skill_tools = {"search_skill", "download_skill", "create_skill", "update_skill"}
        if name not in self.config.tools and name not in dynamic_skill_tools:
            result = ToolResult(name, False, error=f"Tool not allowed for {self.config.name}: {name}")
        elif self.hitl_policy is not None:
            decision = self.hitl_policy.evaluate_tool_call(name, kwargs)
            if decision.requires_approval:
                event = decision.to_event(name, kwargs)
                report.requires_human_approval = True
                report.hitl_events.append(event)
                if self.metrics is not None:
                    self.metrics.hitl_interrupted()
                result = ToolResult(
                    name,
                    False,
                    data={"requires_human_approval": True, "hitl_event": event},
                    error=decision.reason,
                )
            else:
                result = self.tools.run(name, workspace, **kwargs)
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

    def _prepare_skills(self, report: AgentReport) -> None:
        if self.skill_manager is None:
            return
        frontmatter = self.skill_manager.public_frontmatter()
        report.skills_available = frontmatter
        if self.metrics is not None:
            self.metrics.skills_disclosed(len(frontmatter))

        skill_name = self.skill_manager.choose_for_agent(self.config.name)
        if skill_name is None:
            return
        content = self.skill_manager.load_skill(skill_name)
        report.skills_loaded.append(skill_name)
        self._record_thought(
            report,
            f"按需加载 Skill：{skill_name}（{len(content)} 字符），未把全部 Skill 一次性注入。",
        )
        if self.metrics is not None:
            self.metrics.skill_loaded()


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


def _is_timeout_error(exc: Exception) -> bool:
    name = exc.__class__.__name__.lower()
    message = str(exc).lower()
    return "timeout" in name or "timed out" in message
