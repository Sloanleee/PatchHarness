from __future__ import annotations

import json
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


_SOURCE_EXTENSIONS = {".py", ".pyx", ".c", ".cpp", ".h", ".java", ".js", ".ts", ".rs", ".go"}
_NON_SOURCE_PARTS = {"doc", "docs", "test", "tests", ".git", ".venv", "venv", "venv311", "__pycache__"}


def _is_source_path(value: str) -> bool:
    path = Path(value)
    lowered = {part.lower() for part in path.parts}
    return (
        path.suffix.lower() in _SOURCE_EXTENSIONS
        and not lowered.intersection(_NON_SOURCE_PARTS)
        and not path.name.lower().startswith("test_")
    )


def _build_diagnostic_evidence(report: AgentReport) -> dict[str, Any]:
    locations: list[dict[str, Any]] = []
    ranges: list[dict[str, Any]] = []
    seen_locations: set[tuple[str, int]] = set()
    for observation in report.observations:
        if not observation.get("ok"):
            continue
        data = observation.get("data") or {}
        if observation.get("tool") == "grep_search":
            for match in data.get("matches", []):
                path = str(match.get("path", ""))
                key = (path, int(match.get("line", 0)))
                if _is_source_path(path) and key not in seen_locations:
                    seen_locations.add(key)
                    if len(locations) < 20:
                        locations.append({"path": path, "line": key[1], "text": str(match.get("text", ""))[:500]})
        elif observation.get("tool") == "read_file":
            path = str(data.get("path", ""))
            if _is_source_path(path) and len(ranges) < 20:
                ranges.append({"path": path, "start_line": data.get("start_line"), "end_line": data.get("end_line")})
    last_thought = report.thoughts[-1] if report.thoughts else ""
    return {
        "status": "partial",
        "reason": "max_iterations_exhausted",
        "candidate_locations": locations,
        "read_ranges": ranges,
        "last_thought": last_thought[:1000],
        "remaining_work": ("Continue from the candidate source locations and complete the minimal patch." if locations or ranges else "No valid source evidence was found."),
    }


def _compact_prior_reports(reports: list[AgentReport], max_chars: int = 6000) -> str:
    sections: list[str] = []
    for item in reports:
        if item.diagnostic_evidence:
            sections.append(
                f"[{item.agent_name}] "
                + json.dumps(item.diagnostic_evidence, ensure_ascii=False, separators=(",", ":"))
            )
            continue
        observations: list[str] = []
        for observation in item.observations[-4:]:
            data = observation.get("data") or {}
            if isinstance(data, dict):
                path = data.get("path")
                matches = data.get("matches")
                if path:
                    observations.append(f"read: {path}")
                if isinstance(matches, list):
                    for match in matches[:5]:
                        observations.append(
                            f"match: {match.get('path')}:{match.get('line')} {match.get('text', '')}"
                        )
        section = f"[{item.agent_name}] {item.summary}"
        if observations:
            section += "\n" + "\n".join(observations)
        sections.append(section)
    return "\n".join(sections)[:max_chars]


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
        elif self.config.name == "root_cause_analysis":
            report = self._run_root_cause_analysis(request, workspace)
        elif self.config.name == "patch_generation":
            report = self._run_patch_generation(request, workspace, prior_reports)
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

    def _run_root_cause_analysis(self, request: BugfixRequest, workspace: Path) -> AgentReport:
        report = AgentReport(self.config.name, "completed")
        self._prepare_skills(report)
        edit_plan = _parse_edit_plan(request.task_description)
        if edit_plan is not None:
            self._record_thought(
                report,
                f"Detected explicit edit target {edit_plan['path']}; reading it for root cause context.",
            )
            result = self._run_tool(report, workspace, "read_file", path=edit_plan["path"])
            if not result.ok:
                report.status = "failed"
                report.summary = f"Root cause analysis could not read target file: {result.error}"
                return report
            report.summary = (
                f"Likely root cause is in {edit_plan['path']}: "
                f"the code contains `{edit_plan['old']}` and should use `{edit_plan['new']}`."
            )
            return report

        if self.llm_client is not None:
            return self._run_llm_react(request, workspace, report)

        query = _best_query(request.task_description)
        self._record_thought(report, f"Searching for likely root cause with query: {query}")
        result = self._run_tool(report, workspace, "grep_search", query=query, max_results=10)
        matches = result.data.get("matches", []) if result.ok else []
        if matches:
            first_path = str(matches[0]["path"])
            self._record_thought(report, f"Reading first likely source file: {first_path}")
            self._run_tool(report, workspace, "read_file", path=first_path)
            report.summary = f"Found {len(matches)} possible root-cause location(s); first candidate: {first_path}."
        else:
            report.summary = "No clear root-cause location found from deterministic search."
        return report

    def _run_patch_generation(
        self,
        request: BugfixRequest,
        workspace: Path,
        prior_reports: list[AgentReport],
    ) -> AgentReport:
        return self._run_bug_fix(request, workspace, prior_reports)

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

    def _run_bug_fix(
        self,
        request: BugfixRequest,
        workspace: Path,
        prior_reports: list[AgentReport] | None = None,
    ) -> AgentReport:
        report = AgentReport(self.config.name, "completed")
        self._prepare_skills(report)
        edit_plan = _parse_edit_plan(request.task_description)

        if edit_plan is None:
            if self.llm_client is not None:
                return self._run_llm_react(request, workspace, report, prior_reports or [])
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
        prior_reports: list[AgentReport] | None = None,
    ) -> AgentReport:
        self._record_thought(report, "未识别到明确替换指令，进入 LLM ReAct 决策。")
        prior_reports = prior_reports or []
        diagnosis = _compact_prior_reports(prior_reports)
        task_content = request.task_description
        if diagnosis:
            task_content += (
                "\n\nPrior diagnostic evidence (use this before searching again):\n"
                + diagnosis
            )
        messages = [
            {
                "role": "system",
                "content": (
                    f"You are {self.config.name}. Return one JSON object only. "
                    "Required schema: "
                    "{\"thought\": string, \"action\": string|null, "
                    "\"action_input\": object, \"final\": string|null}. "
                    "Never return boolean final. Never return string/list action_input. "
                    "When calling read_file use a narrow range: "
                    "{\"path\": \"...\", \"start_line\": 1, \"end_line\": 200}. "
                    "For grep_search, use path to scope a file/subtree and set regex=true "
                    "when pattern contains regular-expression syntax. "
                    "When calling edit_file use {\"path\": \"...\", \"old\": \"...\", \"new\": \"...\"}. "
                    f"Allowed actions: {', '.join(self.config.tools)}. "
                    "Use final only after the necessary tool actions are complete. "
                    + (
                        "Investigate only; do not edit. Final must name the likely file, "
                        "symbol, and concrete root cause."
                        if self.config.name == "root_cause_analysis"
                        else "Prioritize reaching edit_file; do not repeat completed diagnosis."
                    )
                ),
            },
            {"role": "user", "content": task_content},
        ]
        read_only_actions = 0
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
                if self.config.name in {"bug_fix", "patch_generation"} and not report.changed_files:
                    report.status = "failed"
                    report.summary = (
                        "LLM returned final before producing a code change; "
                        f"{self.config.name} requires edit_file success or explicit changed_files."
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
            if self.config.name == "patch_generation" and action.action in {"grep_search", "read_file"}:
                read_only_actions += 1
            if result.ok and action.action == "edit_file" and "path" in result.data:
                report.changed_files.append(str(result.data["path"]))
                self._run_tool(report, workspace, "git_diff", path=str(result.data["path"]))
            messages.append({"role": "assistant", "content": response.content})
            observation_text = f"Observation: ok={result.ok}, data={result.data}, error={result.error}"
            if self.config.name == "patch_generation" and read_only_actions >= 4 and not report.changed_files:
                directive = (
                    "The read-only investigation budget is exhausted. Your next action must be "
                    "edit_file, or final with a concrete explanation of why no safe edit is possible."
                )
                observation_text += "\n" + directive
                if directive not in report.thoughts:
                    self._record_thought(report, directive)
            messages.append({"role": "user", "content": observation_text})
            if report.requires_human_approval:
                report.status = "failed"
                report.summary = "LLM ReAct paused for human approval."
                return report
        evidence = _build_diagnostic_evidence(report)
        report.diagnostic_evidence = evidence
        report.evidence_count = len(evidence["candidate_locations"]) + len(evidence["read_ranges"])
        report.stop_reason = "max_iterations_exhausted"
        report.status = "partial" if self.config.name == "root_cause_analysis" and report.evidence_count else "failed"
        report.summary = "LLM ReAct reached max_iterations with partial source evidence." if report.status == "partial" else "LLM ReAct reached max_iterations without final answer."
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
