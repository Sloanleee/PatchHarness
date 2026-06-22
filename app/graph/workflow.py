from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from app.agents import AgentRegistry, BaseAgent
from app.context import ContextCompressor, ContextManager
from app.hitl import HitlPolicy
from app.llm import LLMClient, create_llm_client
from app.mcp import MCPClient, MCPServer
from app.metrics import MetricsTracker
from app.planner import LLMFallbackPlanner, RulePlanner
from app.schemas import BugfixRequest, BugfixResponse
from app.skills import SkillManager
from app.tools.file_tools import create_default_tools


class BugfixWorkflow:
    """MVP workflow.

    This class gives us a LangGraph-shaped orchestration boundary while keeping
    the first implementation dependency-light. A later version can replace
    `run` internals with real LangGraph nodes without changing callers.
    """

    def __init__(
        self,
        registry: AgentRegistry,
        planner: RulePlanner | None = None,
        llm_client: LLMClient | None = None,
    ) -> None:
        self.registry = registry
        self.planner = planner or RulePlanner()
        self.llm_client = llm_client

    @classmethod
    def from_default_configs(cls) -> "BugfixWorkflow":
        config_dir = Path(__file__).resolve().parents[1] / "agents" / "configs"
        return cls(AgentRegistry.load_from_dir(config_dir))

    def run(self, request: BugfixRequest) -> BugfixResponse:
        if request.use_langgraph:
            try:
                from app.graph.langgraph_workflow import LangGraphBugfixWorkflow

                return LangGraphBugfixWorkflow(self).run(request)
            except RuntimeError:
                pass
        return self._run_sequential(request)

    def _run_sequential(self, request: BugfixRequest) -> BugfixResponse:
        workspace = Path(request.workspace_path).resolve()
        if not workspace.exists() or not workspace.is_dir():
            raise ValueError(f"workspace_path must be an existing directory: {workspace}")

        metrics = MetricsTracker(planned_by="rule")
        llm_client = self._resolve_llm_client(request)
        planning = self.planner.plan(request)
        if planning.needs_fallback and llm_client is not None:
            planning = LLMFallbackPlanner(
                llm_client,
                metrics=metrics,
                confidence_threshold=request.planning_confidence_threshold,
            ).plan(request)
        metrics.metrics.planned_by = planning.planned_by

        if planning.requires_human_approval:
            return BugfixResponse(
                request_id=str(uuid4()),
                planned_agents=planning.agents,
                agent_reports=[],
                changed_files=[],
                test_result=None,
                metrics=metrics.snapshot(),
                final_summary="Workflow paused for planning-stage human approval.",
                requires_human_approval=True,
                planning=_planning_payload(planning),
                approval_events=[planning.approval_event] if planning.approval_event else [],
            )

        context_manager = ContextManager.from_request(request, planning.agents)
        skill_manager = SkillManager.from_default_dir()
        tools = create_default_tools(metrics=metrics, skill_manager=skill_manager)
        mcp_client = MCPClient(MCPServer(tools), metrics=metrics)
        hitl_policy = HitlPolicy()
        compressor = ContextCompressor(
            llm_client=llm_client if request.enable_llm else None,
            metrics=metrics,
        )

        reports = []
        for agent_name in planning.agents:
            config = self.registry.get(agent_name)
            agent_context = context_manager.fork(agent_name)
            metrics.context_forked()
            agent = BaseAgent(
                config,
                mcp_client,
                skill_manager=skill_manager,
                hitl_policy=hitl_policy,
                metrics=metrics,
                llm_client=llm_client if request.enable_llm else None,
            )
            metrics.agent_called()
            report = agent.run(request, prior_reports=reports, context=agent_context)
            if compressor.maybe_compress_report(report):
                metrics.compressed()
            merge_event = context_manager.merge(agent_context, report)
            metrics.context_merged()
            cleanup_event = context_manager.cleanup(agent_context)
            metrics.context_cleaned()
            report.context_events.extend([merge_event, cleanup_event])
            reports.append(report)
            if report.requires_human_approval or report.status == "failed":
                break

        changed_files = sorted({path for report in reports for path in report.changed_files})
        test_result = next(
            (report.test_result for report in reversed(reports) if report.test_result is not None),
            None,
        )
        final_summary = _build_final_summary(planning.agents, reports)
        requires_human_approval = any(report.requires_human_approval for report in reports)

        return BugfixResponse(
            request_id=str(uuid4()),
            planned_agents=planning.agents,
            agent_reports=reports,
            changed_files=changed_files,
            test_result=test_result,
            metrics=metrics.snapshot(),
            final_summary=final_summary,
            requires_human_approval=requires_human_approval,
            planning=_planning_payload(planning),
            approval_events=[
                event
                for report in reports
                for event in report.hitl_events
            ],
        )

    def _resolve_llm_client(self, request: BugfixRequest) -> LLMClient | None:
        if self.llm_client is not None:
            return self.llm_client
        if not request.enable_llm:
            return None
        try:
            return create_llm_client()
        except (RuntimeError, ValueError):
            return None


def _build_final_summary(planned_agents: list[str], reports: list) -> str:
    failed = [report.agent_name for report in reports if report.status == "failed"]
    changed = sorted({path for report in reports for path in report.changed_files})
    status = "failed" if failed else "completed"
    parts = [
        f"Workflow {status}.",
        f"Planned agents: {', '.join(planned_agents)}.",
    ]
    if changed:
        parts.append(f"Changed files: {', '.join(changed)}.")
    if failed:
        parts.append(f"Failed agents: {', '.join(failed)}.")
    if any(getattr(report, "requires_human_approval", False) for report in reports):
        parts.append("Workflow paused for human approval.")
    return " ".join(parts)


def _planning_payload(planning) -> dict:
    return {
        "planned_by": planning.planned_by,
        "reason": planning.reason,
        "confidence": planning.confidence,
        "needs_fallback": planning.needs_fallback,
        "requires_human_approval": planning.requires_human_approval,
    }
