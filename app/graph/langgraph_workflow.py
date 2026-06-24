from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, TypedDict
from uuid import uuid4

from app.agents import BaseAgent
from app.checkpoints import CheckpointStore
from app.context import ContextCompressor, ContextManager
from app.graph.workflow import _build_final_summary, _planning_payload
from app.hitl import HitlPolicy
from app.mcp import MCPClient, MCPServer
from app.metrics import MetricsTracker
from app.planner import LLMFallbackPlanner
from app.schemas import AgentReport, BugfixRequest, BugfixResponse, WorkflowMetrics
from app.skills import SkillManager
from app.tools.file_tools import create_default_tools


AGENT_NODES = (
    "code_review",
    "root_cause_analysis",
    "patch_generation",
    "bug_fix",
    "test_verify",
    "summary",
)


class _PatchHarnessState(TypedDict, total=False):
    request: BugfixRequest
    workspace: Path
    metrics: MetricsTracker
    llm_client: Any
    planning: Any
    context_manager: ContextManager
    skill_manager: SkillManager
    mcp_client: MCPClient
    hitl_policy: HitlPolicy
    compressor: ContextCompressor
    reports: list[AgentReport]
    response: BugfixResponse
    events: list[dict[str, Any]]
    executed_nodes: list[str]
    run_id: str
    pending_approval: dict[str, Any] | None
    approval_decision: dict[str, Any] | None
    resume_from: str | None


class LangGraphBugfixWorkflow:
    """LangGraph StateGraph implementation for PatchHarness agent orchestration."""

    def __init__(
        self,
        workflow: Any,
        checkpoint_store: CheckpointStore | None = None,
    ) -> None:
        self.workflow = workflow
        self.checkpoint_store = (
            checkpoint_store
            or getattr(workflow, "checkpoint_store", None)
            or CheckpointStore()
        )
        self.graph = self._compile_graph()

    def run(self, request: BugfixRequest) -> BugfixResponse:
        run_id = f"run_{uuid4().hex}"
        state = self.graph.invoke(
            {
                "run_id": run_id,
                "request": request,
                "events": [],
                "reports": [],
                "executed_nodes": [],
                "pending_approval": None,
                "approval_decision": None,
                "resume_from": None,
            }
        )
        response = state["response"]
        response.run_id = state.get("run_id")
        self._attach_langgraph_metadata(response, state)
        return response

    def _compile_graph(self):
        try:
            from langgraph.graph import END, StateGraph
        except ModuleNotFoundError as exc:
            raise RuntimeError("Install langgraph to use LangGraphBugfixWorkflow") from exc

        graph = StateGraph(_PatchHarnessState)
        graph.add_node("validate_workspace", self._validate_workspace)
        graph.add_node("plan_agents", self._plan_agents)
        for agent_name in AGENT_NODES:
            graph.add_node(agent_name, self._make_agent_node(agent_name))
        graph.add_node("hitl_pause", self._hitl_pause)
        graph.add_node("build_response", self._build_response)

        route_targets = {agent_name: agent_name for agent_name in AGENT_NODES}
        route_targets["hitl_pause"] = "hitl_pause"
        route_targets["build_response"] = "build_response"

        graph.set_entry_point("validate_workspace")
        graph.add_edge("validate_workspace", "plan_agents")
        graph.add_conditional_edges("plan_agents", self._route_next, route_targets)
        for agent_name in AGENT_NODES:
            graph.add_conditional_edges(agent_name, self._route_after_agent, route_targets)
        graph.add_edge("hitl_pause", END)
        graph.add_edge("build_response", END)
        return graph.compile()

    @staticmethod
    def _append_event(
        state: _PatchHarnessState,
        event: str,
        **payload: Any,
    ) -> list[dict[str, Any]]:
        events = list(state.get("events", []))
        events.append({"event": event, **payload})
        return events

    def _validate_workspace(self, state: _PatchHarnessState) -> _PatchHarnessState:
        request = state["request"]
        workspace = Path(request.workspace_path).resolve()
        if not workspace.exists() or not workspace.is_dir():
            raise ValueError(f"workspace_path must be an existing directory: {workspace}")
        return {
            **state,
            "workspace": workspace,
            "events": self._append_event(state, "langgraph_node", node="validate_workspace"),
            "executed_nodes": [*state.get("executed_nodes", []), "validate_workspace"],
        }

    def _plan_agents(self, state: _PatchHarnessState) -> _PatchHarnessState:
        request = state["request"]
        metrics = MetricsTracker(planned_by="rule")
        llm_client = self.workflow._resolve_llm_client(request)
        planning = self.workflow.planner.plan(request)
        if planning.needs_fallback and llm_client is not None:
            planning = LLMFallbackPlanner(
                llm_client,
                metrics=metrics,
                confidence_threshold=request.planning_confidence_threshold,
            ).plan(request)
        metrics.metrics.planned_by = planning.planned_by

        updates: _PatchHarnessState = {
            **state,
            "metrics": metrics,
            "llm_client": llm_client,
            "planning": planning,
            "events": self._append_event(
                state,
                "langgraph_node",
                node="plan_agents",
                planned_agents=list(planning.agents),
            ),
            "executed_nodes": [*state.get("executed_nodes", []), "plan_agents"],
        }
        if planning.requires_human_approval:
            return updates

        skill_manager = SkillManager.from_default_dir()
        tools = create_default_tools(metrics=metrics, skill_manager=skill_manager)
        updates.update(
            {
                "context_manager": ContextManager.from_request(request, planning.agents),
                "skill_manager": skill_manager,
                "mcp_client": MCPClient(MCPServer(tools), metrics=metrics),
                "hitl_policy": HitlPolicy(),
                "compressor": ContextCompressor(
                    llm_client=llm_client if request.enable_llm else None,
                    metrics=metrics,
                ),
            }
        )
        return updates

    def _make_agent_node(self, agent_name: str):
        def _run_agent(state: _PatchHarnessState) -> _PatchHarnessState:
            planning = state["planning"]
            if agent_name not in planning.agents:
                return state

            metrics = state["metrics"]
            context_manager = state["context_manager"]
            request = state["request"]
            workspace = state["workspace"]
            reports = list(state.get("reports", []))

            config = self.workflow.registry.get(agent_name)
            agent_context = context_manager.fork(agent_name)
            metrics.context_forked()
            agent = BaseAgent(
                config,
                state["mcp_client"],
                skill_manager=state["skill_manager"],
                hitl_policy=state["hitl_policy"],
                metrics=metrics,
                llm_client=state["llm_client"] if request.enable_llm else None,
            )
            metrics.agent_called()
            report = agent.run(request, prior_reports=reports, context=agent_context)
            if state["compressor"].maybe_compress_report(report):
                metrics.compressed()
            merge_event = context_manager.merge(agent_context, report)
            metrics.context_merged()
            cleanup_event = context_manager.cleanup(agent_context)
            metrics.context_cleaned()
            report.context_events.extend([merge_event, cleanup_event])
            reports.append(report)

            return {
                **state,
                "reports": reports,
                "events": self._append_event(
                    state,
                    "langgraph_node",
                    node=agent_name,
                    status=report.status,
                ),
                "executed_nodes": [*state.get("executed_nodes", []), agent_name],
            }

        return _run_agent

    @staticmethod
    def _route_next(state: _PatchHarnessState) -> str:
        planning = state.get("planning")
        if planning is None:
            return "build_response"
        if planning.requires_human_approval:
            return "hitl_pause"
        reports = state.get("reports", [])
        if len(reports) >= len(planning.agents):
            return "build_response"
        next_agent = planning.agents[len(reports)]
        return next_agent if next_agent in AGENT_NODES else "build_response"

    def _route_after_agent(self, state: _PatchHarnessState) -> str:
        reports = state.get("reports", [])
        if reports:
            last_report = reports[-1]
            if last_report.requires_human_approval:
                return "hitl_pause"
            if last_report.status == "failed":
                return "build_response"
        return self._route_next(state)

    def _hitl_pause(self, state: _PatchHarnessState) -> _PatchHarnessState:
        pending_approval = self._pending_approval_from_state(state)
        updated_state: _PatchHarnessState = {
            **state,
            "pending_approval": pending_approval,
            "resume_from": pending_approval["resume_node"],
            "events": self._append_event(
                state,
                "hitl_pause",
                node="hitl_pause",
                reason=pending_approval["reason"],
                resume_node=pending_approval["resume_node"],
            ),
            "executed_nodes": [*state.get("executed_nodes", []), "hitl_pause"],
        }
        response_state = self._build_response(updated_state)
        response_state = {
            **response_state,
            "events": updated_state["events"],
            "executed_nodes": updated_state["executed_nodes"],
        }
        response = response_state["response"]
        response.run_id = updated_state["run_id"]
        response.requires_human_approval = True
        response.failure_reason = "hitl_pause"
        response.pending_approval = pending_approval
        response.final_summary = "Workflow paused for human approval."
        self._attach_langgraph_metadata(response, response_state)
        checkpoint_state: _PatchHarnessState = {
            **response_state,
            "pending_approval": pending_approval,
            "resume_from": pending_approval["resume_node"],
            "response": response,
        }
        self.checkpoint_store.save(self._checkpoint_payload(checkpoint_state, "paused"))
        return checkpoint_state

    def _build_response(self, state: _PatchHarnessState) -> _PatchHarnessState:
        planning = state["planning"]
        metrics = state["metrics"]
        reports = list(state.get("reports", []))

        if planning.requires_human_approval:
            response = BugfixResponse(
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
        else:
            changed_files = sorted({path for report in reports for path in report.changed_files})
            test_result = next(
                (report.test_result for report in reversed(reports) if report.test_result is not None),
                None,
            )
            response = BugfixResponse(
                request_id=str(uuid4()),
                planned_agents=planning.agents,
                agent_reports=reports,
                changed_files=changed_files,
                test_result=test_result,
                metrics=metrics.snapshot(),
                final_summary=_build_final_summary(planning.agents, reports),
                requires_human_approval=any(report.requires_human_approval for report in reports),
                planning=_planning_payload(planning),
                approval_events=[
                    event
                    for report in reports
                    for event in report.hitl_events
                ],
            )

        return {
            **state,
            "response": response,
            "events": self._append_event(state, "langgraph_node", node="build_response"),
            "executed_nodes": [*state.get("executed_nodes", []), "build_response"],
        }

    @staticmethod
    def _attach_langgraph_metadata(
        response: BugfixResponse,
        state: _PatchHarnessState,
    ) -> None:
        response.planning["langgraph"] = {
            "enabled": True,
            "nodes": list(state.get("executed_nodes", [])),
        }
        response.planning["langgraph_events"] = list(state.get("events", []))

    def _pending_approval_from_state(self, state: _PatchHarnessState) -> dict[str, Any]:
        reports = list(state.get("reports", []))
        last_report = reports[-1] if reports else None
        event = last_report.hitl_events[-1] if last_report and last_report.hitl_events else {}
        action = last_report.actions[-1] if last_report and last_report.actions else {}
        trigger_node = last_report.agent_name if last_report else "plan_agents"
        planning = state.get("planning")
        if not event and planning is not None:
            event = planning.approval_event or {}
        return {
            "approval_id": f"approval_{uuid4().hex}",
            "trigger_node": trigger_node,
            "resume_node": "test_verify",
            "reason": str(event.get("reason") or "Human approval required"),
            "risk": str(event.get("risk") or "high"),
            "action": action,
        }

    def _checkpoint_payload(
        self,
        state: _PatchHarnessState,
        status: str,
    ) -> dict[str, Any]:
        response = state.get("response")
        request = state["request"]
        planning = state.get("planning")
        metrics = state.get("metrics")
        return {
            "run_id": state["run_id"],
            "status": status,
            "request": asdict(request),
            "planning": _planning_payload(planning) if planning is not None else {},
            "reports": [asdict(report) for report in state.get("reports", [])],
            "metrics": (
                asdict(metrics.snapshot())
                if metrics is not None
                else asdict(WorkflowMetrics())
            ),
            "executed_nodes": list(state.get("executed_nodes", [])),
            "events": list(state.get("events", [])),
            "pending_approval": state.get("pending_approval"),
            "resume_from": state.get("resume_from"),
            "approval_decision": state.get("approval_decision"),
            "response": response.to_dict() if response is not None else None,
        }
