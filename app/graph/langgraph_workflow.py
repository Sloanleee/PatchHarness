from __future__ import annotations

from typing import Any, TypedDict

from app.schemas import BugfixRequest, BugfixResponse


class _PatchHarnessState(TypedDict, total=False):
    request: BugfixRequest
    response: BugfixResponse
    events: list[dict[str, Any]]


class LangGraphBugfixWorkflow:
    """LangGraph adapter for PatchHarness.

    The adapter keeps the stable sequential implementation as the execution
    engine while running it through a real LangGraph StateGraph boundary. Later
    phases can split `execute_agents` into per-Agent dynamic nodes without
    changing the public workflow API.
    """

    def __init__(self, workflow: Any) -> None:
        self.workflow = workflow
        self.graph = self._compile_graph()

    def run(self, request: BugfixRequest) -> BugfixResponse:
        state = self.graph.invoke({"request": request, "events": []})
        response = state["response"]
        response.planning["langgraph"] = {
            "enabled": True,
            "nodes": ["validate", "execute_agents"],
        }
        return response

    def _compile_graph(self):
        try:
            from langgraph.graph import END, StateGraph
        except ModuleNotFoundError as exc:
            raise RuntimeError("Install langgraph to use LangGraphBugfixWorkflow") from exc

        graph = StateGraph(_PatchHarnessState)
        graph.add_node("validate", self._validate)
        graph.add_node("execute_agents", self._execute_agents)
        graph.set_entry_point("validate")
        graph.add_edge("validate", "execute_agents")
        graph.add_edge("execute_agents", END)
        return graph.compile()

    @staticmethod
    def _validate(state: _PatchHarnessState) -> _PatchHarnessState:
        events = list(state.get("events", []))
        events.append({"event": "langgraph_validate"})
        return {**state, "events": events}

    def _execute_agents(self, state: _PatchHarnessState) -> _PatchHarnessState:
        events = list(state.get("events", []))
        events.append({"event": "langgraph_execute_agents"})
        response = self.workflow._run_sequential(state["request"])
        response.planning["langgraph_events"] = events
        return {**state, "events": events, "response": response}

