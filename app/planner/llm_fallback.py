from __future__ import annotations

from app.llm import LLMClient
from app.metrics import MetricsTracker
from app.planner.rule_planner import PlanningResult
from app.schemas import BugfixRequest


class LLMFallbackPlanner:
    CATEGORY_TO_AGENTS = {
        "review": ["code_review"],
        "fix": ["root_cause_analysis", "patch_generation", "test_verify"],
        "full": ["code_review", "root_cause_analysis", "patch_generation", "test_verify", "summary"],
    }

    def __init__(
        self,
        llm_client: LLMClient,
        metrics: MetricsTracker | None = None,
        confidence_threshold: float = 0.65,
    ) -> None:
        self.llm_client = llm_client
        self.metrics = metrics
        self.confidence_threshold = confidence_threshold

    def plan(self, request: BugfixRequest) -> PlanningResult:
        response = self.llm_client.complete_json(
            [
                {
                    "role": "system",
                    "content": (
                        "Classify a bugfix automation request. Return JSON only: "
                        "{\"thought\": string, \"category\": \"review|fix|full|unknown\", "
                        "\"confidence\": number}."
                    ),
                },
                {"role": "user", "content": request.task_description},
            ],
            temperature=0.0,
        )
        if self.metrics is not None:
            self.metrics.llm_called(response.prompt_tokens, response.completion_tokens)
            self.metrics.llm_fallback_used()

        import json

        data = json.loads(response.content)
        category = str(data.get("category", "unknown")).lower()
        confidence = float(data.get("confidence", 0.0))
        agents = self.CATEGORY_TO_AGENTS.get(category, ["code_review"])
        requires_approval = confidence < self.confidence_threshold or category == "unknown"
        approval_event = None
        if requires_approval:
            approval_event = {
                "event": "planning_human_approval_required",
                "risk": "medium",
                "reason": "LLM fallback planner confidence is below threshold",
                "category": category,
                "confidence": confidence,
                "threshold": self.confidence_threshold,
            }
            if self.metrics is not None:
                self.metrics.planner_hitl_interrupted()

        return PlanningResult(
            agents=agents,
            planned_by="llm_fallback",
            reason=str(data.get("thought", "LLM fallback classification")),
            confidence=confidence,
            needs_fallback=False,
            requires_human_approval=requires_approval,
            approval_event=approval_event,
        )
