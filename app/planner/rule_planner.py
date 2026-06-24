from __future__ import annotations

from dataclasses import dataclass

from app.schemas import BugfixRequest


@dataclass(slots=True)
class PlanningResult:
    agents: list[str]
    planned_by: str
    reason: str
    confidence: float = 1.0
    needs_fallback: bool = False
    requires_human_approval: bool = False
    approval_event: dict | None = None


class RulePlanner:
    """Zero-LLM planner for deterministic agent routing."""

    REVIEW_KEYWORDS = ("review", "inspect", "audit", "check", "审查", "检查")
    FIX_KEYWORDS = ("fix", "bug", "error", "failure", "repair", "修复", "报错", "失败", "异常")
    FULL_KEYWORDS = ("full", "comprehensive", "complete", "全面", "完整")

    FIX_CHAIN = ["root_cause_analysis", "patch_generation", "test_verify"]
    FULL_CHAIN = [
        "code_review",
        "root_cause_analysis",
        "patch_generation",
        "test_verify",
        "summary",
    ]

    def plan(self, request: BugfixRequest) -> PlanningResult:
        mode = request.mode
        text = request.task_description.lower()

        if mode == "review":
            return PlanningResult(["code_review"], "rule", "mode=review", confidence=1.0)
        if mode == "fix":
            return PlanningResult(list(self.FIX_CHAIN), "rule", "mode=fix", confidence=1.0)
        if mode == "full":
            return PlanningResult(
                list(self.FULL_CHAIN),
                "rule",
                "mode=full",
                confidence=1.0,
            )

        if self._contains(text, self.FULL_KEYWORDS):
            return PlanningResult(
                list(self.FULL_CHAIN),
                "rule",
                "matched full keywords",
                confidence=0.95,
            )
        if self._contains(text, self.REVIEW_KEYWORDS):
            return PlanningResult(
                ["code_review"], "rule", "matched review keywords", confidence=0.9
            )
        if self._contains(text, self.FIX_KEYWORDS):
            return PlanningResult(
                list(self.FIX_CHAIN),
                "rule",
                "matched fix keywords",
                confidence=0.9,
            )

        return PlanningResult(
            ["code_review"],
            "rule",
            "defaulted to review; fallback recommended",
            confidence=0.35,
            needs_fallback=True,
        )

    @staticmethod
    def _contains(text: str, keywords: tuple[str, ...]) -> bool:
        return any(keyword in text for keyword in keywords)
