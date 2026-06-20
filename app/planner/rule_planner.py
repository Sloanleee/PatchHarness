from __future__ import annotations

from dataclasses import dataclass

from app.schemas import BugfixRequest


@dataclass(slots=True)
class PlanningResult:
    agents: list[str]
    planned_by: str
    reason: str


class RulePlanner:
    """Zero-LLM planner for the MVP.

    The planner intentionally starts simple: mode overrides are deterministic,
    then keyword routing covers common Chinese and English task descriptions.
    """

    REVIEW_KEYWORDS = ("审查", "检查", "review", "inspect")
    FIX_KEYWORDS = ("修复", "fix", "bug", "报错", "异常", "失败")
    FULL_KEYWORDS = ("全面", "完整", "full", "comprehensive")

    def plan(self, request: BugfixRequest) -> PlanningResult:
        mode = request.mode
        text = request.task_description.lower()

        if mode == "review":
            return PlanningResult(["code_review"], "rule", "mode=review")
        if mode == "fix":
            return PlanningResult(["bug_fix", "test_verify"], "rule", "mode=fix")
        if mode == "full":
            return PlanningResult(
                ["code_review", "bug_fix", "test_verify", "summary"],
                "rule",
                "mode=full",
            )

        if self._contains(text, self.FULL_KEYWORDS):
            return PlanningResult(
                ["code_review", "bug_fix", "test_verify", "summary"],
                "rule",
                "matched full keywords",
            )
        if self._contains(text, self.FIX_KEYWORDS):
            return PlanningResult(
                ["bug_fix", "test_verify"],
                "rule",
                "matched fix keywords",
            )
        if self._contains(text, self.REVIEW_KEYWORDS):
            return PlanningResult(["code_review"], "rule", "matched review keywords")

        return PlanningResult(["code_review"], "rule", "defaulted to review")

    @staticmethod
    def _contains(text: str, keywords: tuple[str, ...]) -> bool:
        return any(keyword in text for keyword in keywords)

