import unittest

from app.planner import RulePlanner
from app.schemas import BugfixRequest


class RulePlannerTests(unittest.TestCase):
    def test_review_keywords_route_to_review_agent(self):
        result = RulePlanner().plan(BugfixRequest(task_description="请审查登录模块"))
        self.assertEqual(result.agents, ["code_review"])

    def test_fix_keywords_route_to_fix_and_verify(self):
        result = RulePlanner().plan(BugfixRequest(task_description="fix payment bug"))
        self.assertEqual(result.agents, ["root_cause_analysis", "patch_generation", "test_verify"])

    def test_full_keywords_route_to_full_chain(self):
        result = RulePlanner().plan(BugfixRequest(task_description="全面检查项目"))
        self.assertEqual(
            result.agents,
            ["code_review", "root_cause_analysis", "patch_generation", "test_verify", "summary"],
        )

    def test_explicit_fix_mode_uses_root_cause_and_patch_generation(self):
        result = RulePlanner().plan(
            BugfixRequest(task_description="anything", mode="fix")
        )
        self.assertEqual(result.agents, ["root_cause_analysis", "patch_generation", "test_verify"])

    def test_explicit_full_mode_uses_new_full_chain(self):
        result = RulePlanner().plan(
            BugfixRequest(task_description="anything", mode="full")
        )
        self.assertEqual(
            result.agents,
            ["code_review", "root_cause_analysis", "patch_generation", "test_verify", "summary"],
        )


if __name__ == "__main__":
    unittest.main()
