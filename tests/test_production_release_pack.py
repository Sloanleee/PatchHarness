import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


class ProductionReleasePackTests(unittest.TestCase):
    def test_dockerfile_runs_fastapi_api(self):
        dockerfile = ROOT / "Dockerfile"
        self.assertTrue(dockerfile.exists())
        content = dockerfile.read_text(encoding="utf-8")

        self.assertIn("FROM python:", content)
        self.assertIn("pip install", content)
        self.assertIn("requirements.txt", content)
        self.assertIn("uvicorn", content)
        self.assertIn("app.main:app", content)
        self.assertIn("PATCHHARNESS_LLM_PROVIDER=mock", content)
        self.assertIn("--host", content)
        self.assertIn("0.0.0.0", content)
        self.assertIn("--port", content)
        self.assertIn("8000", content)

    def test_dockerignore_excludes_private_and_generated_files(self):
        dockerignore = ROOT / ".dockerignore"
        self.assertTrue(dockerignore.exists())
        ignored = dockerignore.read_text(encoding="utf-8")

        for pattern in [
            ".env",
            ".venv/",
            ".storage/",
            ".runtime/",
            "__pycache__/",
            ".pytest_cache/",
            "results/",
            "docs/",
            "README1.md",
            "READMEv2.md",
            "plan.md",
            "problem-solution.md",
            ".git/",
        ]:
            self.assertIn(pattern, ignored)

        self.assertIn("!demo/hitl_project/.env", ignored)

    def test_compose_exposes_api_service(self):
        compose_path = ROOT / "docker-compose.yml"
        self.assertTrue(compose_path.exists())
        compose = yaml.safe_load(compose_path.read_text(encoding="utf-8"))

        service = compose["services"]["patchharness-api"]
        self.assertEqual(service["build"], ".")
        self.assertIn("8000:8000", service["ports"])
        self.assertNotIn("env_file", service)
        self.assertIn("PATCHHARNESS_LLM_PROVIDER", service["environment"])
        self.assertEqual(
            service["environment"]["PATCHHARNESS_LLM_PROVIDER"],
            "${PATCHHARNESS_LLM_PROVIDER:-mock}",
        )

    def test_github_actions_runs_tests_evidence_and_uploads_artifact(self):
        workflow_path = ROOT / ".github" / "workflows" / "ci.yml"
        self.assertTrue(workflow_path.exists())
        workflow = yaml.safe_load(workflow_path.read_text(encoding="utf-8"))

        self.assertEqual(workflow["name"], "CI")
        self.assertIn("push", workflow[True])
        self.assertIn("pull_request", workflow[True])

        jobs = workflow["jobs"]
        self.assertIn("test-and-evidence", jobs)
        steps = jobs["test-and-evidence"]["steps"]
        step_text = "\n".join(str(step) for step in steps)

        self.assertIn("python -m unittest discover -s tests", step_text)
        self.assertIn(
            "python benchmarks/generate_production_demo_evidence.py --provider mock",
            step_text,
        )
        self.assertIn("actions/upload-artifact", step_text)
        self.assertIn("patchharness-production-demo-evidence", step_text)
        self.assertIn("results/production_demo/runs/**/summary.md", step_text)
        self.assertIn("results/production_demo/runs/**/paused_response.json", step_text)
        self.assertIn("results/production_demo/runs/**/resumed_response.json", step_text)
        self.assertIn("results/production_demo/runs/**/trace.json", step_text)
