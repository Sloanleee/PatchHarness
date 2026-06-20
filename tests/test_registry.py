import unittest
from pathlib import Path

from app.agents import AgentRegistry


class AgentRegistryTests(unittest.TestCase):
    def test_loads_yaml_configs(self):
        config_dir = Path("app/agents/configs")
        registry = AgentRegistry.load_from_dir(config_dir)
        self.assertIn("code_review", registry.names())
        self.assertIn("bug_fix", registry.names())
        self.assertIn("test_verify", registry.names())


if __name__ == "__main__":
    unittest.main()

