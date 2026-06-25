import unittest

from app_config import feature_flag_enabled


class ConfigTests(unittest.TestCase):
    def test_feature_flag_enabled(self):
        self.assertTrue(feature_flag_enabled())


if __name__ == "__main__":
    unittest.main()
