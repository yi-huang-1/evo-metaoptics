import unittest

from evo_metaoptics.mce_env.registry import EnvironmentRegistry


class TestMetaopticsEnvRegistry(unittest.TestCase):
    def test_registry_lists_metaoptics_inverse_design(self) -> None:
        self.assertIn("metaoptics_inverse_design", EnvironmentRegistry.list_environments())

    def test_registry_returns_metaoptics_environment_instance(self) -> None:
        from evo_metaoptics.mce_env.metaoptics_inverse_design import (
            MetaopticsInverseDesignEnvironment,
        )

        env = EnvironmentRegistry.get("metaoptics_inverse_design")
        self.assertIsInstance(env, MetaopticsInverseDesignEnvironment)


if __name__ == "__main__":
    unittest.main()
