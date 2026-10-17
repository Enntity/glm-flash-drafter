"""The `datasets` stand-in: a broken `datasets` install must not break `import speculators`."""

import subprocess
import sys
import textwrap
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


class DatasetsCompatTests(unittest.TestCase):
    def test_broken_datasets_gets_a_loud_stand_in(self):
        # Run in a fresh interpreter: a finder makes `import datasets` fail the way a
        # pandas-less install does, then the package import must still succeed.
        code = textwrap.dedent("""
            import importlib.abc, sys
            class Broken(importlib.abc.MetaPathFinder):
                def find_spec(self, name, path=None, target=None):
                    if name == "datasets" or name.startswith("datasets."):
                        raise ModuleNotFoundError("No module named 'pandas'")
                    return None
            sys.meta_path.insert(0, Broken())
            import enntity_drafter
            import datasets
            assert getattr(datasets, "ENNTITY_STUB", False), "stand-in not installed"
            from datasets import load_from_disk
            try:
                load_from_disk("x")
            except RuntimeError as error:
                assert "not importable" in str(error)
            else:
                raise AssertionError("stand-in must fail loudly")
            print("ok")
        """)
        result = subprocess.run([sys.executable, "-c", code], cwd=REPO, capture_output=True, text=True,
                                env={"PYTHONPATH": str(REPO), "PATH": "/usr/bin:/bin"})
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("ok", result.stdout)

    def test_working_datasets_is_left_alone(self):
        from enntity_drafter._compat import ensure_datasets_importable
        try:
            import datasets  # noqa: F401
        except Exception:  # noqa: BLE001
            self.skipTest("no working datasets in this environment")
        self.assertTrue(ensure_datasets_importable())
        import datasets as again
        self.assertFalse(getattr(again, "ENNTITY_STUB", False))


if __name__ == "__main__":
    unittest.main()
