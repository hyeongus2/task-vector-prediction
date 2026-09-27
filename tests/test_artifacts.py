import tempfile
import unittest
from pathlib import Path
from src.tvp.artifacts import resolve_run_directory, select_fitting_indices


class ArtifactTests(unittest.TestCase):
    def test_nondefault_interval_is_used(self):
        self.assertEqual(select_fitting_indices([0, 20, 40, 60, 80], 3, 20, 1), [1, 2, 3])

    def test_short_run_reports_missing_checkpoints(self):
        with self.assertRaisesRegex(ValueError, "last available step"):
            select_fitting_indices([0, 50, 100], 3, 50, 1)

    def test_sparse_checkpoints_do_not_duplicate_fitting_points(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            select_fitting_indices([0, 100, 200], 3, 50, 1)

    def test_resume_requires_original_reference(self):
        with tempfile.TemporaryDirectory() as directory:
            group = Path(directory)
            run = group / "original"; run.mkdir()
            (run / "checkpoint.pt").write_bytes(b"checkpoint")
            with self.assertRaisesRegex(FileNotFoundError, "theta0.pt"):
                resolve_run_directory(group, "original", True)

    def test_existing_run_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            group = Path(directory); (group / "old").mkdir()
            with self.assertRaises(FileExistsError):
                resolve_run_directory(group, "old", False)

    def test_resume_reuses_exact_run_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            group = Path(directory); run = group / "original"; run.mkdir()
            for name in ("checkpoint.pt", "theta0.pt", "text_features.pt", "effective_config.yaml"):
                (run/name).write_bytes(b"original")
            self.assertEqual(resolve_run_directory(group, "original", True), run)
            self.assertEqual((run/"theta0.pt").read_bytes(), b"original")


if __name__ == "__main__":
    unittest.main()
