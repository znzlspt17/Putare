"""모델을 실행하지 않고 실험 캐시와 실패 경계의 관찰 가능한 동작을 검증한다."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from moe_prune.research import Experiment


class CompletedEngine:
    """외부 실행 대신 계약에 맞는 최소 결과 파일을 쓰는 합성 엔진."""
    pid = 12345
    returncode = 0

    def __init__(self, command, **kwargs):
        output = Path(command[command.index("--output") + 1])
        output.write_text(json.dumps({"type": "result", "logits": [1.0, 2.0]}) + "\n", encoding="utf-8")

    def poll(self):
        return 0

    def wait(self):
        return self.returncode


class RunIntegrationTests(unittest.TestCase):
    def setup_experiment(self, directory):
        root = Path(directory)
        engine, model = root / "engine.exe", root / "model.gguf"
        engine.write_bytes(b"synthetic engine, never executed")
        model.write_bytes(b"synthetic input, not a real model")
        config = {"engine": str(engine), "seed": 42, "context_size": 8192, "ram_budget_bytes": 40_000_000_000}
        return Experiment(root / "run", config), {"model": str(model), "prompt": "hello", "mode": "logits", "allowlist": None}, engine, model

    def test_completed_jobs_resume_without_external_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            experiment, request, _, _ = self.setup_experiment(directory)
            with patch("moe_prune.research.subprocess.Popen", side_effect=CompletedEngine) as process, patch("psutil.Process"):
                first = experiment.run("sample", request)
                second = experiment.run("sample", request)
            self.assertEqual(first, second)
            self.assertEqual(process.call_count, 1)

    def test_same_path_changed_artifact_cannot_reuse_old_measurement(self):
        for role in ("engine", "model"):
            with self.subTest(role=role), tempfile.TemporaryDirectory() as directory:
                experiment, request, engine, model = self.setup_experiment(directory)
                with patch("moe_prune.research.subprocess.Popen", side_effect=CompletedEngine), patch("psutil.Process"):
                    experiment.run("sample", request)
                changed = engine if role == "engine" else model
                changed.write_bytes(b"changed content, cache must not be reused")
                with patch("moe_prune.research.subprocess.Popen") as process:
                    with self.assertRaises(ValueError):
                        experiment.run("sample", request)
                    process.assert_not_called()

    def test_missing_engine_can_be_fixed_and_retried_without_orphan_job(self):
        with tempfile.TemporaryDirectory() as directory:
            experiment, request, engine, _ = self.setup_experiment(directory)
            engine.unlink()
            with self.assertRaises(FileNotFoundError):
                experiment.run("sample", request)
            self.assertFalse((experiment.root / "jobs" / "sample").exists())
            engine.write_bytes(b"restored synthetic executable")
            with patch("moe_prune.research.subprocess.Popen", side_effect=CompletedEngine), patch("psutil.Process"):
                self.assertEqual(experiment.run("sample", request)["logits"], [1.0, 2.0])

    def test_different_settings_require_a_new_run(self):
        with tempfile.TemporaryDirectory() as directory:
            experiment, _, _, _ = self.setup_experiment(directory)
            with self.assertRaises(ValueError):
                Experiment(experiment.root, {**experiment.config, "seed": 99})

    def test_missing_or_failed_evidence_blocks_next_stage(self):
        with tempfile.TemporaryDirectory() as directory:
            experiment, _, _, _ = self.setup_experiment(directory)
            for evidence in (None, {"equivalent": False}):
                if evidence is not None:
                    experiment.record("noop", evidence)
                with self.assertRaises(RuntimeError):
                    experiment.require("noop")


if __name__ == "__main__":
    unittest.main()
