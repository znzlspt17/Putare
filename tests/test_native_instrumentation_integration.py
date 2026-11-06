"""실제 GGUF/HIP 계측 경계 회귀. 명시적으로 실행할 때만 모델을 로드한다."""
import os
import tempfile
import unittest
from pathlib import Path

from moe_prune.datasets import load_snapshot
from moe_prune.research import Experiment

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(os.environ.get("RUN_NATIVE_INSTRUMENTATION_TESTS") == "1", "실제 모델/HIP 검증은 명시 실행")
class NativeInstrumentationIntegrationTests(unittest.TestCase):
    def test_instrumented_logits_and_decode_match_original(self):
        model = ROOT / "models/gemma-4-26B-A4B-it-UD-IQ4_NL.gguf"
        config = {"seed": 42, "context_size": 16384, "n_gpu_layers": 24,
                  "ram_budget_bytes": None, "min_available_ram_bytes": 4 * 1024**3,
                  "min_available_commit_bytes": 4 * 1024**3, "min_free_vram_bytes": 1024**3,
                  "vram_workspace_bytes": 1024**3,
                  "engine": str(ROOT / ".cache/llama-build/bin/llama-expert-research.exe")}
        groups, _ = load_snapshot(ROOT / "runs/datasets/snapshot-001")
        record = next(r for r in groups["selection"] if r["domain"] == "math")
        with tempfile.TemporaryDirectory() as temp:
            exp = Experiment(Path(temp), config)
            result = exp.compare("instrumentation", model, model, [record],
                                 instrumentation_a=False, instrumentation_b=True)
            self.assertTrue(result["equivalent"], result)
            original = exp.collect("generate", model, [record], mode="generate", max_tokens=4)[0]
            profile = exp.collect("decode", model, [record], mode="profile", max_tokens=4)[0]
            self.assertEqual(original["token_ids"], profile["token_ids"])
            self.assertTrue(profile["gpu_fusion_disabled"])
            self.assertGreater(profile["routing_records"], 0)
            events = list(exp.routing("decode", [record]))
            self.assertTrue(any(e["phase"] == "prefill" for e in events))
            self.assertTrue(any(e["phase"] == "decode" for e in events))


if __name__ == "__main__":
    unittest.main()

