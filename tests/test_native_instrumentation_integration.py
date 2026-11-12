"""실제 GGUF/CUDA 계측 경계 회귀. 명시적으로 실행할 때만 모델을 로드한다."""
import os
import tempfile
import unittest
from pathlib import Path

from moe_prune.datasets import load_snapshot
from moe_prune.research import Experiment
from moe_prune.research import read_jsonl
from moe_prune.base_benchmark import validate_routing

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(os.environ.get("RUN_NATIVE_INSTRUMENTATION_TESTS") == "1", "실제 모델/CUDA 검증은 명시 실행")
class NativeInstrumentationIntegrationTests(unittest.TestCase):
    def test_instrumented_logits_and_decode_match_original(self):
        model = ROOT / "models/gemma-4-26B-A4B-it-UD-IQ4_NL.gguf"
        config = {"seed": 42, "context_size": 16384, "n_gpu_layers": 31,
                  "ram_usage_warn_fraction": .80, "vram_usage_warn_fraction": .97,
                  "engine": str(ROOT / ".cache/llama-build/bin/llama-expert-research")}
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

    def test_fusion_enabled_lightweight_routing_covers_all_tokens(self):
        # 명시적 실 GPU 테스트: 결과가 나오더라도 개별 CUDA 커널 융합 자체의 증거는 아니다.
        model = ROOT / "models/gemma-4-26B-A4B-it-UD-IQ4_NL.gguf"
        config = {"seed": 42, "context_size": 16384, "n_gpu_layers": 31,
                  "ram_usage_warn_fraction": .80, "vram_usage_warn_fraction": .97,
                  "engine": str(ROOT / ".cache/llama-build/bin/llama-expert-research")}
        request = {'model': str(model), 'prompt': 'What is two plus two?', 'mode': 'profile',
                   'max_tokens': 4, 'context_size': 16384, 'n_gpu_layers': 31,
                   'instrumentation': True, 'instrumentation_level': 'routing',
                   'fusion_enabled': True, 'all_token_logits': True}
        with tempfile.TemporaryDirectory() as temp:
            exp = Experiment(Path(temp), config)
            result = exp.run('fusion-routing', request)
            events = list(read_jsonl(exp.root / 'jobs/fusion-routing/events.jsonl'))
            self.assertIs(result['fusion_enabled'], True)
            self.assertIs(result['gpu_fusion_disabled'], False)
            self.assertEqual(result['instrumentation_level'], 'routing')
            self.assertEqual(validate_routing(events, result), result['routing_records'])
            rows = [row for row in events if row.get('type') == 'routing']
            self.assertTrue(rows)
            self.assertTrue(all('weighted_output_norms' not in row for row in rows))


if __name__ == "__main__":
    unittest.main()

