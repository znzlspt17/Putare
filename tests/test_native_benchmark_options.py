"""새 추론 설정은 잘못된 값을 모델 적재 전에 명시적으로 거절한다."""
import json
from pathlib import Path
import tempfile
import unittest

from moe_prune.research import spawn_native


ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / '.cache/llama-build/bin/llama-expert-research'


@unittest.skipUnless(ENGINE.exists(), 'native engine not built')
class NativeBenchmarkOptionsTests(unittest.TestCase):
    def test_invalid_sampling_is_rejected_before_loading_model(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            request = root / 'request.json'
            output = root / 'output.jsonl'
            request.write_text(json.dumps({'model': 'not-present.gguf', 'prompt': 'x',
                                          'sampling': {'temperature': -1, 'top_p': .95, 'top_k': 64}}))
            process = spawn_native([str(ENGINE), '--request', str(request), '--output', str(output)],
                                   stdout=-1, stderr=-1)
            _, stderr = process.communicate(timeout=30)
            self.assertNotEqual(process.returncode, 0)
            self.assertIn(b'invalid sampling', stderr)
