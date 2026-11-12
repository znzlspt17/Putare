import json
import tempfile
import unittest
import sys
import types
from pathlib import Path
from unittest.mock import patch

from tools import benchmark175 as bench


class BaseBenchmarkTest(unittest.TestCase):
    def test_base_model_uses_128_experts_and_expected_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp) / 'model.gguf'
            model.touch()
            with patch.object(bench, 'sha256', return_value=bench.BASE_MODEL_SHA), \
                 patch.object(bench, 'GGUF') as gguf:
                gguf.return_value.architecture.return_value = (list(range(30)), 128, 8)
                bench.check_model(model, model_profile='base')
                with self.assertRaises(ValueError):
                    bench.check_model(model)

    def test_base_requires_checkpoint_before_model_access(self):
        with self.assertRaisesRegex(ValueError, '백업'):
            bench.generate(model_profile='base')

    def test_unknown_profile_is_rejected(self):
        with self.assertRaises(ValueError):
            bench.check_model('unused', model_profile='invalid')

    def test_download_verifies_existing_file_and_disables_auth(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / 'model.gguf'
            target.touch()
            with patch.object(bench, 'check_model') as check:
                result = bench.download_base_model(target)
                self.assertEqual(result, target)
                check.assert_called_once_with(target, model_profile='base')

    def test_download_pins_public_revision_without_auth(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / bench.BASE_FILENAME
            calls = []
            def download(**kwargs):
                calls.append(kwargs)
                target.touch()
                return str(target)
            with patch.dict(sys.modules, {'huggingface_hub': types.SimpleNamespace(hf_hub_download=download)}), \
                 patch.object(bench, 'check_model'):
                bench.download_base_model(target)
            self.assertEqual(calls[0]['revision'], bench.BASE_REVISION)
            self.assertIs(calls[0]['token'], False)
            self.assertEqual(calls[0]['repo_id'], bench.BASE_REPO)

    def test_failed_backup_preflight_prevents_model_access(self):
        with patch.object(bench, 'source_identity') as source, \
             patch.object(bench, 'preflight_checkpoint', side_effect=OSError('full')):
            with self.assertRaises(OSError):
                bench.generate(model_profile='base', checkpoint_dir='backup')
            source.assert_not_called()

    def test_base_routing_validation_accepts_experts_80_to_127(self):
        from moe_prune.base_benchmark import validate_routing
        rows = [{'type':'routing', 'phase':'prefill', 'layer':i, 'token_index':0,
                 'expert_ids': list(range(80, 88)), 'weights':[.125]*8} for i in range(30)]
        result = {'prompt_tokens': 1, 'processed_decode_tokens': 0, 'instrumentation_level':'routing'}
        self.assertEqual(validate_routing(rows, result, experts=128), 30)

    def test_base_job_uses_128_expert_routing_validator(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / 'jobs/job'
            folder.mkdir(parents=True)
            (folder / 'stderr.log').write_text('', encoding='utf-8')
            result = {'fusion_enabled': True, 'gpu_fusion_disabled': False,
                      'instrumentation_level': 'routing', 'profiled_layers': list(range(30)),
                      'enable_thinking': True, 'stop_reason': 'eog', 'routing_records': 30}
            observed = {'offloaded_layers': {'gpu': 31, 'total': 31}}
            with patch('moe_prune.base_benchmark.verify_runtime', return_value=observed), \
                 patch('moe_prune.base_benchmark.validate_routing', return_value=30) as validate, \
                 patch.object(bench, 'read_jsonl', return_value=[]):
                bench._validate_job(tmp, 'job', {}, result, model_profile='base')
            self.assertEqual(validate.call_args.kwargs['experts'], 128)

    def test_base_identity_uses_fixed_175_without_e80_manifests(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine = Path(tmp) / 'engine'
            engine.touch()
            absent = Path(tmp) / 'absent.json'
            with patch.object(bench, 'check_model'), patch.object(bench, 'EXPORT', absent), \
                 patch.object(bench, 'PERMUTATION', absent):
                records, identity = bench.source_identity('model.gguf', engine, model_profile='base')
            self.assertEqual(len(records), 175)
            self.assertEqual(identity['ids'], [record['id'] for record in records])
            self.assertEqual(identity['model_sha256'], bench.BASE_MODEL_SHA)
            self.assertEqual(identity['experts'], 128)
            self.assertEqual(identity['huggingface']['revision'], bench.BASE_REVISION)
            self.assertNotIn('export_sha256', identity)


if __name__ == '__main__':
    unittest.main()
