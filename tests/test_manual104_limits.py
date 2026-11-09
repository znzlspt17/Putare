"""Colab 공통 메모리 경고 설정과 모델별 적용을 검증한다."""
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from moe_prune.memory_safety import MemoryGuard

class Limits03Tests(unittest.TestCase):
    def config(self): return {'ram_usage_warn_fraction':.80,'vram_usage_warn_fraction':.97}
    def test_ram_boundary_warns_without_stopping(self):
        with patch('moe_prune.memory_safety._ram_sample',side_effect=[(100,20,'system'),(100,19,'system')]):
            guard=MemoryGuard(self.config(),{})
            self.assertFalse(guard.check()['ram_warning'])
            self.assertTrue(guard.check()['ram_warning'])
    def test_vram_boundary_warns_without_stopping(self):
        with patch('moe_prune.memory_safety._ram_sample',return_value=(100,50,'system')):
            guard=MemoryGuard(self.config(),{});guard.gpu=SimpleNamespace(sample=lambda:(100,97))
            self.assertFalse(guard.check()['vram_warning'])
            guard.gpu=SimpleNamespace(sample=lambda:(100,98))
            self.assertTrue(guard.check()['vram_warning'])
    def test_preflight_reports_cpu_kv_without_fixed_reserve(self):
        request={'n_gpu_layers':0,'cache_type_k':'q8_0','model':'none'}
        with patch('moe_prune.memory_safety._ram_sample',return_value=(100,25,'system')),patch('moe_prune.memory_safety._kv_estimates',return_value=(0,6)):
            result=MemoryGuard(self.config(),request).preflight()
            self.assertEqual(result['ram_estimated_kv_bytes'],6)
            self.assertTrue(any(x['resource']=='ram' and x['kind']=='predicted' for x in result['warning_events']))
    def test_invalid_fractions_rejected(self):
        for value in (True,float('nan'),0,-.1,1.1,'0.7'):
            for field in ('ram_usage_warn_fraction','vram_usage_warn_fraction'):
                with self.subTest(field=field,value=value),self.assertRaises(ValueError):MemoryGuard({field:value},{})
    def test_central_config(self):
        from tools.manual104 import config03,checked_request,MODEL
        from moe_prune.base_benchmark import DEFAULT_CONFIG
        c=config03();self.assertEqual(c['ram_usage_warn_fraction'],.80);self.assertEqual(c['vram_usage_warn_fraction'],.97)
        self.assertEqual(c,DEFAULT_CONFIG)
        self.assertEqual(checked_request({'prompt':'x'},MODEL)['n_gpu_layers'],31)
    def test_grading_guard_receives_same_config(self):
        from moe_prune import execution_safety as s
        with tempfile.TemporaryDirectory() as d,patch.object(s,'_LOCK',Path(d)/'lock'),patch.object(s,'_POISON',Path(d)/'poison'),patch.object(s,'running_inference',return_value=[]),patch('moe_prune.memory_safety.MemoryGuard') as check:
            with s.execution_guard('grading',memory_config=self.config()):pass
            check.assert_called_once_with(self.config(), {'monitor_gpu': True})
            check.return_value.preflight.assert_called_once()

