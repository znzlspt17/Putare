import unittest
from types import SimpleNamespace
from unittest.mock import patch

from moe_prune.memory_safety import MemoryGuard
from tools.native.prepare import build_commands


class ColabRuntimeTests(unittest.TestCase):
    def test_threshold_crossing_warns_once_and_recovers(self):
        samples = [(100, 20, 'fixture'), (100, 19, 'fixture'),
                   (100, 18, 'fixture'), (100, 20, 'fixture'), (100, 19, 'fixture')]
        with patch('moe_prune.memory_safety._ram_sample', side_effect=samples), \
             patch('builtins.print') as printed:
            guard = MemoryGuard({'ram_usage_warn_fraction': .8}, {})
            for _ in samples:
                guard.check()
        self.assertEqual(printed.call_count, 2)

    def test_cuda_configuration_accepts_a100_capability(self):
        commands = build_commands('source', 'build', '80')
        self.assertIn('-DGGML_CUDA=ON', commands[0])
        self.assertIn('-DCMAKE_CUDA_ARCHITECTURES=80', commands[0])
        self.assertNotIn('-DGGML_HIP=ON', commands[0])

    def test_gpu_threshold_warns_only_when_crossing(self):
        gpu = SimpleNamespace(sample=unittest.mock.Mock(side_effect=[(100, 97), (100, 98), (100, 99), (100, 97), (100, 98)]))
        with patch('moe_prune.memory_safety._ram_sample', return_value=(100, 50, 'fixture')), patch('builtins.print') as printed:
            guard = MemoryGuard({}, {})
            guard.gpu = gpu
            for _ in range(5):
                guard.check()
        self.assertEqual(printed.call_count, 2)
        self.assertTrue(guard.last['vram_warning'])

    def test_cgroup_limit_and_parent_available_override_host(self):
        from moe_prune.memory_safety import _ram_sample
        with patch('psutil.virtual_memory', return_value=SimpleNamespace(total=1000, available=800)), \
             patch('moe_prune.memory_safety._cgroup_limits', return_value=[(500, 300), (700, 650)]):
            self.assertEqual(_ram_sample(), (500, 50, 'cgroup'))

    def test_cgroup_v2_reads_child_and_parent_limits(self):
        from pathlib import Path
        from moe_prune.memory_safety import _cgroup_limits
        files = {
            '/proc/self/cgroup': '0::/parent/child\n',
            '/proc/self/mountinfo': '1 0 0:28 / /cg rw - cgroup2 cgroup rw\n',
            '/cg/parent/child/memory.max': '600',
            '/cg/parent/child/memory.current': '400',
            '/cg/parent/memory.max': '500',
            '/cg/parent/memory.current': '300',
            '/cg/memory.max': 'max',
        }
        def read(path, **kwargs):
            return files[str(path).replace('\\', '/')]
        def exists(path):
            return str(path).replace('\\', '/') in files
        with patch.object(Path, 'read_text', autospec=True, side_effect=read), \
             patch.object(Path, 'exists', autospec=True, side_effect=exists):
            self.assertEqual(_cgroup_limits(), [(600, 400), (500, 300)])

    def test_cgroup_v1_reads_memory_controller(self):
        from pathlib import Path
        from moe_prune.memory_safety import _cgroup_limits
        files = {
            '/proc/self/cgroup': '4:memory:/user/session\n',
            '/proc/self/mountinfo': '1 0 0:29 / /mem rw - cgroup cgroup rw,memory\n',
            '/mem/user/session/memory.limit_in_bytes': '600',
            '/mem/user/session/memory.usage_in_bytes': '400',
            '/mem/user/memory.limit_in_bytes': '500',
            '/mem/user/memory.usage_in_bytes': '300',
        }
        def read(path, **kwargs):
            return files[str(path).replace('\\', '/')]
        def exists(path):
            return str(path).replace('\\', '/') in files
        with patch.object(Path, 'read_text', autospec=True, side_effect=read), \
             patch.object(Path, 'exists', autospec=True, side_effect=exists):
            self.assertEqual(_cgroup_limits(), [(600, 400), (500, 300)])

    def test_invalid_measurements_fail(self):
        for sample in ((0, 0, 'fixture'), (100, 101, 'fixture')):
            with self.subTest(sample=sample), patch('moe_prune.memory_safety._ram_sample', return_value=sample):
                with self.assertRaises(RuntimeError):
                    MemoryGuard({}, {}).check()

    def test_legacy_fixed_limits_are_rejected(self):
        for field in ('ram_budget_bytes', 'min_available_ram_bytes', 'min_available_commit_bytes',
                      'min_free_vram_bytes', 'vram_workspace_bytes',
                      'max_ram_usage_fraction', 'max_vram_usage_fraction'):
            with self.subTest(field=field), self.assertRaises(ValueError):
                MemoryGuard({field: 1}, {})

    def test_preflight_estimates_warn_but_do_not_abort(self):
        request = {'model':'fixture', 'n_gpu_layers':31, 'cache_type_k':'q8_0'}
        with patch('moe_prune.memory_safety._ram_sample', return_value=(100, 20, 'fixture')), \
             patch('moe_prune.memory_safety.AdapterMemory') as gpu, \
             patch('moe_prune.memory_safety._kv_estimates', return_value=(20, 1)), \
             patch('moe_prune.memory_safety._gpu_weight_estimate', return_value=60), \
             patch('builtins.print') as printed:
            gpu.return_value.sample.return_value = (100, 20)
            sample = MemoryGuard({}, request).preflight()
        self.assertEqual(printed.call_count, 2)
        self.assertEqual(sample['ram_estimated_kv_bytes'], 1)
        self.assertEqual(sample['vram_estimated_model_bytes'], 60)
        self.assertEqual({event['resource'] for event in sample['warning_events']}, {'ram', 'vram'})

    def test_grading_monitors_gpu_without_estimating_model(self):
        with patch('moe_prune.memory_safety._ram_sample', return_value=(100, 50, 'fixture')), \
             patch('moe_prune.memory_safety.AdapterMemory') as gpu, \
             patch('moe_prune.memory_safety._gpu_weight_estimate') as estimate:
            gpu.return_value.sample.return_value = (100, 10)
            guard = MemoryGuard({}, {'monitor_gpu': True})
            sample = guard.preflight()
            guard.close()
        self.assertEqual(sample['vram_used_bytes'], 10)
        estimate.assert_not_called()

    def test_grading_guard_reuses_one_monitor_and_closes(self):
        import tempfile
        from pathlib import Path
        from moe_prune import execution_safety as safety
        with tempfile.TemporaryDirectory() as folder, \
             patch.object(safety, '_LOCK', Path(folder)/'lock'), \
             patch.object(safety, '_POISON', Path(folder)/'poison'), \
             patch.object(safety, 'running_inference', return_value=[]), \
             patch('moe_prune.memory_safety.MemoryGuard') as factory:
            with safety.execution_guard('grading', memory_config={'ram_usage_warn_fraction': .8}) as guard:
                self.assertIs(guard, factory.return_value)
                guard.check()
            factory.assert_called_once_with({'ram_usage_warn_fraction': .8}, {'monitor_gpu': True})
            factory.return_value.preflight.assert_called_once()
            factory.return_value.check.assert_called_once()
            factory.return_value.close.assert_called_once()
