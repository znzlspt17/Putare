"""메모리 보호와 job 수명은 모델·GPU 실행 없이 검증한다."""
import inspect
import json
from pathlib import Path
import tempfile
import unittest
import sys
import subprocess
from unittest.mock import patch

import psutil
from moe_prune.research import Experiment
from moe_prune.memory_safety import AdapterMemory, GIB, MemoryGuard


class RunningEngine:
    pid = 12345
    returncode = None
    killed = False
    def __init__(self, command, **kwargs):
        type(self).killed = False
    def poll(self):
        return self.returncode
    def kill(self):
        self.killed = True
        type(self).killed = True
        self.returncode = -9
    def wait(self):
        return self.returncode


class Monitor:
    def children(self, recursive=True):
        return []
    def memory_info(self):
        return type('Memory', (), {'rss': 1024})()


class LongEngine(RunningEngine):
    def __init__(self, command, **kwargs):
        super().__init__(command, **kwargs)
        self.polls = 0
        Path(command[command.index('--output') + 1]).write_text('{"type":"result","logits":[1.0]}\n')
    def poll(self):
        self.polls += 1
        if self.polls > 1:
            self.returncode = 0
        return self.returncode


class MemorySafetyTests(unittest.TestCase):
    def make_experiment(self, root):
        model, engine = root / 'model', root / 'engine'
        model.write_bytes(b'model')
        engine.write_bytes(b'engine')
        config = {'engine': str(engine), 'seed': 42, 'context_size': 8192, 'ram_budget_bytes': None}
        return Experiment(root / 'run', config), {'model': str(model), 'prompt': 'hello', 'mode': 'logits'}

    def test_default_timeout_is_unlimited(self):
        self.assertIsNone(inspect.signature(Experiment.run).parameters['timeout_s'].default)

    def test_long_running_job_is_not_killed_after_30_minutes(self):
        with tempfile.TemporaryDirectory() as directory:
            exp, request = self.make_experiment(Path(directory))
            with patch('moe_prune.research.MemoryGuard') as guard, \
                 patch('moe_prune.research.subprocess.Popen', side_effect=LongEngine), \
                 patch('psutil.Process', return_value=Monitor()), \
                 patch('moe_prune.research.time.perf_counter', side_effect=[0, 3600]), \
                 patch('moe_prune.research.time.sleep'):
                guard.return_value.preflight.return_value = {}
                guard.return_value.last = {}
                result = exp.run('long', request)
            self.assertEqual(result['measurement']['wall_s'], 3600)
            self.assertFalse(LongEngine.killed)

    def test_preflight_failure_does_not_launch_or_leave_unfinished_job(self):
        with tempfile.TemporaryDirectory() as directory:
            exp, request = self.make_experiment(Path(directory))
            with patch('moe_prune.research.MemoryGuard', create=True) as guard, patch('moe_prune.research.subprocess.Popen') as spawn:
                guard.return_value.last = {}
                guard.return_value.preflight.side_effect = MemoryError('insufficient GPU reserve')
                with self.assertRaisesRegex(MemoryError, 'GPU reserve'):
                    exp.run('unsafe', request)
                spawn.assert_not_called()
                self.assertFalse((exp.root / 'jobs' / 'unsafe').exists())
                failure = json.loads((exp.root / 'preflight-failures' / 'unsafe.json').read_text(encoding='utf-8'))
                self.assertIn('GPU reserve', failure['reason'])
                guard.return_value.preflight.side_effect = None
                guard.return_value.preflight.return_value = {}
                guard.return_value.last = {}
                spawn.side_effect = LongEngine
                with patch('psutil.Process', return_value=Monitor()), patch('moe_prune.research.time.sleep'):
                    exp.run('unsafe', request)
                self.assertTrue((exp.root / 'jobs' / 'unsafe' / 'measurement.json').exists())

    def test_runtime_pressure_kills_process_and_records_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            exp, request = self.make_experiment(Path(directory))
            with patch('moe_prune.research.MemoryGuard', create=True) as guard, \
                 patch('moe_prune.research.subprocess.Popen', side_effect=RunningEngine), \
                 patch('psutil.Process', return_value=Monitor()):
                guard.return_value.preflight.return_value = {'commit_available_bytes': 100}
                guard.return_value.last = {'commit_available_bytes': 1}
                guard.return_value.check.side_effect = MemoryError('insufficient commit reserve')
                with self.assertRaisesRegex(MemoryError, 'commit reserve'):
                    exp.run('unsafe', request)
            self.assertTrue(RunningEngine.killed)
            failure = json.loads((exp.root / 'jobs' / 'unsafe' / 'failure.json').read_text(encoding='utf-8'))
            self.assertIn('commit reserve', failure['reason'])
            self.assertFalse((exp.root / 'jobs' / 'unsafe' / 'measurement.json').exists())

    def test_cleanup_access_denied_still_kills_parent_and_preserves_error(self):
        with tempfile.TemporaryDirectory() as directory:
            exp, request = self.make_experiment(Path(directory))
            monitor = Monitor()
            monitor.children = lambda recursive: (_ for _ in ()).throw(psutil.AccessDenied())
            with patch('moe_prune.research.MemoryGuard') as guard, \
                 patch('moe_prune.research.subprocess.Popen', side_effect=RunningEngine), \
                 patch('psutil.Process', return_value=monitor):
                guard.return_value.preflight.return_value = {}
                guard.return_value.last = {}
                guard.return_value.check.side_effect = MemoryError('pressure')
                with self.assertRaisesRegex(MemoryError, 'pressure'):
                    exp.run('unsafe', request)
            self.assertTrue(RunningEngine.killed)
            failure = json.loads((exp.root / 'jobs' / 'unsafe' / 'failure.json').read_text(encoding='utf-8'))
            self.assertIn('pressure', failure['reason'])
            self.assertTrue(failure['cleanup_errors'])

    def test_child_kill_failure_does_not_skip_parent_kill(self):
        with tempfile.TemporaryDirectory() as directory:
            exp, request = self.make_experiment(Path(directory))
            monitor = Monitor()
            child = type('Child', (), {'kill': lambda self: (_ for _ in ()).throw(psutil.AccessDenied())})()
            monitor.children = lambda recursive: [child]
            with patch('moe_prune.research.MemoryGuard') as guard, \
                 patch('moe_prune.research.subprocess.Popen', side_effect=RunningEngine), \
                 patch('psutil.Process', return_value=monitor):
                guard.return_value.preflight.return_value = {}
                guard.return_value.last = {}
                guard.return_value.check.side_effect = MemoryError('pressure')
                with self.assertRaisesRegex(MemoryError, 'pressure'):
                    exp.run('unsafe', request)
            self.assertTrue(RunningEngine.killed)
            failure = json.loads((exp.root / 'jobs' / 'unsafe' / 'failure.json').read_text(encoding='utf-8'))
            self.assertIn('child kill', failure['cleanup_errors'][0])

    def test_guard_close_failure_preserves_original_exception(self):
        with tempfile.TemporaryDirectory() as directory:
            exp, request = self.make_experiment(Path(directory))
            with patch('moe_prune.research.MemoryGuard') as guard:
                guard.return_value.last = {}
                guard.return_value.preflight.side_effect = MemoryError('pressure')
                guard.return_value.close.side_effect = OSError('close failed')
                with self.assertRaisesRegex(MemoryError, 'pressure') as caught:
                    exp.run('unsafe', request)
            self.assertIn('close failed', str(caught.exception.__notes__))

    def test_guard_stops_and_reaps_real_sleep_process(self):
        with tempfile.TemporaryDirectory() as directory:
            exp, request = self.make_experiment(Path(directory))
            created = []
            real_spawn = subprocess.Popen
            def launch(*args, **kwargs):
                process = real_spawn([sys.executable, '-c', 'import time; time.sleep(60)'], **kwargs)
                created.append(process)
                return process
            with patch('moe_prune.research.MemoryGuard') as guard, \
                 patch('moe_prune.research.subprocess.Popen', side_effect=launch), \
                 patch('tools.native.gpu_memory.ProcessGpuMemorySampler') as gpu:
                guard.return_value.preflight.return_value = {}
                guard.return_value.last = {}
                guard.return_value.check.side_effect = MemoryError('pressure')
                gpu.return_value.snapshot.return_value = {}
                with self.assertRaisesRegex(MemoryError, 'pressure'):
                    exp.run('unsafe', request)
            self.assertEqual(len(created), 1)
            self.assertIsNotNone(created[0].poll())
            self.assertFalse(psutil.pid_exists(created[0].pid))

    def test_preflight_reserves_full_model_workspace_and_gpu_headroom(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / 'model'
            model.write_bytes(b'12345')
            request = {'model': str(model), 'n_gpu_layers': 99}
            with patch('moe_prune.memory_safety._available_memory', return_value=(8 * GIB, 8 * GIB)), \
                 patch('moe_prune.memory_safety.AdapterMemory') as adapter:
                adapter.return_value.sample.return_value = (3 * GIB, GIB)
                guard = MemoryGuard({}, request)
                with self.assertRaisesRegex(MemoryError, 'GPU reserve'):
                    guard.preflight()
                guard.close()
                adapter.return_value.close.assert_called_once()

    def test_runtime_guard_detects_each_memory_reserve_and_unknown_gpu(self):
        with patch('moe_prune.memory_safety._available_memory') as memory, \
             patch('moe_prune.memory_safety.AdapterMemory') as adapter:
            guard = MemoryGuard({}, {'model': 'unused', 'n_gpu_layers': 1})
            guard.gpu = adapter.return_value
            memory.return_value = (4 * GIB - 1, 8 * GIB)
            with self.assertRaisesRegex(MemoryError, 'RAM'):
                guard.check()
            memory.return_value = (8 * GIB, 4 * GIB - 1)
            with self.assertRaisesRegex(MemoryError, 'commit'):
                guard.check()
            memory.return_value = (8 * GIB, 8 * GIB)
            adapter.return_value.sample.return_value = (16 * GIB, 15 * GIB + 1)
            with self.assertRaisesRegex(MemoryError, 'GPU'):
                guard.check()
            adapter.return_value.sample.side_effect = RuntimeError('GPU counter unavailable')
            with self.assertRaisesRegex(RuntimeError, 'counter unavailable'):
                guard.check()
            guard.close()

    def test_cpu_does_not_require_a_gpu_counter(self):
        with patch('moe_prune.memory_safety._available_memory', return_value=(8 * GIB, 8 * GIB)), \
             patch('moe_prune.memory_safety.AdapterMemory') as adapter:
            guard = MemoryGuard({}, {'model': 'unused', 'n_gpu_layers': 0})
            self.assertEqual(guard.preflight()['ram_available_bytes'], 8 * GIB)
            adapter.assert_not_called()
            guard.close()

    def test_gpu_counter_must_match_dxgi_adapter_and_be_valid(self):
        with patch('moe_prune.memory_safety._amd_adapter', return_value={'capacity': 1000, 'luid': (0, 42)}), \
             patch('moe_prune.memory_safety._PdhCounters') as counters:
            sampler = AdapterMemory()
            counters.return_value.read_named.return_value = {
                r'\GPU Adapter Memory(luid_0x00000000_0x0000002a_phys_0)\Dedicated Usage': (0, 300),
                r'\GPU Adapter Memory(luid_0x00000000_0x0000002b_phys_0)\Dedicated Usage': (0, 600)}
            self.assertEqual(sampler.sample(), (1000, 300))
            counters.return_value.read_named.return_value = {}
            with self.assertRaisesRegex(RuntimeError, 'counter'):
                sampler.sample()
            sampler.close()


if __name__ == '__main__':
    unittest.main()

