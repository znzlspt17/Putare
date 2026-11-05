"""조기 프로세스 종료는 미측정 메모리를 0으로 만들지 않는다."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import psutil
from moe_prune.research import Experiment


class EarlyEngine:
    pid = 12345
    returncode = None
    exitcode = 0
    waited = False

    def __init__(self, command, **kwargs):
        Path(command[command.index('--output') + 1]).write_text('{"type":"result","logits":[1.0]}\n', encoding='utf-8')

    def poll(self):
        return self.returncode

    def wait(self):
        type(self).waited = True
        self.returncode = self.exitcode
        return self.returncode


class CompletedEngine(EarlyEngine):
    returncode = 0


class FailedEngine(EarlyEngine):
    exitcode = 7


class MonitorRaceTests(unittest.TestCase):
    def setup_experiment(self, directory):
        root = Path(directory)
        model, engine = root / 'model.gguf', root / 'engine.exe'
        model.write_bytes(b'synthetic model')
        engine.write_bytes(b'synthetic executable')
        exp = Experiment(root / 'run', {'engine': str(engine), 'seed': 42, 'context_size': 8192, 'ram_budget_bytes': 40_000_000_000})
        return exp, {'model': str(model), 'prompt': 'test', 'mode': 'logits'}

    def test_completed_unsampled_memory_is_null(self):
        with tempfile.TemporaryDirectory() as folder:
            exp, request = self.setup_experiment(folder)
            CompletedEngine.waited = False
            with patch('moe_prune.research.subprocess.Popen', side_effect=CompletedEngine), patch('psutil.Process'):
                result = exp.run('sample', request)
            self.assertIsNone(result['measurement']['peak_process_tree_rss_bytes'])
            self.assertTrue(CompletedEngine.waited)

    def test_process_disappearing_before_monitor_creation_is_reaped(self):
        with tempfile.TemporaryDirectory() as folder:
            exp, request = self.setup_experiment(folder)
            EarlyEngine.waited = False
            with patch('moe_prune.research.subprocess.Popen', side_effect=EarlyEngine), patch('psutil.Process', side_effect=psutil.NoSuchProcess(12345)):
                result = exp.run('sample', request)
            self.assertEqual(result['logits'], [1.0])
            self.assertTrue(EarlyEngine.waited)

    def test_disappearing_failed_process_does_not_record_success(self):
        with tempfile.TemporaryDirectory() as folder:
            exp, request = self.setup_experiment(folder)
            with patch('moe_prune.research.subprocess.Popen', side_effect=FailedEngine), patch('psutil.Process', side_effect=psutil.NoSuchProcess(12345)):
                with self.assertRaises(RuntimeError):
                    exp.run('sample', request)
            failure = json.loads((exp.root / 'jobs/sample/failure.json').read_text(encoding='utf-8'))
            self.assertEqual(failure['returncode'], 7)
            self.assertFalse((exp.root / 'jobs/sample/measurement.json').exists())
