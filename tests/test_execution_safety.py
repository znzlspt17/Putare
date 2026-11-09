"""Linux 추론과 채점 실행 경계."""
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
from moe_prune import execution_safety as s

class ExecutionSafetyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.enterContext(patch.object(s, '_LOCK', Path(self.temp.name)/'lock'))
        self.enterContext(patch.object(s, '_POISON', Path(self.temp.name)/'poison'))
    def test_grading_rejects_active_model(self):
        with patch.object(s, 'running_inference', return_value=['llama-server']):
            with self.assertRaisesRegex(RuntimeError, 'llama-server'):
                with s.execution_guard('grading'): self.fail('must not start')
    def test_grading_checks_memory_first(self):
        with patch.object(s, 'running_inference', return_value=[]), patch('moe_prune.memory_safety.MemoryGuard') as monitor:
            monitor.return_value.preflight.side_effect = RuntimeError('measurement unavailable')
            with self.assertRaisesRegex(RuntimeError, 'measurement unavailable'):
                with s.execution_guard('grading'): self.fail('must not start')
            monitor.return_value.close.assert_called_once()
    def test_unverifiable_process_state_fails_closed(self):
        with patch.object(s, 'running_inference', side_effect=RuntimeError('unavailable')):
            with self.assertRaisesRegex(RuntimeError, 'unavailable'):
                with s.execution_guard('grading'): pass
    def test_unknown_kind_is_rejected(self):
        with self.assertRaises(ValueError):
            with s.execution_guard('wsl'): pass

if __name__ == '__main__': unittest.main()
