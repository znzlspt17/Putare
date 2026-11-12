import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from moe_prune import execution_safety


class GradingGuardTests(unittest.TestCase):
    def test_cpu_only_grading_keeps_lock_and_ram_guard(self):
        fake_fcntl = MagicMock()
        with tempfile.TemporaryDirectory() as tmp, \
             patch.dict(sys.modules, {'fcntl': fake_fcntl}), \
             patch.object(execution_safety, '_LOCK', Path(tmp) / 'lock'), \
             patch.object(execution_safety, '_POISON', Path(tmp) / 'poison'), \
             patch.object(execution_safety, 'running_inference', return_value=[]), \
             patch('moe_prune.memory_safety.MemoryGuard') as guard:
            with execution_safety.execution_guard('grading', memory_config={'seed': 42}, monitor_gpu=False):
                pass
            guard.assert_called_once_with({'seed': 42}, {'monitor_gpu': False})
            guard.return_value.preflight.assert_called_once()
            guard.return_value.close.assert_called_once()
            self.assertGreaterEqual(fake_fcntl.flock.call_count, 2)


if __name__ == '__main__':
    unittest.main()
