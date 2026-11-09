"""Linux 프로세스 간 추론·채점 잠금 및 종료 확인 실패 경계."""
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from moe_prune import execution_safety as safety

@unittest.skipUnless(sys.platform == 'linux', 'Linux flock 필요')
class ExecutionBoundaryRegressions(unittest.TestCase):
    def setUp(self):
        self.directory=self.enterContext(tempfile.TemporaryDirectory())
        self.lock=Path(self.directory)/'execution.lock'; self.poison=Path(self.directory)/'execution.unsafe'
        self.enterContext(patch.object(safety,'_LOCK',self.lock))
        self.enterContext(patch.object(safety,'_POISON',self.poison))
    def child_guard(self, kind):
        script = """import sys
from pathlib import Path
from moe_prune import execution_safety as s, memory_safety as memory
class FakeGuard:
 def __init__(self,*a,**kw): pass
 def preflight(self): pass
 def close(self): pass
memory.MemoryGuard=FakeGuard
s._LOCK=Path(sys.argv[1]);s._POISON=Path(sys.argv[2]);s.running_inference=lambda: []
try:
 with s.execution_guard(sys.argv[3]): pass
except RuntimeError: sys.exit(23)
"""
        return subprocess.run([sys.executable,'-c',script,str(self.lock),str(self.poison),kind],cwd=Path(__file__).resolve().parents[1],capture_output=True,timeout=15)
    def test_native_lease_blocks_grading_then_releases(self):
        with safety.execution_guard('native'):
            self.assertEqual(self.child_guard('grading').returncode,23)
        self.assertEqual(self.child_guard('grading').returncode,0)
    def test_grading_lease_blocks_native(self):
        with patch.object(safety,'running_inference',return_value=[]), patch('moe_prune.memory_safety.MemoryGuard'):
            with safety.execution_guard('grading'):
                self.assertEqual(self.child_guard('native').returncode,23)
    def test_uncertain_cleanup_blocks_fresh_processes(self):
        with safety.execution_guard('native'):
            safety.poison_execution('synthetic uncertain cleanup')
        for kind in ('native','grading'):
            with self.subTest(kind=kind): self.assertEqual(self.child_guard(kind).returncode,23)
    def test_active_model_blocks_grading_before_memory(self):
        with patch.object(safety,'running_inference',return_value=['llama-server']), patch('moe_prune.memory_safety.MemoryGuard') as check:
            with self.assertRaisesRegex(RuntimeError,'llama-server'):
                with safety.execution_guard('grading'): self.fail('must not start')
            check.assert_not_called()
    def test_grading_checks_memory_before_launch(self):
        events=[]
        with patch.object(safety,'running_inference',return_value=[]), patch('moe_prune.memory_safety.MemoryGuard') as monitor:
            monitor.return_value.preflight.side_effect=lambda: events.append('memory')
            with safety.execution_guard('grading'): events.append('launch')
        self.assertEqual(events,['memory','launch'])

if __name__=='__main__': unittest.main()
