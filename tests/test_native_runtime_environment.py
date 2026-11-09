"""자식 추론 프로세스에는 필요한 Linux CUDA 변수만 전달한다."""
import os
import unittest
from unittest.mock import patch
from moe_prune.research import spawn_native

class NativeRuntimeEnvironmentTests(unittest.TestCase):
    def test_allowlisted_cuda_variables_without_unrelated_values(self):
        fake={'PATH':'/bin','CUDA_VISIBLE_DEVICES':'GPU-test','CUDA_DEVICE_ORDER':'PCI_BUS_ID','UNRELATED_TEST_VARIABLE':'synthetic'}
        with patch('moe_prune.research.os.environ',fake), patch('moe_prune.research.subprocess.Popen') as launch:
            spawn_native(['engine'])
        env=launch.call_args.kwargs['env']
        self.assertEqual(env['CUDA_VISIBLE_DEVICES'],'GPU-test');self.assertEqual(env['PATH'],'/bin')
        self.assertNotIn('UNRELATED_TEST_VARIABLE',env)
    def test_spawn_failure_propagates(self):
        with patch('moe_prune.research.subprocess.Popen',side_effect=OSError('launch failed')):
            with self.assertRaisesRegex(OSError,'launch failed'):spawn_native(['engine'])
    def test_no_implicit_rocm_variables(self):
        fake={'PATH':'/bin','HIP_PATH':'fake-hip','ROCM_PATH':'fake-rocm'}
        with patch('moe_prune.research.os.environ',fake), patch('moe_prune.research.subprocess.Popen') as launch:
            spawn_native(['engine'])
        self.assertNotIn('HIP_PATH',launch.call_args.kwargs['env'])
        self.assertNotIn('ROCM_PATH',launch.call_args.kwargs['env'])

if __name__=='__main__':unittest.main()
