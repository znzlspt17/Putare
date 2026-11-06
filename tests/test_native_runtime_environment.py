"""ROCm DLL 검색 경로는 자식 시작에만 적용하고 원래 PATH를 복원한다."""
import os
from pathlib import Path
import tempfile
import sys
import unittest
from unittest.mock import patch

from moe_prune.research import native_runtime_directories, spawn_native


class NativeRuntimeEnvironmentTests(unittest.TestCase):
    def test_discovers_only_existing_sdk_directories(self):
        with tempfile.TemporaryDirectory() as directory:
            prefix = Path(directory)
            target = prefix / 'Lib/site-packages/_rocm_sdk_devel/bin'
            target.mkdir(parents=True)
            self.assertEqual(native_runtime_directories(prefix), [str(target)])

    def test_child_sees_sdk_path_and_parent_path_is_restored(self):
        old = os.environ.get('PATH')
        def launch(*args, **kwargs):
            self.assertEqual(kwargs['env']['PATH'].split(os.pathsep)[0], 'sdk-bin')
            self.assertEqual(os.environ.get('PATH'), old)
            self.assertNotIn('UNRELATED_TEST_VARIABLE', kwargs['env'])
            return object()
        os.environ['UNRELATED_TEST_VARIABLE'] = 'synthetic'
        try:
            with patch('moe_prune.research.native_runtime_directories', return_value=['sdk-bin']), \
                 patch('moe_prune.research.subprocess.Popen', side_effect=launch):
                spawn_native(['engine'])
        finally:
            os.environ.pop('UNRELATED_TEST_VARIABLE')
        self.assertEqual(os.environ.get('PATH'), old)

    def test_path_is_restored_if_spawn_raises(self):
        old = os.environ.get('PATH')
        with patch('moe_prune.research.native_runtime_directories', return_value=['sdk-bin']), \
             patch('moe_prune.research.subprocess.Popen', side_effect=OSError('launch failed')):
            with self.assertRaises(OSError):
                spawn_native(['engine'])
        self.assertEqual(os.environ.get('PATH'), old)

    def test_real_subprocess_inherits_temporary_path(self):
        old = os.environ.get('PATH')
        script = 'import os,sys; raise SystemExit(not os.environ["PATH"].startswith(sys.argv[1] + os.pathsep))'
        with patch('moe_prune.research.native_runtime_directories', return_value=['sdk-bin']):
            process = spawn_native([sys.executable, '-c', script, 'sdk-bin'])
            self.assertEqual(process.wait(timeout=10), 0)
        self.assertEqual(os.environ.get('PATH'), old)
