from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tools import lcb_prepare


class GraderPipPreparationTests(unittest.TestCase):
    def _fixture(self, root):
        python = root / '.cache/putare-lcb/venv/bin/python'
        python.parent.mkdir(parents=True)
        python.write_bytes(b'fixture')
        source = root / '.cache/putare-lcb/livecodebench' / lcb_prepare.COMMIT / 'lcb_runner/evaluation/testing_util.py'
        source.parent.mkdir(parents=True)
        source.write_text('# fixture', encoding='utf-8')
        return python

    def _path(self, path):
        if path == '/proc/version':
            return SimpleNamespace(is_file=lambda: True)
        return Path(path)

    def test_existing_venv_without_pip_is_repaired_before_install(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            venv_python = self._fixture(root)
            installed = False
            def run(args, **kwargs):
                nonlocal installed
                if '--version' in args and '-m' in args and 'pip' in args:
                    return subprocess.CompletedProcess(args, 0 if installed else 1,
                                                       stdout='', stderr='' if installed else 'No module named pip')
                if '--python' in args and args[-1] == 'pip':
                    installed = True
                    return subprocess.CompletedProcess(args, 0, stdout='', stderr='')
                if 'numpy>=2,<3' in args and not installed:
                    raise subprocess.CalledProcessError(1, args, stderr='No module named pip')
                return subprocess.CompletedProcess(args, 0, stdout='', stderr='')
            with patch.object(lcb_prepare, 'os', SimpleNamespace(name='posix')), \
                 patch.object(lcb_prepare, 'Path', side_effect=self._path), \
                 patch('moe_prune.grading.worker_environment', return_value={}), \
                 patch.object(lcb_prepare.subprocess, 'run', side_effect=run) as runner:
                result = lcb_prepare.prepare(root)
            self.assertTrue(installed)
            self.assertEqual(result['venv'], str(venv_python.parent.parent))
            self.assertTrue(venv_python.is_file())
            calls = [item.args[0] for item in runner.call_args_list]
            bootstrap = next(cmd for cmd in calls if '--python' in cmd)
            self.assertEqual(bootstrap[bootstrap.index('--python') + 1], str(venv_python))
            self.assertTrue(any('numpy>=2,<3' in cmd for cmd in calls))

    def test_other_interpreter_failure_is_not_treated_as_missing_pip(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self._fixture(root)
            def run(args, **kwargs):
                if '--version' in args:
                    return subprocess.CompletedProcess(args, 1, stdout='', stderr='Fatal Python error: broken interpreter')
                raise AssertionError('실패한 인터프리터를 자동 수정하면 안 됩니다')
            with patch.object(lcb_prepare, 'os', SimpleNamespace(name='posix')), \
                 patch.object(lcb_prepare, 'Path', side_effect=self._path), \
                 patch('moe_prune.grading.worker_environment', return_value={}), \
                 patch.object(lcb_prepare.subprocess, 'run', side_effect=run) as runner:
                with self.assertRaisesRegex(RuntimeError, 'broken interpreter'):
                    lcb_prepare.prepare(root)
            self.assertFalse(any('--python' in call.args[0] for call in runner.call_args_list))

    def test_healthy_pip_is_not_bootstrapped(self):
        result = subprocess.CompletedProcess(['python'], 0, stdout='pip available', stderr='')
        with patch.object(lcb_prepare.subprocess, 'run', return_value=result) as run:
            lcb_prepare._ensure_pip(Path('venv/python'), {'PATH': '/bin'})
        self.assertEqual(run.call_count, 1)
        self.assertEqual(run.call_args.kwargs['env'], {'PATH': '/bin'})

    def test_bootstrap_failure_is_reported(self):
        failed_probe = subprocess.CompletedProcess(['probe'], 1, stdout='', stderr='No module named pip')
        failed_install = subprocess.CompletedProcess(['install'], 1, stdout='', stderr='package index unavailable')
        with patch.object(lcb_prepare.subprocess, 'run', side_effect=[failed_probe, failed_install]):
            with self.assertRaisesRegex(RuntimeError, 'package index unavailable'):
                lcb_prepare._ensure_pip(Path('venv/python'), {})

    def test_new_venv_is_created_without_ensurepip_then_bootstrapped(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            venv_python = self._fixture(root)
            venv_python.unlink()
            installed = False
            def run(args, **kwargs):
                nonlocal installed
                if '-m' in args and 'venv' in args:
                    self.assertIn('--without-pip', args)
                    venv_python.write_bytes(b'fixture')
                elif '--version' in args:
                    return subprocess.CompletedProcess(args, 0 if installed else 1,
                                                       stdout='', stderr='' if installed else 'No module named pip')
                elif '--python' in args:
                    installed = True
                return subprocess.CompletedProcess(args, 0, stdout='', stderr='')
            with patch.object(lcb_prepare, 'os', SimpleNamespace(name='posix')), \
                 patch.object(lcb_prepare, 'Path', side_effect=self._path), \
                 patch('moe_prune.grading.worker_environment', return_value={}), \
                 patch.object(lcb_prepare.subprocess, 'run', side_effect=run):
                lcb_prepare.prepare(root)
            self.assertTrue(installed)


if __name__ == '__main__':
    unittest.main()
