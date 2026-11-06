"""2026-11-06: 공식 채점 코드를 실행할 격리 경계를 검증한다."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
import sys
import json
import subprocess


ROOT = Path(__file__).resolve().parents[1]


class SandboxContractTests(unittest.TestCase):
    def module(self):
        spec = importlib.util.spec_from_file_location('lcb_sandbox', ROOT / 'tools/lcb_sandbox.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_command_isolates_network_mounts_processes_and_environment(self):
        from unittest.mock import patch
        module = self.module()
        with patch.object(module.shutil, 'which', return_value='/usr/bin/unshare'):
            command = module.namespace_command(Path('/tmp/work'), Path('/source'), Path('/venv'), Path('/adapter.py'))
        for option in ('--user', '--map-root-user', '--mount', '--pid', '--fork', '--net', '--ipc', '--uts', '--kill-child'):
            self.assertIn(option, command)
        self.assertEqual(module.worker_environment().get('HOME'), '/tmp')
        self.assertNotIn('HF_TOKEN', module.worker_environment())
        self.assertEqual(command[-1], str(Path('/adapter.py')))

    def test_result_symlink_and_oversize_are_rejected(self):
        module = self.module()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'result.json'
            path.write_bytes(b'12345')
            with self.assertRaises(ValueError):
                module.read_result(path, maximum_bytes=4)
            self.assertEqual(module.read_result(path, maximum_bytes=5), b'12345')
            target = Path(directory) / 'elsewhere'
            target.write_bytes(b'{}')
            path.unlink()
            try:
                path.symlink_to(target)
            except OSError:
                self.skipTest('symlink 권한 없음')
            with self.assertRaises((OSError, ValueError)):
                module.read_result(path)

    def test_missing_namespace_tool_never_runs_worker(self):
        from unittest.mock import patch
        module = self.module()
        with patch.object(module.shutil, 'which', return_value=None):
            with self.assertRaisesRegex(RuntimeError, 'unshare'):
                module.namespace_command(Path('/tmp/work'), Path('/source'), Path('/venv'), Path('/adapter.py'))

    @unittest.skipUnless(sys.platform == 'linux', '실제 namespace는 WSL Linux에서 검증')
    def test_real_worker_cannot_see_host_paths_or_retain_capabilities(self):
        module = self.module()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, venv = root / 'source', root / 'venv'
            source.mkdir()
            subprocess.run([sys.executable, '-m', 'venv', '--without-pip', str(venv)], check=True)
            adapter = root / 'adapter.py'
            adapter.write_text('''import argparse,json,os,pathlib,socket
p=argparse.ArgumentParser()
p.add_argument('--worker',action='store_true')
for key in ['official-source','input','output']: p.add_argument('--'+key)
a=p.parse_args()
s=socket.socket(); s.settimeout(.1)
try: s.connect(('1.1.1.1',9)); network=True
except OSError: network=False
finally: s.close()
caps=next(line.split(':')[1].strip() for line in pathlib.Path('/proc/self/status').read_text().splitlines() if line.startswith('CapEff:'))
r={'host_home_visible':pathlib.Path('/home/vivy').exists(),'host_drive_visible':pathlib.Path('/mnt/d').exists(),'network':network,'capabilities':caps,'input':json.loads(pathlib.Path(a.input).read_text())}
pathlib.Path(a.output).write_text(json.dumps(r))
''', encoding='utf-8')
            input_path, output = root / 'input.json', root / 'result.json'
            input_path.write_text('{"test":42}', encoding='utf-8')
            module.run(source, venv, input_path, output, adapter=adapter, timeout_s=30)
            result = json.loads(output.read_text())
            self.assertFalse(result['host_home_visible'])
            self.assertFalse(result['host_drive_visible'])
            self.assertFalse(result['network'])
            self.assertEqual(int(result['capabilities'], 16), 0)
            self.assertEqual(result['input'], {'test': 42})


if __name__ == '__main__':
    unittest.main()
