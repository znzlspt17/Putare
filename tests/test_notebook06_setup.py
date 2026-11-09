import ast
import json
from pathlib import Path
import unittest
import tempfile
import hashlib
import io
import zipfile
from types import SimpleNamespace
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / 'ipynb/06_expert80_benchmark.ipynb'


class Notebook06SetupTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.notebook = json.loads(NOTEBOOK.read_text(encoding='utf-8'))

    def tagged_source(self, tag):
        cells = [c for c in self.notebook['cells'] if c.get('metadata', {}).get('putare_setup') == tag]
        self.assertEqual(len(cells), 1)
        return ''.join(cells[0]['source'])

    def restore_function(self):
        for cell in self.notebook['cells']:
            if cell['cell_type'] != 'code':
                continue
            if cell.get('metadata', {}).get('putare_setup') == 'payload':
                continue
            source = ''.join(cell['source'])
            tree = ast.parse(source)
            functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'restore_embedded_bundle']
            if functions:
                ns = {}
                exec(compile(ast.Module(body=functions, type_ignores=[]), '<restore>', 'exec'), ns)
                return ns['restore_embedded_bundle']
        self.fail('노트북에 번들 복원 함수가 없습니다')

    def archive(self, files):
        import base64
        entries = {name: hashlib.sha256(data).hexdigest() for name, data in files.items()}
        output = io.BytesIO()
        with zipfile.ZipFile(output, 'w', zipfile.ZIP_DEFLATED) as z:
            z.writestr('_putare_inventory.json', json.dumps(entries))
            for name, content in files.items():
                z.writestr(name, content)
        payload = output.getvalue()
        return base64.b64encode(payload).decode(), hashlib.sha256(payload).hexdigest()

    def test_bundle_restore_reuses_identical_and_rejects_conflict(self):
        restore = self.restore_function()
        encoded, digest = self.archive({'moe_prune/a.py': b'hello', 'tools/b.py': b'world'})
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            restore(root, encoded, digest)
            restore(root, encoded, digest)
            self.assertEqual((root / 'moe_prune/a.py').read_bytes(), b'hello')
            (root / 'tools/b.py').write_bytes(b'changed')
            with self.assertRaises((ValueError, FileExistsError)):
                restore(root, encoded, digest)
            self.assertEqual((root / 'tools/b.py').read_bytes(), b'changed')

    def test_bundle_rejects_corruption_and_unsafe_paths_before_write(self):
        restore = self.restore_function()
        for files, corrupt in [({'../escape': b'x'}, False), ({'a': b'x'}, True)]:
            encoded, digest = self.archive(files)
            with tempfile.TemporaryDirectory() as temp:
                root = Path(temp) / 'new'
                with self.assertRaises(ValueError):
                    restore(root, encoded, '0' * 64 if corrupt else digest)
                self.assertFalse(root.exists())

    def test_setup_is_present_and_disabled_by_default(self):
        assignments = {}
        for cell in self.notebook['cells']:
            if cell['cell_type'] != 'code':
                continue
            if cell.get('metadata', {}).get('putare_setup') == 'payload':
                continue
            for node in ast.walk(ast.parse(''.join(cell['source']))):
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name) and isinstance(node.value, ast.Constant):
                            assignments[target.id] = node.value.value
        self.assertIn('SETUP_COLAB', assignments)
        self.assertIs(assignments['SETUP_COLAB'], False)

    def test_setup_detects_both_gpu_architectures_and_allows_missing_model(self):
        import os
        import shutil
        import subprocess
        import sys
        from moe_prune import benchmark_data
        from tools.native import gpu_memory
        for major, minor in ((8, 0), (12, 0)):
            with self.subTest(capability=(major, minor)):
                fake_nvml = SimpleNamespace(nvmlDeviceGetCudaComputeCapability=lambda handle: (major, minor))
                fake_device = SimpleNamespace(name='GPU', nvml=fake_nvml, handle=0, close=Mock())
                run = Mock(return_value=SimpleNamespace(stdout='sm_80\nsm_120\n'))
                namespace = {'SETUP_COLAB': True, '_BUNDLE_DATA': 'synthetic', '_BUNDLE_SHA256': 'synthetic',
                             'MODEL_PATH': '/missing/model.gguf', '_PUTARE_SETUP_READY': False,
                             'restore_embedded_bundle': Mock(return_value=1)}
                with patch.object(sys, 'version_info', (3, 12, 0)), patch.object(sys, 'path', list(sys.path)), \
                     patch.object(os, 'chdir'), patch.object(shutil, 'which', return_value='/bin/tool'), \
                     patch.object(subprocess, 'run', run), patch.object(gpu_memory, 'NvmlDevice', return_value=fake_device), \
                     patch.object(benchmark_data, 'prepare_grader'):
                    exec(compile(self.tagged_source('setup'), '<setup>', 'exec'), namespace)
                self.assertTrue(namespace['_PUTARE_SETUP_READY'])
                commands = [c.args[0] for c in run.call_args_list]
                build = next(cmd for cmd in commands if any('prepare.py' in str(x) for x in cmd))
                self.assertEqual(build[build.index('--gpu-target') + 1], f'{major}{minor}')
                self.assertIn('--execute', build)
                self.assertEqual(fake_device.close.call_count, 1)

    def test_failed_setup_blocks_generation_even_if_model_exists(self):
        generation = self.tagged_source('generation')
        ns = {'RUN_GENERATION': True, '_PUTARE_SETUP_READY': False}
        with self.assertRaisesRegex(RuntimeError, '환경 준비'):
            exec(compile(generation, '<generation>', 'exec'), ns)

    def test_preflight_failure_stays_blocked_when_setup_is_switched_off(self):
        import shutil
        import sys
        ns = {'SETUP_COLAB': True, '_PUTARE_SETUP_READY': False,
              '_BUNDLE_DATA': 'unused', '_BUNDLE_SHA256': 'unused', 'MODEL_PATH': None}
        with patch.object(sys, 'version_info', (3, 12, 0)), patch.object(shutil, 'which', return_value=None):
            with self.assertRaisesRegex(RuntimeError, '빌드 도구'):
                exec(compile(self.tagged_source('setup'), '<setup>', 'exec'), ns)
        exec(compile(self.tagged_source('settings'), '<settings>', 'exec'), ns)
        self.assertFalse(ns['_PUTARE_SETUP_READY'])
        ns['RUN_GENERATION'] = True
        with self.assertRaisesRegex(RuntimeError, '환경 준비'):
            exec(compile(self.tagged_source('generation'), '<generation>', 'exec'), ns)

    def test_repeated_setup_failure_clears_prior_success(self):
        import os
        import shutil
        import subprocess
        import sys
        from tools.native import gpu_memory
        nvml = SimpleNamespace(nvmlDeviceGetCudaComputeCapability=lambda handle: (12, 0))
        device = SimpleNamespace(name='GPU', nvml=nvml, handle=0, close=Mock())
        ns = {'SETUP_COLAB': True, '_PUTARE_SETUP_READY': True, 'MODEL_PATH': None,
              '_BUNDLE_DATA': 'unused', '_BUNDLE_SHA256': 'unused',
              'restore_embedded_bundle': Mock(return_value=1)}
        with patch.object(sys, 'version_info', (3, 12, 0)), patch.object(sys, 'path', list(sys.path)), \
             patch.object(os, 'chdir'), patch.object(shutil, 'which', return_value='/bin/tool'), \
             patch.object(subprocess, 'run', return_value=SimpleNamespace(stdout='sm_80\n')), \
             patch.object(gpu_memory, 'NvmlDevice', return_value=device):
            with self.assertRaisesRegex(RuntimeError, 'sm_120'):
                exec(compile(self.tagged_source('setup'), '<setup>', 'exec'), ns)
        self.assertFalse(ns['_PUTARE_SETUP_READY'])
        ns['RUN_GENERATION'] = True
        with self.assertRaisesRegex(RuntimeError, '환경 준비'):
            exec(compile(self.tagged_source('generation'), '<generation>', 'exec'), ns)

    def test_generation_reports_missing_model(self):
        generation = self.tagged_source('generation')
        with tempfile.TemporaryDirectory() as temp:
            ns = {'RUN_GENERATION': True, '_PUTARE_SETUP_READY': True, 'MODEL': Path(temp) / 'missing.gguf'}
            with self.assertRaisesRegex(FileNotFoundError, '모델 파일'):
                exec(compile(generation, '<generation>', 'exec'), ns)

    def test_embedded_bootstrap_matches_updater(self):
        from tools.notebook06_bundle import RESTORE_SOURCE, SETUP_SOURCE, GENERATION_SOURCE
        self.assertEqual(self.tagged_source('restore'), RESTORE_SOURCE)
        self.assertEqual(self.tagged_source('setup'), SETUP_SOURCE)
        self.assertEqual(self.tagged_source('generation'), GENERATION_SOURCE)


if __name__ == '__main__':
    unittest.main()
