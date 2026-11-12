import ast
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import zipfile

from tools import colab_archive, colab_setup, notebook06_bundle


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / 'ipynb/06_expert80_benchmark.ipynb'


def make_zip(path, files, *, bad_digest=False, duplicate=False, link=None):
    inventory = {name: hashlib.sha256(data).hexdigest() for name, data in files.items()}
    if bad_digest:
        inventory[next(iter(inventory))] = '0' * 64
    with zipfile.ZipFile(path, 'w') as archive:
        archive.writestr('_putare_inventory.json', json.dumps(inventory))
        for name, data in files.items():
            item = zipfile.ZipInfo(name)
            if name == link:
                item.external_attr = 0o120777 << 16
            archive.writestr(item, data)
            if duplicate:
                archive.writestr(item, data)
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Notebook06SetupTests(unittest.TestCase):
    def test_restore_both_archives_is_idempotent_and_collision_safe(self):
        with tempfile.TemporaryDirectory() as temp:
            temp = Path(temp)
            source, data, root = temp / 's.zip', temp / 'd.zip', temp / 'out'
            source_sha = make_zip(source, {'tools/x.py': b'source'})
            data_sha = make_zip(data, {'data/y.json': b'data'})
            archives = [(source, source_sha), (data, data_sha)]
            self.assertEqual(colab_archive.restore_archives(root, archives), 2)
            self.assertEqual(colab_archive.restore_archives(root, archives), 2)
            (root / 'data/y.json').write_bytes(b'changed')
            (root / 'tools/x.py').unlink()
            with self.assertRaises(FileExistsError):
                colab_archive.restore_archives(root, archives)
            self.assertFalse((root / 'tools/x.py').exists())
            self.assertEqual((root / 'data/y.json').read_bytes(), b'changed')

    def test_rejects_invalid_archives_before_writing(self):
        for kind in ('outer_digest', 'inner_digest', 'traversal', 'absolute', 'link', 'duplicate'):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as temp:
                temp = Path(temp)
                source, data, root = temp / 's.zip', temp / 'd.zip', temp / 'out'
                source_sha = make_zip(source, {'good/file': b'good'})
                name = '../escape' if kind == 'traversal' else '/escape' if kind == 'absolute' else 'bad/file'
                with __import__('warnings').catch_warnings():
                    __import__('warnings').simplefilter('ignore', UserWarning)
                    data_sha = make_zip(data, {name: b'data'}, bad_digest=kind == 'inner_digest',
                                        duplicate=kind == 'duplicate', link=name if kind == 'link' else None)
                if kind == 'outer_digest':
                    data_sha = '0' * 64
                with self.assertRaises(ValueError):
                    colab_archive.restore_archives(root, [(source, source_sha), (data, data_sha)])
                self.assertFalse(root.exists())

    def test_missing_data_archive_does_not_restore_source(self):
        with tempfile.TemporaryDirectory() as temp:
            temp = Path(temp)
            source, root = temp / 's.zip', temp / 'out'
            source_sha = make_zip(source, {'good/file': b'good'})
            with self.assertRaises(FileNotFoundError):
                colab_archive.restore_archives(root, [(source, source_sha), (temp / 'missing.zip', '0' * 64)])
            self.assertFalse(root.exists())

    def test_source_only_rebuild_reuses_identical_data_zip(self):
        with tempfile.TemporaryDirectory() as temp:
            temp = Path(temp)
            (temp / 'source.py').write_text('one', encoding='utf-8')
            (temp / 'data.bin').write_bytes(b'data')
            with patch.object(notebook06_bundle, 'SOURCE_FILES', ['source.py']), \
                 patch.object(notebook06_bundle, 'data_paths', return_value=['data.bin']):
                first = notebook06_bundle.build_artifacts(temp, temp / 'dist')
                stamp = first['data']['path'].stat().st_mtime_ns
                (temp / 'source.py').write_text('two', encoding='utf-8')
                second = notebook06_bundle.build_artifacts(temp, temp / 'dist')
            self.assertNotEqual(first['source']['sha256'], second['source']['sha256'])
            self.assertEqual(first['data']['path'], second['data']['path'])
            self.assertFalse(second['data']['created'])
            self.assertEqual(stamp, second['data']['path'].stat().st_mtime_ns)

    def test_existing_custom_zip_locations_are_preserved(self):
        artifacts = {'source': {'path': Path('putare-source-222222222222.zip')},
                     'data': {'path': Path('putare-data-333333333333.zip')}}
        source = "SOURCE_ZIP = '/custom/source.zip'\nDATA_ZIP = '/content/drive/MyDrive/Putare/putare-data-111111111111.zip'\ncustom = 42\n"
        updated = notebook06_bundle._update_settings(source, {'data': '1' * 64}, artifacts)
        self.assertIn("SOURCE_ZIP = '/custom/source.zip'", updated)
        self.assertIn("DATA_ZIP = '/content/drive/MyDrive/Putare/putare-data-333333333333.zip'", updated)
        self.assertIn('custom = 42', updated)

    def test_custom_expressions_and_notebook_magics_are_preserved(self):
        artifacts = {'source': {'path': Path('s.zip')}, 'data': {'path': Path('d.zip')}}
        source = "%pip install something\nSOURCE_ZIP = str(my_custom_path)\ncustom = 42\n"
        updated = notebook06_bundle._update_settings(source, {}, artifacts)
        self.assertIn('%pip install something', updated)
        self.assertIn('SOURCE_ZIP = str(my_custom_path)', updated)
        self.assertEqual(updated.count('SOURCE_ZIP ='), 1)
        self.assertIn("DATA_ZIP = '/content/drive/MyDrive/Putare/d.zip'", updated)

    def test_valid_same_kernel_state_reuses_prepared_environment(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            engine = root / '.cache/llama-build/bin/llama-expert-research'
            grader = root / '.cache/putare-lcb/venv/bin/python'
            engine.parent.mkdir(parents=True)
            grader.parent.mkdir(parents=True)
            engine.write_bytes(b'engine')
            grader.write_bytes(b'python')
            state = {'root': str(root), 'python': sys.executable, 'gpu_uuid': 'uuid', 'gpu_target': '120',
                     'engine_sha256': hashlib.sha256(b'engine').hexdigest()}
            with patch.object(colab_setup.shutil, 'which', return_value='/tool'), \
                 patch.object(colab_setup.importlib.util, 'find_spec', return_value=object()), \
                 patch.object(colab_setup, '_gpu', return_value={'uuid': 'uuid', 'target': '120'}), \
                 patch.object(colab_setup.subprocess, 'run', side_effect=AssertionError('unexpected reinstall')):
                self.assertIs(colab_setup.prepare_environment(root, state), state)
                grader.unlink()
                self.assertFalse(colab_setup._reusable(root, state))
                grader.write_bytes(b'python')
                with patch.object(colab_setup.importlib.util, 'find_spec', return_value=None):
                    self.assertFalse(colab_setup._reusable(root, state))
                engine.write_bytes(b'changed')
                self.assertFalse(colab_setup._reusable(root, state))

    def test_missing_ninja_is_installed_and_detected_gpu_builds(self):
        for target in ('80', '120'):
            with self.subTest(target=target), tempfile.TemporaryDirectory() as temp:
                root = Path(temp).resolve()
                installed = False
                def which(name):
                    return None if name == 'ninja' and not installed else '/tool'
                def run(command, **kwargs):
                    nonlocal installed
                    if command[-1] == 'ninja':
                        installed = True
                    if '--execute' in command:
                        engine = root / '.cache/llama-build/bin/llama-expert-research'
                        engine.parent.mkdir(parents=True, exist_ok=True)
                        engine.write_bytes(b'engine')
                    return SimpleNamespace(stdout=f'sm_{target}\n')
                gpu = {'name': 'gpu', 'uuid': 'uuid', 'target': target}
                with patch.object(colab_setup.shutil, 'which', side_effect=which), \
                     patch.object(colab_setup, '_gpu', return_value=gpu), \
                     patch.object(colab_setup.subprocess, 'run', side_effect=run) as runner, \
                     patch('moe_prune.benchmark_data.prepare_grader'):
                    state = colab_setup.prepare_environment(root)
                self.assertEqual(state['gpu_target'], target)
                commands = [call.args[0] for call in runner.call_args_list]
                self.assertTrue(any(cmd[-3:] == ['--isolated', 'install', 'ninja'] for cmd in commands))
                self.assertTrue(any('--execute' in cmd and cmd[cmd.index('--gpu-target') + 1] == target for cmd in commands))

    def test_unsupported_gpu_stops_before_build(self):
        with tempfile.TemporaryDirectory() as temp, \
             patch.object(colab_setup.shutil, 'which', return_value='/tool'), \
             patch.object(colab_setup, '_gpu', return_value={'name': 'gpu', 'uuid': 'uuid', 'target': '120'}), \
             patch.object(colab_setup.subprocess, 'run', return_value=SimpleNamespace(stdout='sm_80\n')) as runner:
            with self.assertRaisesRegex(RuntimeError, 'sm_120'):
                colab_setup.prepare_environment(Path(temp))
            self.assertFalse(any('--execute' in call.args[0] for call in runner.call_args_list))

    def test_bootstrap_restores_versions_and_rejects_stale_modules(self):
        notebook = json.loads(NOTEBOOK.read_text(encoding='utf-8'))
        setup = ''.join(next(c for c in notebook['cells'] if c.get('metadata', {}).get('putare_setup') == 'setup')['source'])
        previous = notebook06_bundle._old_hashes(notebook['cells'])
        with tempfile.TemporaryDirectory() as temp:
            temp = Path(temp)
            source, data = temp / 's.zip', temp / 'd.zip'
            source_sha = make_zip(source, {'tools/colab_setup.py': b'def prepare_environment(root, state=None):\n    return {"root": str(root)}\n'})
            data_sha = make_zip(data, {'data/x': b'hello'})
            setup = setup.replace(previous['source'], source_sha).replace(previous['data'], data_sha)
            setup = setup.replace("ROOT = Path('/content/Putare')", f'ROOT = Path({str(temp / "runtime")!r})')
            script = """import os, sys\nos.path.ismount = lambda path: True\nns = {'SETUP_COLAB': True, 'MODEL_PATH': None, 'SOURCE_ZIP': sys.argv[1], 'DATA_ZIP': sys.argv[2]}\nexec(sys.argv[3], ns)\nassert ns['_PUTARE_SETUP_READY']\nassert (ns['ROOT'] / 'data/x').read_bytes() == b'hello'\nns['MODEL_PATH'] = object()\ntry:\n    exec(sys.argv[3], ns)\n    raise AssertionError('invalid model path accepted')\nexcept TypeError:\n    pass\nassert not ns['_PUTARE_SETUP_READY']\nns['MODEL_PATH'] = None\nsys.modules['tools.colab_setup'].__file__ = '/different/old/colab_setup.py'\ntry:\n    exec(sys.argv[3], ns)\n    raise AssertionError('stale module accepted')\nexcept RuntimeError as error:\n    assert '커널' in str(error)\nassert not ns['_PUTARE_SETUP_READY']\n"""
            result = subprocess.run([sys.executable, '-c', script, str(source), str(data), setup],
                                    cwd=temp, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr.decode(errors='replace'))

    def test_notebook_uses_two_zip_settings(self):
        notebook = json.loads(NOTEBOOK.read_text(encoding='utf-8'))
        settings = next(c for c in notebook['cells'] if c.get('metadata', {}).get('putare_setup') == 'settings')
        source = ''.join(settings['source'])
        self.assertIn('SOURCE_ZIP = ', source)
        self.assertIn('DATA_ZIP = ', source)

    def test_updater_preserves_user_cells_and_settings(self):
        settings = "SETUP_COLAB = True\nfrom google.colab import drive\ndrive.mount('/content/drive')\nMODEL_PATH = '/custom/model.gguf'\nuser_note = 42\n"
        diagnostic = {'cell_type': 'code', 'metadata': {}, 'source': ['print(123)\n'],
                      'execution_count': 2, 'outputs': [{'output_type': 'stream', 'name': 'stdout', 'text': ['123\n']}]}
        notebook = {'cells': [{'cell_type': 'code', 'source': [settings], 'metadata': {'putare_setup': 'settings'}},
                              diagnostic, {'cell_type': 'code', 'source': ['old'], 'metadata': {'putare_setup': 'setup'},
                                           'execution_count': 1, 'outputs': [{'output_type': 'stream', 'name': 'stdout', 'text': ['old\n']}]}]}
        artifacts = {'source': {'path': Path('s.zip'), 'sha256': '1' * 64, 'created': True},
                     'data': {'path': Path('d.zip'), 'sha256': '2' * 64, 'created': True}}
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'notebook.ipynb'
            path.write_text(json.dumps(notebook), encoding='utf-8')
            with patch.object(notebook06_bundle, 'NOTEBOOK', path), patch.object(notebook06_bundle, 'build_artifacts', return_value=artifacts):
                notebook06_bundle.update_notebook()
            updated = json.loads(path.read_text(encoding='utf-8'))
        settings_result = ''.join(updated['cells'][0]['source'])
        self.assertIn(settings, settings_result)
        self.assertEqual(updated['cells'][1], diagnostic)
        self.assertEqual(updated['cells'][2]['outputs'], notebook['cells'][2]['outputs'])
        self.assertEqual(updated['cells'][2]['execution_count'], 1)

    def test_packager_refuses_concurrent_notebook_edit(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'notebook.ipynb'
            notebook = {'cells': [{'cell_type': 'code', 'source': [''], 'metadata': {'putare_setup': 'settings'}},
                                  {'cell_type': 'code', 'source': [''], 'metadata': {'putare_setup': 'setup'}}]}
            path.write_text(json.dumps(notebook), encoding='utf-8')
            artifacts = {'source': {'path': Path('s.zip'), 'sha256': '1' * 64, 'created': True},
                         'data': {'path': Path('d.zip'), 'sha256': '2' * 64, 'created': True}}
            def modify():
                path.write_text('changed', encoding='utf-8')
                return artifacts
            with patch.object(notebook06_bundle, 'NOTEBOOK', path), patch.object(notebook06_bundle, 'build_artifacts', side_effect=modify):
                with self.assertRaisesRegex(RuntimeError, '변경'):
                    notebook06_bundle.update_notebook()
            self.assertEqual(path.read_text(encoding='utf-8'), 'changed')

    def test_failed_atomic_replace_preserves_original_notebook(self):
        notebook = {'cells': [{'cell_type': 'code', 'source': [''], 'metadata': {'putare_setup': 'settings'}},
                              {'cell_type': 'code', 'source': [''], 'metadata': {'putare_setup': 'setup'}}]}
        artifacts = {'source': {'path': Path('s.zip'), 'sha256': '1' * 64, 'created': True},
                     'data': {'path': Path('d.zip'), 'sha256': '2' * 64, 'created': True}}
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'notebook.ipynb'
            original = json.dumps(notebook)
            path.write_text(original, encoding='utf-8')
            with patch.object(notebook06_bundle, 'NOTEBOOK', path), \
                 patch.object(notebook06_bundle, 'build_artifacts', return_value=artifacts), \
                 patch('os.replace', side_effect=OSError('locked')):
                with self.assertRaisesRegex(OSError, 'locked'):
                    notebook06_bundle.update_notebook()
            self.assertEqual(path.read_text(encoding='utf-8'), original)


if __name__ == '__main__':
    unittest.main()
