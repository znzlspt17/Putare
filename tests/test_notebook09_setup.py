import contextlib
import io
import json
import hashlib
import socket
import subprocess
import sys
import tempfile
import types
import unittest
import zipfile
from pathlib import Path
from unittest.mock import Mock, patch

from tools import notebook09_bundle as bundle


def artifacts():
    return {name: {'path': Path(f'putare09-{name}-' + 'a' * 12 + '.zip'), 'sha256': 'a' * 64}
            for name in ('source', 'data')}


class Notebook09Test(unittest.TestCase):
    def test_default_execution_is_inert_and_all_code_compiles(self):
        notebook = bundle.notebook_document(artifacts())
        cells = [cell for cell in notebook['cells'] if cell['cell_type'] == 'code']
        self.assertGreaterEqual(len(cells), 6)
        source = '\n'.join(''.join(cell['source']) for cell in cells)
        for flag in ('SETUP_COLAB', 'DOWNLOAD_MODEL', 'RUN_PREFLIGHT',
                     'RUN_GENERATION', 'RUN_GRADING', 'EXPORT_RESULTS', 'DOWNLOAD_ARCHIVE'):
            self.assertIn(flag + ' = False', source)
        with patch.object(subprocess, 'run', side_effect=AssertionError('process')), \
             patch.object(subprocess, 'Popen', side_effect=AssertionError('process')), \
             patch.object(socket.socket, 'connect', side_effect=AssertionError('network')), \
             contextlib.redirect_stdout(io.StringIO()):
            scope = {}
            for cell in cells:
                code = compile(''.join(cell['source']), cell['id'], 'exec')
                exec(code, scope)

    def test_bundle_has_required_allowlisted_inputs_only(self):
        self.assertIn('tools/comparison205.py', bundle.SOURCE_FILES)
        self.assertIn('tools/native/expert_research.cpp', bundle.SOURCE_FILES)
        self.assertIn('runs/gemma4-ordered-e80-001/export-manifest.json', bundle.DATA_FILES)
        self.assertIn('runs/gemma4-lcb100-reordered-001/permutation-manifest.json', bundle.DATA_FILES)
        self.assertTrue(any(p.endswith('/records.jsonl') for p in bundle.DATA_FILES))
        self.assertFalse(any(Path(name).name in ('.env', '.npmrc', 'token.json', 'credentials.json')
                             for name in bundle.SOURCE_FILES + bundle.DATA_FILES))

    def test_generation_requires_setup(self):
        notebook = bundle.notebook_document(artifacts())
        cell = next(c for c in notebook['cells'] if c.get('metadata', {}).get('putare09') == 'generation')
        with self.assertRaisesRegex(RuntimeError, '준비'):
            exec(''.join(cell['source']), {'RUN_GENERATION': True})

    def test_setup_installs_dependencies_before_importing_benchmark_modules(self):
        notebook = bundle.notebook_document(artifacts())
        cell = next(c for c in notebook['cells'] if c.get('metadata', {}).get('putare09') == 'setup')
        source = ''.join(cell['source'])
        self.assertLess(source.index('prepare_environment(ROOT'), source.index('from tools import benchmark175'))

    def test_failed_drive_mount_clears_a_prior_ready_state(self):
        notebook = bundle.notebook_document(artifacts())
        cell = next(c for c in notebook['cells'] if c.get('metadata', {}).get('putare09') == 'setup')
        google = types.ModuleType('google')
        colab = types.ModuleType('google.colab')
        def fail_mount(_path):
            raise RuntimeError('drive unavailable')
        colab.drive = types.SimpleNamespace(mount=fail_mount)
        scope = {'SETUP_COLAB': True, '_PUTARE_SETUP_READY': True}
        with patch.dict(sys.modules, {'google': google, 'google.colab': colab}):
            with self.assertRaisesRegex(RuntimeError, 'drive unavailable'):
                exec(''.join(cell['source']), scope)
        self.assertFalse(scope['_PUTARE_SETUP_READY'])

    def test_download_verifies_archive_before_transferring_it(self):
        notebook = bundle.notebook_document(artifacts())
        cell = next(c for c in notebook['cells'] if c.get('metadata', {}).get('putare09') == 'download-archive')
        google = types.ModuleType('google')
        colab = types.ModuleType('google.colab')
        colab.files = types.SimpleNamespace(download=Mock())
        verify = Mock(side_effect=ValueError('archive damaged'))
        scope = {'DOWNLOAD_ARCHIVE': True, '_PUTARE_SETUP_READY': True, 'Path': Path,
                 'ARCHIVE_PATH': 'results.zip', 'comparison205': types.SimpleNamespace(verify_archive=verify)}
        with patch.dict(sys.modules, {'google': google, 'google.colab': colab}):
            with self.assertRaisesRegex(ValueError, 'archive damaged'):
                exec(''.join(cell['source']), scope)
        colab.files.download.assert_not_called()

    def test_generation_uses_the_durable_run_and_both_models(self):
        notebook = bundle.notebook_document(artifacts())
        cell = next(c for c in notebook['cells'] if c.get('metadata', {}).get('putare09') == 'generation')
        generate = Mock(return_value={'complete': True, 'answers': 410})
        scope = {'RUN_GENERATION': True, '_PUTARE_SETUP_READY': True, 'Path': Path,
                 '_validate_run_directory': lambda: Path('durable/run'),
                 'BASE_MODEL': 'models/base.gguf', 'E80_MODEL': 'models/e80.gguf',
                 'ENGINE': Path('engine'), 'comparison205': types.SimpleNamespace(generate=generate)}
        with contextlib.redirect_stdout(io.StringIO()):
            exec(''.join(cell['source']), scope)
        generate.assert_called_once_with(Path('durable/run'), base_model=Path('models/base.gguf'),
                                         e80_model=Path('models/e80.gguf'), engine=Path('engine'))

    def test_grading_prints_summary_without_dumping_per_problem_results(self):
        notebook = bundle.notebook_document(artifacts())
        cell = next(c for c in notebook['cells'] if c.get('metadata', {}).get('putare09') == 'grading')
        report = {'complete': True, 'domains': {'math': 'MATH_SUMMARY'},
                  'paired': {'base_only': 'PAIR_SUMMARY'}, 'efficiency': {'base': 'SPEED_SUMMARY'},
                  'cases': [{'answer': 'VERY_LONG_ANSWER_DO_NOT_PRINT'}]}
        scope = {'RUN_GRADING': True, '_PUTARE_SETUP_READY': True, 'Path': Path,
                 '_validate_run_directory': lambda: Path('durable/run'),
                 'BASE_MODEL': 'base.gguf', 'E80_MODEL': 'e80.gguf', 'ENGINE': Path('engine'),
                 'comparison205': types.SimpleNamespace(grade=Mock(return_value=report))}
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            exec(''.join(cell['source']), scope)
        self.assertIn('MATH_SUMMARY', output.getvalue())
        self.assertIn('PAIR_SUMMARY', output.getvalue())
        self.assertIn('SPEED_SUMMARY', output.getvalue())
        self.assertNotIn('VERY_LONG_ANSWER_DO_NOT_PRINT', output.getvalue())

    def test_actual_bundles_are_hash_identified_and_contain_only_allowlisted_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            built = bundle.build_artifacts(output=tmp)
            for name, allowed in (('source', bundle.SOURCE_FILES), ('data', bundle.DATA_FILES)):
                item = built[name]
                self.assertEqual(hashlib.sha256(item['path'].read_bytes()).hexdigest(), item['sha256'])
                with zipfile.ZipFile(item['path']) as archive:
                    inventory = json.loads(archive.read('_putare_inventory.json'))
                    self.assertEqual(set(inventory), set(allowed))
                    self.assertEqual(set(archive.namelist()), set(allowed) | {'_putare_inventory.json'})
                    for filename, expected in inventory.items():
                        self.assertEqual(hashlib.sha256(archive.read(filename)).hexdigest(), expected)

    def test_existing_notebook_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(bundle, 'NOTEBOOK', Path(tmp) / '09.ipynb'):
            bundle.write_notebook(artifacts())
            content = json.loads(bundle.NOTEBOOK.read_text(encoding='utf-8'))
            content['cells'][0]['source'].append('사용자 수정\n')
            bundle.NOTEBOOK.write_text(json.dumps(content, ensure_ascii=False), encoding='utf-8')
            with self.assertRaises(FileExistsError):
                bundle.write_notebook(artifacts())
            self.assertIn('사용자 수정', bundle.NOTEBOOK.read_text(encoding='utf-8'))

    def test_unmodified_generated_notebook_can_update_its_bundle_digests(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(bundle, 'NOTEBOOK', Path(tmp) / '09.ipynb'):
            bundle.write_notebook(artifacts())
            newer = {name: {'path': Path(f'putare09-{name}-' + 'b' * 12 + '.zip'), 'sha256': 'b' * 64}
                     for name in ('source', 'data')}
            bundle.write_notebook(newer)
            text = bundle.NOTEBOOK.read_text(encoding='utf-8')
            self.assertIn('putare09-source-bbbbbbbbbbbb.zip', text)
            self.assertIn('putare09-data-bbbbbbbbbbbb.zip', text)


if __name__ == '__main__':
    unittest.main()
