import contextlib
import io
import socket
import subprocess
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from tools import notebook09_bundle as bundle


def artifacts():
    return {name: {'path': Path(f'putare09-{name}-' + 'a' * 12 + '.zip'), 'sha256': 'a' * 64}
            for name in ('source', 'data')}


class SeparateNotebookTests(unittest.TestCase):
    def cell(self, notebook, name):
        return next(''.join(c['source']) for c in notebook['cells']
                    if c.get('metadata', {}).get('putare_single') == name)

    def test_independent_notebooks_default_to_inert_single_model(self):
        for profile in ('base', 'e80'):
            with self.subTest(profile=profile):
                notebook = bundle.single_notebook_document(profile, artifacts())
                self.assertIn(f"MODEL_PROFILE = '{profile}'", self.cell(notebook, 'settings'))
                with patch.object(subprocess, 'run', side_effect=AssertionError('process')), \
                     patch.object(subprocess, 'Popen', side_effect=AssertionError('process')), \
                     patch.object(socket.socket, 'connect', side_effect=AssertionError('network')), \
                     contextlib.redirect_stdout(io.StringIO()):
                    scope = {}
                    for cell in notebook['cells']:
                        if cell['cell_type'] == 'code':
                            exec(compile(''.join(cell['source']), cell['id'], 'exec'), scope)
                self.assertFalse(scope['RUN_GENERATION'])
                self.assertFalse(scope['RUN_GRADING'])

    def test_each_stage_requires_only_its_profile_model(self):
        for profile in ('base', 'e80'):
            with self.subTest(profile=profile):
                notebook = bundle.single_notebook_document(profile, artifacts())
                compare = types.SimpleNamespace(prepare_single=Mock(return_value={}),
                    generate_single=Mock(return_value={}), grade_single=Mock(return_value={}),
                    archive_single=Mock(return_value={}))
                scope = {'_PUTARE_SETUP_READY': True, '_validate_run_directory': lambda: Path('drive/run'),
                         'Path': Path, 'MODEL_PATH': 'only-this-model.gguf', 'MODEL_PROFILE': profile,
                         'ENGINE': Path('engine'), 'ARCHIVE_PATH': 'results.zip', 'comparison205': compare}
                for flag, name in [('RUN_PREFLIGHT', 'preflight'), ('RUN_GENERATION', 'generation'),
                                   ('RUN_GRADING', 'grading'), ('EXPORT_RESULTS', 'export')]:
                    scope[flag] = True
                    with contextlib.redirect_stdout(io.StringIO()):
                        exec(self.cell(notebook, name), scope)
                for method in (compare.prepare_single, compare.generate_single, compare.grade_single):
                    method.assert_called_once_with(Path('drive/run'), model=Path('only-this-model.gguf'),
                                                   engine=Path('engine'), model_profile=profile)
                compare.archive_single.assert_called_once_with(Path('drive/run'), Path('results.zip'))

    def test_e80_optional_compare_uses_both_run_folders_only_when_enabled(self):
        notebook = bundle.single_notebook_document('e80', artifacts())
        code = self.cell(notebook, 'compare')
        compare = Mock(return_value={'complete': True})
        scope = {'RUN_COMPARE': False, 'comparison205': types.SimpleNamespace(compare_runs=compare)}
        exec(code, scope)
        compare.assert_not_called()
        scope.update(RUN_COMPARE=True, _PUTARE_SETUP_READY=True, Path=Path,
                     _validate_run_directory=lambda: Path('drive/e80'),
                     BASE_RUN_DIR='drive/base', COMPARE_OUTPUT='drive/report.json')
        with contextlib.redirect_stdout(io.StringIO()):
            exec(code, scope)
        compare.assert_called_once_with(Path('drive/base'), Path('drive/e80'), output_path=Path('drive/report.json'))

    def test_invalid_profile_is_rejected(self):
        with self.assertRaises(ValueError):
            bundle.single_notebook_document('unknown', artifacts())

    def test_compare_switch_is_present_only_in_e80_notebook(self):
        self.assertNotIn('RUN_COMPARE', self.cell(bundle.single_notebook_document('base', artifacts()), 'settings'))
        self.assertIn('RUN_COMPARE = False', self.cell(bundle.single_notebook_document('e80', artifacts()), 'settings'))

    def test_e80_introduction_explains_offline_comparison_of_verified_archives(self):
        notebook = bundle.single_notebook_document('e80', artifacts())
        intro = self.cell(notebook, 'intro')
        self.assertIn('검증한 ZIP', intro)
        self.assertIn('GPU와 GGUF 없이', intro)
        self.assertIn('compare_runs', intro)

    def test_rebuild_updates_only_unmodified_generated_notebooks(self):
        with tempfile.TemporaryDirectory() as temp:
            paths = {'base': Path(temp) / '09.ipynb', 'e80': Path(temp) / '10.ipynb'}
            with patch.object(bundle, 'SINGLE_NOTEBOOKS', paths):
                for profile in paths:
                    bundle.write_single_notebook(profile, artifacts())
                changed = {name: {'path': Path(f'putare09-{name}-' + 'b' * 12 + '.zip'),
                                  'sha256': 'b' * 64} for name in ('source', 'data')}
                bundle.write_single_notebook('base', changed)
                self.assertIn('putare09-source-bbbbbbbbbbbb.zip', paths['base'].read_text(encoding='utf-8'))
                original = paths['e80'].read_text(encoding='utf-8')
                import json
                notebook = json.loads(original)
                notebook['cells'][0]['source'].append('사용자 수정\n')
                paths['e80'].write_text(json.dumps(notebook, ensure_ascii=False), encoding='utf-8')
                with self.assertRaises(FileExistsError):
                    bundle.write_single_notebook('e80', changed)
                self.assertIn('사용자 수정', paths['e80'].read_text(encoding='utf-8'))


if __name__ == '__main__':
    unittest.main()
