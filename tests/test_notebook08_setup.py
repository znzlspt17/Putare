import contextlib
import io
import json
import socket
import subprocess
import tempfile
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch
from tools import notebook08_bundle as bundle


class Notebook08Test(unittest.TestCase):
    def notebook(self):
        return json.loads(Path('ipynb/08_base_model_lcb175.ipynb').read_text(encoding='utf-8'))

    def test_safe_default_run_all(self):
        notebook = self.notebook()
        source = '\n'.join(''.join(cell['source']) for cell in notebook['cells'])
        for name in ('SETUP_COLAB', 'DOWNLOAD_MODEL', 'RESTORE_CHECKPOINT', 'RUN_GENERATION', 'RUN_GRADING'):
            self.assertIn(name + ' = False', source)
        with patch.object(subprocess, 'run', side_effect=AssertionError('process')), \
             patch.object(subprocess, 'Popen', side_effect=AssertionError('process')), \
             patch.object(socket.socket, 'connect', side_effect=AssertionError('network')), \
             contextlib.redirect_stdout(io.StringIO()):
            scope = {}
            for cell in notebook['cells']:
                if cell['cell_type'] == 'code':
                    exec(''.join(cell['source']), scope)

    def test_generation_requires_setup(self):
        notebook = self.notebook()
        generation = next(''.join(c['source']) for c in notebook['cells'] if c.get('metadata', {}).get('putare08') == 'generation')
        with self.assertRaisesRegex(RuntimeError, '준비'):
            exec(generation, {'RUN_GENERATION': True})

    def test_failed_setup_invalidates_previous_ready_state(self):
        notebook = self.notebook()
        setup = next(''.join(c['source']) for c in notebook['cells'] if c.get('metadata', {}).get('putare08') == 'setup')
        mod = types.ModuleType('tools.colab_setup')
        def fail(*args):
            raise RuntimeError('installation failed')
        mod.prepare_environment = fail
        scope = {'SETUP_COLAB': True, '_PUTARE_SETUP_READY': True, 'ROOT': Path('.')}
        with patch.dict(sys.modules, {'tools.colab_setup': mod}):
            with self.assertRaisesRegex(RuntimeError, 'installation failed'):
                exec(setup, scope)
        self.assertFalse(scope['_PUTARE_SETUP_READY'])

    def test_bundle_regeneration_updates_default_paths_and_preserves_user_settings(self):
        def artifacts(letter):
            return {k: {'path': Path(f'putare08-{k}-{letter * 12}.zip'), 'sha256': letter * 64}
                    for k in ('source', 'data')}
        with tempfile.TemporaryDirectory() as temp, patch.object(bundle, 'NOTEBOOK', Path(temp) / 'notebook.ipynb'):
            bundle.write_notebook(artifacts('a'))
            original = json.loads(bundle.NOTEBOOK.read_text(encoding='utf-8'))
            settings = original['cells'][1]
            settings['source'].append('USER_SETTING = 123\n')
            bundle.NOTEBOOK.write_text(json.dumps(original), encoding='utf-8')
            bundle.write_notebook(artifacts('b'))
            output = json.loads(bundle.NOTEBOOK.read_text(encoding='utf-8'))
            text = ''.join(output['cells'][1]['source'])
            self.assertIn('putare08-source-bbbbbbbbbbbb.zip', text)
            self.assertIn('putare08-data-bbbbbbbbbbbb.zip', text)
            self.assertIn('USER_SETTING = 123', text)


if __name__ == '__main__':
    unittest.main()
