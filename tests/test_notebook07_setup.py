import contextlib
import io
import json
from pathlib import Path
import subprocess
import unittest
from unittest.mock import patch


class Notebook07Test(unittest.TestCase):
    def notebook(self):
        path = Path('ipynb/07_expert80_lcb175.ipynb')
        return path, json.loads(path.read_text(encoding='utf-8'))

    def test_safe_defaults_and_no_embedded_archive(self):
        path, notebook = self.notebook()
        source = '\n'.join(''.join(c['source']) for c in notebook['cells'])
        for name in ('SETUP_COLAB', 'RUN_GENERATION', 'RUN_GRADING'):
            self.assertIn(name + ' = False', source)
        self.assertLess(path.stat().st_size, 100000)
        self.assertNotIn('base64.b64decode', source)

    def test_default_run_all_does_not_start_process_or_network(self):
        import socket
        _, notebook = self.notebook()
        scope = {}
        with patch.object(subprocess, 'run', side_effect=AssertionError('process')), \
             patch.object(subprocess, 'Popen', side_effect=AssertionError('process')), \
             patch.object(socket.socket, 'connect', side_effect=AssertionError('network')), \
             contextlib.redirect_stdout(io.StringIO()):
            for cell in notebook['cells']:
                if cell['cell_type'] == 'code':
                    exec(''.join(cell['source']), scope)

    def test_generation_blocked_without_setup(self):
        _, notebook = self.notebook()
        run = ''.join(notebook['cells'][3]['source'])
        with self.assertRaisesRegex(RuntimeError, '환경 준비'):
            exec(run, {'RUN_GENERATION': True, 'RUN_GRADING': False, '_PUTARE_SETUP_READY': False})

    def test_regeneration_preserves_user_settings_and_outputs(self):
        import tempfile
        from tools import notebook07_bundle as bundle
        from unittest.mock import patch
        path, notebook = self.notebook()
        notebook['cells'][1]['source'].append("CUSTOM_FLAG = True\n")
        notebook['cells'][1]['outputs'] = [{'output_type': 'stream', 'name': 'stdout', 'text': ['kept\n']}]
        notebook['cells'].append({'cell_type': 'markdown', 'metadata': {}, 'source': ['custom']})
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'ipynb').mkdir()
            target = root / path
            target.write_text(json.dumps(notebook), encoding='utf-8')
            artifacts = {k: {'path': root / f'{k}.zip', 'sha256': '0' * 64} for k in ('source', 'data')}
            with patch.object(bundle, 'ROOT', root):
                bundle.write_notebook(artifacts)
            updated = json.loads(target.read_text(encoding='utf-8'))
            self.assertIn('CUSTOM_FLAG = True', ''.join(updated['cells'][1]['source']))
            self.assertEqual(updated['cells'][1]['outputs'], notebook['cells'][1]['outputs'])
            self.assertEqual(updated['cells'][-1]['source'], ['custom'])

    def test_checkpoint_is_opt_in(self):
        _, notebook = self.notebook()
        source = '\n'.join(''.join(c['source']) for c in notebook['cells'])
        self.assertIn('CHECKPOINT_DIR = None', source)
        self.assertIn('checkpoint_dir=CHECKPOINT_DIR', source)
