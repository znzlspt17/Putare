import ast
import json
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import nbformat


class Trial80Tests(unittest.TestCase):
    def test_notebook_defaults_do_not_execute(self):
        path = Path(__file__).resolve().parents[1] / 'ipynb/06_expert80_benchmark.ipynb'
        self.assertTrue(path.is_file(), '세 문항 시험용 06번 노트북이 필요합니다')
        notebook = nbformat.read(path, as_version=4)
        nbformat.validate(notebook)
        flags = {}
        for cell in notebook.cells:
            if cell.cell_type != 'code':
                continue
            self.assertIsNone(cell.execution_count)
            self.assertEqual(cell.outputs, [])
            tree = ast.parse(cell.source)
            for node in ast.walk(tree):
                if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                    name = node.targets[0].id
                    if name.startswith(('RUN_', 'PREPARE_', 'BUILD_', 'INSTALL_')) and name != 'RUN_DIR':
                        self.assertIsInstance(node.value, ast.Constant)
                        flags[name] = node.value.value
        self.assertIn('RUN_GENERATION', flags)
        self.assertIn('RUN_GRADING', flags)
        self.assertTrue(all(value is False for value in flags.values()))

    def test_math_fraction_and_unparsed_are_distinct(self):
        from tools import benchmark80 as trial
        for text in (r'답은 \boxed{\frac{3}{56}}', r'<|channel>final<channel|>\boxed{3/56}'):
            self.assertEqual(trial.grade_math(text)['passed'], True)
        self.assertEqual(trial.grade_math(r'\boxed{3/57}')['passed'], False)
        self.assertTrue(trial.grade_math('<|channel>final\n' + r'\boxed{3/56}')['passed'])
        self.assertTrue(trial.grade_math('<|channel>thought\n' + r'\boxed{1}' + '<channel|>' + r'\boxed{3/56}')['passed'])
        self.assertIsNone(trial.grade_math('<|channel>thought\n' + r'\boxed{3/56}')['passed'])
        self.assertEqual(trial.grade_math('숫자 3과 56만 있습니다')['status'], 'unparsed')
        self.assertIsNone(trial.grade_math(r'\boxed{\sqrt{9}/56}')['passed'])

    def test_default_notebook_run_does_not_start_work(self):
        import socket
        import subprocess
        import IPython.display
        from tools import benchmark80 as trial
        from tools import manual80
        notebook = nbformat.read(Path(__file__).resolve().parents[1] / 'ipynb/06_expert80_benchmark.ipynb', as_version=4)
        with tempfile.TemporaryDirectory() as folder, \
             patch.object(trial, 'DIRECTORY', Path(folder) / 'unused'), \
             patch.object(trial, 'generate', side_effect=AssertionError('unexpected generation')) as generate, \
             patch.object(trial, 'grade', side_effect=AssertionError('unexpected grading')) as grade, \
             patch.object(subprocess, 'run', side_effect=AssertionError('unexpected subprocess')), \
             patch.object(subprocess, 'Popen', side_effect=AssertionError('unexpected process')), \
             patch.object(socket, 'socket', side_effect=AssertionError('unexpected network')):
            namespace = {}
            for cell in notebook.cells:
                if cell.cell_type == 'code':
                    exec(compile(cell.source, 'notebook06', 'exec'), namespace)
            generate.assert_not_called()
            grade.assert_not_called()

    def test_three_cases_use_one_session_and_validate_before_grading(self):
        from tools import benchmark80 as trial
        from moe_prune.research import canonical_hash, write_json
        from tools import manual80

        records = [trial.MATH_RECORD, *[
            {'id': ident, 'domain': 'code', 'dataset': 'lcb', 'prompt': 'solve',
             'lcb_problem': {'question_title': 'q'}} for ident in manual80.IDS
        ]]
        identity = {'model_sha256': manual80.MODEL_SHA, 'engine_sha256': 'engine',
                    'manifest_sha256': 'manifest', 'records_sha256': 'records',
                    'bundle_inventory_sha256': 'bundle', 'engine_files': {'libggml.so': 'original'},
                    'script_sha256': 'script', 'source_hashes': {'base_benchmark.py': 'original'}}
        calls = []
        closed = []
        math_text = [r'\boxed{\frac{3}{56}}']

        class FakeExperiment:
            def __init__(self, root, config):
                self.root = Path(root)
                calls.append(('init', config['engine']))
            def __enter__(self):
                return self
            def __exit__(self, *args):
                closed.append(True)
            def check(self):
                pass
            def run(self, name, request):
                self_test.assertNotIn('reference', request)
                self_test.assertNotIn('lcb_problem', request)
                calls.append(('run', request['prompt']))
                folder = self.root / 'jobs' / name
                folder.mkdir(parents=True)
                write_json(folder / 'request.json', request)
                native = {'type': 'result', 'text': math_text[0] if name == 'problem-001' else 'code', 'token_ids': [1],
                          'stop_reason': 'eog', 'timings': {}}
                (folder / 'events.jsonl').write_text(json.dumps(native) + '\n', encoding='utf-8')
                (folder / 'stderr.log').write_text('', encoding='utf-8')
                write_json(folder / 'identity.json', {'model': {'sha256': manual80.MODEL_SHA},
                                                      'engine': {'sha256': 'engine'}})
                return {**native, 'measurement': {}}

        self_test = self
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            model, engine, output = root / 'model.gguf', root / 'engine', root / 'run'
            model.write_bytes(b'model'); engine.write_bytes(b'engine')
            with patch.object(trial, 'source_identity', return_value=(records, identity)), \
                 patch('moe_prune.resident.ResidentExperiment', FakeExperiment), \
                 patch('tools.manual104.validate_job', return_value={}):
                trial.generate(output, model=model, engine=engine)
                self.assertEqual(len([c for c in calls if c[0] == 'init']), 1)
                self.assertEqual(len([c for c in calls if c[0] == 'run']), 3)
                self.assertEqual(closed, [True])
                self.assertEqual([x['id'] for x in trial.verify_inputs(output, model=model, engine=engine)[1]],
                                 [trial.MATH_RECORD['id'], *manual80.IDS])
                status = json.loads((output / 'status.json').read_text(encoding='utf-8'))
                self.assertTrue(status['complete'])
                saved_request = output / 'jobs/problem-001/request.json'
                request = json.loads(saved_request.read_text(encoding='utf-8'))
                changed = {**request, 'prompt': '다른 문제'}
                write_json(saved_request, changed)
                with self.assertRaises(ValueError):
                    trial.verify_inputs(output, model=model, engine=engine)
                write_json(saved_request, request)
                changed_source = {**identity, 'engine_files': {'libggml.so': 'tampered'}}
                with patch.object(trial, 'source_identity', return_value=(records, changed_source)):
                    with self.assertRaises(ValueError):
                        trial.verify_inputs(output, model=model, engine=engine)
                events = []
                active_guard = object()
                @contextmanager
                def guard(kind, *, memory_config=None):
                    self.assertTrue(closed, '채점 전에 생성 세션이 종료되어야 합니다')
                    events.append(('enter', kind))
                    yield active_guard
                    events.append(('exit', kind))
                def grade_worker(source, venv, payload, target, **kwargs):
                    self.assertIs(kwargs['guard'], active_guard)
                    self.assertGreater(kwargs['timeout_s'], 0)
                    item = json.loads(Path(payload).read_text(encoding='utf-8'))
                    events.append(('grade', item['id']))
                    result = {'id': item['id'], 'passed': True, 'pass_at_1': 1.0, 'test_count': 2}
                    write_json(target, result)
                    return result
                with patch('moe_prune.execution_safety.execution_guard', guard), \
                     patch('moe_prune.grading.run', side_effect=grade_worker), \
                     patch.object(trial, '_verify_official', return_value={'source_digest': 'test'}):
                    summary = trial.grade(output, model=model, engine=engine, venv=root / 'venv')
                self.assertTrue(summary['complete'])
                self.assertEqual([row['id'] for row in summary['cases']], [trial.MATH_RECORD['id'], *manual80.IDS])
                self.assertTrue(summary['cases'][0]['passed'])
                self.assertEqual(events, [('enter', 'grading'), ('grade', manual80.IDS[0]),
                                          ('grade', manual80.IDS[1]), ('exit', 'grading')])
                math_text[0] = '답을 찾지 못했습니다.'
                other = root / 'unparsed'
                trial.generate(other, model=model, engine=engine)
                with patch('moe_prune.execution_safety.execution_guard', guard), \
                     patch('moe_prune.grading.run', side_effect=grade_worker), \
                     patch.object(trial, '_verify_official', return_value={'source_digest': 'test'}):
                    unparsed = trial.grade(other, model=model, engine=engine, venv=root / 'venv')
                self.assertEqual(unparsed['cases'][0]['status'], 'unparsed')
                self.assertIsNone(unparsed['cases'][0]['passed'])
                raw = output / 'jobs/problem-001/events.jsonl'
                raw.write_text('{}\n', encoding='utf-8')
                with self.assertRaises(ValueError):
                    trial.verify_inputs(output, model=model, engine=engine)
                status['complete'] = False
                write_json(output / 'status.json', status)
                with patch('moe_prune.grading.run') as grading, self.assertRaises(ValueError):
                    trial.grade(output, model=model, engine=engine)
                grading.assert_not_called()

                class BrokenExperiment(FakeExperiment):
                    def run(self, name, request):
                        if name == 'problem-002':
                            raise RuntimeError('synthetic failure')
                        return super().run(name, request)
                with patch('moe_prune.resident.ResidentExperiment', BrokenExperiment), self.assertRaises(RuntimeError):
                    trial.generate(root / 'failed', model=model, engine=engine)
                self.assertEqual(closed, [True, True, True])
                failed = json.loads((root / 'failed/status.json').read_text(encoding='utf-8'))
                self.assertFalse(failed['complete'])
                self.assertEqual(failed['completed'], 1)

    def test_custom_model_and_engine_paths_are_verified(self):
        from tools import manual80
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            bundle = root / 'bundle'; bundle.mkdir()
            (bundle / 'lcb_portable.py').write_text('def verify_inventory(path): pass\n', encoding='utf-8')
            manifest = root / 'manifest.json'
            manifest.write_text(json.dumps({'output_sha256': manual80.MODEL_SHA,
                                            'file_validation_status': 'passed',
                                            'reorder_manifest_sha256': 'hash'}), encoding='utf-8')
            reorder = root / 'runs/gemma4-lcb100-reordered-001/permutation-manifest.json'
            reorder.parent.mkdir(parents=True)
            reorder.write_text(json.dumps({'profile': {'records_sha256': 'hash'}}), encoding='utf-8')
            model, engine = root / 'custom.gguf', root / 'custom-engine'
            seen = []
            def fake_hash(path):
                path = Path(path); seen.append(path)
                return manual80.MODEL_SHA if path == model else 'hash'
            class FakeGGUF:
                def __init__(self, path): self.path = path
                def architecture(self):
                    self_test.assertEqual(self.path, model)
                    return (list(range(30)), 80, 8)
            self_test = self
            with patch.object(manual80, 'ROOT', root), patch.object(manual80, 'BUNDLE', bundle), \
                 patch.object(manual80, 'MANIFEST', manifest), patch.object(manual80, 'sha256', side_effect=fake_hash), \
                 patch.object(manual80, 'GGUF', FakeGGUF), \
                 patch.object(manual80, 'selected_records', return_value=[{'id': ident} for ident in manual80.IDS]):
                _, identity = manual80.source_identity(model=model, engine=engine)
            self.assertIn(model, seen)
            self.assertIn(engine, seen)
            self.assertEqual(identity['engine_sha256'], 'hash')


if __name__ == '__main__':
    unittest.main()
