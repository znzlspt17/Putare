import gzip
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from moe_prune.research import canonical_hash, write_json
from tools import benchmark175 as module


class Benchmark175Test(unittest.TestCase):
    def test_request(self):
        request = module.make_request({'id': 'x', 'prompt': 'hello'}, 'model')
        self.assertEqual((request['context_size'], request['max_tokens']), (262144, 65536))
        self.assertTrue(request['fusion_enabled'] and request['instrumentation'])
        self.assertEqual(request['instrumentation_level'], 'routing')
        self.assertEqual((request['cache_type_k'], request['cache_type_v']), ('q8_0', 'q8_0'))

    def test_missing_model(self):
        with tempfile.TemporaryDirectory() as tmp, self.assertRaises(FileNotFoundError):
            module.check_model(Path(tmp) / 'missing')

    def test_partial_run_is_not_graded(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            write_json(path / 'status.json', {'complete': False, 'completed': 2, 'total': 175})
            with self.assertRaisesRegex(ValueError, '미완료'):
                module.grade(path)

    def test_compression_preserves_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / 'events.jsonl'
            source.write_bytes(b'{"type":"routing"}\n' * 100)
            target = Path(tmp) / 'events.jsonl.gz'
            info = module.compress_events(source, target)
            self.assertEqual(gzip.decompress(target.read_bytes()), source.read_bytes())
            self.assertEqual(len(info['uncompressed_sha256']), 64)

    def test_resume_rejects_changed_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            write_json(path / 'identity.json', {'source': 'old'})
            with patch.object(module, 'source_identity', return_value=([], {'source': 'new'})):
                with self.assertRaisesRegex(ValueError, '기존 실행'):
                    module.generate(path)

    def test_compressed_tamper_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            answer = {'id': 'x', 'identity_hash': 'id', 'text': 'ok', 'job': 'problem-001-attempt-001',
                      'routing_integrity': {'compressed_sha256': '0' * 64, 'uncompressed_sha256': '0' * 64}}
            answer['answer_sha256'] = canonical_hash(answer)
            write_json(path / 'answers/001.json', answer)
            routing = path / 'routing/problem-001-attempt-001.jsonl.gz'
            routing.parent.mkdir()
            routing.write_bytes(b'tampered')
            with self.assertRaisesRegex(ValueError, '압축 계측'):
                module._verified_answer(path, 1, {'id': 'x', 'lcb_problem': {}}, 'id')

    def test_native_result_requires_final_event(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'events.gz'
            with gzip.open(path, 'wt', encoding='utf-8') as stream:
                stream.write('{"type":"result","text":"native"}\n')
            self.assertEqual(module.native_result(path)['text'], 'native')
            with gzip.open(path, 'at', encoding='utf-8') as stream:
                stream.write('{"type":"routing"}\n')
            with self.assertRaises(ValueError):
                module.native_result(path)

    def test_resume_175_after_interruption(self):
        records = [{'id': str(i), 'prompt': 'x', 'lcb_problem': {}} for i in range(175)]
        calls = []
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model, engine = root / 'model', root / 'engine'
            identity = {'model_path': str(model), 'engine_path': str(engine), 'model_sha256': 'm',
                        'engine_files': {'engine': 'e'}}
            class FakeExperiment:
                def __init__(self, path, cfg): self.root = Path(path)
                def __enter__(self): return self
                def __exit__(self, *args): pass
                def check(self): pass
                def run(self, name, request):
                    calls.append(name)
                    if len(calls) == 4: raise RuntimeError('interrupted')
                    folder = self.root / 'jobs' / name
                    write_json(folder / 'request.json', request)
                    write_json(folder / 'identity.json', {'model': {'sha256': 'm'}, 'engine': {'sha256': 'e'}})
                    result = {'type': 'result', 'text': 'ok', 'token_ids': [1]}
                    (folder / 'events.jsonl').write_text(json.dumps(result) + '\n')
                    return result
            with patch.object(module, 'source_identity', return_value=(records, identity)), \
                 patch.object(module, '_validate_job', return_value={}), \
                 patch.object(module, 'gpu_info', return_value={'name': 'fake'}), \
                 patch('moe_prune.resident.ResidentExperiment', FakeExperiment), \
                 patch.object(module, 'checkpoint_problem') as checkpoint:
                with self.assertRaisesRegex(RuntimeError, 'interrupted'):
                    module.generate(root / 'run', model=model, engine=engine, checkpoint_dir=root / 'backup')
                module.generate(root / 'run', model=model, engine=engine, checkpoint_dir=root / 'backup')
            answer = json.loads((root / 'run/answers/001.json').read_text(encoding='utf-8'))
            self.assertEqual(answer['gpu'], {'name': 'fake'})
            self.assertEqual(checkpoint.call_count, 178)
            self.assertEqual(len(calls), 176)
            self.assertEqual(json.loads((root / 'run/status.json').read_text()),
                             {'complete': True, 'completed': 175, 'total': 175})

    def test_grading_infrastructure_failure_has_no_score(self):
        from contextlib import nullcontext
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'tools').mkdir()
            (root / 'tools/lcb_grade.py').write_text('adapter')
            from moe_prune.gguf import sha256
            write_json(root / 'snapshot/manifest.json', {'grader_identity': {'tools/lcb_grade.py': sha256(root / 'tools/lcb_grade.py')}})
            run = root / 'run'
            write_json(run / 'status.json', {'complete': True, 'completed': 175, 'total': 175})
            identity = {'official_identity': {'source_digest': 'expected'}}
            write_json(run / 'identity.json', identity)
            records = [{'id': str(i)} for i in range(175)]
            probe = SimpleNamespace(stdout=json.dumps({'source_digest': 'expected'}))
            with patch.object(module, 'ROOT', root), patch.object(module, 'SNAPSHOT', root / 'snapshot'), \
                 patch.object(module, 'source_identity', return_value=(records, identity)), \
                 patch.object(module, '_verified_answer'), patch.object(module.subprocess, 'run', return_value=probe), \
                 patch('moe_prune.execution_safety.execution_guard', return_value=nullcontext(object())), \
                 patch('moe_prune.grading.run', side_effect=RuntimeError('worker failed')), \
                 patch.object(module, 'checkpoint_files') as checkpoint:
                summary = module.grade(run, checkpoint_dir=root / 'backup')
            self.assertTrue(checkpoint.called)
            self.assertFalse(summary['complete'])
            self.assertEqual(summary['cases'][0]['status'], 'infrastructure_error')
            self.assertIsNone(summary['cases'][0]['passed'])
            self.assertNotIn('passed', summary)


    def test_checkpoint_copies_and_validates_before_marker(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, backup = root / 'run', root / 'backup'
            write_json(source / 'identity.json', {'run': 1})
            write_json(source / 'answers/001.json', {'id': 'x'})
            files = ['identity.json', 'answers/001.json']
            module.checkpoint_files(source, backup, files, 'problems/001.json')
            self.assertTrue((backup / 'problems/001.json').is_file())
            module.checkpoint_files(source, backup, files, 'problems/001.json')
            (backup / 'answers/001.json').write_text('changed')
            with self.assertRaises(ValueError):
                module.checkpoint_files(source, backup, files, 'problems/001.json')

    def test_checkpoint_never_publishes_marker_when_copy_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, backup = root / 'run', root / 'backup'
            write_json(source / 'identity.json', {'run': 1})
            with patch('shutil.copyfileobj', side_effect=OSError('quota')):
                with self.assertRaises(OSError):
                    module.checkpoint_files(source, backup, ['identity.json'], 'problems/001.json')
            self.assertFalse((backup / 'problems/001.json').exists())




    def test_restore_rejects_damaged_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, backup = root / 'run', root / 'backup'
            write_json(source / 'identity.json', {'run': 1})
            write_json(source / 'answers/001.json', {'id': 'x'})
            module.checkpoint_files(source, backup, ['identity.json', 'answers/001.json'], 'problems/001.json')
            (backup / 'answers/001.json').write_text('damaged')
            with self.assertRaises(ValueError):
                module.restore_checkpoint(backup, root / 'restored')

    def test_checkpoint_problem_rejects_mismatched_gpu_observation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_json(root / 'gpu-observations/001.json', {'name': 'other'})
            answer = {'job': 'problem-001-attempt-001', 'gpu_observation': 'gpu-observations/001.json',
                      'gpu': {'name': 'expected'}}
            with self.assertRaisesRegex(ValueError, 'GPU'):
                module.checkpoint_problem(root, root / 'backup', 1, answer)

    def test_backup_failure_preserves_completed_local_answer_and_stops(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model, engine = root / 'model', root / 'engine'
            records = [{'id': str(i), 'prompt': 'x', 'lcb_problem': {}} for i in range(175)]
            identity = {'model_path': str(model), 'engine_path': str(engine), 'model_sha256': 'm',
                        'engine_files': {'engine': 'e'}}
            calls = []
            class Experiment:
                def __init__(self, path, cfg): self.root = Path(path)
                def __enter__(self): return self
                def __exit__(self, *args): pass
                def check(self): pass
                def run(self, name, request):
                    calls.append(name)
                    folder = self.root / 'jobs' / name
                    write_json(folder / 'request.json', request)
                    write_json(folder / 'identity.json', {'model': {'sha256': 'm'}, 'engine': {'sha256': 'e'}})
                    result = {'type': 'result', 'text': 'ok', 'token_ids': [1]}
                    (folder / 'events.jsonl').write_text(json.dumps(result) + '\n')
                    return result
            with patch.object(module, 'source_identity', return_value=(records, identity)), \
                 patch.object(module, '_validate_job', return_value={}), \
                 patch.object(module, 'gpu_info', return_value={'name': 'fake'}), \
                 patch('moe_prune.resident.ResidentExperiment', Experiment), \
                 patch.object(module, 'checkpoint_problem', side_effect=OSError('quota')):
                with self.assertRaisesRegex(OSError, 'quota'):
                    module.generate(root / 'run', model=model, engine=engine, checkpoint_dir=root / 'backup')
            self.assertEqual(len(calls), 1)
            self.assertTrue((root / 'run/answers/001.json').is_file())
            status = json.loads((root / 'run/status.json').read_text())
            self.assertEqual(status['completed'], 1)
            self.assertFalse(status['complete'])
