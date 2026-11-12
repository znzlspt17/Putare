import copy
import json
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch

from tools import comparison205 as module


class SingleComparisonTests(unittest.TestCase):
    def test_gpu_contract_allows_different_physical_device_but_not_class(self):
        first = {'name': 'RTX PRO 6000 Blackwell', 'uuid': 'GPU-1', 'total_bytes': 96, 'driver': '1', 'compute_capability': '12.0'}
        second = {**first, 'uuid': 'GPU-2'}
        self.assertEqual(module.single_gpu_contract(first), module.single_gpu_contract(second))
        self.assertNotEqual(module.single_gpu_contract(first), module.single_gpu_contract({**second, 'total_bytes': 48}))

    def test_merge_rejects_different_shared_contract_or_prompt_tokens(self):
        base = {'model_profile': 'base', 'shared': {'engine_sha256': 'x'}}
        e80 = {'model_profile': 'e80', 'shared': {'engine_sha256': 'y'}}
        tokens = [{'id': 'one', 'tokens': [1], 'rendered_prompt': 'p'}]
        with self.assertRaisesRegex(ValueError, '조건'):
            module.validate_single_merge(base, e80, tokens, tokens)
        e80['shared'] = copy.deepcopy(base['shared'])
        with self.assertRaisesRegex(ValueError, '토큰'):
            module.validate_single_merge(base, e80, tokens, [{'id': 'one', 'tokens': [2], 'rendered_prompt': 'p'}])
        self.assertTrue(module.validate_single_merge(base, e80, tokens, tokens))

    def test_base_identity_inspects_only_base_model(self):
        gpu = {'name': 'RTX PRO 6000 Blackwell', 'uuid': 'GPU-1', 'total_bytes': 96, 'driver': '1', 'compute_capability': '12.0'}
        model_id = {'model_sha256': 'base-hash', 'engine_files': {'engine': 'engine-hash'},
                    'source_hashes': {}, 'official_identity': {'source_digest': 'x'},
                    'snapshot_records': 'r', 'snapshot_manifest': 'm', 'id_proof': 'p'}
        fake_model = type('FakeModel', (), {'metadata': {'tokenizer.chat_template': 'template'}})()
        with patch.object(module.benchmark175, 'source_identity', return_value=([], model_id)) as source, \
             patch.object(module, 'GGUF', return_value=fake_model) as gguf, \
             patch.object(module, 'sha256', return_value='comparison-source-hash'):
            identity = module.single_source_identity([{'id': 'one'}], {'official_identity': {}},
                                                       'base.gguf', 'engine', 'base', gpu)
        source.assert_called_once_with('base.gguf', 'engine', model_profile='base')
        gguf.assert_called_once_with('base.gguf')
        self.assertEqual(identity['model_profile'], 'base')
        self.assertNotIn('other_model', identity)
        self.assertNotIn('uuid', identity['shared']['gpu'])

    def test_each_profile_can_grade_205_without_the_other_model(self):
        records = [{'id': str(i), 'domain': 'math' if i < 30 else 'code',
                    'reference': '42', 'lcb_problem': {}} for i in range(205)]
        identity = {'model_profile': 'base', 'shared': {'official_identity': {'source_digest': 'official'}}}
        result = {'text': '<|channel>thought\nx<channel|>\\boxed{42}', 'token_ids': [1],
                  'channel_tokens': None, 'stop_reason': 'eog', 'timings': {'prefill_s': 1., 'decode_s': 1.},
                  'measurement': {'wall_s': 2., 'os_gpu_memory': {'peak_bytes': 2, 'scope': 'session_to_date'}}}
        def answer(directory, number, record, identity, dry):
            return {'id': record['id'], 'answer_sha256': str(number), 'text': result['text'], 'result': result,
                    'gpu': {'uuid': 'GPU-1'}, 'session_id': 'one'}
        def fake_run(source, venv, input_path, target, **kwargs):
            value = json.loads(Path(input_path).read_text(encoding='utf8'))
            return {'id': value['id'], 'passed': True, 'pass_at_1': 1.0}
        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(module, '_load_stored_single', return_value=(records, identity, {}), create=True), \
             patch.object(module, 'checked_gpu', side_effect=AssertionError('GPU must not be queried')), \
             patch.object(module, 'verified_single_answer', side_effect=answer, create=True), \
             patch.object(module, 'case_state', return_value='complete'), \
             patch.object(module, 'sha256', return_value='grader-hash'), \
             patch.object(module, '_single_manifest', return_value={'grader_identity': {'tools/lcb_grade.py': 'grader-hash'}}, create=True), \
             patch('moe_prune.execution_safety.execution_guard', return_value=nullcontext(object())), \
             patch('moe_prune.grading.run', side_effect=fake_run), \
             patch.object(module.subprocess, 'run', return_value=type('Probe', (), {'stdout': '{"source_digest":"official"}'})()):
            report = module.grade_single(tmp, model_profile='base')
            self.assertTrue(report['complete'])
            self.assertEqual(report['domains']['math']['passed'], 30)
            self.assertEqual(report['domains']['code']['passed'], 175)
            self.assertEqual(json.loads((Path(tmp) / 'grading-status.json').read_text(encoding='utf8'))['completed'], 205)

    def test_new_uuid_resumes_only_unstarted_problems(self):
        records = [{'id': str(i), 'prompt': f'p{i}'} for i in range(3)]
        identity = {'model_profile': 'base', 'shared': {}}
        rows = [{'id': r['id'], 'tokens': [1], 'rendered_prompt': r['prompt']} for r in records]
        dry = {'tokens': rows, 'requests': [
            {k: v for k, v in module.make_request(r, 'm', 'e', [1]).items() if k != 'model'} for r in records]}
        calls = []
        class FakeExperiment:
            def __init__(self, root, config): self.root = Path(root)
            def __enter__(self): return self
            def __exit__(self, *args): return False
            def check(self): pass
            def run(self, name, request):
                calls.append(name)
                if name == 'problem-002':
                    raise RuntimeError('interrupted')
                folder = self.root / 'jobs' / name
                folder.mkdir(parents=True)
                result = {'type': 'result', 'text': 'answer', 'token_ids': [1]}
                (folder / 'events.jsonl').write_text(json.dumps(result) + '\n', encoding='utf8')
                return {**result, 'measurement': {'wall_s': .1}}
        def tokenize(rows, model, engine):
            return [{'id': row['id'], 'tokens': [1], 'rendered_prompt': row['prompt']} for row in rows]
        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(module, '_check_single', side_effect=[(records, identity, dry, {'uuid': 'GPU-1'}),
                                                                (records, identity, dry, {'uuid': 'GPU-2'})]), \
             patch.object(module, 'tokenize_records', side_effect=tokenize), \
             patch.object(module, 'ensure_gpu'), patch.object(module, '_validate_result', return_value={}), \
             patch.object(module, 'verified_single_answer', side_effect=lambda directory, number, record, identity, dry:
                          module.verified_answer(directory, number, record['id'], module.canonical_hash(identity))), \
             patch('moe_prune.resident.ResidentExperiment', FakeExperiment), patch('builtins.print'):
            args = {'model': 'base', 'engine': 'engine', 'model_profile': 'base'}
            with self.assertRaisesRegex(RuntimeError, 'interrupted'):
                module.generate_single(tmp, **args)
            with self.assertRaisesRegex(RuntimeError, '미완료'):
                module.generate_single(tmp, **args)
            self.assertEqual(calls.count('problem-001'), 1)
            self.assertEqual(calls.count('problem-002'), 1)
            self.assertEqual(calls.count('problem-003'), 1)
            a = json.loads((Path(tmp) / 'answers/001.json').read_text(encoding='utf8'))
            b = json.loads((Path(tmp) / 'answers/003.json').read_text(encoding='utf8'))
            self.assertEqual((a['gpu']['uuid'], b['gpu']['uuid']), ('GPU-1', 'GPU-2'))

    def test_completed_cache_with_bad_session_blocks_next_dispatch(self):
        records = [{'id': 'a'}, {'id': 'b'}]
        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(module, '_check_single', return_value=(records, {'model_profile': 'base'}, {}, {})), \
             patch.object(module, 'case_state', side_effect=['complete', 'unstarted']), \
             patch.object(module, 'verified_answer', return_value={}), \
             patch.object(module, 'verified_single_answer', side_effect=ValueError('session mismatch')), \
             patch.object(module, 'ensure_gpu'), \
             patch.object(module, 'tokenize_records', side_effect=RuntimeError('new work started')):
            with self.assertRaisesRegex(ValueError, 'session mismatch'):
                module.generate_single(tmp, model='a', engine='b', model_profile='base')

    def test_single_profile_dispatches_205_once(self):
        records = [{'id': str(i), 'prompt': f'p{i}'} for i in range(205)]
        identity = {'model_profile': 'base', 'shared': {}}
        rows = [{'id': row['id'], 'tokens': [1], 'rendered_prompt': row['prompt']} for row in records]
        dry = {'tokens': rows, 'requests': [
            {k: v for k, v in module.make_request(r, 'm', 'e', [1]).items() if k != 'model'} for r in records]}
        calls = []
        class FakeExperiment:
            def __init__(self, root, config): self.root = Path(root)
            def __enter__(self): return self
            def __exit__(self, *args): return False
            def check(self): pass
            def run(self, name, request):
                calls.append(name)
                folder = self.root / 'jobs' / name
                folder.mkdir(parents=True)
                result = {'type': 'result', 'text': 'answer', 'token_ids': [1]}
                (folder / 'events.jsonl').write_text(json.dumps(result) + '\n', encoding='utf8')
                return {**result, 'measurement': {'wall_s': .1}}
        def tokenize(rows, model, engine):
            return [{'id': row['id'], 'tokens': [1], 'rendered_prompt': row['prompt']} for row in rows]
        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(module, '_check_single', return_value=(records, identity, dry, {'uuid': 'GPU-1'})), \
             patch.object(module, 'tokenize_records', side_effect=tokenize), \
             patch.object(module, 'ensure_gpu'), patch.object(module, '_validate_result', return_value={}), \
             patch.object(module, 'verified_single_answer', side_effect=lambda directory, number, record, identity, dry:
                          module.verified_answer(directory, number, record['id'], module.canonical_hash(identity))), \
             patch('moe_prune.resident.ResidentExperiment', FakeExperiment), patch('builtins.print'):
            args = {'model': 'base', 'engine': 'engine', 'model_profile': 'base'}
            self.assertEqual(module.generate_single(tmp, **args)['completed'], 205)
            module.generate_single(tmp, **args)
            self.assertEqual(len([x for x in calls if x.startswith('problem-')]), 205)
            self.assertEqual(len(set(calls)), 206)

    def test_compare_uses_saved_runs_without_gpu_or_model(self):
        records = [{'id': 'one', 'domain': 'math'}, {'id': 'two', 'domain': 'code'}]
        tokens = [{'id': row['id'], 'tokens': [i + 1], 'rendered_prompt': 'p'} for i, row in enumerate(records)]
        base = {'model_profile': 'base', 'shared': {'engine': 'same'}}
        e80 = {'model_profile': 'e80', 'shared': {'engine': 'same'}, 'selection_ids': ['two']}
        base_report = {'complete': True, 'cases': [{'id': 'one', 'passed': True, 'gpu': {'uuid': 'GPU-1'}},
                                                   {'id': 'two', 'passed': False, 'gpu': {'uuid': 'GPU-1'}}]}
        e80_report = {'complete': True, 'cases': [{'id': 'one', 'passed': False, 'gpu': {'uuid': 'GPU-2'}},
                                                  {'id': 'two', 'passed': True, 'gpu': {'uuid': 'GPU-2'}}]}
        with patch.object(module, '_load_stored_single', side_effect=[(records, base, {'tokens': tokens, 'requests': [1, 2]}),
                                                                     (records, e80, {'tokens': tokens, 'requests': [1, 2]})]), \
             patch.object(module, 'verified_single_report', side_effect=[base_report, e80_report], create=True), \
             patch.object(module, 'checked_gpu', side_effect=AssertionError('GPU must not be queried')):
            report = module.compare_runs('base', 'e80')
        self.assertEqual(report['paired']['base_only'], 1)
        self.assertEqual(report['paired']['e80_only'], 1)
        self.assertEqual(report['gpu_uuids'], {'base': ['GPU-1'], 'e80': ['GPU-2']})

    def test_single_archive_rejects_unrelated_files_before_opening_them(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / 'run'
            directory.mkdir()
            (directory / '.env').touch()
            with patch.object(module, 'verified_single_report', return_value={'complete': True}, create=True):
                with self.assertRaisesRegex(ValueError, '허용'):
                    module.archive_single(directory, Path(tmp) / 'backup.zip')

    def test_single_archive_contains_only_run_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / 'run'
            items = ['identity.json', 'single-report.json', 'session-checks/' + 'a' * 32 + '.json',
                     'checks/' + 'b' * 32 + '/jobs/allocation-check/request.json',
                     'jobs/problem-001/request.json', 'started/001.json', 'routing/001.jsonl.gz']
            for name in items:
                path = directory / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b'fixture')
            with patch.object(module, 'verified_single_report', return_value={'complete': True}):
                verified = module.archive_single(directory, Path(tmp) / 'backup.zip')
            self.assertEqual(verified['files_verified'], len(items))

    def test_failed_warmup_load_is_unknown_without_blocking_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'jobs' / ('warmup-' + 'a' * 32)
            path.mkdir(parents=True)
            (path / 'events.jsonl').write_text('{broken', encoding='utf8')
            self.assertEqual(module._single_load_timings(tmp), [None])


if __name__ == '__main__':
    unittest.main()
