import tempfile
import unittest
import json
import os
import zipfile
from pathlib import Path
from unittest.mock import patch

from tools import comparison205 as module


class Comparison205Tests(unittest.TestCase):
    def test_budget_uses_remaining_context_and_rejects_full_input(self):
        self.assertEqual(module.generation_budget(570), 262144 - 570 - 1)
        with self.assertRaises(ValueError):
            module.generation_budget(262143)

    def test_started_case_is_never_dispatched_again(self):
        with tempfile.TemporaryDirectory() as tmp:
            module.reserve_case(Path(tmp), 1, 'aime2026:1')
            with self.assertRaisesRegex(RuntimeError, '시작'):
                module.reserve_case(Path(tmp), 1, 'aime2026:1')

    def test_preflight_detects_changed_token_sequence(self):
        base = [{'id': 'x', 'tokens': [1, 2], 'rendered_prompt': 'p'}]
        e80 = [{'id': 'x', 'tokens': [1, 3], 'rendered_prompt': 'p'}]
        with self.assertRaisesRegex(ValueError, '토큰'):
            module.compare_preflight(base, e80)

    def test_started_without_answer_blocks_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            module.reserve_case(tmp, 1, 'aime2026:1')
            self.assertEqual(module.case_state(tmp, 1, 'aime2026:1'), 'interrupted')

    def test_math_only_uses_final_answer(self):
        self.assertFalse(module.grade_math('<|channel>thought\n\\boxed{042}', 42))
        self.assertTrue(module.grade_math('<|channel>thought\n42<channel|>\\boxed{042}', 42))

    def test_cached_grade_rejects_changed_answer(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'grade.json'
            path.write_text(json.dumps({'answer_sha256': 'old', 'passed': True}), encoding='utf8')
            with self.assertRaisesRegex(ValueError, '답안'):
                module.grade_cached(path, {'answer_sha256': 'new'}, lambda: False)

    def test_cached_grade_detects_tampered_verdict(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'grade.json'
            answer = {'answer_sha256': 'answer-hash'}
            module.grade_cached(path, answer, lambda: True)
            cached = json.loads(path.read_text(encoding='utf8'))
            cached['passed'] = False
            path.write_text(json.dumps(cached), encoding='utf8')
            with self.assertRaisesRegex(ValueError, '채점'):
                module.grade_cached(path, answer, lambda: False)

    def test_grading_failure_is_persisted_and_raised(self):
        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(module, '_grade_impl', side_effect=TimeoutError('grader failed')):
            with self.assertRaises(TimeoutError):
                module.grade(tmp, base_model='a', e80_model='b', engine='engine')
            status = json.loads((Path(tmp) / 'grading-status.json').read_text(encoding='utf8'))
            self.assertFalse(status['complete'])
            self.assertEqual(status['status'], 'infrastructure_error')

    def test_report_preserves_pairs_and_selection_overlap(self):
        records = [{'id': 'a', 'domain': 'code'}, {'id': 'b', 'domain': 'code'}]
        base = {'a': {'passed': True}, 'b': {'passed': False}}
        e80 = {'a': {'passed': False}, 'b': {'passed': True}}
        report = module.comparison_report(records, base, e80, ['a'])
        self.assertEqual(report['paired']['base_only'], 1)
        self.assertEqual(report['paired']['e80_only'], 1)
        self.assertEqual(report['selection_overlap']['used']['total'], 1)
        self.assertEqual(report['selection_overlap']['unused']['total'], 1)

    def test_efficiency_counts_unclassified_and_separates_timing(self):
        answers = [
            {'generated_tokens': 7, 'channel_tokens': {'thinking': 3, 'final': 2, 'control': 2, 'unclassified': 0},
             'stop_reason': 'eog', 'final_answer_present': True, 'timings': {'prefill_s': 2, 'decode_s': 4},
             'measurement': {'wall_s': 7, 'os_gpu_memory': {'peak_bytes': 80, 'scope': 'session_to_date'}}},
            {'generated_tokens': 5, 'channel_tokens': {'thinking': None, 'final': None, 'control': None, 'unclassified': 5},
             'stop_reason': 'length', 'final_answer_present': False, 'timings': {'prefill_s': 1, 'decode_s': 3},
             'measurement': {'wall_s': 5, 'os_gpu_memory': {'peak_bytes': 100, 'scope': 'session_to_date'}}},
        ]
        summary = module.efficiency_summary(answers)
        self.assertEqual(summary['generated_tokens'], 12)
        self.assertEqual(summary['channel_tokens']['unclassified'], 5)
        self.assertEqual(summary['length_stops'], 1)
        self.assertEqual(summary['missing_final_answers'], 1)
        self.assertEqual(summary['decode_s'], 7)
        self.assertEqual(summary['gpu_memory_session_peak_bytes'], 100)

    def test_report_contains_domain_efficiency_and_rates(self):
        record = {'id': 'a', 'domain': 'math'}
        answer = {'passed': True, 'generated_tokens': 2,
                  'channel_tokens': {'thinking': None, 'final': None, 'control': None, 'unclassified': 2},
                  'stop_reason': 'length', 'final_answer_present': False,
                  'timings': {'prefill_s': 1, 'decode_s': 2},
                  'measurement': {'wall_s': 4, 'os_gpu_memory': {'peak_bytes': 9, 'scope': 'session_to_date'}}}
        report = module.comparison_report([record], {'a': answer}, {'a': answer}, [])
        self.assertEqual(report['domains']['math']['base_percentage'], 100.0)
        self.assertEqual(report['efficiency_by_domain']['math']['base']['length_stops'], 1)

    def test_archive_detects_tampered_payload(self):
        with tempfile.TemporaryDirectory() as tmp:
            archive = Path(tmp) / 'results.zip'
            with zipfile.ZipFile(archive, 'w') as z:
                z.writestr('result.json', 'tampered')
                z.writestr('manifest.json', json.dumps({'result.json': '0' * 64}))
            with self.assertRaisesRegex(ValueError, '해시'):
                module.verify_archive(archive)

    def test_archive_rejects_unknown_secret_filename_before_reading(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'run'
            root.mkdir()
            (root / '.env').touch()
            with patch.object(module, 'verified_report', return_value={'complete': True}):
                with self.assertRaisesRegex(ValueError, '허용'):
                    module.archive_results(root, Path(tmp) / 'result.zip')

    def test_archive_publish_waits_for_verification_and_allows_retry(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, target = Path(tmp) / 'run', Path(tmp) / 'result.zip'
            root.mkdir()
            (root / 'identity.json').write_text('{}', encoding='utf8')
            with patch.object(module, 'verified_report', return_value={'complete': True}), \
                 patch.object(module, 'verify_archive', side_effect=ValueError('bad archive')):
                with self.assertRaises(ValueError):
                    module.archive_results(root, target)
            self.assertFalse(target.exists())
            with patch.object(module, 'verified_report', return_value={'complete': True}):
                self.assertEqual(module.archive_results(root, target)['files_verified'], 1)
            self.assertTrue(target.is_file())

    def test_archive_publish_does_not_overwrite_racing_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, target = Path(tmp) / 'run', Path(tmp) / 'result.zip'
            root.mkdir()
            (root / 'identity.json').write_text('{}', encoding='utf8')
            real_link = os.link
            def racing_link(source, destination):
                Path(destination).write_text('other result', encoding='utf8')
                return real_link(source, destination)
            with patch.object(module, 'verified_report', return_value={'complete': True}), \
                 patch.object(module.os, 'link', side_effect=racing_link):
                with self.assertRaises(FileExistsError):
                    module.archive_results(root, target)
            self.assertEqual(target.read_text(encoding='utf8'), 'other result')

    def test_runtime_rejects_context_mismatch_before_reading_log(self):
        with self.assertRaisesRegex(ValueError, '조건'):
            module._validate_result('.', 'job', {'expected_prompt_tokens': [1],
                'expected_rendered_prompt': 'p', 'max_tokens': 2},
                {'actual_context_size': 8192}, 128)

    def test_runtime_rejects_cpu_kv_buffer(self):
        request = {'expected_prompt_tokens': [1], 'expected_rendered_prompt': 'p', 'max_tokens': 2}
        result = {'actual_context_size': 262144, 'enable_thinking': True, 'apply_chat_template': True,
                  'instrumentation_level': 'routing', 'masked_layers': [], 'profiled_layers': list(range(30)),
                  'stop_reason': 'eog', 'prompt_tokens': 1, 'rendered_prompt': 'p', 'routing_records': 0}
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / 'jobs' / 'job'
            folder.mkdir(parents=True)
            (folder / 'stderr.log').touch()
            for backend in ('CPU', 'CUDA_Host'):
                observed = {'offloaded_layers': {'gpu': 31, 'total': 31},
                            'kv_buffers_mib': [{'backend': backend, 'mib': 2}]}
                with self.subTest(backend=backend), \
                     patch.object(module, 'verify_runtime', return_value=observed), \
                     patch.object(module, 'validate_routing', return_value=0):
                    with self.assertRaisesRegex(ValueError, 'KV'):
                        module._validate_result(tmp, 'job', request, result, 128)
            observed = {'offloaded_layers': {'gpu': 31, 'total': 31},
                        'kv_buffers_mib': [{'backend': 'CUDA0', 'mib': 2}]}
            with patch.object(module, 'verify_runtime', return_value=observed), \
                 patch.object(module, 'validate_routing', return_value=0):
                self.assertEqual(module._validate_result(tmp, 'job', request, result, 128), observed)

    def test_dry_run_produces_410_identical_requests_without_answers(self):
        records = [{'id': str(i), 'domain': 'math' if i < 30 else 'code', 'prompt': f'p{i}',
                    'reference': 'secret answer'} for i in range(205)]
        def tokenize(rows, model, engine):
            return [{'id': row['id'], 'tokens': [1, i + 2], 'rendered_prompt': row['prompt']}
                    for i, row in enumerate(rows)]
        with patch.object(module, 'tokenize_records', side_effect=tokenize):
            plan = module.dry_run(records, 'base.gguf', 'e80.gguf', 'engine')
        self.assertEqual(sum(len(rows) for rows in plan['requests'].values()), 410)
        for profile in ('base', 'e80'):
            for request in plan['requests'][profile]:
                self.assertEqual(request['max_tokens'], 262141)
                self.assertNotIn('reference', request)
                self.assertEqual(request['sampling'], {'temperature': 1.0, 'top_p': .95, 'top_k': 64})

    def test_all_410_are_dispatched_once_and_completed_answers_resume(self):
        records = [{'id': str(i), 'domain': 'math' if i < 30 else 'code', 'prompt': f'p{i}'}
                   for i in range(205)]
        identity = {'gpu': {'uuid': 'fake'}}
        rows = [{'id': r['id'], 'tokens': [1], 'rendered_prompt': r['prompt']} for r in records]
        dry = {'base_tokens': rows, 'e80_tokens': rows, 'requests': {
            profile: [module.make_request(r, profile, 'engine', [1]) for r in records]
            for profile in ('base', 'e80')}}
        calls = []
        class FakeExperiment:
            def __init__(self, root, config): self.root = Path(root)
            def __enter__(self): return self
            def __exit__(self, *args): return False
            def check(self): pass
            def run(self, name, request):
                calls.append((self.root.name, name))
                folder = self.root / 'jobs' / name
                folder.mkdir(parents=True)
                result = {'type': 'result', 'text': 'answer', 'token_ids': [1]}
                (folder / 'events.jsonl').write_text(json.dumps(result) + '\n', encoding='utf-8')
                return {**result, 'measurement': {'wall_s': .1}}
        def tokenize(rows, model, engine):
            return [{'id': row['id'], 'tokens': [1], 'rendered_prompt': row['prompt']} for row in rows]
        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(module, '_check_prepared', return_value=(records, identity, dry)), \
             patch.object(module, 'tokenize_records', side_effect=tokenize), \
             patch.object(module, 'ensure_gpu'), patch.object(module, '_validate_result', return_value={}), \
             patch('moe_prune.resident.ResidentExperiment', FakeExperiment):
            args = {'base_model': 'base', 'e80_model': 'e80', 'engine': 'engine'}
            self.assertEqual(module.generate(tmp, **args)['answers'], 410)
            problem_calls = [x for x in calls if x[1].startswith('problem-')]
            self.assertEqual(len(problem_calls), 410)
            self.assertEqual(len(set(problem_calls)), 410)
            module.generate(tmp, **args)
            self.assertEqual(len(calls), 412)  # 각 모델 1회 워밍업을 포함한다.

    def test_interrupted_case_is_not_retried_but_unstarted_cases_resume(self):
        records = [{'id': str(i), 'domain': 'math', 'prompt': f'p{i}'} for i in range(3)]
        identity = {'gpu': {'uuid': 'fake'}}
        rows = [{'id': r['id'], 'tokens': [1], 'rendered_prompt': r['prompt']} for r in records]
        dry = {'base_tokens': rows, 'e80_tokens': rows, 'requests': {
            profile: [module.make_request(r, profile, 'engine', [1]) for r in records]
            for profile in ('base', 'e80')}}
        calls = []
        class FakeExperiment:
            def __init__(self, root, config): self.root = Path(root)
            def __enter__(self): return self
            def __exit__(self, *args): return False
            def check(self): pass
            def run(self, name, request):
                calls.append((self.root.name, name))
                if name == 'problem-002':
                    raise RuntimeError('injected failure')
                folder = self.root / 'jobs' / name
                folder.mkdir(parents=True)
                result = {'type': 'result', 'text': 'answer', 'token_ids': [1]}
                (folder / 'events.jsonl').write_text(json.dumps(result) + '\n', encoding='utf-8')
                return {**result, 'measurement': {'wall_s': .1}}
        def tokenize(rows, model, engine):
            return [{'id': row['id'], 'tokens': [1], 'rendered_prompt': row['prompt']} for row in rows]
        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(module, '_check_prepared', return_value=(records, identity, dry)), \
             patch.object(module, 'tokenize_records', side_effect=tokenize), \
             patch.object(module, 'ensure_gpu'), patch.object(module, '_validate_result', return_value={}), \
             patch('moe_prune.resident.ResidentExperiment', FakeExperiment), patch('builtins.print'):
            args = {'base_model': 'base', 'e80_model': 'e80', 'engine': 'engine'}
            with self.assertRaisesRegex(RuntimeError, 'injected failure'):
                module.generate(tmp, **args)
            with self.assertRaisesRegex(RuntimeError, '미완료'):
                module.generate(tmp, **args)
            self.assertEqual(calls.count(('base', 'problem-001')), 1)
            self.assertEqual(calls.count(('base', 'problem-002')), 1)
            self.assertEqual(calls.count(('base', 'problem-003')), 1)
            self.assertFalse(any(profile == 'e80' for profile, _ in calls))


if __name__ == '__main__':
    unittest.main()
