"""벤치마크 데이터와 답변의 관찰 가능한 계약을 검증한다."""
import unittest
import tempfile
import base64
import pickle
import zlib
import io
import json
import hashlib
from pathlib import Path
from unittest.mock import patch

from moe_prune.benchmark_data import (aime_answer, final_answer,
                                      select_v6_new, windows_to_wsl_path,
                                      prepare_snapshot, load_snapshot, grade_snapshot, _lcb_rows, validate_id_proof,
                                      grading_cache_key)
from tools.lcb_grade import decode_cases


class BenchmarkDataTests(unittest.TestCase):
    def test_v6_only_uses_ids_missing_from_v5(self):
        old = [{'question_id': '1'}, {'question_id': '2'}]
        all_rows = old + [{'question_id': '3'}, {'question_id': '4'}]
        self.assertEqual([r['question_id'] for r in select_v6_new(old, all_rows, expected=2)], ['3', '4'])
        with self.assertRaises(ValueError):
            select_v6_new(old, all_rows, expected=175)
        with self.assertRaises(ValueError):
            select_v6_new(old, all_rows + [all_rows[-1]], expected=2)

    def test_aime_final_boxed_integer_only(self):
        self.assertEqual(aime_answer(r'풀이 10, 답은 \boxed{042}'), 42)
        self.assertIsNone(aime_answer('계산 결과 42입니다'))
        self.assertIsNone(aime_answer(r'\boxed{1000}'))
        self.assertIsNone(aime_answer(r'\boxed{-1}'))
        self.assertIsNone(aime_answer(r'\boxed{1.5}'))

    def test_reasoning_channel_is_excluded(self):
        text = '<|channel>thought\n\\boxed{42}<channel|>\\boxed{27}'
        self.assertEqual(aime_answer(final_answer(text)), 27)
        self.assertIsNone(aime_answer(final_answer('<|channel>thought\n\\boxed{42}')))

    def test_windows_paths_map_to_wsl_mount(self):
        self.assertEqual(windows_to_wsl_path(r'D:\Repos\wemakemodel\runs\my run'),
                         '/mnt/d/Repos/wemakemodel/runs/my run')
        with self.assertRaises(ValueError):
            windows_to_wsl_path(r'\\server\share\file')

    def test_snapshot_uses_frozen_sources_and_detects_record_tampering(self):
        math = [{'problem_idx': i, 'problem': f'question {i}', 'answer': i} for i in range(30)]
        code = [{'question_id': str(i), 'question_content': f'problem {i}', 'starter_code': '',
                 'private_test_cases': 'secret-test-not-in-prompt'} for i in range(175)]
        prompts = [{'prompt': 'official format ' + row['question_content'], 'system_prompt': 'official system'}
                   for row in code]
        with tempfile.TemporaryDirectory() as folder, \
             patch('moe_prune.benchmark_data._revision', side_effect=[('aime-rev', 'a'), ('lcb-rev', 'b')]), \
             patch('moe_prune.benchmark_data.official_identity', return_value={'source_digest': 'trusted'}), \
             patch('moe_prune.benchmark_data.grader_identity', return_value={'adapter': 'trusted'}), \
             patch('moe_prune.benchmark_data._aime_rows', return_value=(math, 'aime-sha')), \
             patch('moe_prune.benchmark_data._lcb_rows', return_value=([], code, {'test6.jsonl': 'sha'})), \
             patch('moe_prune.benchmark_data._official_prompts', return_value=prompts):
            destination = Path(folder) / 'snapshot'
            manifest = prepare_snapshot(destination, folder)
            records, observed = load_snapshot(destination)
            self.assertEqual(manifest['counts'], {'math': 30, 'code': 175})
            self.assertEqual(observed['datasets']['lcb']['revision'], 'lcb-rev')
            self.assertTrue(all('secret-test-not-in-prompt' not in r['prompt'] for r in records))
            self.assertEqual(records[-1]['lcb_problem']['private_test_cases'], 'secret-test-not-in-prompt')
            with (destination / 'records.jsonl').open('ab') as stream:
                stream.write(b' ')
            with self.assertRaises(ValueError):
                load_snapshot(destination)

    def test_private_test_pickle_cannot_load_executable_globals(self):
        encoded = base64.b64encode(zlib.compress(pickle.dumps(eval))).decode()
        with self.assertRaises(pickle.UnpicklingError):
            decode_cases(encoded)

    def test_grading_rejects_generations_from_other_run(self):
        with tempfile.TemporaryDirectory() as folder, \
             patch('moe_prune.benchmark_data.load_snapshot', return_value=([], {})):
            path = Path(folder)
            with self.assertRaisesRegex(ValueError, '실행 폴더'):
                grade_snapshot(path, path / 'other.jsonl', path / 'run', path)

    def test_official_v5_index_and_v6_loader_are_pinned_and_checked(self):
        old_files = ['test.jsonl'] + [f'test{i}.jsonl' for i in range(2, 6)]
        allowed = {'release_v5': old_files, 'release_v6': old_files + ['test6.jsonl']}
        loader = ('ALLOWED_FILES = ' + repr(allowed)).encode()
        space = json.dumps({'sha': 'official-index-revision'}).encode()
        index = json.dumps([{'question_id': f'old-{i}'} for i in range(880)]).encode()
        data = b''.join(json.dumps({'question_id': f'new-{i}'}).encode() + b'\n' for i in range(175))
        with patch('moe_prune.benchmark_data._public', side_effect=[loader, space, index]), \
             patch('moe_prune.benchmark_data._open_public', return_value=io.BytesIO(data)):
            old, current, provenance = _lcb_rows('dataset-revision')
        self.assertEqual((len(old), len(current)), (880, 1055))
        self.assertEqual(len(select_v6_new(old, current)), 175)
        self.assertEqual(provenance['v5_ids_source_revision'], 'official-index-revision')
        self.assertEqual(provenance['release_v6_files'], old_files + ['test6.jsonl'])

    def test_v5_id_proof_is_bound_to_exact_snapshot(self):
        with tempfile.TemporaryDirectory() as folder:
            snapshot = Path(folder) / 'snapshot'
            snapshot.mkdir()
            records = b'example records'
            (snapshot / 'records.jsonl').write_bytes(records)
            record_hash = hashlib.sha256(records).hexdigest()
            filenames = ['test.jsonl'] + [f'test{i}.jsonl' for i in range(2, 6)]
            manifest = {'datasets': {'lcb': {'revision': 'dataset-revision',
                         'source_sha256': {'v5_ids_source_revision': 'index-revision',
                                           'release_v5_files': filenames}}},
                        'file_sha256': {'records': record_hash}}
            raw = json.dumps(manifest).encode()
            (snapshot / 'manifest.json').write_bytes(raw)
            proof = {'status': 'verified', 'snapshot_manifest_sha256': hashlib.sha256(raw).hexdigest(),
                     'snapshot_records_sha256': record_hash, 'dataset_revision': 'dataset-revision',
                     'v5_index_revision': 'index-revision', 'v5_count': 880, 'v6_new_count': 175,
                     'v5_files': {name: {'rows': 176, 'sha256': 'a' * 64} for name in filenames}}
            path = Path(folder) / 'snapshot-id-validation.json'
            path.write_text(json.dumps(proof), encoding='utf-8')
            self.assertEqual(validate_id_proof(snapshot)['status'], 'verified')
            (snapshot / 'manifest.json').write_bytes(raw + b' ')
            with self.assertRaises(ValueError):
                validate_id_proof(snapshot)

    def test_grading_cache_is_bound_to_exact_id_proof(self):
        first = grading_cache_key({'id': 'q1'}, {'source': 'a'}, {'adapter': 'b'}, {'proof': 'v1'})
        second = grading_cache_key({'id': 'q1'}, {'source': 'a'}, {'adapter': 'b'}, {'proof': 'v2'})
        self.assertNotEqual(first, second)


if __name__ == '__main__':
    unittest.main()
