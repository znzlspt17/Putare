"""공개 snapshot은 인증 없이 읽고 분할 무결성을 검증한다."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from moe_prune.datasets import load_snapshot, prepare_datasets


class DatasetPreparationTests(unittest.TestCase):
    def test_public_datasets_load_anonymously(self):
        calls = []

        def load(repo, subset, *, split, revision, **kwargs):
            calls.append((repo, subset, split, revision, kwargs))
            if repo == 'openai/gsm8k':
                return [{'question': f'gsm {split} {i}', 'answer': '#### 42'} for i in range(4)]
            if repo == 'google-research-datasets/mbpp':
                return [{'task_id': i, 'text': f'mbpp {i}', 'code': 'return 42'} for i in range(4)]
            if repo == 'Salesforce/wikitext':
                return [{'text': f'wiki {i} ' + 'text ' * 40} for i in range(5)]
            return [{'problem': 'symbolic evaluation problem', 'answer': 'x'}]

        human = {'HumanEval/0': {'prompt': 'human eval prompt', 'canonical_solution': 'pass'}}
        mbpp = {'Mbpp/0': {'prompt': 'mbpp plus prompt', 'canonical_solution': 'pass'}}
        with tempfile.TemporaryDirectory() as folder, \
             patch('moe_prune.datasets._revision', return_value=('revision', 'license')), \
             patch('datasets.load_dataset', side_effect=load), \
             patch('evalplus.data.get_human_eval_plus', return_value=human), \
             patch('evalplus.data.get_mbpp_plus', return_value=mbpp):
            destination = Path(folder) / 'snapshot'
            expected, _ = prepare_datasets(destination, 1, 1, 1, 42)
            self.assertTrue(calls)
            self.assertTrue(all(call[-1].get('token') is False for call in calls))
            actual, _ = load_snapshot(destination)
            self.assertEqual(actual, expected)


if __name__ == '__main__':
    unittest.main()
