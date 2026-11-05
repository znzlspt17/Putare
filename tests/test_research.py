import unittest

from moe_prune.gguf import validate_allowlist
from moe_prune.research import split_records, aggregate_routing, select_experts, compare_logits, paired_quality


class AllowlistTests(unittest.TestCase):
    def test_valid_mapping_is_sorted_and_normalized(self):
        self.assertEqual(validate_allowlist({0: [3, 1], 1: [0, 2]}, [0, 1], 4, 2), {0: [1, 3], 1: [0, 2]})

    def test_invalid_mapping_is_rejected(self):
        for mapping in ({0: [0, 0]}, {0: [0]}, {0: [0, 4]}, {1: [0, 1]}):
            with self.subTest(mapping=mapping), self.assertRaises(ValueError):
                validate_allowlist(mapping, [0], 4, 2)


class ProtocolTests(unittest.TestCase):
    def test_splits_are_disjoint_and_deterministic(self):
        records = [{"id": str(i), "prompt": f"question {i}"} for i in range(10)]
        a = split_records(records, 4, 3)
        self.assertEqual({k: len(v) for k, v in a.items()}, {"calibration": 4, "selection": 3, "evaluation": 3})
        self.assertEqual(a, split_records(records, 4, 3))

    def test_duplicate_prompt_is_rejected(self):
        with self.assertRaises(ValueError):
            split_records([{"id": "1", "prompt": "a b"}, {"id": "2", "prompt": "a  b"}], 1, 1)

    def test_native_routing_aggregation_and_ties(self):
        events = [{"type": "routing", "domain": d, "layer": 0, "phase": "decode", "expert_ids": [0, 2], "weights": [0.4, 0.6], "weighted_output_norms": [2., 1.]} for d in ("math", "code")]
        p = aggregate_routing(events, 4)
        self.assertEqual(select_experts(p, 2, "frequency"), {0: [0, 2]})
        self.assertEqual(select_experts(p, 2, "contribution"), {0: [0, 2]})

    def test_nan_logits_do_not_pass_equivalence(self):
        self.assertFalse(compare_logits([float("nan")], [0.])["equivalent"])

    def test_contribution_is_not_fabricated(self):
        events = [{"type": "routing", "domain": d, "layer": 0, "phase": "decode", "expert_ids": [0], "weights": [1.]} for d in ("math", "code")]
        with self.assertRaises(ValueError):
            select_experts(aggregate_routing(events, 2), 1, "contribution")

    def test_paired_quality_does_not_invent_code_accuracy(self):
        source = [{'id': 'm', 'domain': 'math', 'dataset': 'gsm8k', 'reference': '#### 2', 'text': '2'},
                  {'id': 'c', 'domain': 'code', 'dataset': 'humanevalplus', 'text': 'print(1)'}]
        variant = [dict(r) for r in source]
        variant[0]['text'] = '3'
        result = paired_quality(source, variant, bootstrap_samples=100)
        self.assertEqual(result['gsm8k']['delta_accuracy'], -1)
        self.assertIsNone(result['humanevalplus']['code_execution_delta'])
        self.assertIsNone(result['humanevalplus']['delta_accuracy'])


if __name__ == "__main__":
    unittest.main()
