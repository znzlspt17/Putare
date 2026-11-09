import unittest

class CostRoutingTests(unittest.TestCase):
    def test_predictor_position_first_and_later_tokens(self):
        from tools.analyze_cost_routing import predictor
        self.assertEqual(predictor(570,0),('prefill',569))
        self.assertEqual(predictor(570,1),('decode',570))
        self.assertEqual(predictor(570,7),('decode',576))
    def test_overlap_multibyte_and_byte_fallback(self):
        from tools.analyze_cost_routing import pieces_and_ranges,overlap_tokens
        data,ranges=pieces_and_ranges([0,1,2],['▁가','<0xEB>','<0x82>'],[1,6,6])
        self.assertEqual(data,' 가'.encode()+bytes([0xeb,0x82]))
        self.assertEqual(overlap_tokens(ranges,2,3),[0])
        self.assertEqual(overlap_tokens(ranges,4,6),[1,2])
    def test_mapping_and_values_reject_malformed(self):
        from tools.analyze_cost_routing import validate_mapping,validate_vectors
        import math
        manifest={'expert_count':2,'top_k':8,'layers':[{'layer':0,'ids':[5,6]}]}
        self.assertEqual(validate_mapping(manifest,'ids',2,layers=1),[[5,6]])
        for ids in ([5,5],[5,128],[5]):
            with self.subTest(ids=ids),self.assertRaises(ValueError):validate_mapping({**manifest,'layers':[{'layer':0,'ids':ids}]},'ids',2,layers=1)
        with self.assertRaises(ValueError):validate_vectors(list(range(8)),[float('nan')]*8,[1.]*8,80)
        with self.assertRaises(ValueError):validate_vectors(list(range(8)),None,[1.]*8,80)
