from pathlib import Path
import tempfile
import unittest
from test_gguf_integration import fixture, payloads
from moe_prune.gguf import GGUF

class PermutationTests(unittest.TestCase):
    def test_permutation_preserves_blocks_metadata_and_shared_tensors(self):
        with tempfile.TemporaryDirectory() as d:
            src, dst = Path(d)/'source.gguf', Path(d)/'ordered.gguf'
            original = fixture(src)
            result = GGUF(src).rewrite(dst, permutation={0:[3,1,0,2]})
            model=GGUF(dst); actual=payloads(model)
            self.assertEqual(model.metadata, GGUF(src).metadata)
            for name in ('blk.0.ffn_gate_inp.weight','blk.0.ffn_gate_up_exps.weight','blk.0.ffn_down_exps.weight','blk.0.ffn_down_exps.scale'):
                unit=len(original[name])//4
                self.assertEqual(actual[name],b''.join(original[name][i*unit:(i+1)*unit] for i in (3,1,0,2)))
            for name in ('token_embd.weight','blk.0.ffn_gate_inp.scale','blk.0.ffn_up.weight','blk.0.attn_q.weight'):
                self.assertEqual(actual[name],original[name])
            self.assertEqual(result['old_to_new'], {'0': {'3':0,'1':1,'0':2,'2':3}})
    def test_invalid_or_conflicting_permutations_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            src=Path(d)/'source.gguf'; fixture(src); model=GGUF(src)
            for n, mapping in enumerate(({},{0:[0,1,2]},{0:[0,1,2,2]},{0:[0,1,2,4]},{1:[0,1,2,3]})):
                with self.subTest(mapping=mapping), self.assertRaises(ValueError):
                    model.rewrite(Path(d)/f'{n}.gguf',permutation=mapping)
            with self.assertRaises(ValueError):
                model.rewrite(Path(d)/'both.gguf',allowlist={0:[0,1]},permutation={0:[0,1,2,3]})
    def test_unknown_expert_tensor_outside_routed_layers_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            src=Path(d)/'source.gguf'; dst=Path(d)/'output.gguf'
            fixture(src, ('blk.7.ffn_down_exps.weight', (32,4), 20, bytes(72)))
            with self.assertRaises(ValueError): GGUF(src).rewrite(dst,permutation={0:[3,1,0,2]})
            self.assertFalse(dst.exists())

class ProfileBindingTests(unittest.TestCase):
    def test_rejects_model_snapshot_or_generation_hash_mismatch(self):
        from tools.reorder_experts import validate_binding, SOURCE_HASH
        bundle={'generations_sha256':'gen', 'source_manifest_sha256':'snapshot'}
        identity={'model':SOURCE_HASH, 'snapshot_manifest':'snapshot'}
        validate_binding(bundle,identity,'gen')
        for bad_bundle, bad_identity, gen in ((bundle,{**identity,'model':'wrong'},'gen'), ({**bundle,'source_manifest_sha256':'wrong'},identity,'gen'), (bundle,identity,'wrong')):
            with self.assertRaises(ValueError): validate_binding(bad_bundle,bad_identity,gen)
