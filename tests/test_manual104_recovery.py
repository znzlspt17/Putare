import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from moe_prune.gguf import GGUF,sha256
from test_gguf_integration import fixture

class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup); self.root=Path(self.tmp.name)
        self.src=self.root/'original.gguf';self.dst=self.root/'pruned.gguf';fixture(self.src);GGUF(self.src).rewrite(self.dst,{0:[0,1]})
        self.mapping=self.root/'mapping.json';self.mapping.write_text(json.dumps({'output_sha256':sha256(self.src),'layers':[{'layer':0,'new_to_old':[2,0,3,1]}]}))
        self.run=self.root/'run';self.kw={'source':self.src,'output':self.dst,'run':self.run,'mapping_path':self.mapping,'source_sha':sha256(self.src),'pruned_sha':sha256(self.dst),'layers':[0],'experts':4,'keep':2,'top_k':2,'expected_size':self.dst.stat().st_size}
    def test_recovers_only_record_from_verified_files(self):
        from tools.manual104 import _recover_existing_manifest
        before=self.dst.read_bytes(); result=_recover_existing_manifest(**self.kw)
        self.assertEqual(self.dst.read_bytes(),before);self.assertTrue(result['verification']['all_tensor_bytes_verified']);self.assertEqual(result['layers'][0]['pruned_to_original'],[2,0]);self.assertIn('recovery_provenance',result)
        with self.assertRaises(FileExistsError):_recover_existing_manifest(**self.kw)
    def test_bad_hash_and_corrupt_tensor_rejected(self):
        from tools.manual104 import _recover_existing_manifest
        with self.assertRaises(ValueError):_recover_existing_manifest(**{**self.kw,'pruned_sha':'bad'})
        m=GGUF(self.dst);t=next(x for x in m.tensors if x.name=='blk.0.ffn_down_exps.scale')
        with self.dst.open('r+b') as f:f.seek(m.data_offset+t.offset);f.write(b'\xff')
        with self.assertRaises(ValueError):_recover_existing_manifest(**{**self.kw,'pruned_sha':sha256(self.dst)})
        self.assertFalse((self.run/'export-manifest.json').exists())
    def test_missing_record_reports_recovery_before_bundle_work(self):
        import tools.manual104 as m
        with patch.object(m,'EXPORT_RUN',self.run):
            with self.assertRaisesRegex(FileNotFoundError,'recover-existing'):m.execution_identity(self.root/'other')
