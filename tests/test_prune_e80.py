import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from moe_prune.gguf import GGUF,sha256
from test_gguf_integration import fixture

class PruneE80Tests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        self.source=self.root/'e104.gguf';fixture(self.source,experts=104,top_k=8)
        self.reorder=self.root/'reorder.json';self.reorder.write_text(json.dumps({'output_sha256':'parent','layers':[{'layer':0,'new_to_old':list(reversed(range(128)))}]}))
        self.manifest=self.root/'source.json';self.manifest.write_text(json.dumps({'source_sha256':'parent','mapping_manifest_sha256':sha256(self.reorder),'output_sha256':sha256(self.source),'expert_count':104,'top_k':8,'file_validation_status':'passed','layers':[{'layer':0,'pruned_to_ordered':list(range(104)),'pruned_to_original':list(reversed(range(128)))[:104]}]}))
        self.target=self.root/'e80.gguf';self.run=self.root/'run'
        self.kw={'source':self.source,'output':self.target,'run':self.run,'source_manifest':self.manifest,'expected_sha':sha256(self.source),'expected_bytes':self.source.stat().st_size,'expected_layers':[0],'reorder_manifest':self.reorder}
    def test_104_to_80_bytes_and_mapping(self):
        from tools.prune_e80 import _export_e80
        old=self.source.read_bytes();result=_export_e80(**self.kw)
        self.assertEqual(old,self.source.read_bytes());self.assertEqual(GGUF(self.target).architecture(),([0],80,8));self.assertTrue(result['verification']['all_tensor_bytes_verified'])
        self.assertEqual(result['layers'][0]['e80_to_original'],list(reversed(range(128)))[:80]);self.assertEqual(result['layers'][0]['removed_source104'],list(range(80,104)))
        self.assertEqual(result['output_bytes'],result['expected_output_bytes']);self.assertEqual(result['inference_status'],'unverified')
    def test_wrong_hash_or_mapping_rejected(self):
        from tools.prune_e80 import _export_e80
        with self.assertRaises(ValueError):_export_e80(**{**self.kw,'expected_sha':'bad'})
        x=json.loads(self.manifest.read_text());x['layers'][0]['pruned_to_original'][1]=x['layers'][0]['pruned_to_original'][0];self.manifest.write_text(json.dumps(x))
        with self.assertRaises(ValueError):_export_e80(**self.kw)
        self.assertFalse(self.target.exists())
    def test_existing_output_manifest_or_partial_rejected(self):
        from tools.prune_e80 import _export_e80
        for path in (self.target,self.target.with_name(self.target.name+'.partial'),self.run/'export-manifest.json'):
            path.parent.mkdir(parents=True,exist_ok=True);path.write_text('untouched')
            with self.assertRaises(FileExistsError):_export_e80(**self.kw)
            self.assertEqual(path.read_text(),'untouched');path.unlink()
    def test_missing_weight_rejected_before_write(self):
        from tools.prune_e80 import _export_e80
        real=GGUF(self.source);real.tensors=[t for t in real.tensors if t.name!='blk.0.ffn_down_exps.weight']
        with patch('tools.prune_e80.GGUF',return_value=real),self.assertRaises(ValueError):_export_e80(**self.kw)
        self.assertFalse(self.target.exists())
    def test_detects_corruption_after_rewrite(self):
        from tools.prune_e80 import _export_e80
        original=GGUF.rewrite
        def corrupt(model,destination,**kwargs):
            value=original(model,destination,**kwargs);m=GGUF(destination);t=next(x for x in m.tensors if x.name=='blk.0.ffn_down_exps.scale')
            with Path(destination).open('r+b') as f:f.seek(m.data_offset+t.offset);f.write(b'\xff')
            return value
        with patch.object(GGUF,'rewrite',corrupt),self.assertRaises(ValueError):_export_e80(**self.kw)
        self.assertFalse((self.run/'export-manifest.json').exists())
