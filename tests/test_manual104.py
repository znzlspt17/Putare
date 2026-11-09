import json
from pathlib import Path
import tempfile
import unittest
import nbformat
from test_gguf_integration import fixture
from moe_prune.gguf import GGUF

class Pruning104Tests(unittest.TestCase):
    def test_slice_verifier_checks_all_payloads_and_mapping(self):
        from tools.manual104 import verify_sliced, derive_mapping
        with tempfile.TemporaryDirectory() as d:
            src, dst=Path(d)/'src.gguf',Path(d)/'pruned.gguf'; fixture(src)
            GGUF(src).rewrite(dst,{0:[0,1]})
            info=verify_sliced(src,dst,{0:[0,1]})
            self.assertTrue(info['all_tensor_bytes_verified'])
            self.assertEqual(derive_mapping([{'layer':0,'new_to_old':[2,0,3,1]}],[0],4,2)[0]['pruned_to_original'],[2,0])
            with dst.open('r+b') as f:
                t=next(t for t in GGUF(dst).tensors if t.name=='blk.0.ffn_down_exps.scale'); f.seek(GGUF(dst).data_offset+t.offset);f.write(b'\xff')
            with self.assertRaises(ValueError): verify_sliced(src,dst,{0:[0,1]})
    def test_mapping_rejects_duplicates(self):
        from tools.manual104 import derive_mapping
        with self.assertRaises(ValueError): derive_mapping([{'layer':0,'new_to_old':[0,0,2,3]}],[0],4,2)
    def test_notebook_execution_flags_are_boolean(self):
        nb=nbformat.read(Path(__file__).resolve().parents[1]/'ipynb/04_expert_pruning_104.ipynb',as_version=4); cells=[c.source for c in nb.cells if c.cell_type=='code']
        import ast
        tree=ast.parse('\n'.join(cells)); flags={}
        for n in ast.walk(tree):
            if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name) and n.targets[0].id.startswith('RUN_'):
                self.assertIsInstance(n.value,ast.Constant);flags[n.targets[0].id]=n.value.value
        self.assertGreaterEqual(len(flags),4);self.assertTrue(all(type(v) is bool for v in flags.values()))
    def test_orphan_expert_layer_rejected(self):
        from tools.manual104 import validate_expert_layout
        with tempfile.TemporaryDirectory() as d:
            src=Path(d)/'x.gguf'; fixture(src,('blk.7.ffn_down_exps.weight',(32,4),20,bytes(72)))
            with self.assertRaises(ValueError): validate_expert_layout(GGUF(src))

class WorkflowTests(unittest.TestCase):
    def test_real_128_to_104_synthetic_fixture(self):
        from tools.manual104 import verify_sliced,validate_expert_layout
        with tempfile.TemporaryDirectory() as d:
            src,dst=Path(d)/'s.gguf',Path(d)/'d.gguf';fixture(src,experts=128,top_k=8)
            model=GGUF(src);validate_expert_layout(model);model.rewrite(dst,{0:list(range(104))})
            self.assertEqual(GGUF(dst).architecture(),([0],104,8));self.assertTrue(verify_sliced(src,dst,{0:list(range(104))})['all_tensor_bytes_verified'])
            with self.assertRaises(FileExistsError):model.rewrite(dst,{0:list(range(104))})
    def test_routing_104_rejects_outside_missing_duplicate(self):
        from moe_prune.base_benchmark import validate_routing
        event={'type':'routing','phase':'prefill','layer':0,'token_index':0,'expert_ids':list(range(8)),'weights':[.125]*8,'weighted_output_norms':[1.]*8}
        result={'prompt_tokens':1,'processed_decode_tokens':0}
        self.assertEqual(validate_routing([event],result,layers=1,experts=104),1)
        for events in ([],[event,event],[{**event,'expert_ids':[104]+list(range(1,8))}]):
            with self.subTest(events=events),self.assertRaises(ValueError):validate_routing(events,result,layers=1,experts=104)
    def test_comparison_closes_resident_between_different_models(self):
        from unittest.mock import patch
        import tools.manual104 as w
        from moe_prune import resident,research
        instances=[]
        class Fake:
            def __init__(self,*a):self.model=None;instances.append(self)
            def __enter__(self):return self
            def __exit__(self,*a):pass
            def run(self,name,request):
                if self.model is not None and self.model!=request['model']:raise ValueError('resident identity changed')
                self.model=request['model'];return {'logits':[1.,2.],'masked_layers':list(range(30)) if 'allowlist' in request else []}
            def close(self):self.model=None
        record={'id':'x','prompt':'hi'}
        with tempfile.TemporaryDirectory() as d, patch.object(w,'selected_records',return_value=[record]), patch.object(w,'execution_identity',return_value=({},{})), patch.object(w,'validate_job'),patch.object(resident,'ResidentExperiment',Fake),patch.object(research,'read_jsonl',return_value=iter([{'type':'routing','expert_ids':list(range(8))}])):
            result=w.compare_masked((1,),Path(d));self.assertTrue(result[0]['equivalent']);self.assertIsNone(instances[0].model)
    def test_notebook_runall_with_flags_disabled_does_not_execute_actions(self):
        from unittest.mock import patch
        nb=nbformat.read(Path(__file__).resolve().parents[1]/'ipynb/04_expert_pruning_104.ipynb',as_version=4)
        import tools.manual104 as w
        names=['export','compare_masked','run_generation','grade','render_profiles']
        with __import__('contextlib').ExitStack() as stack:
            mocks=[stack.enter_context(patch.object(w,n,side_effect=AssertionError(n))) for n in names]
            stack.enter_context(patch('subprocess.Popen',side_effect=AssertionError('subprocess')))
            stack.enter_context(patch('socket.socket',side_effect=AssertionError('network')))
            g={}
            for c in nb.cells:
                if c.cell_type=='code':
                    import ast
                    tree=ast.parse(c.source)
                    for node in ast.walk(tree):
                        if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name) and node.targets[0].id.startswith('RUN_'):
                            node.value=ast.copy_location(ast.Constant(False),node.value)
                    exec(compile(tree,'notebook','exec'),g)
            self.assertTrue(all(not x.called for x in mocks))
    def test_generation_uses_complete_104_contract_with_synthetic_engine(self):
        from unittest.mock import patch
        import tools.manual104 as w
        from moe_prune import resident
        requests=[]
        class Fake:
            def __init__(self,root,config):self.root=Path(root)
            def __enter__(self):return self
            def __exit__(self,*a):pass
            def run(self,name,request):
                requests.append(request);p=self.root/'jobs'/name;p.mkdir(parents=True);(p/'events.jsonl').write_text('test')
                return {'stop_reason':'eog','token_ids':[1,2],'text':'result','masked_layers':[]}
            def check(self):pass
        record={'id':'livecodebench:3674','prompt':'hi','domain':'code','dataset':'test','lcb_problem':{'question_title':'x'}}
        with tempfile.TemporaryDirectory() as d,patch.object(w,'selected_records',return_value=[record]),patch.object(w,'execution_identity',return_value=({},{})),patch.object(w,'validate_job',return_value={}),patch.object(w,'compress_events',return_value={'compressed_sha256':'x'}),patch.object(resident,'ResidentExperiment',Fake):
            w.run_generation((1,),Path(d));status=json.loads((Path(d)/'status.json').read_text());self.assertTrue(status['complete']);self.assertEqual(status['completed'],1)
            self.assertTrue(requests[0]['instrumentation'] and requests[0]['all_token_logits']);self.assertEqual(requests[0]['sampling'],{'temperature':1.,'top_p':.95,'top_k':64})
            self.assertEqual(json.loads((Path(d)/'grade-inputs/001.json').read_text())['text'],'result')
    def test_comparison_rejects_expert_outside_mask(self):
        from unittest.mock import patch
        import tools.manual104 as w
        from moe_prune import resident,research
        class Fake:
            def __init__(self,*a):pass
            def __enter__(self):return self
            def __exit__(self,*a):pass
            def run(self,name,request):return {'logits':[1.],'masked_layers':list(range(30))}
            def close(self):pass
        with tempfile.TemporaryDirectory() as d,patch.object(w,'selected_records',return_value=[{'id':'x','prompt':'p'}]),patch.object(w,'execution_identity',return_value=({},{})),patch.object(w,'validate_job'),patch.object(resident,'ResidentExperiment',Fake),patch.object(research,'read_jsonl',return_value=iter([{'type':'routing','expert_ids':[104]}])):
            with self.assertRaises(ValueError):w.compare_masked((1,),Path(d))
    def test_manual104_requests_whole_gpu_model(self):
        import tools.manual104 as w
        request=w.checked_request({'prompt':'question'},'model.gguf')
        self.assertEqual((request['n_gpu_layers'],request['offload_kqv']),(31,True))
    def test_rejects_unknown_axis(self):
        from tools.manual104 import validate_expert_layout
        with tempfile.TemporaryDirectory() as d:
            src=Path(d)/'s.gguf';fixture(src,('blk.0.ffn_down_exps.bias',(4,),0,bytes(16)))
            with self.assertRaises(ValueError):validate_expert_layout(GGUF(src))
    def test_prefill_decode_combination_and_104_expert_plot(self):
        from unittest.mock import patch
        import tools.manual104 as w
        from moe_prune import base_benchmark
        with tempfile.TemporaryDirectory() as d:
            base=Path(d);run=base/'run';run.mkdir();directory=base/'eval';directory.mkdir();(directory/'answers').mkdir();(directory/'routing').mkdir()
            (directory/'status.json').write_text(json.dumps({'complete':True,'indices':[1]}))
            (directory/'answers/001.json').write_text(json.dumps({'id':'sample','routing_path':'routing/001.jsonl.gz','routing_integrity':{'compressed_sha256':'ok'}}))
            (directory/'routing/001.jsonl.gz').write_bytes(b'fixture')
            (run/'export-manifest.json').write_text(json.dumps({'layers':[{'pruned_to_original':list(range(104))} for _ in range(30)]}))
            c=[0]*104;c[1]=1;c[2]=2
            phase={'0':{'tokens':10,'counts':c,'token_rate':[x/10 for x in c]}}
            profile={'sample':{'prefill':phase,'decode':phase}}
            with patch.object(w,'RUN',run),patch.object(w,'sha256',return_value='ok'),patch.object(base_benchmark,'aggregate_profiles',return_value=profile):out=w.render_profiles(directory)
            svg=next(out.glob('*.svg')).read_text(encoding='utf-8')
            self.assertEqual(svg.count('<rect x='),30*104)
            for item in ('#000000','#ff0000','#808080','original 2','prefill + decode','4/20'):
                self.assertIn(item,svg)
            self.assertEqual(w.combined_rows(profile['sample'])['0']['tokens'],20)
    def test_invalid_allowlists_and_missing_weight_fail_without_output(self):
        with tempfile.TemporaryDirectory() as d:
            src=Path(d)/'s.gguf';fixture(src,experts=128,top_k=8)
            for n,bad in enumerate(({0:[0]*104},{0:list(range(103))+[128]},{0:list(range(7))},{})):
                dst=Path(d)/f'{n}.gguf'
                with self.subTest(allowlist=bad),self.assertRaises(ValueError): GGUF(src).rewrite(dst,bad)
                self.assertFalse(dst.exists())
            model=GGUF(src);model.tensors=[x for x in model.tensors if x.name!='blk.0.ffn_down_exps.weight']
            dst=Path(d)/'missing.gguf'
            with self.assertRaises(ValueError):model.rewrite(dst,{0:list(range(104))})
            self.assertFalse(dst.exists())
