import unittest
from unittest.mock import patch
from moe_prune.base_benchmark import validate_routing

class Manual80Tests(unittest.TestCase):
    def test_whole_gpu_placement_closes_guard(self):
        from tools.manual80 import choose_placement
        seen=[];closed=[]
        class Guard:
            def __init__(self,c,r):self.n=r['n_gpu_layers'];seen.append(self.n)
            def preflight(self):
                return {'free':123}
            def close(self):closed.append(self.n)
        chosen,rows=choose_placement('model',guard_cls=Guard)
        self.assertEqual(chosen,31);self.assertEqual(seen,[31]);self.assertEqual(closed,seen);self.assertEqual(len(rows),1)
    def test_unexpected_error_stops_search_and_closes(self):
        from tools.manual80 import choose_placement
        seen=[]
        class Guard:
            def __init__(self,c,r):pass
            def preflight(self):raise RuntimeError('counter unavailable')
            def close(self):seen.append('closed')
        with self.assertRaises(RuntimeError):choose_placement('model',guard_cls=Guard)
        self.assertEqual(seen,['closed'])
    def test_payload_must_match_native_text_and_tokens(self):
        from tools.manual80 import verify_payload
        from moe_prune.research import canonical_hash
        native={'text':'answer','token_ids':[1,2]};record={'id':'one','lcb_problem':{'x':1}}
        answer={'id':'one',**native,'native_result_sha256':canonical_hash(native)};payload={'id':'one','problem':record['lcb_problem'],'text':'answer'}
        verify_payload(record,answer,payload,native)
        for altered in ({**answer,'text':'other'},{**answer,'token_ids':[2]},{**answer,'native_result_sha256':'bad'}):
            with self.assertRaises(ValueError):verify_payload(record,altered,payload,native)
    def test_80_routing_range(self):
        event={'type':'routing','phase':'prefill','layer':0,'token_index':0,'expert_ids':[80,1,2,3,4,5,6,7],'weights':[.125]*8,'weighted_output_norms':[1.]*8}
        with self.assertRaises(ValueError):validate_routing([event],{'prompt_tokens':1,'processed_decode_tokens':0},layers=1,experts=80)
    def test_80_profile_combines_prefill_and_decode(self):
        from tools.manual80 import combined_80
        counts=[0]*80;counts[1]=2
        phase={'0':{'tokens':4,'counts':counts}}
        x=combined_80({'prefill':phase,'decode':phase})['0']
        self.assertEqual(len(x['counts']),80);self.assertEqual(x['counts'][1],4);self.assertEqual(x['tokens'],8)
    def test_fake_native_session_finishes_before_status_complete(self):
        import tempfile,json
        from pathlib import Path
        import tools.manual80 as w
        import moe_prune.resident as resident, moe_prune.base_benchmark as benchmark
        closed=[];requests=[]
        records=[{'id':x,'prompt':'p','lcb_problem':{'question_title':'q'}} for x in w.IDS]
        class Fake:
            def __init__(self,root,config):self.root=Path(root)
            def __enter__(self):return self
            def __exit__(self,*args):closed.append(True)
            def check(self):pass
            def run(self,name,request):
                requests.append(request);folder=self.root/'jobs'/name;folder.mkdir(parents=True)
                rows=[]
                for phase,token in [('prefill',0),('decode',1)]:
                    for layer in range(30):rows.append({'type':'routing','phase':phase,'layer':layer,'token_index':token,'expert_ids':list(range(8)),'weights':[.125]*8,'weighted_output_norms':[1.]*8})
                native={'type':'result','text':'ok','token_ids':[1],'stop_reason':'eog','prompt_tokens':1,'processed_decode_tokens':1,'routing_records':60,'profiled_layers':list(range(30)),'gpu_fusion_disabled':True,'masked_layers':[],'enable_thinking':True,'requested_n_gpu_layers':31,'actual_context_size':65536,'cache_type_k':'q8_0','cache_type_v':'q8_0'}
                with (folder/'events.jsonl').open('w') as f:
                    for row in [*rows,native]:f.write(json.dumps(row)+'\n')
                (folder/'stderr.log').write_text('fixture');return {**native,'measurement':{}}
        real_write=w.write_json
        def write(path,value):
            if Path(path).name=='status.json' and value.get('complete'):self.assertTrue(closed)
            real_write(path,value)
        with tempfile.TemporaryDirectory() as d,patch.object(w,'source_identity',return_value=(records,{})),patch.object(w,'choose_placement',return_value=(31,[])),patch.object(resident,'ResidentExperiment',Fake),patch.object(benchmark,'verify_runtime',return_value={}),patch.object(w,'write_json',side_effect=write):
            out=Path(d)/'run';w.generate(out);status=json.loads((out/'status.json').read_text());self.assertTrue(status['complete']);self.assertEqual(len(requests),2)
            self.assertTrue(all(x['n_gpu_layers']==31 and x['all_token_logits'] and x['instrumentation'] and x['seed']==42 for x in requests))
            answer=json.loads((out/'answers/001.json').read_text());native=w.native_result(out/'jobs/problem-001/events.jsonl');w.verify_payload(records[0],answer,json.loads((out/'grade-inputs/001.json').read_text()),native)
    def test_grade_does_not_launch_worker_for_incomplete_generation(self):
        import json,tempfile
        from pathlib import Path
        import tools.manual80 as w
        with tempfile.TemporaryDirectory() as d, patch('subprocess.run',side_effect=AssertionError('launched')):
            path=Path(d);(path/'status.json').write_text(json.dumps({'complete':False}))
            with patch.object(w.sys,'platform','linux'),self.assertRaises(ValueError):w.grade(path)
    def test_grade_inputs_reject_tampered_native_payload_and_raw_log(self):
        import tempfile,json,gzip
        from pathlib import Path
        from moe_prune.gguf import sha256
        from moe_prune.research import canonical_hash
        import tools.manual80 as w
        with tempfile.TemporaryDirectory() as d:
            p=Path(d); bundle=p/'bundle';bundle.mkdir();(bundle/'records.jsonl').write_text('fixture');(bundle/'SHA256SUMS.json').write_text('{}');manifest=p/'manifest.json';manifest.write_text('{}')
            run=p/'run';run.mkdir();(run/'status.json').write_text(json.dumps({'complete':True,'completed':2,'total':2}))
            identity={'model_sha256':w.MODEL_SHA,'ids':w.IDS,'manifest_sha256':sha256(manifest),'records_sha256':sha256(bundle/'records.jsonl'),'bundle_inventory_sha256':sha256(bundle/'SHA256SUMS.json'),'engine_sha256':'engine'}
            (run/'identity.json').write_text(json.dumps(identity));(run/'answers').mkdir();(run/'routing').mkdir();(run/'grade-inputs').mkdir()
            records=[]
            for n,ident in enumerate(w.IDS,1):
                record={'id':ident,'lcb_problem':{'question_title':'q'}};records.append(record)
                folder=run/'jobs'/f'problem-{n:03d}';folder.mkdir(parents=True);(folder/'identity.json').write_text(json.dumps({'model':{'sha256':w.MODEL_SHA},'engine':{'sha256':'engine'}}))
                native={'type':'result','text':'ok','token_ids':[n]};raw=folder/'events.jsonl';raw.write_text(json.dumps(native)+'\n');compressed=run/'routing'/f'{n:03d}.jsonl.gz'
                with gzip.open(compressed,'wb') as f:f.write(raw.read_bytes())
                answer={'id':ident,'text':'ok','token_ids':[n],'native_result_sha256':canonical_hash(native),'routing_path':f'routing\\{n:03d}.jsonl.gz','raw_events_path':f'jobs\\problem-{n:03d}\\events.jsonl','routing_integrity':{'uncompressed_sha256':sha256(raw),'compressed_sha256':sha256(compressed)}}
                (run/'answers'/f'{n:03d}.json').write_text(json.dumps(answer));(run/'grade-inputs'/f'{n:03d}.json').write_text(json.dumps({'id':ident,'problem':record['lcb_problem'],'text':'ok'}))
            with patch.object(w,'MANIFEST',manifest),patch.object(w,'BUNDLE',bundle),patch.object(w,'selected_records',return_value=records):
                self.assertEqual(len(w.verified_grade_inputs(run)[1]),2)
                path=run/'grade-inputs/001.json';x=json.loads(path.read_text());x['text']='tampered';path.write_text(json.dumps(x))
                with self.assertRaises(ValueError):w.verified_grade_inputs(run)
                x['text']='ok';path.write_text(json.dumps(x));(run/'jobs/problem-001/events.jsonl').write_text('{}')
                with self.assertRaises(ValueError):w.verified_grade_inputs(run)
    def test_plot_has_80_columns_zero_black_highest_red_and_missing_gray(self):
        import tempfile,json
        from pathlib import Path
        import tools.manual80 as w
        import moe_prune.base_benchmark as benchmark
        from moe_prune.research import write_json
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);m=p/'manifest.json';m.write_text(json.dumps({'layers':[{'e80_to_original':list(range(80))} for _ in range(30)]}))
            c=[0]*80;c[1]=1;c[2]=2;phase={'0':{'tokens':10,'counts':c,'token_rate':[x/10 for x in c]}}
            profile={w.IDS[0]:{'prefill':phase,'decode':phase}};record={'id':w.IDS[0]}
            with patch.object(w,'MANIFEST',m),patch.object(w,'verified_grade_inputs',return_value=({'manifest_sha256':'x'},[(1,record,{},None)])),patch.object(benchmark,'aggregate_profiles',return_value=profile):out=w.plots(p)
            svg=(out/'001-overall.svg').read_text(encoding='utf-8');self.assertEqual(svg.count('<rect x='),30*80)
            for label in ('#000000','#ff0000','#808080','original ID 2','4/20','prefill + decode'):self.assertIn(label,svg)
    def test_grader_import_does_not_require_inference_packages(self):
        import subprocess,sys
        code='''import sys,importlib.abc
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in ('pynvml','numpy','torch'): raise ImportError('blocked '+fullname)
sys.meta_path.insert(0,Block());sys.path.insert(0,sys.argv[1]);import tools.manual80
'''
        from pathlib import Path
        r=subprocess.run([sys.executable,'-I','-B','-c',code,str(Path(__file__).resolve().parents[1])],capture_output=True,text=True)
        self.assertEqual(r.returncode,0,r.stderr)

