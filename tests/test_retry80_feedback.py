import unittest
class FeedbackRetryTests(unittest.TestCase):
    def test_prompt_has_only_original_code_and_generic_feedback(self):
        from tools.retry80_feedback import feedback_prompt, FEEDBACK
        record={'prompt':'Original problem','system_prompt':'System'};code='print(1)\n'
        p=feedback_prompt(record,code)
        self.assertEqual(p,'Original problem\n\nPrevious answer:\n```python\nprint(1)\n```\n\n'+FEEDBACK)
        self.assertEqual(FEEDBACK,'Your previous answer was incorrect. Solve the problem again.')
        for bad in ('counterexample','expected','segment tree','reset cost','expert'):self.assertNotIn(bad,p.lower())
    def test_rejects_official_code_mismatch(self):
        from tools.retry80_feedback import verify_previous_code
        import unittest.mock as mock
        with self.assertRaises(ValueError):verify_previous_code('print(1)\n',{'code':'print(2)','id':'livecodebench:3674','passed':False},'```python\nprint(1)\n```')
    def test_previous_code_ast_and_official_newline_match(self):
        from tools.retry80_feedback import verify_previous_code
        verify_previous_code('print(1)\n',{'code':'print(1)','id':'livecodebench:3674','passed':False},'```python\nprint(1)\n```')
    def test_fake_generation_one_problem_and_close_before_complete(self):
        import tempfile,json
        from pathlib import Path
        from unittest.mock import patch
        import tools.retry80_feedback as t
        import moe_prune.base_benchmark as benchmark,moe_prune.resident as resident
        closed=[];requests=[]
        record={'id':t.ID,'prompt':'Original problem','system_prompt':'System','lcb_problem':{'question_title':'x'}}
        code='print(1)\n';prior={'previous_code_sha256':'x'}
        class Fake:
            def __init__(self,root,config):self.root=Path(root)
            def __enter__(self):return self
            def __exit__(self,*args):closed.append(True)
            def check(self):pass
            def run(self,name,request):
                requests.append(request);folder=self.root/'jobs'/name;folder.mkdir(parents=True);rows=[]
                for phase,token in [('prefill',0),('decode',1)]:
                    for layer in range(30):rows.append({'type':'routing','phase':phase,'layer':layer,'token_index':token,'expert_ids':list(range(8)),'weights':[.125]*8,'weighted_output_norms':[1.]*8})
                native={'type':'result','text':'ok','token_ids':[1],'stop_reason':'eog','prompt_tokens':1,'processed_decode_tokens':1,'routing_records':60,'profiled_layers':list(range(30)),'gpu_fusion_disabled':True,'masked_layers':[],'enable_thinking':True,'requested_n_gpu_layers':31,'actual_context_size':65536,'cache_type_k':'q8_0','cache_type_v':'q8_0'}
                with (folder/'events.jsonl').open('w') as f:
                    for row in [*rows,native]:f.write(json.dumps(row)+'\n')
                (folder/'stderr.log').write_text('fixture');return {**native,'measurement':{}}
        real=t.write_json
        def write(path,value):
            if Path(path).name=='status.json' and value.get('complete'):self.assertTrue(closed)
            real(path,value)
        with tempfile.TemporaryDirectory() as d,patch.object(t.w,'source_identity',return_value=([record],{})),patch.object(t,'previous_source',return_value=(record,code,prior)),patch.object(t.w,'choose_placement',return_value=(31,[])),patch.object(resident,'ResidentExperiment',Fake),patch.object(benchmark,'verify_runtime',return_value={}),patch.object(t,'write_json',side_effect=write):
            out=Path(d)/'run';t.generate(out);self.assertTrue(json.loads((out/'status.json').read_text())['complete']);self.assertEqual(len(requests),1)
            self.assertEqual(requests[0]['prompt'],t.feedback_prompt(record,code));self.assertEqual(requests[0]['system_prompt'],'System');self.assertTrue(requests[0]['instrumentation'] and requests[0]['all_token_logits']);self.assertEqual(requests[0]['seed'],42)
    def test_rejects_80_out_of_range_route(self):
        from moe_prune.base_benchmark import validate_routing
        row={'type':'routing','phase':'prefill','layer':0,'token_index':0,'expert_ids':[80,1,2,3,4,5,6,7],'weights':[.125]*8,'weighted_output_norms':[1.]*8}
        with self.assertRaises(ValueError):validate_routing([row],{'prompt_tokens':1,'processed_decode_tokens':0},layers=1,experts=80)
    def test_grade_does_not_launch_worker_for_incomplete_generation(self):
        import json,tempfile
        from pathlib import Path
        from unittest.mock import patch
        import tools.retry80_feedback as t
        with tempfile.TemporaryDirectory() as d, patch('subprocess.run',side_effect=AssertionError('launched')):
            path=Path(d);(path/'status.json').write_text(json.dumps({'complete':False}))
            with patch.object(t.sys,'platform','linux'),self.assertRaises(ValueError):t.grade(path)
    def test_grading_binding_rejects_prompt_and_payload_tampering(self):
        import tempfile,json,gzip
        from pathlib import Path
        from unittest.mock import patch
        from moe_prune.gguf import sha256
        from moe_prune.research import canonical_hash
        import tools.retry80_feedback as t
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);manifest=p/'manifest.json';manifest.write_text('{}');run=p/'run';run.mkdir()
            record={'id':t.ID,'prompt':'Original problem','system_prompt':'System','lcb_problem':{'question_title':'q'}};code='print(1)\n';prior={'previous_code_sha256':'x'}
            identity={'model_sha256':t.w.MODEL_SHA,'ids':[t.ID],'prior':prior,'manifest_sha256':sha256(manifest),'engine_sha256':'engine'}
            (run/'identity.json').write_text(json.dumps(identity));(run/'status.json').write_text(json.dumps({'complete':True,'completed':1}))
            folder=run/'jobs/problem-001';folder.mkdir(parents=True);(run/'answers').mkdir();(run/'grade-inputs').mkdir();(run/'routing').mkdir()
            (folder/'identity.json').write_text(json.dumps({'model':{'sha256':t.w.MODEL_SHA},'engine':{'sha256':'engine'}}))
            req={'prompt':t.feedback_prompt(record,code),'system_prompt':'System'};(folder/'request.json').write_text(json.dumps(req))
            native={'type':'result','text':'new','token_ids':[1]};raw=folder/'events.jsonl';raw.write_text(json.dumps(native)+'\n');compressed=run/'routing/001.jsonl.gz'
            with gzip.open(compressed,'wb') as f:f.write(raw.read_bytes())
            answer={'id':t.ID,'text':'new','token_ids':[1],'native_result_sha256':canonical_hash(native),'routing_path':'routing\\001.jsonl.gz','raw_events_path':'jobs\\problem-001\\events.jsonl','routing_integrity':{'uncompressed_sha256':sha256(raw),'compressed_sha256':sha256(compressed)}}
            (run/'answers/001.json').write_text(json.dumps(answer));payload={'id':t.ID,'problem':record['lcb_problem'],'text':'new'};(run/'grade-inputs/001.json').write_text(json.dumps(payload))
            with patch.object(t.w,'MANIFEST',manifest),patch.object(t,'previous_source',return_value=(record,code,prior)):
                self.assertEqual(t.verified_retry(run)[2]['text'],'new')
                payload['text']='tampered';(run/'grade-inputs/001.json').write_text(json.dumps(payload))
                with self.assertRaises(ValueError):t.verified_retry(run)
                payload['text']='new';(run/'grade-inputs/001.json').write_text(json.dumps(payload));req['prompt']+=' bad reason';(folder/'request.json').write_text(json.dumps(req))
                with self.assertRaises(ValueError):t.verified_retry(run)
    def test_linux_grader_import_needs_no_inference_packages(self):
        import subprocess,sys
        from pathlib import Path
        code='''import sys,importlib.abc
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in ('pynvml','numpy','torch'): raise ImportError('blocked '+fullname)
sys.meta_path.insert(0,Block());sys.path.insert(0,sys.argv[1]);import tools.retry80_feedback
'''
        r=subprocess.run([sys.executable,'-I','-B','-c',code,str(Path(__file__).resolve().parents[1])],capture_output=True,text=True)
        self.assertEqual(r.returncode,0,r.stderr)

