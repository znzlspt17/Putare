import ast
from pathlib import Path
import unittest
import json
import os
import sys
import tempfile
from contextlib import nullcontext
from unittest.mock import patch

source = Path(__file__).resolve().parents[1] / 'moe_prune/base_benchmark.py'
tree = ast.parse(source.read_text(encoding='utf-8'))
selected = [n for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in ('DEFAULT_CONFIG','PLACEMENTS') for t in n.targets) or isinstance(n, ast.FunctionDef) and n.name == 'run_with_offload']
namespace = {}
exec(compile(ast.Module(body=selected, type_ignores=[]), str(source), 'exec'), namespace)

class ColabDefaults(unittest.TestCase):
    def test_warning_policy_uses_percentages_without_byte_reservations(self):
        config=namespace['DEFAULT_CONFIG']
        self.assertEqual(config.get('ram_usage_warn_fraction'), .80)
        self.assertEqual(config.get('vram_usage_warn_fraction'), .97)
        self.assertEqual(config['n_gpu_layers'], 31)
        for key in ('min_available_ram_bytes','min_available_commit_bytes','min_free_vram_bytes','vram_workspace_bytes','ram_budget_bytes'):
            self.assertNotIn(key, config)
    def test_allocation_failure_does_not_retry_cpu(self):
        calls=[]
        def fail(placement):
            calls.append(placement)
            raise MemoryError('allocation failed')
        with self.assertRaises(MemoryError): namespace['run_with_offload'](fail)
        self.assertEqual(calls, [(31, True)])


@unittest.skipUnless(os.name == 'posix', '실제 프로세스 테스트는 Linux에서 실행')
class GradeWorkerTests(unittest.TestCase):
    class Guard:
        def __init__(self,*args): pass
        def preflight(self): return {}
        def check(self): return {}
        def close(self): pass

    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.source_dir=self.root/'source'; self.source_dir.mkdir()
        self.input=self.root/'input.json'; self.input.write_text(json.dumps({'id':'case'}),encoding='utf-8')
        self.output=self.root/'result.json'; self.adapter=self.root/'worker.py'
        self.venv=Path(sys.executable).parent.parent

    def invoke(self,script,timeout=5):
        from moe_prune.grading import run
        self.adapter.write_text(script,encoding='utf-8')
        with patch('moe_prune.execution_safety.execution_guard',return_value=nullcontext(self.Guard())), patch('moe_prune.memory_safety.MemoryGuard',self.Guard):
            return run(self.source_dir,self.venv,self.input,self.output,adapter=self.adapter,timeout_s=timeout)

    def test_worker_success_and_wrong_answer_are_valid_results(self):
        for passed in (True,False):
            with self.subTest(passed=passed):
                script="import json,sys\np=sys.argv[sys.argv.index('--output')+1]\nopen(p,'w').write(json.dumps({'id':'case','passed':%r,'pass_at_1':%r}))\n" % (passed,float(passed))
                self.assertEqual(self.invoke(script)['passed'],passed)
                self.assertEqual(json.loads(self.output.read_text())['passed'],passed)
                self.output.unlink()

    def test_existing_grading_lease_is_reused_for_each_worker(self):
        from moe_prune.grading import run
        script="import json,sys\np=sys.argv[sys.argv.index('--output')+1]\nopen(p,'w').write(json.dumps({'id':'case','passed':True,'pass_at_1':1.0}))\n"
        self.adapter.write_text(script,encoding='utf-8')
        guard=self.Guard()
        with patch('moe_prune.execution_safety.execution_guard') as acquire:
            for i in range(2):
                run(self.source_dir,self.venv,self.input,self.root/f'result-{i}.json',adapter=self.adapter,guard=guard)
            acquire.assert_not_called()

    def test_manual_grader_holds_one_lease_for_whole_run(self):
        import tools.manual80 as manual
        from contextlib import contextmanager
        events=[];guard=self.Guard()
        @contextmanager
        def lease(kind,*,memory_config=None):
            events.append(('enter',kind));yield guard;events.append(('exit',kind))
        (self.root/'status.json').write_text(json.dumps({'complete':True,'completed':2,'total':2}))
        def work(directory,venv,active_guard,deadline):
            self.assertIs(active_guard,guard)
            self.assertGreater(deadline.remaining(),0)
            self.assertEqual(events,[('enter','grading')])
            return 'done'
        with patch('moe_prune.execution_safety.execution_guard',lease),patch.object(manual,'_grade',side_effect=work):
            self.assertEqual(manual.grade(self.root),'done')
        self.assertEqual(events,[('enter','grading'),('exit','grading')])

    def test_whole_run_deadline_stops_before_next_worker(self):
        from moe_prune.grading import GradeDeadline
        ticks=iter((0,10,7200))
        deadline=GradeDeadline(clock=lambda:next(ticks))
        launched=[]
        launched.append(deadline.remaining())
        with self.assertRaises(TimeoutError):
            launched.append(deadline.remaining())
        self.assertEqual(launched,[7190])

    def test_manual80_stops_before_second_worker_when_deadline_expires(self):
        from types import SimpleNamespace
        import tools.manual80 as manual
        import moe_prune.grading as grading
        bundle=self.root/'bundle';bundle.mkdir()
        (bundle/'lcb_portable.py').write_text('def verify_inventory(root): pass\n')
        (bundle/'original-manifest.json').write_text(json.dumps({'official_identity':{'source_digest':'trusted'}}))
        (bundle/'SHA256SUMS.json').write_text('{}')
        (self.root/'identity.json').write_text('{}')
        items=[]
        for i in (1,2):
            record={'id':f'case{i}','lcb_problem':{'question_title':'q'}}
            answer={'stop_reason':'eog','token_ids':[1],'timings':{},'measurement':{}}
            items.append((i,record,answer,self.root/f'payload{i}.json'))
        class Deadline:
            calls=0
            def remaining(self):
                self.calls+=1
                if self.calls==3:raise TimeoutError('deadline')
                return 10 if self.calls==1 else 4
        deadline=Deadline(); launched=[]
        def fake_run(source,venv,payload,target,**kwargs):
            launched.append(kwargs['timeout_s'])
            Path(target).write_text(json.dumps({'id':'case1','passed':True,'pass_at_1':1.0,'test_count':1}))
        probe=SimpleNamespace(stdout=json.dumps({'source_digest':'trusted'}))
        with patch.object(manual,'BUNDLE',bundle),patch.object(manual,'verified_grade_inputs',return_value=({},items)), \
             patch('subprocess.run',return_value=probe),patch.object(grading,'run',side_effect=fake_run):
            summary=manual._grade(self.root,self.venv,self.Guard(),deadline)
        self.assertEqual(launched,[4]);self.assertEqual(deadline.calls,3)
        self.assertFalse(summary['complete'])
        self.assertEqual([x['status'] for x in summary['cases']],['graded','infrastructure_error'])

    def test_symlink_result_is_rejected_without_copying_it(self):
        victim=self.root/'victim.json'; victim.write_text(json.dumps({'id':'case','passed':True,'pass_at_1':1.0}))
        script="import os,sys\np=sys.argv[sys.argv.index('--output')+1]\nos.symlink(%r,p)\n" % str(victim)
        with self.assertRaises((ValueError,OSError)): self.invoke(script)
        self.assertFalse(self.output.exists())

    def test_timeout_leaves_no_result(self):
        with self.assertRaises(TimeoutError): self.invoke('import time\ntime.sleep(10)\n',.2)
        self.assertFalse(self.output.exists())

    def test_worker_error_or_invalid_id_is_not_scored(self):
        scripts=['raise SystemExit(3)',
                 "import json,sys\np=sys.argv[sys.argv.index('--output')+1]\nopen(p,'w').write(json.dumps({'id':'other','passed':False,'pass_at_1':0.0}))"]
        for script in scripts:
            with self.subTest(script=script), self.assertRaises((RuntimeError,ValueError)):
                self.invoke(script)
            self.assertFalse(self.output.exists())

    def test_cleanup_failure_poison_marks_execution_and_never_publishes_result(self):
        from moe_prune.grading import run
        self.adapter.write_text("import json,sys\np=sys.argv[sys.argv.index('--output')+1]\nopen(p,'w').write(json.dumps({'id':'case','passed':True,'pass_at_1':1.0}))",encoding='utf-8')
        with patch('moe_prune.execution_safety.execution_guard',return_value=nullcontext(self.Guard())), \
             patch('moe_prune.execution_safety.poison_execution') as poison, \
             patch('moe_prune.memory_safety.MemoryGuard',self.Guard), \
             patch('moe_prune.grading._cleanup',side_effect=RuntimeError('cleanup incomplete')):
            with self.assertRaises(RuntimeError):
                run(self.source_dir,self.venv,self.input,self.output,adapter=self.adapter,timeout_s=5)
            poison.assert_called_once()
        self.assertFalse(self.output.exists())

    def test_timeout_terminates_launched_child(self):
        import psutil
        marker=self.root/'child.pid'
        script=("import subprocess,sys,time\np=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)'])\n"
                "open(%r,'w').write(str(p.pid))\ntime.sleep(30)\n") % str(marker)
        with self.assertRaises(TimeoutError): self.invoke(script,.6)
        pid=int(marker.read_text())
        if psutil.pid_exists(pid):
            self.assertEqual(psutil.Process(pid).status(),psutil.STATUS_ZOMBIE)

    def test_worker_exit_does_not_leave_signal_resistant_child(self):
        import psutil
        import signal
        marker=self.root/'orphan.pid'
        child_code='import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); print("ready",flush=True); time.sleep(30)'
        script=("import json,subprocess,sys\np=subprocess.Popen([sys.executable,'-c',%r],stdout=subprocess.PIPE,text=True)\n"
                "p.stdout.readline()\nopen(%r,'w').write(str(p.pid))\n"
                "out=sys.argv[sys.argv.index('--output')+1]\nopen(out,'w').write(json.dumps({'id':'case','passed':True,'pass_at_1':1.0}))\n") % (child_code,str(marker))
        pid=None
        try:
            self.invoke(script)
            pid=int(marker.read_text())
            if psutil.pid_exists(pid):
                self.assertEqual(psutil.Process(pid).status(),psutil.STATUS_ZOMBIE)
        finally:
            if pid is None and marker.exists(): pid=int(marker.read_text())
            if pid and psutil.pid_exists(pid):
                try: os.kill(pid,signal.SIGKILL)
                except ProcessLookupError: pass

    def test_cancellation_terminates_launched_child(self):
        import psutil
        from moe_prune.grading import run
        marker=self.root/'cancel-child.pid'
        self.adapter.write_text(("import subprocess,sys,time\np=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)'])\n"
                "open(%r,'w').write(str(p.pid))\ntime.sleep(30)\n") % str(marker),encoding='utf-8')
        class Interrupting(self.Guard):
            def check(self):
                if marker.exists(): raise KeyboardInterrupt
        with patch('moe_prune.execution_safety.execution_guard',return_value=nullcontext(Interrupting())), patch('moe_prune.memory_safety.MemoryGuard',Interrupting):
            with self.assertRaises(KeyboardInterrupt):
                run(self.source_dir,self.venv,self.input,self.output,adapter=self.adapter,timeout_s=5)
        pid=int(marker.read_text())
        if psutil.pid_exists(pid):
            self.assertEqual(psutil.Process(pid).status(),psutil.STATUS_ZOMBIE)
        self.assertFalse(self.output.exists())

    @unittest.skipUnless(os.environ.get('RUN_COLAB_LCB_SYNTHETIC') == '1', '공식 채점기 synthetic 검사는 명시 실행')
    def test_official_lcb_correct_wrong_and_timeout(self):
        from moe_prune.grading import run
        root=Path(__file__).resolve().parents[1]
        bundle=root/'download/gemma4-lcb-first100-local-grading'
        source=bundle/'official-source';adapter=bundle/'lcb_grade.py'
        tests=json.dumps([{'input':'3\n','output':'6\n','testtype':'stdin'}])
        problem={'public_test_cases':tests,'private_test_cases':tests,'metadata':'{}'}
        for i,(code,expected) in enumerate((('print(int(input())*2)',True),('print(9)',False),('while True: pass',False))):
            payload=self.root/f'official-{i}.json';output=self.root/f'official-out-{i}.json'
            payload.write_text(json.dumps({'id':str(i),'problem':problem,'text':f'```python\n{code}\n```'}),encoding='utf-8')
            result=run(source,self.venv,payload,output,adapter=adapter,guard=self.Guard(),timeout_s=45)
            self.assertEqual(result['passed'],expected)
            self.assertEqual(result['pass_at_1'],float(expected))
            if i==2:self.assertEqual(result['details']['error_code'],-3)

