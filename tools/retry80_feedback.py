"""E80 첫 문제를 이전 코드와 일반적 오답 피드백만 제공해 재시도한다."""
import argparse,ast,gzip,hashlib,json,re,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from moe_prune.gguf import sha256
from moe_prune.research import canonical_hash,write_json,write_jsonl
from tools import manual80 as w
from tools.manual104 import config03

HERE=w.RUN/'retry-first1-feedback-001'
PRIOR=w.EXPORT_RUN/'generation-maxgpu-001'
EXPLICIT=PRIOR/'grading-explicit-code-first1-001'
FEEDBACK='Your previous answer was incorrect. Solve the problem again.'
ID='livecodebench:3674'

def feedback_prompt(record,code):
    if not isinstance(code,str) or not code:raise ValueError('previous code missing')
    sep='' if code.endswith('\n') else '\n'
    return record['prompt']+'\n\nPrevious answer:\n```python\n'+code+sep+'```\n\n'+FEEDBACK

def verify_previous_code(code,grade,text):
    if grade.get('id')!=ID or grade.get('passed') is not False or not isinstance(grade.get('code'),str):raise ValueError('이전 공식 채점 결과 불일치')
    if grade['code']!=code and grade['code']+'\n'!=code:raise ValueError('제출 코드와 공식 추출 코드 불일치')
    blocks=list(re.finditer(r'```python\s*\n(.*?)```',text,re.IGNORECASE|re.DOTALL))
    if not blocks or ast.dump(ast.parse(blocks[-1].group(1)))!=ast.dump(ast.parse(code)):raise ValueError('이전 native 최종 코드와 제출 코드 불일치')

# 2026-11-09: 이전 native 답안·공식 제출물 해시를 결속하고 오류 사유는 읽어 프롬프트에 전달하지 않는다.
def previous_source():
    identity,items=w.verified_grade_inputs(PRIOR)
    n,record,answer,payload=items[0]
    if n!=1 or record['id']!=ID:raise ValueError('이전 문제 ID 불일치')
    code_path=EXPLICIT/'submitted-code.py';code=code_path.read_text(encoding='utf-8')
    grade=json.loads((EXPLICIT/'001.json').read_text(encoding='utf-8'));submission=json.loads((EXPLICIT/'submission.json').read_text(encoding='utf-8'))
    if submission.get('submitted_code_sha256')!=sha256(code_path) or not submission.get('matches_last_model_code_block_ast'):raise ValueError('이전 제출 코드 해시 불일치')
    verify_previous_code(code,grade,answer['text'])
    provenance={'prior_run_identity_sha256':sha256(PRIOR/'identity.json'),'prior_answer_sha256':sha256(PRIOR/'answers/001.json'),'previous_code_sha256':sha256(code_path),'official_grade_sha256':sha256(EXPLICIT/'001.json'),'submission_sha256':sha256(EXPLICIT/'submission.json'),'feedback':FEEDBACK}
    return record,code,provenance

def generate(directory=HERE):
    from moe_prune.base_benchmark import build_request,validate_routing,verify_runtime
    from moe_prune.resident import ResidentExperiment
    directory=Path(directory)
    if directory.exists():raise FileExistsError('기존 재시도 결과 보존')
    records,identity=w.source_identity();record,code,provenance=previous_source()
    if records[0]['id']!=ID or records[0]['prompt']!=record['prompt'] or records[0].get('system_prompt')!=record.get('system_prompt'):raise ValueError('원본 문제 불일치')
    directory.mkdir(parents=True,exist_ok=False)
    try: chosen,rows=w.choose_placement(w.MODEL)
    except Exception as error:
        write_json(directory/'placement-preflight.json',{'complete':False,'candidates':getattr(error,'candidate_observations',[]),'error_type':type(error).__name__});raise
    write_json(directory/'placement-preflight.json',{'complete':True,'candidates':rows,'chosen':chosen})
    config={**config03(),'n_gpu_layers':chosen,'engine':str(w.ENGINE)}
    identity.update({'ids':[ID],'config':config,'placement':[chosen,True],'engine_batch':64,'engine_ubatch':64,'retry_script_sha256':sha256(Path(__file__)),'prior':provenance})
    write_json(directory/'identity.json',identity)
    request=build_request({**record,'prompt':feedback_prompt(record,code)},w.MODEL,(chosen,True),config)
    try:
        with ResidentExperiment(directory,config) as exp:
            name='problem-001';folder=directory/'jobs'/name
            print(f'재시도 {ID} 시작, GPU layers={chosen}',flush=True);result=exp.run(name,request)
            if result.get('stop_reason') not in ('eog','length') or (result['stop_reason']=='length' and len(result['token_ids'])!=32768):raise ValueError('생성 종료 사유 오류')
            if not result.get('gpu_fusion_disabled') or result.get('profiled_layers')!=list(range(30)) or result.get('masked_layers') or result.get('enable_thinking') is not True or result.get('requested_n_gpu_layers')!=chosen or result.get('actual_context_size')!=65536 or (result.get('cache_type_k'),result.get('cache_type_v'))!=('q8_0','q8_0'):raise ValueError('실제 runtime 설정 불일치')
            observed=verify_runtime(result,(folder/'stderr.log').read_text(encoding='utf-8',errors='replace'),request)
            events=folder/'events.jsonl';count=validate_routing(w.read_jsonl(events),result,experts=80)
            if count!=result.get('routing_records'):raise ValueError('라우팅 수 불일치')
            native=w.native_result(events)
            if native!={k:v for k,v in result.items() if k!='measurement'}:raise ValueError('native 최종 결과 불일치')
            target=directory/'routing/001.jsonl.gz';integrity=w.compress_events(events,target);exp.check()
            answer={'id':ID,'number':1,**result,'native_result_sha256':canonical_hash(native),'observed_runtime':observed,'routing_path':str(target.relative_to(directory)),'routing_integrity':integrity,'raw_events_path':str(events.relative_to(directory))}
            write_json(directory/'answers/001.json',answer);write_json(directory/'grade-inputs/001.json',{'id':ID,'problem':record['lcb_problem'],'text':native['text']});write_jsonl(directory/'generations.jsonl',[answer])
        write_json(directory/'status.json',{'complete':True,'completed':1,'total':1,'grading':'pending'})
    except BaseException:
        write_json(directory/'status.json',{'complete':False,'completed':0,'total':1});raise
    return directory

def verified_retry(directory=HERE):
    directory=Path(directory).resolve();status=json.loads((directory/'status.json').read_text(encoding='utf-8'))
    if status.get('complete') is not True or status.get('completed')!=1:raise ValueError('재시도 미완료')
    identity=json.loads((directory/'identity.json').read_text(encoding='utf-8'));record,code,prior=previous_source()
    if identity.get('model_sha256')!=w.MODEL_SHA or identity.get('ids')!=[ID] or identity.get('prior')!=prior or identity.get('manifest_sha256')!=sha256(w.MANIFEST):raise ValueError('재시도 identity 불일치')
    folder=directory/'jobs/problem-001';job=json.loads((folder/'identity.json').read_text(encoding='utf-8'));request=json.loads((folder/'request.json').read_text(encoding='utf-8'))
    if job['model']['sha256']!=w.MODEL_SHA or job['engine']['sha256']!=identity['engine_sha256'] or request['prompt']!=feedback_prompt(record,code) or request.get('system_prompt')!=record.get('system_prompt'):raise ValueError('실행 모델 또는 프롬프트 불일치')
    answer=json.loads((directory/'answers/001.json').read_text(encoding='utf-8'));payload_path=directory/'grade-inputs/001.json';payload=json.loads(payload_path.read_text(encoding='utf-8'))
    if answer['routing_path'].replace('\\','/')!='routing/001.jsonl.gz' or answer['raw_events_path'].replace('\\','/')!='jobs/problem-001/events.jsonl':raise ValueError('라우팅 경로 오류')
    raw=(folder/'events.jsonl').resolve();compressed=(directory/'routing/001.jsonl.gz').resolve()
    if not raw.is_relative_to(directory) or not compressed.is_relative_to(directory):raise ValueError('경로 범위 오류')
    if sha256(raw)!=answer['routing_integrity']['uncompressed_sha256'] or sha256(compressed)!=answer['routing_integrity']['compressed_sha256']:raise ValueError('라우팅 해시 불일치')
    h=hashlib.sha256()
    with gzip.open(compressed,'rb') as f:
        while chunk:=f.read(1024*1024):h.update(chunk)
    if h.hexdigest()!=answer['routing_integrity']['uncompressed_sha256']:raise ValueError('압축 내용 불일치')
    native=w.native_result(raw);w.verify_payload(record,answer,payload,native)
    return identity,record,answer,payload_path

# 2026-11-09: 공통 실행기를 사용하여 재시도도 같은 비격리 채점 출처를 기록한다.
def grade(directory=HERE,venv=ROOT/'.cache/putare-lcb/venv'):
    from moe_prune.execution_safety import execution_guard
    from moe_prune.grading import GradeDeadline
    directory=Path(directory).resolve()
    status=json.loads((directory/'status.json').read_text(encoding='utf-8'))
    if status.get('complete') is not True or status.get('completed')!=1:raise ValueError('생성 미완료')
    with execution_guard('grading',memory_config=config03()) as guard:
        return _grade(directory,venv,guard,GradeDeadline())


def _grade(directory,venv,guard,deadline):
    import importlib.util,subprocess
    from moe_prune.grading import run as run_grade, identity as grader_identity, worker_environment, GradingCleanupError
    if sys.platform!='linux':raise RuntimeError('Colab Linux 환경 필요')
    directory=Path(directory).resolve();venv=Path(venv)
    identity,record,answer,payload=verified_retry(directory)
    def load(name,path):
        spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
    portable=load('_retry_portable',w.BUNDLE/'lcb_portable.py');portable.verify_inventory(w.BUNDLE)
    env=worker_environment()
    probe=subprocess.run([str(venv/'bin/python'),'-I','-B',str(w.BUNDLE/'lcb_grade.py'),'--verify','--official-source',str(w.BUNDLE/'official-source')],env=env,check=True,capture_output=True,text=True,timeout=deadline.remaining())
    official=json.loads(probe.stdout);original=json.loads((w.BUNDLE/'original-manifest.json').read_text(encoding='utf-8'))
    if official.get('source_digest')!=original['official_identity'].get('source_digest'):raise ValueError('공식 채점기 source 불일치')
    out=directory/'grading';out.mkdir(exist_ok=False)
    write_json(out/'identity.json',{'official':official,'inventory_sha256':sha256(w.BUNDLE/'SHA256SUMS.json'),'run_identity_sha256':sha256(directory/'identity.json'),'orchestrator_sha256':sha256(Path(__file__)),**grader_identity()})
    try:
        target=out/'001.json';run_grade(w.BUNDLE/'official-source',venv,payload,target,adapter=w.BUNDLE/'lcb_grade.py',memory_config=config03(),guard=guard,timeout_s=deadline.remaining());result=json.loads(target.read_text(encoding='utf-8'))
        if result.get('id')!=ID or type(result.get('passed')) is not bool or result.get('pass_at_1')!=float(result['passed']):raise ValueError('공식 결과 오류')
        row={'status':'graded','passed':result['passed'],'planned_test_count':result.get('test_count'),'executed_test_count':None}
        if isinstance(result.get('code'),str) and result['code'].strip():
            solutions=directory/'solutions';solutions.mkdir(exist_ok=True)
            with (solutions/'001.py').open('x',encoding='utf-8') as f:f.write(result['code'])
    except GradingCleanupError:
        raise
    except Exception as error:
        row={'status':'infrastructure_error','passed':None,'error_type':type(error).__name__}
    row.update({'id':ID,'title':record['lcb_problem']['question_title'],'stop_reason':answer['stop_reason'],'generated_tokens':len(answer['token_ids']),'timings':answer['timings'],'measurement':answer['measurement']})
    summary={'complete':row['status']=='graded','scope':'E80 first problem retry after generic incorrect feedback','model_sha256':w.MODEL_SHA,'feedback':FEEDBACK,'case':row,'limitations':['Test count is planned, executed count is not established.','Instrumented throughput includes monitoring overhead.']}
    write_json(out/'summary.json',summary)
    verdict=('통과' if row['passed'] else '실패') if row['status']=='graded' else '환경 오류(판정 없음)'
    (out/'RESULTS-KO.txt').write_text(f"E80 첫 문제 재시도: {verdict}\n토큰 {row['generated_tokens']}, 종료 {row['stop_reason']}\n",encoding='utf-8')
    print(f'공식 채점: {verdict}',flush=True)

def plots(directory=HERE):
    import html
    from moe_prune.base_benchmark import aggregate_profiles
    directory=Path(directory).resolve();identity,record,answer,payload=verified_retry(directory)
    def events(path):
        with gzip.open(path,'rt',encoding='utf-8') as f:
            for line in f:
                if line.strip():yield json.loads(line)
    profile=aggregate_profiles([(ID,events(directory/'routing/001.jsonl.gz'))],experts=80)[ID]
    mapping=json.loads(w.MANIFEST.read_text(encoding='utf-8'))['layers'];rows=w.combined_80(profile)
    vmax=max((v for row in rows.values() for v in row['token_rate']),default=0.0)
    out=directory/'visualizations';out.mkdir(exist_ok=False)
    write_json(out/'profiles.json',{'profile':profile,'model_sha256':w.MODEL_SHA,'manifest_sha256':identity['manifest_sha256'],'interpretation':'selection frequency, not contribution or expertise'})
    parts=['<svg xmlns="http://www.w3.org/2000/svg" width="950" height="650">','<rect width="950" height="650" fill="white"/>',f'<text x="70" y="24">{html.escape(ID)} retry: prefill + decode selection frequency</text>',f'<text x="70" y="45">black=zero; white-red=positive up to observed max {100*vmax:.3f}%; gray=no data</text>']
    for layer in range(30):
        row=rows.get(str(layer));y=65+layer*17;parts.append(f'<text x="30" y="{y+12}">{layer}</text>')
        for expert in range(80):
            count=row['counts'][expert] if row else None;den=row['tokens'] if row else 0;shade=round(255*(1-(count/den)/vmax)) if den and vmax else 0
            color='#808080' if den==0 else '#000000' if count==0 else f'#ff{shade:02x}{shade:02x}'
            parts.append(f'<rect x="{70+expert*10}" y="{y}" width="10" height="16" fill="{color}"><title>layer {layer}, e80 ID {expert}, original ID {mapping[layer]["e80_to_original"][expert]}: {count}/{den}</title></rect>')
    for expert in list(range(0,80,8))+[79]:parts.append(f'<text x="{70+expert*10}" y="590" font-size="10">{expert}</text>')
    parts.extend(['<text x="70" y="610">Selection frequency is not contribution or semantic expertise.</text>','</svg>'])
    (out/'001-overall.svg').write_text(''.join(parts),encoding='utf-8');return out

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('generate','grade','plots'));args=parser.parse_args()
    if args.action=='generate':generate()
    elif args.action=='grade':grade()
    else:print(plots())
