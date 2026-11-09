"""80 전문가 모델의 두 문제 계측 생성과 비격리 채점 실행 도구."""
import argparse, gzip, hashlib, json, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from moe_prune.gguf import GGUF,sha256
from moe_prune.memory_safety import MemoryGuard
from moe_prune.research import read_jsonl,write_json,write_jsonl,canonical_hash
from tools.manual104 import config03,compress_events,selected_records

MODEL=ROOT/'models/gemma-4-26B-A4B-it-UD-IQ4_NL-lcb100-activity-ordered-e80.gguf'
MODEL_SHA='760a302c4f67cb99dc28b6d195ae0ec6a623d8c37614fedb20df555b92180321'
EXPORT_RUN=ROOT/'runs/gemma4-ordered-e80-001'
RUN=ROOT/'runs/gemma4-ordered-e80-colab-a100-001'
MANIFEST=EXPORT_RUN/'export-manifest.json'
DIRECTORY=RUN/'generation-maxgpu-001'
ENGINE=ROOT/'.cache/llama-build/bin/llama-expert-research'
BUNDLE=ROOT/'download/gemma4-lcb-first100-local-grading'
IDS=['livecodebench:3674','livecodebench:3677']

# 2026-11-09: 모든 실험을 전체 GPU 배치로 고정하고 초과는 guard의 경고로 남긴다.
def choose_placement(model,guard_cls=MemoryGuard):
    from moe_prune.base_benchmark import build_request
    n=31
    config={**config03(),'n_gpu_layers':n}
    request=build_request({'prompt':'preflight'},model,(n,True),config)
    guard=guard_cls(config,request)
    try:
        sample=guard.preflight()
        return n,[{'n_gpu_layers':n,'offload_kqv':True,'observation':sample}]
    finally:
        guard.close()

def verify_payload(record,answer,payload,native):
    if answer.get('id')!=record['id'] or payload!={'id':record['id'],'problem':record['lcb_problem'],'text':native.get('text')}:
        raise ValueError('채점 입력이 native 결과와 다릅니다')
    if answer.get('text')!=native.get('text') or answer.get('token_ids')!=native.get('token_ids') or answer.get('native_result_sha256')!=canonical_hash(native):
        raise ValueError('답안과 native 결과 해시가 다릅니다')

def native_result(path):
    result=None;seen=False
    for row in read_jsonl(path):
        if seen: raise ValueError('native result 뒤에 추가 기록이 있습니다')
        if row.get('type')=='result':result=row;seen=True
        elif row.get('type')!='routing':raise ValueError('알 수 없는 native 이벤트')
    if result is None:raise ValueError('native result 누락')
    return result

def source_identity(*, model=None, engine=None):
    import importlib.util
    # 2026-11-09: 모델·엔진 위치를 바꿔도 동일한 출처 검증을 적용한다.
    model=Path(MODEL if model is None else model);engine=Path(ENGINE if engine is None else engine)
    spec=importlib.util.spec_from_file_location('_e80_bundle',BUNDLE/'lcb_portable.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);m.verify_inventory(BUNDLE)
    manifest=json.loads(MANIFEST.read_text(encoding='utf-8'))
    if manifest.get('output_sha256')!=MODEL_SHA or manifest.get('file_validation_status')!='passed' or sha256(model)!=MODEL_SHA or GGUF(model).architecture()!=(list(range(30)),80,8):raise ValueError('E80 모델 해시 또는 구조 불일치')
    records=selected_records((1,2))
    if [r['id'] for r in records]!=IDS:raise ValueError('첫 두 문제 ID 불일치')
    reorder=ROOT/'runs/gemma4-lcb100-reordered-001/permutation-manifest.json'
    if sha256(reorder)!=manifest['reorder_manifest_sha256'] or json.loads(reorder.read_text(encoding='utf-8'))['profile']['records_sha256']!=sha256(BUNDLE/'records.jsonl'):raise ValueError('문제 또는 매핑 출처 불일치')
    files=[engine]+sorted(p for p in engine.parent.glob('*.so*') if p.name.lower().startswith(('libllama','libggml')))
    identity={'model_sha256':MODEL_SHA,'engine_sha256':sha256(engine),'engine_files':{p.name:sha256(p) for p in files},'records_sha256':sha256(BUNDLE/'records.jsonl'),'bundle_inventory_sha256':sha256(BUNDLE/'SHA256SUMS.json'),'manifest_sha256':sha256(MANIFEST),'script_sha256':sha256(Path(__file__)),'source_hashes':{str(p.relative_to(ROOT)):sha256(p) for p in [ROOT/'moe_prune/base_benchmark.py',ROOT/'moe_prune/gguf.py',ROOT/'tools/manual104.py',ROOT/'tools/manual80_grade.py',ROOT/'moe_prune/memory_safety.py',ROOT/'moe_prune/resident.py',ROOT/'moe_prune/execution_safety.py',ROOT/'tools/native/expert_research.cpp']},'ids':IDS,'limitations':['Instrumentation affects throughput.','GPU memory is sampled and session-to-date, not physical peak.','Hardware and CPU/GPU placement differ from the original Vast run.']}
    return records,identity

def generate(directory=DIRECTORY):
    from moe_prune.base_benchmark import build_request,validate_routing,verify_runtime
    from moe_prune.resident import ResidentExperiment
    directory=Path(directory)
    if directory.exists():raise FileExistsError('기존 실행 폴더 보존')
    records,identity=source_identity();directory.mkdir(parents=True,exist_ok=False)
    try:
        chosen,rows=choose_placement(MODEL)
    except Exception as error:
        write_json(directory/'placement-preflight.json',{'complete':False,'candidates':getattr(error,'candidate_observations',[]),'error_type':type(error).__name__})
        raise
    write_json(directory/'placement-preflight.json',{'complete':True,'candidates':rows,'chosen':chosen})
    config={**config03(),'n_gpu_layers':chosen,'engine':str(ENGINE)};identity.update({'config':config,'placement':[chosen,True],'engine_batch':64,'engine_ubatch':64})
    write_json(directory/'identity.json',identity);answers=[]
    try:
        with ResidentExperiment(directory,config) as exp:
            for n,record in enumerate(records,1):
                name=f'problem-{n:03d}';folder=directory/'jobs'/name
                request=build_request(record,MODEL,(chosen,True),config)
                print(f'{n}/2 {record["id"]} 생성 시작, GPU layers={chosen}',flush=True)
                result=exp.run(name,request)
                if result.get('stop_reason') not in ('eog','length') or (result['stop_reason']=='length' and len(result['token_ids'])!=32768):raise ValueError('생성 종료 사유 오류')
                if not result.get('gpu_fusion_disabled') or result.get('profiled_layers')!=list(range(30)) or result.get('masked_layers') or result.get('enable_thinking') is not True or result.get('requested_n_gpu_layers')!=chosen or result.get('actual_context_size')!=65536 or (result.get('cache_type_k'),result.get('cache_type_v'))!=('q8_0','q8_0'):
                    raise ValueError('실제 runtime 설정 불일치')
                observed=verify_runtime(result,(folder/'stderr.log').read_text(encoding='utf-8',errors='replace'),request)
                events=folder/'events.jsonl'; count=validate_routing(read_jsonl(events),result,experts=80)
                if count!=result.get('routing_records'):raise ValueError('라우팅 개수 불일치')
                native=native_result(events);actual={k:v for k,v in result.items() if k!='measurement'}
                if native!=actual:raise ValueError('native 최종 결과 불일치')
                target=directory/'routing'/f'{n:03d}.jsonl.gz';integrity=compress_events(events,target);exp.check()
                answer={'id':record['id'],'number':n,**result,'native_result_sha256':canonical_hash(native),'observed_runtime':observed,'routing_path':str(target.relative_to(directory)),'routing_integrity':integrity,'raw_events_path':str(events.relative_to(directory))}
                write_json(directory/'answers'/f'{n:03d}.json',answer);write_json(directory/'grade-inputs'/f'{n:03d}.json',{'id':record['id'],'problem':record['lcb_problem'],'text':native['text']})
                answers.append(answer);write_jsonl(directory/'generations.jsonl',answers);write_json(directory/'status.json',{'complete':False,'completed':len(answers),'total':2})
                print(f'{n}/2 완료: {record["id"]}, tokens={len(native["token_ids"])}',flush=True)
        write_json(directory/'status.json',{'complete':True,'completed':2,'total':2,'grading':'pending'})
    except BaseException:
        write_json(directory/'status.json',{'complete':False,'completed':len(answers),'total':2});raise
    return directory

# 2026-11-09: 채점 전 원본 이벤트의 최종 결과·압축 무결성·문제별 실행 ID를 다시 연결한다.
def verified_grade_inputs(directory):
    directory=Path(directory).resolve();status=json.loads((directory/'status.json').read_text(encoding='utf-8'))
    if status.get('complete') is not True or status.get('completed')!=2 or status.get('total')!=2:raise ValueError('생성 미완료')
    identity=json.loads((directory/'identity.json').read_text(encoding='utf-8'))
    if identity.get('model_sha256')!=MODEL_SHA or identity.get('ids')!=IDS or identity.get('manifest_sha256')!=sha256(MANIFEST) or identity.get('records_sha256')!=sha256(BUNDLE/'records.jsonl') or identity.get('bundle_inventory_sha256')!=sha256(BUNDLE/'SHA256SUMS.json'):raise ValueError('실행 출처 불일치')
    records=selected_records((1,2));items=[]
    if [r['id'] for r in records]!=IDS:raise ValueError('문제 ID 불일치')
    for n,record in enumerate(records,1):
        folder=directory/'jobs'/f'problem-{n:03d}'
        job=json.loads((folder/'identity.json').read_text(encoding='utf-8'))
        if job['model']['sha256']!=MODEL_SHA or job['engine']['sha256']!=identity['engine_sha256']:raise ValueError('native job identity 불일치')
        answer=json.loads((directory/'answers'/f'{n:03d}.json').read_text(encoding='utf-8'))
        payload_path=directory/'grade-inputs'/f'{n:03d}.json';payload=json.loads(payload_path.read_text(encoding='utf-8'))
        if answer['routing_path'].replace('\\','/')!=f'routing/{n:03d}.jsonl.gz' or answer['raw_events_path'].replace('\\','/')!=f'jobs/problem-{n:03d}/events.jsonl':raise ValueError('라우팅 경로 불일치')
        raw=(folder/'events.jsonl').resolve();compressed=(directory/'routing'/f'{n:03d}.jsonl.gz').resolve()
        if not raw.is_relative_to(directory) or not compressed.is_relative_to(directory):raise ValueError('경로 범위 오류')
        if sha256(raw)!=answer['routing_integrity']['uncompressed_sha256'] or sha256(compressed)!=answer['routing_integrity']['compressed_sha256']:raise ValueError('라우팅 해시 불일치')
        h=hashlib.sha256()
        with gzip.open(compressed,'rb') as f:
            while chunk:=f.read(1024*1024):h.update(chunk)
        if h.hexdigest()!=answer['routing_integrity']['uncompressed_sha256']:raise ValueError('압축 내용 불일치')
        native=native_result(raw);verify_payload(record,answer,payload,native)
        items.append((n,record,answer,payload_path))
    return identity,items

# 2026-11-09: 실행기 공통 잠금과 메모리 감시를 사용하며 격리는 제공하지 않는다.
def grade(directory=DIRECTORY,venv=ROOT/'.cache/putare-lcb/venv'):
    from moe_prune.execution_safety import execution_guard
    from moe_prune.grading import GradeDeadline
    directory=Path(directory).resolve()
    status=json.loads((directory/'status.json').read_text(encoding='utf-8'))
    if status.get('complete') is not True or status.get('completed')!=2 or status.get('total')!=2:raise ValueError('생성 미완료')
    with execution_guard('grading',memory_config=config03()) as guard:
        return _grade(directory,venv,guard,GradeDeadline())


def _grade(directory,venv,guard,deadline):
    import importlib.util,subprocess
    from moe_prune.grading import run as run_grade, identity as grader_identity, worker_environment, GradingCleanupError
    if sys.platform!='linux':raise RuntimeError('Colab Linux 환경 필요')
    venv=Path(venv);directory=Path(directory).resolve()
    identity,items=verified_grade_inputs(directory)
    def load(name,path):
        spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
    portable=load('_e80_portable',BUNDLE/'lcb_portable.py');portable.verify_inventory(BUNDLE)
    env=worker_environment()
    probe=subprocess.run([str(venv/'bin/python'),'-I','-B',str(BUNDLE/'lcb_grade.py'),'--verify','--official-source',str(BUNDLE/'official-source')],env=env,check=True,capture_output=True,text=True,timeout=deadline.remaining())
    official=json.loads(probe.stdout);original=json.loads((BUNDLE/'original-manifest.json').read_text(encoding='utf-8'))
    if official.get('source_digest')!=original['official_identity'].get('source_digest'):raise ValueError('공식 채점기 source 불일치')
    out=directory/'grading';out.mkdir(exist_ok=False)
    write_json(out/'identity.json',{'official':official,'inventory_sha256':sha256(BUNDLE/'SHA256SUMS.json'),'run_identity_sha256':sha256(directory/'identity.json'),'orchestrator_sha256':sha256(Path(__file__)),**grader_identity()})
    rows=[]
    for n,record,answer,payload in items:
        timed_out=False
        target=out/f'{n:03d}.json'
        try:
            run_grade(BUNDLE/'official-source',venv,payload,target,adapter=BUNDLE/'lcb_grade.py',memory_config=config03(),guard=guard,timeout_s=deadline.remaining())
            result=json.loads(target.read_text(encoding='utf-8'))
            if result.get('id')!=record['id'] or type(result.get('passed')) is not bool or result.get('pass_at_1')!=float(result['passed']):raise ValueError('채점 결과 오류')
            row={'number':n,'id':record['id'],'title':record['lcb_problem']['question_title'],'status':'graded','passed':result['passed'],'planned_test_count':result.get('test_count'),'executed_test_count':None}
            if isinstance(result.get('code'),str) and result['code'].strip():
                solutions=directory/'solutions';solutions.mkdir(exist_ok=True)
                with (solutions/f'{n:03d}.py').open('x',encoding='utf-8') as f:f.write(result['code'])
        except GradingCleanupError:
            raise
        except TimeoutError as error:
            row={'number':n,'id':record['id'],'title':record['lcb_problem']['question_title'],'status':'infrastructure_error','passed':None,'error_type':type(error).__name__}
            timed_out=True
        except Exception as error:
            row={'number':n,'id':record['id'],'title':record['lcb_problem']['question_title'],'status':'infrastructure_error','passed':None,'error_type':type(error).__name__}
        row.update({'stop_reason':answer['stop_reason'],'generated_tokens':len(answer['token_ids']),'timings':answer['timings'],'measurement':answer['measurement']});rows.append(row)
        print(f'{n}/2 {record["id"]} {row["status"]}',flush=True)
        if timed_out:break
    summary={'complete':all(x['status']=='graded' for x in rows),'scope':'E80, first two problems only','cases':rows,'model_sha256':MODEL_SHA,'limitations':['Test count is planned, executed count is not established.','Instrumented throughput includes monitoring overhead.','Original model benchmark scores do not apply.']}
    write_json(out/'summary.json',summary)
    text=['E80 모델 LCB 첫 두 문제 비격리 채점','']
    for row in rows:
        verdict=('통과' if row['passed'] else '실패') if row['status']=='graded' else '환경 오류(판정 없음)'
        timing=row.get('timings') or {};measurement=row.get('measurement') or {};gpu=measurement.get('os_gpu_memory') or {}
        text.append(f"{row['number']:03d} {row['id']} {row['title']}: {verdict}; 토큰 {row['generated_tokens']}; 종료 {row['stop_reason']}; 시간 {measurement.get('wall_s')}초; decode {timing.get('decode_tokens_s')} tok/s; RAM {measurement.get('peak_process_tree_rss_bytes')} bytes; GPU 표본 {gpu.get('peak_bytes')} bytes")
    text.append('테스트 수는 예정 수이며 실제 실행 수는 확인되지 않았습니다. GPU 수치는 OS 표본입니다.')
    (out/'RESULTS-KO.txt').write_text('\n'.join(text)+'\n',encoding='utf-8')
    return summary

def combined_80(profile):
    out={}
    for layer in range(30):
        parts=[profile.get(phase,{}).get(str(layer)) for phase in ('prefill','decode')];parts=[x for x in parts if x is not None]
        if not parts:continue
        tokens=sum(x['tokens'] for x in parts);counts=[sum(x['counts'][i] for x in parts) for i in range(80)]
        out[str(layer)]={'tokens':tokens,'counts':counts,'token_rate':[x/tokens for x in counts]}
    return out

# 2026-11-09: 새 E80 기록만 집계하고 문제별 prefill+decode 전체 선택률을 표시한다.
def plots(directory=DIRECTORY):
    import html
    from moe_prune.base_benchmark import aggregate_profiles
    directory=Path(directory).resolve();identity,items=verified_grade_inputs(directory)
    def events(path):
        with gzip.open(path,'rt',encoding='utf-8') as f:
            for line in f:
                if line.strip():yield json.loads(line)
    profiles=aggregate_profiles(((record['id'],events(directory/'routing'/f'{n:03d}.jsonl.gz')) for n,record,answer,payload in items),experts=80)
    mapping=json.loads(MANIFEST.read_text(encoding='utf-8'))['layers']
    out=directory/'visualizations';out.mkdir(exist_ok=False)
    write_json(out/'profiles.json',{'profiles':profiles,'model_sha256':MODEL_SHA,'manifest_sha256':identity['manifest_sha256'],'interpretation':'selection frequency, not contribution or expertise'})
    for n,record,answer,payload in items:
        rows=combined_80(profiles[record['id']]);vmax=max((v for row in rows.values() for v in row['token_rate']),default=0.0)
        parts=['<svg xmlns="http://www.w3.org/2000/svg" width="950" height="650">','<rect width="950" height="650" fill="white"/>',f'<text x="70" y="24">{html.escape(record["id"])}: prefill + decode selection frequency</text>',f'<text x="70" y="45">black=zero; white-red=positive up to observed max {100*vmax:.3f}%; gray=no data</text>']
        for l in range(30):
            row=rows.get(str(l));y=65+l*17;parts.append(f'<text x="30" y="{y+12}">{l}</text>')
            for e in range(80):
                count=row['counts'][e] if row else None;den=row['tokens'] if row else 0
                shade=round(255*(1-(count/den)/vmax)) if den and vmax else 0
                color='#808080' if den==0 else '#000000' if count==0 else f'#ff{shade:02x}{shade:02x}'
                old=mapping[l]['e80_to_original'][e]
                parts.append(f'<rect x="{70+e*10}" y="{y}" width="10" height="16" fill="{color}"><title>layer {l}, e80 ID {e}, original ID {old}: {count}/{den}</title></rect>')
        for e in list(range(0,80,8))+[79]:parts.append(f'<text x="{70+e*10}" y="590" font-size="10">{e}</text>')
        parts.extend(['<text x="70" y="610">Selection frequency is not contribution or semantic expertise.</text>','</svg>'])
        (out/f'{n:03d}-overall.svg').write_text(''.join(parts),encoding='utf-8')
    return out

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('generate','grade','plots','check'));args=p.parse_args()
    if args.action=='generate':generate()
    elif args.action=='grade':grade()
    elif args.action=='plots':print(plots())
    else:
        from moe_prune.base_benchmark import build_request
        c=config03();r=build_request({'prompt':'contract'},MODEL,(31,True),{**c,'n_gpu_layers':31})
        assert r['instrumentation'] and r['all_token_logits'] and r['enable_thinking'] and r['max_tokens']==32768 and r['context_size']==65535 and r['max_routing_records']==65535*30
    print(json.dumps({'config':c,'placement':[31,True],'experts':80,'top_k':8,'indices':[1,2]},ensure_ascii=False))
