"""수동 104 Expert 절삭: 양자화 블록을 보존하고 안전 게이트를 분리한다."""
from pathlib import Path
import argparse, gzip, hashlib, html, json, re, sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from moe_prune.gguf import GGUF,sha256

ORDERED=ROOT/'models/gemma-4-26B-A4B-it-UD-IQ4_NL-lcb100-activity-ordered.gguf'
ORDERED_SHA='90f69b9b8cb35f117ab49051f0dd1db8d3610d59185ecf2f914afa514851022f'
MAP=ROOT/'runs/gemma4-lcb100-reordered-001/permutation-manifest.json'
MODEL=ROOT/'models/gemma-4-26B-A4B-it-UD-IQ4_NL-lcb100-activity-ordered-e104.gguf'
EXPORT_RUN=ROOT/'runs/gemma4-ordered-e104-001'
RUN=ROOT/'runs/gemma4-ordered-e104-colab-a100-001'
ENGINE=ROOT/'.cache/llama-build/bin/llama-expert-research'
DATA=ROOT/'download/gemma4-lcb-first100-local-grading/records.jsonl'
BUNDLE=DATA.parent

# 2026-11-08: 새 ID가 기존 및 원본 ID의 어느 블록인지 명시하고 불완전한 순열을 거부한다.
def derive_mapping(entries,layers,experts,keep):
    if not 0<keep<=experts or len(entries)!=len(layers): raise ValueError('mapping dimensions')
    result=[]
    for layer,item in zip(layers,entries):
        ids=item['new_to_old']
        if item['layer']!=layer or len(ids)!=experts or any(type(x) is not int for x in ids) or set(ids)!=set(range(experts)):
            raise ValueError('invalid expert permutation')
        result.append({'layer':layer,'pruned_to_ordered':list(range(keep)),'pruned_to_original':ids[:keep], 'removed_ordered':list(range(keep,experts)), 'removed_original':ids[keep:]})
    return result

# 2026-11-08: 메타데이터와 모든 tensor를 원본 블록과 직접 비교하여 잘못된 축 절삭을 차단한다.
def verify_sliced(source,output,allowlist):
    a,b=GGUF(source),GGUF(output); layers,experts,topk=a.architecture(); keep=len(allowlist[layers[0]])
    if a.metadata.keys()!=b.metadata.keys() or any(a.metadata[k]!=b.metadata[k] for k in a.metadata if k!='gemma4.expert_count') or b.metadata['gemma4.expert_count']!=keep or len(a.tensors)!=len(b.tensors): raise ValueError('metadata mismatch')
    if [(k,t,r) for k,t,r in a.metadata_raw if k!='gemma4.expert_count'] != [(k,t,r) for k,t,r in b.metadata_raw if k!='gemma4.expert_count']: raise ValueError('metadata raw mismatch')
    with Path(source).open('rb') as src,Path(output).open('rb') as dst:
        for x,y in zip(a.tensors,b.tensors):
            m=re.fullmatch(r'blk\.(\d+)\.(ffn_gate_inp\.weight|ffn_(?:gate|up|down|gate_up)_exps\.(?:weight|scale))',x.name)
            ids=allowlist[int(m[1])] if m else [0]; unit=x.nbytes//experts if m else x.nbytes
            shape=x.shape[:-1]+(keep,) if m else x.shape
            if (y.name,y.kind,y.shape,y.nbytes)!=(x.name,x.kind,shape,unit*len(ids)): raise ValueError('tensor structure mismatch: '+x.name)
            for new,old in enumerate(ids):
                src.seek(a.data_offset+x.offset+old*unit);dst.seek(b.data_offset+y.offset+new*unit); remaining=unit
                while remaining:
                    n=min(4*1024*1024,remaining); p=src.read(n);q=dst.read(n)
                    if len(p)!=n or p!=q: raise ValueError(f'tensor bytes mismatch: {x.name} expert {old}')
                    remaining-=n
    return {'tensor_count':len(a.tensors),'all_tensor_bytes_verified':True,'other_metadata_raw_identical':True}

# 2026-11-08: router 없는 계층과 미지 Expert 텐서를 공유 값으로 착각하지 않는다.
def validate_expert_layout(model):
    layers,experts,_=model.architecture()
    for t in model.tensors:
        if '_exps' not in t.name and 'ffn_gate_inp.' not in t.name: continue
        m=re.fullmatch(r'blk\.(\d+)\.(ffn_gate_inp\.(?:weight|scale)|ffn_(?:gate|up|down|gate_up)_exps\.(?:weight|scale))',t.name)
        if not m or int(m[1]) not in layers: raise ValueError('unknown expert tensor: '+t.name)
        if m[2]!='ffn_gate_inp.scale' and t.shape[-1]!=experts: raise ValueError('unexpected expert axis: '+t.name)

def export(source=ORDERED,output=MODEL,run=EXPORT_RUN,mapping_path=MAP):
    source,output,run,mapping_path=map(Path,(source,output,run,mapping_path))
    manifest=run/'export-manifest.json'
    if output.exists() or manifest.exists() or output.with_name(output.name+'.partial').exists(): raise FileExistsError('기존 모델/manifest 보존')
    mapping=json.loads(mapping_path.read_text(encoding='utf-8'))
    if mapping['output_sha256']!=ORDERED_SHA or sha256(source)!=ORDERED_SHA: raise ValueError('정렬 모델 해시 불일치')
    model=GGUF(source); layers,experts,topk=model.architecture()
    if (layers,experts,topk)!=(list(range(30)),128,8): raise ValueError('expert architecture mismatch')
    validate_expert_layout(model)
    maps=derive_mapping(mapping['layers'],layers,128,104); allowlist={layer:list(range(104)) for layer in layers}
    print('모델 절삭 시작',flush=True); output_info=model.rewrite(output,allowlist=allowlist)
    if output.stat().st_size!=11538949024: raise ValueError('예상 GGUF 크기와 다릅니다')
    print('전체 tensor 바이트 검증',flush=True); verification=verify_sliced(source,output,allowlist)
    document={'source':str(source),'source_sha256':output_info['source_sha256'],'output':str(output),'output_sha256':output_info['sha256'],'output_bytes':output.stat().st_size,'source_bytes':source.stat().st_size,'savings_bytes':source.stat().st_size-output.stat().st_size,'mapping_manifest_sha256':sha256(mapping_path),'expert_count':104,'top_k':8,'layers':maps,'verification':verification,'file_validation_status':'passed','inference_status':'unverified','quality_status':'unverified','runtime_equivalence':'unverified','limits':'정적 블록 검증은 추론 동등성이나 품질을 증명하지 않습니다.'}
    run.mkdir(parents=True,exist_ok=True)
    with manifest.open('x',encoding='utf-8') as f: json.dump(document,f,indent=2,ensure_ascii=False)
    return document

# 2026-11-09: 기존 모델의 해시·구조·전체 바이트를 재검증한 뒤 누락된 기록만 복원한다.
def _recover_existing_manifest(*,source,output,run,mapping_path,source_sha,pruned_sha,layers,experts,keep,top_k,expected_size):
    from datetime import datetime,timezone
    source,output,run,mapping_path=map(Path,(source,output,run,mapping_path))
    manifest=run/'export-manifest.json'
    if manifest.exists() or manifest.is_symlink(): raise FileExistsError('기존 manifest 보존')
    mapping=json.loads(mapping_path.read_text(encoding='utf-8'))
    if mapping.get('output_sha256')!=source_sha or sha256(source)!=source_sha or sha256(output)!=pruned_sha: raise ValueError('기존 모델 해시 불일치')
    original,pruned=GGUF(source),GGUF(output)
    if original.architecture()!=(layers,experts,top_k) or pruned.architecture()!=(layers,keep,top_k) or output.stat().st_size!=expected_size:
        raise ValueError('기존 모델 구조 또는 크기 불일치')
    validate_expert_layout(original)
    validate_expert_layout(pruned)
    maps=derive_mapping(mapping['layers'],layers,experts,keep)
    allowlist={layer:list(range(keep)) for layer in layers}
    verification=verify_sliced(source,output,allowlist)
    document={'source':str(source),'source_sha256':source_sha,'output':str(output),'output_sha256':pruned_sha,'output_bytes':output.stat().st_size,'source_bytes':source.stat().st_size,'savings_bytes':source.stat().st_size-output.stat().st_size,'mapping_manifest_sha256':sha256(mapping_path),'expert_count':keep,'top_k':top_k,'layers':maps,'verification':verification,'file_validation_status':'passed','inference_status':'unverified','quality_status':'unverified','runtime_equivalence':'unverified','limits':'정적 블록 검증은 추론 동등성이나 품질을 증명하지 않습니다.','recovery_provenance':{'action':'reconstructed from existing files with fresh static verification','reconstructed_at_utc':datetime.now(timezone.utc).isoformat(),'historical_execution_claimed':False}}
    run.mkdir(parents=True,exist_ok=True)
    with manifest.open('x',encoding='utf-8') as f: json.dump(document,f,indent=2,ensure_ascii=False)
    return document

def recover_existing_manifest():
    return _recover_existing_manifest(source=ORDERED,output=MODEL,run=EXPORT_RUN,mapping_path=MAP,source_sha=ORDERED_SHA,pruned_sha='d4f395700b4a1739007c6214a5c256e73bc4f48c71b2ee7528dafa76be0652f2',layers=list(range(30)),experts=128,keep=104,top_k=8,expected_size=11538949024)

if __name__=='__main__':
    p=argparse.ArgumentParser();group=p.add_mutually_exclusive_group();group.add_argument('--export',action='store_true');group.add_argument('--recover-existing',action='store_true');a=p.parse_args()
    if a.export: print(json.dumps(export(),ensure_ascii=False,indent=2))
    elif a.recover_existing: print(json.dumps(recover_existing_manifest(),ensure_ascii=False,indent=2))
    else: p.print_help()

# 2026-11-09: 모든 Colab 추론·채점에 동일한 메모리 경고 비율을 전달한다.
def config03():
    from moe_prune.base_benchmark import DEFAULT_CONFIG
    return dict(DEFAULT_CONFIG)

# 2026-11-08: 실제 실행마다 결과·모델·설정을 묶어 다른 실험 산출물의 재사용을 차단한다.
def execution_identity(directory,model=MODEL):
    from moe_prune.base_benchmark import DEFAULT_CONFIG
    from moe_prune.research import write_json
    import platform
    if not (EXPORT_RUN/'export-manifest.json').is_file():
        raise FileNotFoundError('104 모델 기록이 없습니다. 기존 모델을 검증해 복원하려면 tools/manual104.py --recover-existing 을 실행하세요.')
    import importlib.util
    spec=importlib.util.spec_from_file_location('_manual104_bundle',BUNDLE/'lcb_portable.py');portable=importlib.util.module_from_spec(spec);spec.loader.exec_module(portable)
    portable.verify_inventory(BUNDLE)
    export_info=json.loads((EXPORT_RUN/'export-manifest.json').read_text(encoding='utf-8'))
    if export_info['source_sha256']!=ORDERED_SHA or export_info['expert_count']!=104 or sha256(model)!=export_info['output_sha256']:
        raise ValueError('104 모델 또는 출처 해시 불일치')
    if sha256(MAP)!=export_info['mapping_manifest_sha256'] or json.loads(MAP.read_text(encoding='utf-8'))['profile']['records_sha256']!=sha256(DATA): raise ValueError('원본 100문항 및 순열 출처 불일치')
    if sha256(ORDERED)!=ORDERED_SHA: raise ValueError('정렬 원본 모델 해시 불일치')
    if GGUF(model).architecture()!=(list(range(30)),104,8) or GGUF(ORDERED).architecture()!=(list(range(30)),128,8): raise ValueError('GGUF 전문가 구조 불일치')
    directory=Path(directory); directory.mkdir(parents=True,exist_ok=False)
    config={**config03(),'engine':str(ENGINE)}
    identity={'ordered_model_sha256':ORDERED_SHA,'model_sha256':export_info['output_sha256'],'engine_sha256':sha256(ENGINE),'records_sha256':sha256(DATA),'bundle_inventory_sha256':sha256(BUNDLE/'SHA256SUMS.json'),'export_manifest_sha256':sha256(EXPORT_RUN/'export-manifest.json'),'config':config,'placement':[31,True],'runtime_equivalence':'unverified','engine_batch':'native fixed n_batch=64, n_ubatch=64','platform':platform.platform()}
    write_json(directory/'identity.json',identity)
    return config,identity

def selected_records(indices):
    from moe_prune.research import read_jsonl
    wanted=list(indices)
    if not wanted or len(set(wanted))!=len(wanted) or any(type(x) is not int or x<1 or x>100 for x in wanted): raise ValueError('1..100의 중복 없는 문제 번호 필요')
    records=[]
    for i,item in enumerate(read_jsonl(DATA),1):
        records.append(item)
        if i==100: break
    if len(records)!=100 or len({x['id'] for x in records})!=100: raise ValueError('원본 100개 문제 불일치')
    return [records[i-1] for i in wanted]

def checked_request(record,model):
    from moe_prune.base_benchmark import DEFAULT_CONFIG,build_request
    return build_request(record,model,(31,True),config03())

def validate_job(directory,name,request,result,experts):
    from moe_prune.base_benchmark import validate_routing,verify_runtime
    from moe_prune.research import read_jsonl
    folder=Path(directory)/'jobs'/name
    if not result.get('gpu_fusion_disabled') or result.get('profiled_layers')!=list(range(30)) or result.get('enable_thinking') is not True:
        raise ValueError('fusion 또는 계측 설정 불일치')
    if result.get('actual_context_size')!=65536 or result.get('requested_n_gpu_layers')!=request['n_gpu_layers'] or (result.get('cache_type_k'),result.get('cache_type_v'))!=('q8_0','q8_0'):raise ValueError('실제 context/GPU/KV 설정 불일치')
    observed=verify_runtime(result,(folder/'stderr.log').read_text(encoding='utf-8',errors='replace'),request)
    count=validate_routing(read_jsonl(folder/'events.jsonl'),result,experts=experts)
    if result['routing_records']!=count: raise ValueError('라우팅 이벤트 수 불일치')
    return observed

def compress_events(source,target):
    target.parent.mkdir(parents=True,exist_ok=True);h=hashlib.sha256();v=hashlib.sha256()
    with source.open('rb') as src,target.open('xb') as raw,gzip.GzipFile(fileobj=raw,mode='wb',compresslevel=3) as out:
        while chunk:=src.read(1024*1024): h.update(chunk);out.write(chunk)
    with gzip.open(target,'rb') as f:
        while chunk:=f.read(1024*1024):v.update(chunk)
    if h.digest()!=v.digest():raise ValueError('압축 검증 실패')
    return {'uncompressed_sha256':h.hexdigest(),'compressed_sha256':sha256(target)}

def run_generation(indices=(1,2),directory=RUN/'generation-001'):
    from moe_prune.resident import ResidentExperiment
    from moe_prune.research import write_json,write_jsonl
    records=selected_records(indices);directory=Path(directory);config,identity=execution_identity(directory)
    answers=[]
    try:
        with ResidentExperiment(directory,config) as exp:
            for i,record in zip(indices,records):
                name=f'problem-{i:03d}';request=checked_request(record,MODEL);result=exp.run(name,request)
                if result.get('stop_reason') not in ('eog','length') or (result['stop_reason']=='length' and len(result['token_ids'])!=32768): raise ValueError('종료 사유 오류')
                if result.get('masked_layers'):raise ValueError('물리 모델에 마스크가 적용됨')
                observed=validate_job(directory,name,request,result,104);events=directory/'jobs'/name/'events.jsonl';target=directory/'routing'/f'{i:03d}.jsonl.gz'
                integrity=compress_events(events,target);exp.check()
                answer={'id':record['id'],'number':i,**result,'observed_runtime':observed,'routing_path':str(target.relative_to(directory)),'routing_integrity':integrity,'raw_events_path':str(events.relative_to(directory))}
                write_json(directory/'answers'/f'{i:03d}.json',answer)
                write_json(directory/'grade-inputs'/f'{i:03d}.json',{'id':record['id'],'problem':record['lcb_problem'],'text':result['text']})
                answers.append(answer);write_jsonl(directory/'generations.jsonl',answers)
                write_json(directory/'status.json',{'complete':False,'completed':len(answers),'total':len(records),'indices':list(indices)})
                print(f'{len(answers)}/{len(records)} {record["id"]} 완료',flush=True)
        write_json(directory/'status.json',{'complete':True,'completed':len(answers),'total':len(records),'indices':list(indices),'grading':'pending'})
    except BaseException:
        write_json(directory/'status.json',{'complete':False,'completed':len(answers),'total':len(records),'indices':list(indices)})
        raise

def compare_masked(indices=(1,2),directory=RUN/'comparison-001'):
    from moe_prune.resident import ResidentExperiment
    from moe_prune.research import compare_logits,write_json,read_jsonl
    directory=Path(directory); records=selected_records(indices);config,identity=execution_identity(directory)
    rows=[];mask={str(layer):list(range(104)) for layer in range(30)}
    with ResidentExperiment(directory,config) as exp:
        for number,record in zip(indices,records):
            results=[]
            for label,model,expert_count in (('masked',ORDERED,128),('physical',MODEL,104)):
                request=checked_request(record,model);request.update({'mode':'logits','max_tokens':0})
                if label=='masked':request['allowlist']=mask
                name=f'{label}-{number:03d}';result=exp.run(name,request)
                if label=='masked' and result.get('masked_layers')!=list(range(30)):raise ValueError('30층 마스크 불일치')
                if label=='physical' and result.get('masked_layers'):raise ValueError('물리 모델 마스크 불일치')
                validate_job(directory,name,request,result,expert_count)
                if label=='masked':
                    for row in read_jsonl(directory/'jobs'/name/'events.jsonl'):
                        if row.get('type')=='routing' and any(i>=104 for i in row['expert_ids']):raise ValueError('마스크 밖 Expert 선택')
                results.append(result)
                exp.close()
            rows.append({'number':number,'id':record['id'],**compare_logits(results[0]['logits'],results[1]['logits'],atol=1e-4,rtol=1e-4)})
            write_json(directory/'comparison.json',{'complete':False,'comparisons':rows})
    write_json(directory/'comparison.json',{'complete':True,'comparisons':rows,'all_equivalent':all(x['equivalent'] for x in rows),'scope':'masked ordered 0..103 vs physical104; limited prompts only'})
    return rows

# 2026-11-09: 생성 코드는 별도 프로세스에서 비격리 실행하며 환경 오류를 오답과 구분한다.
def grade(directory,venv=ROOT/'.cache/putare-lcb/venv'):
    from moe_prune.execution_safety import execution_guard
    from moe_prune.grading import GradeDeadline
    directory=Path(directory).resolve()
    status=json.loads((directory/'status.json').read_text(encoding='utf-8'))
    if not status.get('complete') or status['completed']!=status['total']:raise ValueError('생성 미완료')
    with execution_guard('grading',memory_config=config03()) as guard:
        return _grade(directory,venv,guard,GradeDeadline())


def _grade(directory,venv,guard,deadline):
    import importlib.util,subprocess
    from moe_prune.grading import run as run_grade, identity as grader_identity, worker_environment, GradingCleanupError
    if sys.platform!='linux': raise RuntimeError('채점은 Colab Linux에서 실행합니다')
    directory=Path(directory).resolve(); status=json.loads((directory/'status.json').read_text(encoding='utf-8'))
    if not status.get('complete') or status['completed']!=status['total']: raise ValueError('생성 미완료')
    def module(name,path):
        spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
    portable=module('_m104_portable',BUNDLE/'lcb_portable.py');portable.verify_inventory(BUNDLE)
    env=worker_environment()
    venv=Path(venv);p=subprocess.run([str(venv/'bin/python'),'-I','-B',str(BUNDLE/'lcb_grade.py'),'--verify','--official-source',str(BUNDLE/'official-source')],env=env,check=True,capture_output=True,text=True,timeout=deadline.remaining())
    official=json.loads(p.stdout);original=json.loads((BUNDLE/'original-manifest.json').read_text(encoding='utf-8'))
    if official.get('source_digest')!=original['official_identity'].get('source_digest'):raise ValueError('공식 채점기 source 불일치')
    identity=json.loads((directory/'identity.json').read_text(encoding='utf-8'))
    export_info=json.loads((EXPORT_RUN/'export-manifest.json').read_text(encoding='utf-8'))
    if identity['bundle_inventory_sha256']!=sha256(BUNDLE/'SHA256SUMS.json') or identity['records_sha256']!=sha256(DATA) or identity['export_manifest_sha256']!=sha256(EXPORT_RUN/'export-manifest.json') or identity['model_sha256']!=export_info['output_sha256']:raise ValueError('실행 identity 불일치')
    records=selected_records(status['indices'])
    # 모든 입력을 먼저 검증해야 부분 채점 결과와 잘못된 source가 섞이지 않는다.
    items=[]
    for i,record in zip(status['indices'],records):
        payload_path=directory/'grade-inputs'/f'{i:03d}.json';payload=json.loads(payload_path.read_text(encoding='utf-8'));answer=json.loads((directory/'answers'/f'{i:03d}.json').read_text(encoding='utf-8'))
        job=json.loads((directory/'jobs'/f'problem-{i:03d}'/'identity.json').read_text(encoding='utf-8'))
        if payload!={'id':record['id'],'problem':record['lcb_problem'],'text':answer['text']} or answer['id']!=record['id'] or job['model']['sha256']!=identity['model_sha256'] or job['engine']['sha256']!=identity['engine_sha256']:raise ValueError('답안 출처 불일치')
        if answer['routing_path'].replace('\\','/')!=f'routing/{i:03d}.jsonl.gz':raise ValueError('라우팅 경로 오류')
        path=(directory/'routing'/f'{i:03d}.jsonl.gz').resolve()
        if not path.is_relative_to(directory) or sha256(path)!=answer['routing_integrity']['compressed_sha256']:raise ValueError('라우팅 해시 불일치')
        items.append((i,record,payload_path,answer))
    out=directory/'grading';out.mkdir(exist_ok=False)
    (out/'identity.json').write_text(json.dumps({'official':official,'inventory_sha256':sha256(BUNDLE/'SHA256SUMS.json'),'run_identity_sha256':sha256(directory/'identity.json'),**grader_identity()},indent=2),encoding='utf-8')
    rows=[]
    for i,record,payload_path,answer in items:
        target=out/f'{i:03d}.json'
        try:
            run_grade(BUNDLE/'official-source',venv,payload_path,target,adapter=BUNDLE/'lcb_grade.py',memory_config=config03(),guard=guard,timeout_s=deadline.remaining())
            result=json.loads(target.read_text(encoding='utf-8'))
            if result.get('id')!=record['id'] or type(result.get('passed')) is not bool or result.get('pass_at_1')!=float(result['passed']):raise ValueError('채점 결과 오류')
            rows.append({'number':i,'id':record['id'],'status':'graded','passed':result['passed'],'planned_test_count':result.get('test_count'),'executed_test_count':None})
        except GradingCleanupError:
            raise
        except TimeoutError as error:
            rows.append({'number':i,'id':record['id'],'status':'infrastructure_error','passed':None,'error_type':type(error).__name__})
            break
        except Exception as error:
            rows.append({'number':i,'id':record['id'],'status':'infrastructure_error','passed':None,'error_type':type(error).__name__})
    summary={'complete':all(x['status']=='graded' for x in rows),'cases':rows,'scope':'selected subset, not LCB benchmark score','model_sha256':identity['model_sha256']}
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    return summary

def combined_rows(profile):
    # 2026-11-08: 문제별 전체 사용량은 입력과 생성 단계의 실제 이벤트를 함께 분모로 쓴다.
    out={}
    for layer in range(30):
        parts=[profile.get(phase,{}).get(str(layer)) for phase in ('prefill','decode')]
        parts=[x for x in parts if x is not None]
        if not parts:continue
        tokens=sum(x['tokens'] for x in parts)
        counts=[sum(x['counts'][i] for x in parts) for i in range(104)]
        out[str(layer)]={'tokens':tokens,'counts':counts,'token_rate':[x/tokens for x in counts]}
    return out

def render_profiles(directory):
    from moe_prune.base_benchmark import aggregate_profiles
    directory=Path(directory);status=json.loads((directory/'status.json').read_text(encoding='utf-8'))
    if not status.get('complete'):raise ValueError('생성 미완료')
    samples=[]
    def events(path):
        with gzip.open(path,'rt',encoding='utf-8') as f:
            for line in f:
                if line.strip():yield json.loads(line)
    for i in status['indices']:
        a=json.loads((directory/'answers'/f'{i:03d}.json').read_text(encoding='utf-8'))
        if a['routing_path'].replace('\\','/')!=f'routing/{i:03d}.jsonl.gz':raise ValueError('경로 오류')
        p=(directory/'routing'/f'{i:03d}.jsonl.gz').resolve()
        if not p.is_relative_to(directory.resolve()) or sha256(p)!=a['routing_integrity']['compressed_sha256']:raise ValueError('라우팅 무결성 오류')
        samples.append((a['id'],p))
    profiles=aggregate_profiles(((name,events(path)) for name,path in samples),experts=104)
    out=directory/'visualizations';out.mkdir(exist_ok=False)
    (out/'profiles.json').write_text(json.dumps(profiles,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    mapping=json.loads((EXPORT_RUN/'export-manifest.json').read_text(encoding='utf-8'))['layers']
    for name,_ in samples:
        rows=combined_rows(profiles[name]); vmax=max((v for row in rows.values() for v in row['token_rate']),default=0.0);parts=['<svg xmlns="http://www.w3.org/2000/svg" width="1140" height="650">','<rect width="1140" height="650" fill="white"/>',f'<text x="70" y="24">{html.escape(name)}: selection frequency: prefill + decode</text>',f'<text x="70" y="45">black=zero; white-red=positive 0–{vmax*100:.3f}%; red=observed max; gray=no data; IDs 0–103</text>']
        for l in range(30):
            row=rows.get(str(l));y=65+l*17;parts.append(f'<text x="30" y="{y+12}">{l}</text>')
            for e in range(104):
                count=row['counts'][e] if row else None;n=row['tokens'] if row else 0
                shade=round(255*(1-(count/n)/vmax)) if n and vmax else 0;color='#808080' if n==0 else '#000000' if count==0 else f'#ff{shade:02x}{shade:02x}'
                parts.append(f'<rect x="{70+e*10}" y="{y}" width="10" height="16" fill="{color}"><title>layer {l} expert {e} (original {mapping[l]['pruned_to_original'][e]}): {count}/{n}</title></rect>')
        for e in list(range(0,104,8))+[103]:parts.append(f'<text x="{70+e*10}" y="590" font-size="10">{e}</text>')
        parts.extend(['<text x="70" y="605">Selection count / observed prefill + decode events; not contribution or semantic expertise.</text>','</svg>'])
        (out/(hashlib.sha256(name.encode()).hexdigest()+'.svg')).write_text(''.join(parts),encoding='utf-8')
    return out
