"""정렬된 104 Expert GGUF에서 0–79만 보존하는 정적 내보내기."""
from pathlib import Path
import argparse
import json
import re
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from moe_prune.gguf import GGUF,sha256
from tools.manual104 import validate_expert_layout,verify_sliced

SOURCE=ROOT/'models/gemma-4-26B-A4B-it-UD-IQ4_NL-lcb100-activity-ordered-e104.gguf'
SOURCE_SHA='d4f395700b4a1739007c6214a5c256e73bc4f48c71b2ee7528dafa76be0652f2'
SOURCE_BYTES=11538949024
PARENT=ROOT/'runs/gemma4-ordered-e104-001/export-manifest.json'
REORDER=ROOT/'runs/gemma4-lcb100-reordered-001/permutation-manifest.json'
OUTPUT=ROOT/'models/gemma-4-26B-A4B-it-UD-IQ4_NL-lcb100-activity-ordered-e80.gguf'
RUN=ROOT/'runs/gemma4-ordered-e80-001'

# 2026-11-09: 저장된 104 모델 ID와 이전 128 순열의 연결이 올바른지 확인한다.
def derive_maps(parent,reorder,layers):
    a,b=parent['layers'],reorder['layers']
    if len(a)!=len(layers) or len(b)!=len(layers):raise ValueError('layer mapping count mismatch')
    result=[]
    for layer,x,y in zip(layers,a,b):
        ids=y['new_to_old']
        if x['layer']!=layer or y['layer']!=layer or len(ids)!=128 or any(type(i) is not int for i in ids) or set(ids)!=set(range(128)):
            raise ValueError('invalid 128 expert permutation')
        if x['pruned_to_ordered']!=list(range(104)) or x['pruned_to_original']!=ids[:104]:raise ValueError('104 to original mapping mismatch')
        result.append({'layer':layer,'e80_to_source104':list(range(80)),'e80_to_ordered128':x['pruned_to_ordered'][:80],'e80_to_original':ids[:80], 'removed_source104':list(range(80,104)),'removed_ordered128':x['pruned_to_ordered'][80:104],'removed_original':ids[80:104]})
    return result

# 2026-11-09: 각 tensor별 GGUF alignment를 포함해 예상 출력 크기를 계산한다.
def expected_size(model,keep):
    layers,experts,_=model.architecture();offset=0
    for tensor in model.tensors:
        offset=(offset+model.alignment-1)//model.alignment*model.alignment
        m=re.fullmatch(r'blk\.(\d+)\.(ffn_gate_inp\.weight|ffn_(?:gate|up|down|gate_up)_exps\.(?:weight|scale))',tensor.name)
        offset+=tensor.nbytes//experts*keep if m else tensor.nbytes
    return model.data_offset+offset

def _export_e80(*,source,output,run,source_manifest,reorder_manifest,expected_sha,expected_bytes,expected_layers):
    source,output,run,source_manifest,reorder_manifest=map(Path,(source,output,run,source_manifest,reorder_manifest))
    manifest=run/'export-manifest.json';partial=output.with_name(output.name+'.partial')
    if any(p.exists() or p.is_symlink() for p in (output,manifest,partial)):raise FileExistsError('기존 출력 또는 부분 파일 보존')
    parent=json.loads(source_manifest.read_text(encoding='utf-8'));reorder=json.loads(reorder_manifest.read_text(encoding='utf-8'))
    if parent.get('output_sha256')!=expected_sha or parent.get('expert_count')!=104 or parent.get('top_k')!=8 or parent.get('file_validation_status')!='passed':raise ValueError('source manifest mismatch')
    if parent.get('source_sha256')!=reorder.get('output_sha256') or parent.get('mapping_manifest_sha256')!=sha256(reorder_manifest):raise ValueError('reorder provenance mismatch')
    maps=derive_maps(parent,reorder,expected_layers)
    if source.stat().st_size!=expected_bytes or sha256(source)!=expected_sha:raise ValueError('104 source hash or size mismatch')
    model=GGUF(source)
    if model.architecture()!=(expected_layers,104,8):raise ValueError('104 source architecture mismatch')
    validate_expert_layout(model)
    predicted=expected_size(model,80)
    print(f'80 Expert model export; expected bytes={predicted}',flush=True)
    info=model.rewrite(output,allowlist={layer:list(range(80)) for layer in expected_layers})
    if output.stat().st_size!=predicted:raise ValueError('output size mismatch')
    print('verifying all tensor bytes and metadata',flush=True)
    verification=verify_sliced(source,output,{layer:list(range(80)) for layer in expected_layers})
    if sha256(output)!=info['sha256']:raise ValueError('output changed during validation')
    doc={'source':str(source),'source_sha256':expected_sha,'source_bytes':expected_bytes,'source_manifest_sha256':sha256(source_manifest),'reorder_manifest_sha256':sha256(reorder_manifest),'tool_sha256':sha256(Path(__file__)),'output':str(output),'output_sha256':info['sha256'],'output_bytes':output.stat().st_size,'expected_output_bytes':predicted,'savings_bytes':expected_bytes-predicted,'expert_count':80,'top_k':8,'layers':maps,'verification':verification,'file_validation_status':'passed','inference_status':'unverified','quality_status':'unverified','runtime_equivalence':'unverified','limits':'정적 바이트 검증은 추론 동등성이나 품질을 증명하지 않습니다.'}
    run.mkdir(parents=True,exist_ok=True)
    with manifest.open('x',encoding='utf-8') as f:json.dump(doc,f,ensure_ascii=False,indent=2)
    return doc

def export():
    return _export_e80(source=SOURCE,output=OUTPUT,run=RUN,source_manifest=PARENT,reorder_manifest=REORDER,expected_sha=SOURCE_SHA,expected_bytes=SOURCE_BYTES,expected_layers=list(range(30)))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--export',action='store_true');args=parser.parse_args()
    if args.export:print(json.dumps(export(),ensure_ascii=False,indent=2))
    else:parser.print_help()
