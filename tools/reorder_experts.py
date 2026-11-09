"""LCB 100문항 decode 기록으로 Gemma4 Expert 블록을 재정렬한다."""
from pathlib import Path
import argparse
import hashlib
import json
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from moe_prune.gguf import GGUF, sha256

SOURCE_HASH = 'eeb867f279ea5a3d52a0dc15fe8ada677b3328a328530957f3f9a5da93cb10b8'
PROFILE = ROOT / 'download/lcb-first100-results/expert-usage'
INPUT = ROOT / 'download/gemma4-lcb-first100-local-grading'
SOURCE = ROOT/'models/gemma-4-26B-A4B-it-UD-IQ4_NL.gguf'
OUTPUT = ROOT/'models/gemma-4-26B-A4B-it-UD-IQ4_NL-lcb100-activity-ordered.gguf'
MANIFEST = ROOT/'runs/gemma4-lcb100-reordered-001/permutation-manifest.json'

# 2026-11-08: 기록이 대상 모델과 실제 생성본에 연결되는지 해시로 확인한다.
def validate_binding(bundle, identity, generations_hash):
    if identity.get('model') != SOURCE_HASH or bundle.get('source_manifest_sha256') != identity.get('snapshot_manifest') or bundle.get('generations_sha256') != generations_hash:
        raise ValueError('모델 또는 생성 데이터 출처 해시 불일치')

# 2026-11-08: 기록·답변·개별 집계를 교차 확인해 임의 행렬이 모델 순서를 결정하지 못하게 한다.
def profile():
    index_path = PROFILE / 'index.json'
    index = json.loads(index_path.read_text(encoding='utf-8'))
    bundle_path = INPUT/'bundle.json'; identity_path = INPUT/'generation-identity.json'
    bundle = json.loads(bundle_path.read_text(encoding='utf-8')); identity = json.loads(identity_path.read_text(encoding='utf-8'))
    generations_hash = sha256(INPUT/'generations.jsonl')
    validate_binding(bundle, identity, generations_hash)
    def jsonl(path):
        with path.open(encoding='utf-8') as f:
            return [json.loads(line) for line in f]
    records = jsonl(INPUT/'records.jsonl'); generations = jsonl(INPUT/'generations.jsonl')
    entries = index['entries']
    if index['problem_count'] != 100 or len(entries) != 100 or len(records) != 100 or len(generations) != 100:
        raise ValueError('첫 100문항 입력 개수가 일치하지 않습니다')
    counts = [[0]*128 for _ in range(30)]; denom = [0]*30; hashes = []
    ids = set()
    for number, (entry, record, generation) in enumerate(zip(entries, records, generations), 1):
        ident = record['id']; path = PROFILE / f'{number:03d}.json'
        if ident in ids or entry['number'] != number or entry['id'] != ident or generation['id'] != ident or entry['json'] != path.name:
            raise ValueError(f'문항 순서/ID 불일치: {number}')
        ids.add(ident); item = json.loads(path.read_text(encoding='utf-8'))
        result_path = ROOT / 'download/lcb-first100-results' / hashlib.sha256(ident.encode()).hexdigest() / 'result.json'
        result = json.loads(result_path.read_text(encoding='utf-8'))
        if item['id'] != ident or item['number'] != number or result['id'] != ident or item['passed'] != result['passed'] or entry['passed'] != result['passed']:
            raise ValueError(f'개별 결과 불일치: {ident}')
        if not item['hash_verified'] or not item['top8_unique_valid_and_reconciled'] or item['routing_sha256_actual'] != generation['routing_sha256'] or item['routing_sha256_expected'] != generation['routing_sha256'] or entry['routing_sha256'] != generation['routing_sha256']:
            raise ValueError(f'라우팅 해시 검증 근거 불일치: {ident}')
        phase = item['phases']['decode']; rows = phase['counts']; events = phase['events']
        if len(rows) != 30 or len(events) != 30:
            raise ValueError(f'층 개수 오류: {ident}')
        for layer, row in enumerate(rows):
            n = events[layer]
            if len(row) != 128 or type(n) is not int or n <= 0 or any(type(v) is not int or v < 0 or v > n for v in row) or sum(row) != 8*n:
                raise ValueError(f'집계 오류: {ident}, layer={layer}')
            denom[layer] += n
            for expert, count in enumerate(row): counts[layer][expert] += count
        hashes.append({'id':ident, 'profile_sha256':sha256(path), 'routing_sha256':generation['routing_sha256']})
    if counts != index['overall']['counts'] or denom != index['overall']['decode_events_per_layer']:
        raise ValueError('전체 집계가 개별 집계 합계와 다릅니다')
    permutations = {layer: sorted(range(128), key=lambda e: (-counts[layer][e], e)) for layer in range(30)}
    return counts, denom, permutations, {'index_sha256':sha256(index_path), 'records_sha256':sha256(INPUT/'records.jsonl'), 'generations_sha256':generations_hash, 'bundle_sha256':sha256(bundle_path), 'generation_identity_sha256':sha256(identity_path), 'generation_model_sha256':identity['model'], 'source_manifest_sha256':identity['snapshot_manifest'], 'problems':hashes}

# 2026-11-08: 역양자화나 float 비교 없이 이동한 각 블록과 불변 텐서를 원본 바이트로 대조한다.
def verify(source, output, perm):
    a,b = GGUF(source), GGUF(output)
    if a.metadata_raw != b.metadata_raw or len(a.tensors) != len(b.tensors): raise ValueError('metadata/tensor 개수 변경')
    checked = 0
    with source.open('rb') as src, output.open('rb') as dst:
        for x,y in zip(a.tensors,b.tensors):
            if (x.name,x.shape,x.kind,x.nbytes)!=(y.name,y.shape,y.kind,y.nbytes): raise ValueError(f'tensor 구조 불일치: {x.name}')
            m=re.fullmatch(r'blk\.(\d+)\.(ffn_gate_inp\.weight|ffn_(?:gate|up|down|gate_up)_exps\.(?:weight|scale))',x.name)
            ids = perm[int(m[1])] if m else [0]
            unit = x.nbytes // 128 if m else x.nbytes
            for new,old in enumerate(ids):
                src.seek(a.data_offset+x.offset+old*unit); dst.seek(b.data_offset+y.offset+new*unit)
                remain=unit
                while remain:
                    n=min(remain,4*1024*1024)
                    if src.read(n)!=dst.read(n): raise ValueError(f'바이트 불일치: {x.name}, new={new}, old={old}')
                    remain-=n
            checked+=1
            print(f'verified tensor {checked}/{len(a.tensors)} {x.name}',flush=True)
    return {'tensor_count':checked, 'all_tensor_bytes_verified':True, 'metadata_raw_identical':True}

# 2026-11-09: 노트북과 CLI가 동일한 출처 검증 및 덮어쓰기 방지 절차를 공유한다.
def export(source=SOURCE, output=OUTPUT, manifest=MANIFEST):
    source, output, manifest = map(Path, (source, output, manifest))
    if output.exists() or manifest.exists(): raise FileExistsError('출력 또는 manifest가 이미 있습니다')
    counts, denom, perm, provenance=profile()
    print('profile verified; hashing source',flush=True)
    if sha256(source)!=SOURCE_HASH: raise ValueError('원본 모델 SHA-256 불일치')
    model=GGUF(source)
    if model.architecture() != (list(range(30)),128,8): raise ValueError('모델 아키텍처 불일치')
    print('source verified; writing model',flush=True)
    result=model.rewrite(output,permutation=perm)
    print('model written; verifying every tensor byte',flush=True)
    validation=verify(source,output,perm)
    document={'method':'descending first-100 decode selection count; ties original expert ID ascending', 'source_path':str(source), 'source_sha256':result['source_sha256'], 'output_path':str(output), 'output_sha256':result['sha256'], 'output_bytes':result['file_bytes'], 'profile':provenance, 'decode_events_per_layer':denom, 'layers': [{'layer':l, 'new_to_old':perm[l], 'old_to_new':[perm[l].index(i) for i in range(128)], 'sorted_counts':[counts[l][i] for i in perm[l]]} for l in range(30)], 'verification':validation, 'limitation':'Physical byte equivalence under permutation does not prove runtime logit or generation equivalence.'}
    manifest.parent.mkdir(parents=True,exist_ok=True)
    with manifest.open('x',encoding='utf-8') as f: json.dump(document,f,ensure_ascii=False,indent=2)
    print('complete',result['sha256'],flush=True)
    return document

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--source', type=Path, default=SOURCE); ap.add_argument('--output', type=Path, default=OUTPUT); ap.add_argument('--manifest',type=Path,default=MANIFEST); args=ap.parse_args()
    export(source=args.source, output=args.output, manifest=args.manifest)

if __name__ == '__main__': main()
