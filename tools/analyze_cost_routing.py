"""104/80 모델의 서로 다른 cost 코드 구간을 예측한 Expert 기록을 비교한다."""
import csv,gzip,hashlib,json,math,re,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from moe_prune.gguf import GGUF,sha256

RUN104=ROOT/'runs/gemma4-ordered-e104-001/generation-001'
RUN80=ROOT/'runs/gemma4-ordered-e80-001/generation-maxgpu-001'
MAP104=ROOT/'runs/gemma4-ordered-e104-001/export-manifest.json'
MAP80=ROOT/'runs/gemma4-ordered-e80-001/export-manifest.json'
MODEL=ROOT/'models/gemma-4-26B-A4B-it-UD-IQ4_NL-lcb100-activity-ordered-e80.gguf'
DEST=RUN80/'analysis-cost-routing'
BAD=b'st.update(1, 0, n-1, new_l, right, -current_val_at_new_l)'

# 2026-11-09: 출력 토큰을 낸 직전 forward의 위치를 사용하며 첫 토큰은 prompt 마지막을 사용한다.
def predictor(prompt,index):
    if type(prompt) is not int or prompt<=0 or type(index) is not int or index<0:raise ValueError('invalid token position')
    return ('prefill',prompt-1) if index==0 else ('decode',prompt+index-1)

# 2026-11-09: UTF-8 바이트 오프셋을 보존해 멀티바이트 경계에서 토큰을 누락하지 않는다.
def pieces_and_ranges(ids,tokens,types):
    result=bytearray();ranges=[]
    for idx in ids:
        if type(idx) is not int or not 0<=idx<len(tokens):raise ValueError('bad token id')
        token=tokens[idx];kind=types[idx]
        if kind==1:piece=token.replace('▁',' ').encode('utf-8')
        elif kind==4:piece=token.encode('utf-8')
        elif kind==6 and re.fullmatch(r'<0x[0-9A-Fa-f]{2}>',token):piece=bytes([int(token[3:5],16)])
        else:raise ValueError(f'unsupported token kind: {kind}')
        start=len(result);result.extend(piece);ranges.append((start,len(result)))
    return bytes(result),ranges

def overlap_tokens(ranges,start,end):
    if start<0 or end<=start:raise ValueError('invalid byte span')
    return [i for i,(a,b) in enumerate(ranges) if a<end and b>start]

def final_python(data):
    blocks=list(re.finditer(rb'```python\s*\n(.*?)```',data,flags=re.DOTALL|re.IGNORECASE))
    if not blocks:raise ValueError('final Python code fence missing')
    return blocks[-1].start(1),blocks[-1].end(1)

def one_span(data,start,end,needle):
    part=data[start:end];found=[m.start() for m in re.finditer(re.escape(needle),part)]
    return [(start+i,start+i+len(needle)) for i in found]

def validate_mapping(manifest,key,experts,layers=30):
    if manifest.get('expert_count')!=experts or manifest.get('top_k')!=8 or len(manifest.get('layers',[]))!=layers:raise ValueError('mapping architecture mismatch')
    result=[]
    for layer,item in enumerate(manifest['layers']):
        ids=item.get(key)
        if item.get('layer')!=layer or not isinstance(ids,list) or len(ids)!=experts or any(type(x) is not int or not 0<=x<128 for x in ids) or len(set(ids))!=experts:raise ValueError('invalid layer mapping')
        result.append(ids)
    return result

def validate_vectors(ids,weights,norms,experts):
    if not all(isinstance(x,list) and len(x)==8 for x in (ids,weights,norms)):raise ValueError('invalid routing vector shape')
    if len(set(ids))!=8 or any(type(x) is not int or not 0<=x<experts for x in ids):raise ValueError('invalid expert IDs')
    if any(isinstance(x,bool) or not isinstance(x,(int,float)) or not math.isfinite(x) or x<0 for x in weights+norms):raise ValueError('invalid routing weights/norms')

def collect(run,answer,selection,experts):
    rel=answer['routing_path'].replace('\\','/')
    if rel!='routing/001.jsonl.gz':raise ValueError('unexpected routing path')
    path=(run/'routing/001.jsonl.gz').resolve()
    if not path.is_relative_to(run.resolve()) or sha256(path)!=answer['routing_integrity']['compressed_sha256']:raise ValueError('compressed routing hash mismatch')
    wanted={predictor(answer['prompt_tokens'],i) for values in selection.values() for i in values}
    captured={};total=0
    with gzip.open(path,'rb') as f:
        for raw in f:
            row=json.loads(raw)
            if row.get('type')!='routing':continue
            total+=1;key=(row.get('phase'),row.get('token_index'))
            if key not in wanted:continue
            layer=row.get('layer');ids=row.get('expert_ids');weights=row.get('weights');norms=row.get('weighted_output_norms')
            if type(layer) is not int or not 0<=layer<30:raise ValueError('invalid selected layer')
            validate_vectors(ids,weights,norms,experts)
            event_key=(key[0],key[1],layer)
            if event_key in captured:raise ValueError('duplicate selected routing event')
            captured[event_key]=row
    if total!=answer['routing_records']:raise ValueError('routing total mismatch')
    expected={(phase,pos,layer) for phase,pos in wanted for layer in range(30)}
    if set(captured)!=expected:raise ValueError('selected event coverage incomplete')
    return captured,total

def summarize(answer,selection,captured,mapping,experts):
    result={}
    for label,indices in selection.items():
        if not indices or indices!=sorted(set(indices)):raise ValueError('invalid selected token indices')
        if label!='wrong_lines_combined' and indices!=list(range(indices[0],indices[-1]+1)):raise ValueError('noncontiguous selected span')
        rows=[]
        for layer in range(30):
            counts=[0]*experts;gates=[0.]*experts;norms=[0.]*experts
            for i in indices:
                phase,pos=predictor(answer['prompt_tokens'],i);event=captured[(phase,pos,layer)]
                for expert,gate,norm in zip(event['expert_ids'],event['weights'],event['weighted_output_norms']):counts[expert]+=1;gates[expert]+=gate;norms[expert]+=norm
            for expert,count in enumerate(counts):
                if count:rows.append({'layer':layer,'expert_id':expert,'original_id':mapping[layer][expert],'count':count,'denominator':len(indices),'frequency':count/len(indices),'gate_sum':gates[expert],'norm_sum':norms[expert],'mean_gate':gates[expert]/count,'mean_weighted_norm':norms[expert]/count})
        result[label]={'output_token_start':indices[0],'output_token_end_inclusive':indices[-1],'tokens':len(indices),'token_indices':indices,'routing_events':30*len(indices),'experts':rows}
    return result

def main():
    if DEST.exists():raise FileExistsError('existing analysis preserved')
    answers=[json.loads((run/'answers/001.json').read_text(encoding='utf-8')) for run in (RUN104,RUN80)]
    tokenizer=GGUF(MODEL).metadata
    if tokenizer.get('tokenizer.ggml.model')!='gemma4':raise ValueError('unexpected tokenizer')
    mappings=[];model_sha=[]
    for path,key,experts,run in ((MAP104,'pruned_to_original',104,RUN104),(MAP80,'e80_to_original',80,RUN80)):
        manifest=json.loads(path.read_text(encoding='utf-8'));mappings.append(validate_mapping(manifest,key,experts));model_sha.append(manifest['output_sha256'])
        job=json.loads((run/'jobs/problem-001/identity.json').read_text(encoding='utf-8'))
        if job['model']['sha256']!=manifest['output_sha256']:raise ValueError('model and native job identity mismatch')
    decoded=[]
    for a in answers:
        data,ranges=pieces_and_ranges(a['token_ids'],tokenizer['tokenizer.ggml.tokens'],tokenizer['tokenizer.ggml.token_type'])
        if data!=a['text'].encode('utf-8'):raise ValueError('token bytes != saved output')
        decoded.append((data,ranges))
    data104,ranges104=decoded[0];data80,ranges80=decoded[1]
    l104,r104=final_python(data104);l80,r80=final_python(data80)
    bad=one_span(data80,l80,r80,BAD)
    if len(bad)!=2:raise ValueError(f'expected two target lines, observed {len(bad)}')
    returns=one_span(data104,l104,r104,b'return s - sum_nums')
    starts=one_span(data104,l104,r104,b'def get_cost(i, j):'); ends=one_span(data104,l104,r104,b'# Two-pointer')
    if len(returns)!=1 or len(starts)!=1 or len(ends)!=1 or not starts[0][0]<returns[0][0]<ends[0][0]:raise ValueError('reference function span mismatch')
    correct_minus=data104.index(b'-',*returns[0])
    sel104={'correct_return':overlap_tokens(ranges104,*returns[0]),'correct_minus':overlap_tokens(ranges104,correct_minus,correct_minus+1),'correct_cost_function':overlap_tokens(ranges104,starts[0][0],ends[0][0])}
    sel80={f'wrong_line_{i+1}':overlap_tokens(ranges80,*span) for i,span in enumerate(bad)}
    sel80['wrong_lines_combined']=sorted(set(sel80['wrong_line_1']+sel80['wrong_line_2']))
    for i,(start,end) in enumerate(bad,1):
        minus=data80.index(b'-current_val_at_new_l',start,end)
        sel80[f'wrong_minus_{i}']=overlap_tokens(ranges80,minus,minus+1)
    cap104,total104=collect(RUN104,answers[0],sel104,104);cap80,total80=collect(RUN80,answers[1],sel80,80)
    result104=summarize(answers[0],sel104,cap104,mappings[0],104);result80=summarize(answers[1],sel80,cap80,mappings[1],80)
    deleted={name:[row for row in x['experts'] if row['expert_id']>=80] for name,x in result104.items()}
    minus_events={}
    for model,label,selection,answer,captured,mapping in (('104','correct_minus',sel104,answers[0],cap104,mappings[0]),('80','wrong_minus_1',sel80,answers[1],cap80,mappings[1]),('80','wrong_minus_2',sel80,answers[1],cap80,mappings[1])):
        minus_events[label]=[]
        for token in selection[label]:
            phase,pos=predictor(answer['prompt_tokens'],token)
            for layer in range(30):
                event=captured[(phase,pos,layer)];minus_events[label].append({'model':model,'output_token_index':token,'predictor_phase':phase,'predictor_token_index':pos,'layer':layer,'expert_ids':event['expert_ids'],'original_ids':[mapping[layer][x] for x in event['expert_ids']],'weights':event['weights'],'weighted_output_norms':event['weighted_output_norms']})
    output={'tool_sha256':sha256(Path(__file__)),'answer_sha256':{'104':sha256(RUN104/'answers/001.json'),'80':sha256(RUN80/'answers/001.json')},'limitations':['Selection does not prove causation. The code algorithms, model weights and GPU placements differ.','Reference cost computation is semantic only; there is no one-to-one code span.','Routing position is the forward that predicts the output token.'],'token_alignment_verified':True,'models':{'104':{'model_sha256_recorded':model_sha[0],'manifest_sha256':sha256(MAP104),'routing_sha256':answers[0]['routing_integrity']['compressed_sha256'],'routing_records':total104,'spans':result104},'80':{'model_sha256_recorded':model_sha[1],'manifest_sha256':sha256(MAP80),'routing_sha256':answers[1]['routing_integrity']['compressed_sha256'],'routing_records':total80,'spans':result80}},'byte_spans':{'104':{'final_python':[l104,r104],'correct_return':list(returns[0]),'correct_minus':[correct_minus,correct_minus+1],'correct_cost_function':[starts[0][0],ends[0][0]]},'80':{'final_python':[l80,r80],'wrong_line_1':list(bad[0]),'wrong_line_2':list(bad[1])}},'deleted_experts_in_reference':deleted,'deleted_ranked':{name:sorted(rows,key=lambda x:(-x['count'],x['layer'],x['expert_id'])) for name,rows in deleted.items()},'wrong_minus_events':minus_events}
    DEST.mkdir(parents=True,exist_ok=False)
    (DEST/'analysis.json').write_text(json.dumps(output,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    with (DEST/'experts.csv').open('w',encoding='utf-8',newline='') as f:
        fields=['model','span','layer','expert_id','original_id','count','denominator','frequency','gate_sum','norm_sum','mean_gate','mean_weighted_norm'];writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for model,results in (('104',result104),('80',result80)):
            for name,part in results.items():
                for row in part['experts']:writer.writerow({'model':model,'span':name,**row})
    with (DEST/'wrong-minus-routing.csv').open('w',encoding='utf-8',newline='') as f:
        fields=['model','span','output_token_index','predictor_phase','predictor_token_index','layer','rank','expert_id','original_id','gate','weighted_output_norm'];writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for label,rows in minus_events.items():
            for row in rows:
                for rank,(expert,original,gate,norm) in enumerate(zip(row['expert_ids'],row['original_ids'],row['weights'],row['weighted_output_norms']),1):
                    writer.writerow({'model':row['model'],'span':label,'output_token_index':row['output_token_index'],'predictor_phase':row['predictor_phase'],'predictor_token_index':row['predictor_token_index'],'layer':row['layer'],'rank':rank,'expert_id':expert,'original_id':original,'gate':gate,'weighted_output_norm':norm})
    print('complete',DEST,flush=True)

if __name__=='__main__':main()
