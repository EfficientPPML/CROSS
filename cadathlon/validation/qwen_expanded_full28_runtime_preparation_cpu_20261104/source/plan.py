"""Public all28 decoder-tail example scale replay; no parameter bodies/HE."""
from pathlib import Path
import hashlib,json,math,sys
P=Path(__file__).resolve().parent;sys.path.insert(0,str(P/'checkout/cadathlon/common'))
from encrypted_qwen_rmsnorm import QwenRMSContract,rms_scale_plan
from qwen_silu_contract import QwenSiLUContract,silu_scale_plan,DELTA
from encrypted_scale_planning import drop_divisor,weighted_sum_scale,multiply_scale

def derive():
    m=(P/'inputs/candidate_manifest.json').read_bytes();c=(P/'inputs/certificates.json').read_bytes();sm=(P/'inputs/site_metadata.json').read_bytes()
    norm_rows={r['site']:r for r in json.loads((P/'inputs/join_plan.json').read_bytes())['normalizations']}
    parameter_bytes=(P/'inputs/decoder_parameters.json').read_bytes()
    assert hashlib.sha256(parameter_bytes).hexdigest()=='60c556376e42358f227a2978edb6b2addf59dc9e694b65e17265b15e813d65c3'
    matrices=json.loads(parameter_bytes)['matrices']
    q=next(x['q'] for x in json.loads((P/'inputs/profile_audit.json').read_bytes())['profiles'] if x['name']=='depth49_p60_d8')
    adjacent=(math.nextafter(DELTA,-math.inf),DELTA,math.nextafter(DELTA,math.inf));layers=[];examples=[]
    for layer in range(28):
        norm=QwenRMSContract(m,sm,f'{layer}.post.inverse_sqrt');silu=QwenSiLUContract(m,c,layer);nr=norm_rows[norm.site];maps={}
        for role,name,shape in [('o','self_attn.o_proj',(1024,2048)),('gate','mlp.gate_proj',(3072,1024)),('up','mlp.up_proj',(3072,1024)),('down','mlp.down_proj',(1024,3072))]:
            row=matrices[f'model.layers.{layer}.{name}.weight'];assert row['shape']==list(shape)
            maps[role]=dict(shape=row['shape'],sha256=row['sha256'])
        layers.append(dict(layer=layer,maps=maps,matrix_f64_bytes=88*1024**2,raw_post_bound=nr['raw_input_bound'],normalized_post_bound=nr['output_bound'],norm_output_level=norm.output_level,norm_events=len(norm.events),silu_events=silu.doublings,outer_events=2,mandatory_original_mask_levels=1))
        for attention_level in (20,22):
            for variant,source_scale in enumerate(adjacent):
                stages=[]
                def weighted(name,v):
                    out=dict(level=v['level']-1,scale=weighted_sum_scale([v['scale']],DELTA,drop_divisor(q,v['level']))['output_scale']);stages.append(dict(name=name,kind='weighted',**out));return out
                def add(name,x,y):
                    assert math.isclose(x['scale'],y['scale'],rel_tol=1e-12)
                    out=dict(level=min(x['level'],y['level']),scale=x['scale']);stages.append(dict(name=name,kind='add-left-scale',**out));return out
                def refresh(name,v,bound,index):
                    assert v['level']>=2
                    out=dict(level=25,scale=adjacent[index%3]);stages.append(dict(name=name,kind='conditional-refresh',input_level=v['level'],public_bound=bound,**out));return out
                source=dict(level=25,scale=source_scale);attended=dict(level=attention_level,scale=adjacent[(variant+1)%3])
                masked=weighted('original_residual_mask',source)
                residual=add('attention_residual',masked,weighted('o',attended))
                post=refresh('raw_post',residual,nr['raw_input_bound'],variant)
                np=rms_scale_plan(q,norm,input_scale=post['scale'],refresh_scales=tuple(adjacent[(variant+1+i)%3] for i in range(len(norm.events))))
                normalized=dict(level=np['output_level'],scale=np['output_scale']);stages.append(dict(name='post_rms',kind='frozen-original',**normalized))
                fresh=refresh('normalized_post',normalized,nr['output_bound'],variant+len(norm.events)+1)
                gate=weighted('gate',fresh);up=weighted('up',fresh)
                sp=silu_scale_plan(q,silu,input_scale=gate['scale'],refresh_scales=tuple(adjacent[(variant+i)%3] for i in range(silu.doublings)))
                act=dict(level=sp['output_level'],scale=sp['output_scale']);stages.append(dict(name='silu',kind='frozen-original',**act))
                level=min(act['level'],up['level']);product=dict(level=level-1,scale=multiply_scale(act['scale'],up['scale'],drop_divisor(q,level)));stages.append(dict(name='gated',kind='multiply',**product))
                output=add('final_residual',residual,weighted('down',product));assert output['level']==4 and output['scale']==masked['scale']
                examples.append(dict(layer=layer,attention_level=attention_level,scale_variant=variant,stages=stages,norm_plan=np,silu_plan=sp,output=output))
    return dict(passed=True,scope='168 sample public scale schedules, not exhaustive reachability/domain/CKKS proof; no tensors/model/HE',layers=layers,examples=examples,output_level=4,actual_keys='Use independently audited actual234-matrix key inventory; not provisional166-key shape union')
if __name__=='__main__':
    d=derive();(P/'report.json').write_text(json.dumps(d,indent=2,allow_nan=False)+'\n');print(dict(layers=len(d['layers']),examples=len(d['examples']),output_level=d['output_level']))
