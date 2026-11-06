"""Public full28 level/interface/scale plan; no model or tensor bodies."""
from pathlib import Path
import hashlib,json,math,sys
P=Path(__file__).resolve().parent
sys.path.insert(0,str(P/'checkout/cadathlon/common'))
from full_contract import FullQwenContract,KEY_PLAN
from encrypted_qwen_rmsnorm import QwenRMSContract,rms_scale_plan
from encrypted_qwen_attention_join import QwenAttentionJoinContract
from encrypted_qwen_decoder_tail import QwenDecoderTailContract
from encrypted_scale_planning import weighted_sum_scale,drop_divisor

def contract(positions):
    names=('inputs/candidate_manifest.json','inputs/config.json','inputs/site_metadata.json','inputs/certificates.json',
           'inputs/join_plan.json','qkv-export-v1/parameter_manifest.json','inputs/decoder_parameters.json','inputs/actual_key_plan.json')
    return FullQwenContract(*( (P/n).read_bytes() for n in names),positions)

def derive():
    # Authenticate inherited public bytes before parsing any dependent plan.
    for name,h in json.loads((P/'predecessor87.json').read_bytes())['sources'].items():
        if hashlib.sha256((P/'before87'/name).read_bytes()).hexdigest()!=h:raise ValueError('preserved predecessor changed: '+name)
    base=contract((0,));manifest=json.loads(base.manifest_bytes);site=json.loads(base.site_bytes)
    final=QwenRMSContract(base.manifest_bytes,base.site_bytes,'final.inverse_sqrt');head_level=base.vocabulary_blocks[0].output_level
    profile=next(r for r in json.loads((P/'inputs/profile_audit.json').read_bytes())['profiles'] if r['name']=='depth49_p60_d8')
    delta=profile['canonical_scale'];q=tuple(profile['q']);keys=json.loads(base.key_plan_bytes)
    layers=[]
    for layer in range(28):
        attention=QwenAttentionJoinContract(base.manifest_bytes,base.site_bytes,base.certificate_bytes,base.continuation_bytes,layer)
        tail=QwenDecoderTailContract(base.manifest_bytes,base.site_bytes,base.certificate_bytes,base.continuation_bytes,layer,8)
        norms=[site[f'{layer}.{kind}.inverse_sqrt'] for kind in ('input','q','k','post')]
        silu=manifest['nonlinear'][f'{layer}.silu'];doublings=silu.get('doublings',0)
        layers.append(dict(layer=layer,input_level=25,attention_output_level_multi=20,attention_output_level_single=22,
            original_residual_mask=[25,24],tail_output_level=4,
            attention_outer_events=[dict(stage=e.stage,input_level=e.input_level,bound=e.public_bound,output_level=e.output_level) for e in attention.outer_events],
            tail_outer_events=[dict(stage=e.stage,input_level=e.input_level,bound=e.public_bound,output_level=e.output_level) for e in tail.outer_events],
            rms_inverse_refreshes_per_tile=sum(len(n['refresh_events']) for n in norms),silu_refreshes_per_tile=doublings,
            causal_squarings=manifest['softmax'][str(layer)]['squarings']))
    workloads=[]
    for sequence in range(1,33):
        c=contract(tuple(3*i for i in range(sequence)));tiles=(sequence+15)//16
        inverse=tiles*sum(r['rms_inverse_refreshes_per_tile'] for r in layers)
        silu=tiles*sum(r['silu_refreshes_per_tile'] for r in layers)
        attention_outer=tiles*sum(len(r['attention_outer_events']) for r in layers)
        tail_outer=56*tiles
        # Each literal nontrivial query uses three initial events and two per
        # original squaring round. K1 remains the direct V0 branch.
        softmax=(sequence-1)*sum(3+2*r['causal_squarings'] for r in layers)
        continuation=27*tiles+1
        workloads.append(dict(sequence=sequence,tiles=tiles,original_positions=list(c.positions),last_position=c.last_position,
            last_tile=c.last_tile,last_local_row=c.last_local_row,last_row_rotation=c.last_local_row*4096,
            refresh_counts=dict(rms_inverse=inverse,silu=silu,attention_outer=attention_outer,tail_outer=tail_outer,interlayer=27*tiles,final_input=1,softmax=softmax,
                total=inverse+silu+attention_outer+tail_outer+continuation+softmax),
            entry_events=[dict(stage=e.stage,layer=e.layer,input_level=e.input_level,bound=e.bound,output_level=e.output_level) for e in c.entry_events],
            final_stages=[dict(name='select_last_hidden',input_level=4,output_level=3),dict(name='row_to_zero',input_level=3,output_level=3),
                dict(name='final_raw_refresh',input_level=3,output_level=25,bound=c.entry_events[-1].bound),dict(name='final_rms',input_level=25,output_level=c.final_norm_level),dict(name='all38_vocabulary_maps',input_level=c.final_norm_level,output_level=c.vocabulary_blocks[0].output_level)],
            layer_matrix_rotations=tiles*keys['all_layer_linear_rotation_calls_per_row_tile'],vocabulary_rotations=keys['vocabulary_rotation_calls_last_row'],
            final_select_rotations=int(c.last_local_row!=0),extra_final_select_weighted_stages=1,output_ciphertexts=38))
    scales=[]
    adjacent=(math.nextafter(delta,-math.inf),delta,math.nextafter(delta,math.inf))
    for source in adjacent:
        selected=weighted_sum_scale([source],delta,drop_divisor(q,4))['output_scale']
        for refreshed in adjacent:
            norm=rms_scale_plan(q,final,input_scale=refreshed)
            output=weighted_sum_scale([norm['output_scale']],delta,drop_divisor(q,final.output_level))['output_scale']
            scales.append(dict(source_L4_scale=source,masked_L3_scale=selected,rotation_preserves_scale=selected,
                actual_refresh_L25_scale=refreshed,actual_final_norm_level=final.output_level,actual_final_norm_scale=norm['output_scale'],actual_all38_level=head_level,actual_all38_scale=output,
                final_norm_plan=norm))
    result=dict(passed=True,scope='Public schedule/interface only; no parameter bodies, model/reference execution, backend, keygen or HE.',
        candidate_sha256=hashlib.sha256(base.manifest_bytes).hexdigest(),actual_key_plan_sha256=KEY_PLAN,
        layers=layers,workloads=workloads,final_scale_examples=scales,
        final_norm=dict(site='final.inverse_sqrt',domain=list(final.domain),degree=final.degree,method='balanced-chebyshev',input_level=25,output_level=final.output_level,
            gamma_sha256=final.gamma_sha256,raw_input_bound=base.entry_events[-1].bound,output_bound=next(r['output_bound'] for r in json.loads(base.continuation_bytes)['normalizations'] if r['site']==final.site),inverse_refresh_count=len(final.events)),
        vocabulary=dict(tied_embedding_sha256=base.vocabulary_sha256,full_shape=[151936,1024],blocks=[vars(b) for b in base.vocabulary_blocks],
            output_count=38,output_level=head_level,all_logits=151936,last_block_width=384,only_last_original_position=True,physical_output_row=0),
        resources=dict(rotation_keys=166,rotation_indices=list(base.required_keys),raw_rotation_store_bytes=keys['raw_rotation_keys_bytes'],
            raw_rotation_pair_bytes=keys['raw_rotation_pair_bytes'],final_ciphertext_bytes=2*131072*(2*(head_level+1))*4,all38_final_ciphertext_bytes=38*2*131072*(2*(head_level+1))*4,
            full_promoted_unique_parameter_bytes=28*120*1024**2+151936*1024*8,largest_streamed_vocabulary_block_bytes=4096*1024*8,
            exclusions=['Relin/public/secret keys, gamma arrays, controls/compilers, temporary and duplicate host/device buffers.','Storage is not peak resident memory; no128GiB or latency budget pass.']),
        execution_authorized=False,scientific_calibration_verified=False,first_pass_freeze_sha256=None,
        reference=dict(frozen_arithmetic_sha256='9689b843b4ec2d0a0ef724dbdca7a516e9a4f23da83f83af5571da13380e9a6f',
            input='Only original embeddings; public parameter dictionary and exact expanded candidate/config. Actual inputs remain unopened.',
            outputs='All28 block outputs,421 operator labels and final151936 logits from unchanged approximate_forward.',
            independent_hook='Separate dense NumPy Chebval/original-Newton layer loop; starts original embeddings, never supplied native or owner intermediates.',
            planned_fixtures=['Deferred until independent first complete calibration PASS/freeze and a separate execution grant; row0 remains separate post-freeze regression.'],
            executed=False),
        limitations=['Refresh/domain bounds remain conditional and exclude CKKS/floating/refresh perturbation.','The broad public score bounds still do not establish868 frozen centered-domain implications.',
            'No wholegraph ideal or native/HE numerical pass is implied.','Actual returned refresh scales must be replayed; nine examples are not all possible scales.','S1..32/batch1 covers current prefill and repeated-prefill decode geometry; file-stage privacy/lifecycle and benchmark gates remain separate.'])
    if any(n in sys.modules for n in ('jax','jaxlib','torch','transformers','safetensors')):raise RuntimeError('public proof imported forbidden backend/model module')
    return result

if __name__=='__main__':
    report=derive();(P/'public_plan.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(dict(layers=len(report['layers']),workloads=len(report['workloads']),keys=166,final_level=report['vocabulary']['output_level'],S32_refreshes=report['workloads'][-1]['refresh_counts']['total']))
