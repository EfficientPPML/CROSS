"""Public full28 join contract only: no ciphertext or parameter-body reads."""
from dataclasses import dataclass,field
import hashlib,json
from encrypted_qwen_head_routing import QwenHeadLayout,MANIFEST,CONFIG
from encrypted_qwen_rms_refresh import typed
from encrypted_qwen_rmsnorm import QwenRMSContract
from encrypted_qwen_attention_join import QwenAttentionJoinContract,JOIN_PLAN
from encrypted_qwen_decoder_tail import QwenDecoderTailContract,PARAMETERS

KEY_PLAN='bac55debfe95752a16a9a2e849e66488f4afbb576959403a2af4698358dd32ca'
QKV='c707068e1a247f4ef14fda3493a3e2edf9dea2f11ec19e94531a64d80dbcf48b'
sha=lambda raw:hashlib.sha256(raw).hexdigest()

@dataclass(frozen=True)
class EntryRefresh:
    stage:str
    layer:int
    input_level:int
    bound:float
    output_level:int=25

@dataclass(frozen=True)
class VocabularyBlock:
    index:int
    start:int
    stop:int
    input_level:int
    output_level:int
    input_width:int=1024
    physical_row:int=0

@dataclass(frozen=True)
class FullQwenContract:
    manifest_bytes:bytes=field(repr=False)
    config_bytes:bytes=field(repr=False)
    site_bytes:bytes=field(repr=False)
    certificate_bytes:bytes=field(repr=False)
    continuation_bytes:bytes=field(repr=False)
    qkv_bytes:bytes=field(repr=False)
    decoder_bytes:bytes=field(repr=False)
    key_plan_bytes:bytes=field(repr=False)
    positions:tuple
    entry_events:tuple=field(init=False)
    vocabulary_blocks:tuple=field(init=False)
    final_gamma_sha256:str=field(init=False)
    final_norm_level:int=field(init=False)
    final_norm_degree:int=field(init=False)
    vocabulary_sha256:str=field(init=False)
    last_position:int=field(init=False)
    last_tile:int=field(init=False)
    last_local_row:int=field(init=False)
    required_keys:tuple=field(init=False)
    def __post_init__(self):
        for raw,pin in ((self.manifest_bytes,MANIFEST),(self.config_bytes,CONFIG),(self.continuation_bytes,JOIN_PLAN),
                        (self.qkv_bytes,QKV),(self.decoder_bytes,PARAMETERS),(self.key_plan_bytes,KEY_PLAN)):
            if type(raw)is not bytes or sha(raw)!=pin:raise ValueError('exact frozen full-model public inputs required')
        layout=QwenHeadLayout(self.positions)
        if type(self.positions)is not tuple:raise TypeError('exact original-position tuple required')
        plan=json.loads(self.continuation_bytes);norms={r['site']:r for r in plan['normalizations']}
        # Both public child constructors recheck original coefficients/counts,
        # complete typed event plans and all selected per-site RMS rules.
        for layer in range(28):
            QwenAttentionJoinContract(self.manifest_bytes,self.site_bytes,self.certificate_bytes,self.continuation_bytes,layer)
            QwenDecoderTailContract(self.manifest_bytes,self.site_bytes,self.certificate_bytes,self.continuation_bytes,layer,len(self.positions))
        final=QwenRMSContract(self.manifest_bytes,self.site_bytes,'final.inverse_sqrt')
        if final.method!='polynomial' or final.events or type(final.output_level)is not int or final.output_level<2 or norms['final.inverse_sqrt']['output_level']!=final.output_level:raise ValueError('authenticated final norm circuit differs')
        decoder=json.loads(self.decoder_bytes);head=decoder['matrices']['lm_head.weight'];embedding=decoder['matrices']['model.embed_tokens.weight']
        if head['shape']!=[151936,1024] or embedding['shape']!=head['shape'] or embedding['storage_alias_of']!='lm_head.weight' or any(embedding[n]!=head[n] for n in ('file','sha256','file_sha256','source_raw_sha256')):raise ValueError('authenticated tied full vocabulary required')
        keys=json.loads(self.key_plan_bytes)
        if keys['passed']is not True or keys['matrix_map_count']!=234 or keys['combined_key_count']!=166:raise ValueError('complete actual matrix-key plan required')
        events=tuple(EntryRefresh(f'layer{layer}.input',layer,4,float(norms[f'{layer}.input.inverse_sqrt']['raw_input_bound'])) for layer in range(1,28))
        events+=(EntryRefresh('final.input',28,3,float(norms['final.inverse_sqrt']['raw_input_bound'])),)
        blocks=tuple(VocabularyBlock(i,start,min(start+4096,151936),final.output_level,final.output_level-1) for i,start in enumerate(range(0,151936,4096)))
        last=len(self.positions)-1;shift=(last%16)*4096
        if shift and shift not in keys['combined_keys']:raise ValueError('last-row extraction requires an unplanned key')
        values=dict(entry_events=events,vocabulary_blocks=blocks,final_gamma_sha256=final.gamma_sha256,final_norm_level=final.output_level,final_norm_degree=final.degree,vocabulary_sha256=head['sha256'],
            last_position=self.positions[-1],last_tile=last//16,last_local_row=last%16,required_keys=tuple(keys['combined_keys']))
        for name,value in values.items():object.__setattr__(self,name,value)

def state(c):
    def fields(value):return type(value),tuple(typed(getattr(value,n,None)) for n in value.__dataclass_fields__)
    return (tuple(typed(getattr(c,n,None)) for n in ('manifest_bytes','config_bytes','site_bytes','certificate_bytes','continuation_bytes','qkv_bytes','decoder_bytes','key_plan_bytes','positions',
                'final_gamma_sha256','final_norm_level','final_norm_degree','vocabulary_sha256','last_position','last_tile','last_local_row','required_keys')),
        type(c.entry_events),tuple(fields(e) for e in c.entry_events),type(c.vocabulary_blocks),tuple(fields(b) for b in c.vocabulary_blocks))
def verify(c):
    if type(c)is not FullQwenContract:raise TypeError('exact full-model contract required')
    expected=FullQwenContract(c.manifest_bytes,c.config_bytes,c.site_bytes,c.certificate_bytes,c.continuation_bytes,c.qkv_bytes,c.decoder_bytes,c.key_plan_bytes,c.positions)
    if state(c)!=state(expected):raise ValueError('typed full-model contract changed')
    return state(c)
