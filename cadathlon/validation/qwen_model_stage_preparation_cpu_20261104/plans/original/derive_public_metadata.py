"""Public-metadata derivation only; never opens any descriptor's body path."""
from pathlib import Path
import ast,hashlib,json,math
ROOT=Path('/tmp/qwen-organizer-file-engine-integration-v2-20261104')
PUBLIC=ROOT/'adapter238/parent208'
OUT=Path(__file__).resolve().parent

def canonical(value):
    return (json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
def sha(raw):return hashlib.sha256(raw).hexdigest()
def load(path):return json.loads(path.read_bytes())
def digest(value):return sha(canonical(value))
def constants(path):
    result={}
    for node in ast.parse(path.read_text()).body:
        if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name):
            try:result[node.targets[0].id]=ast.literal_eval(node.value)
            except (ValueError,TypeError):pass
    return result

def derive():
    pins=constants(ROOT/'public_source_pins.py')
    values={}
    for name,expected in pins['PUBLIC_PINS'].items():
        raw=(PUBLIC/name).read_bytes()
        if sha(raw)!=expected:raise ValueError('public source pin changed: '+name)
        values[name]=json.loads(raw)
    q=values['qkv-export-v1/parameter_manifest.json'];d=values['inputs/decoder_parameters.json'];g=values['inputs/gamma_preparation_report.json']
    expected={}
    for layer in range(28):
        for role,width in (('q',2048),('k',1024),('v',1024)):
            expected[f'model.layers.{layer}.self_attn.{role}_proj.weight']=('qkv',[width,1024])
        for role,shape in (('self_attn.o_proj',[1024,2048]),('mlp.gate_proj',[3072,1024]),('mlp.up_proj',[3072,1024]),('mlp.down_proj',[1024,3072])):
            expected[f'model.layers.{layer}.{role}.weight']=('decoder',shape)
    for name in ('lm_head.weight','model.embed_tokens.weight'):expected[name]=('decoder',[151936,1024])
    matrices={};content_matrices={};physical={}
    for family,manifest in (('qkv',q),('decoder',d)):
        for name,row in manifest['matrices'].items():
            if name in matrices or name not in expected or expected[name]!=(family,row['shape']):raise ValueError('matrix inventory/shape differs')
            if row['name']!=name or row['source_dtype']!='BF16' or row['promotion_exact'] is not True or row['dtype']not in ('float64','<f8'):raise ValueError('promotion descriptor differs')
            matrices[name]={'family':family,'descriptor':row}
            key=(family,row['file'])
            previous=physical.setdefault(key,row)
            for field in ('shape','sha256','file_sha256','source_raw_sha256'):
                if previous[field]!=row[field]:raise ValueError('physical alias differs')
            content_matrices[name]={'shape':row['shape'],'dtype':'<f8','raw_sha256':row['sha256'],'source_dtype':'BF16','source_raw_sha256':row['source_raw_sha256'],'storage_alias_of':row.get('storage_alias_of')}
    if set(matrices)!=set(expected)or len(physical)!=197:raise ValueError('complete matrix inventory required')
    head=d['matrices']['lm_head.weight'];embedding=d['matrices']['model.embed_tokens.weight']
    if head['storage_alias_of'] is not None or embedding['storage_alias_of']!='lm_head.weight' or embedding['exact_bf16_alias_verified'] is not True:raise ValueError('tied alias differs')
    for field in ('file','shape','dtype','sha256','file_sha256','source_raw_sha256'):
        if embedding[field]!=head[field]:raise ValueError('tied descriptor differs')
    names={f'{i}.{role}.inverse_sqrt'for i in range(28)for role in ('input','q','k','post')}|{'final.inverse_sqrt'}
    if set(g['arrays'])!=names:raise ValueError('gamma inventory differs')
    content_gammas={}
    for name,row in g['arrays'].items():
        if row['shape']!=[128 if name.split('.')[1]in ('q','k')else 1024]or row['dtype']!='float64':raise ValueError('gamma shape/type differs')
        content_gammas[name]={'shape':row['shape'],'dtype':'<f8','raw_sha256':row['sha256']}
    if q['checkpoint_sha256']!=d['checkpoint_sha256']or q['config_sha256']!=d['config_sha256']:raise ValueError('export model identity differs')
    if g['asset_sha256_before']!=g['asset_sha256_after']or g['asset_sha256_before']!={'config.json':q['config_sha256'],'model.safetensors':q['checkpoint_sha256']}:raise ValueError('gamma asset identity differs')
    provenance={'qkv_metadata_sha256':pins['PUBLIC_PINS']['qkv-export-v1/parameter_manifest.json'],'decoder_metadata_sha256':pins['PUBLIC_PINS']['inputs/decoder_parameters.json'],'gamma_metadata_sha256':pins['PUBLIC_PINS']['inputs/gamma_preparation_report.json'],'historical_qkv_identity':{k:q[k]for k in ('candidate_sha256','checkpoint_sha256','config_sha256')},'historical_decoder_identity':{k:d[k]for k in ('candidate_sha256','checkpoint_sha256','config_sha256')},'gamma_asset_sha256':g['asset_sha256_before']}
    # This is a JSON representation of the four selected ParameterCatalog fields.
    catalog={'matrices':matrices,'gammas':g['arrays'],'gamma_container_sha256':g['fixture_npz_sha256'],'provenance':provenance}
    content={'schema':'qwen-learned-model-content-v1','checkpoint_sha256':q['checkpoint_sha256'],'config_sha256':q['config_sha256'],'matrices':content_matrices,'gammas':content_gammas,'logical_matrices':198,'physical_matrices':197,'gamma_vectors':113}
    old=load(Path('/tmp/qwen-parameter-provenance-security-review-20261104/review.json'))['pins']
    roles={'qkv':{'manifest':'qkv_manifest','report':'qkv_export_report','source_archive':'qkv_export_sources'},'decoder':{'manifest':'decoder_manifest','parent_report':'decoder_parent','worker_report':'decoder_worker','source_archive':'decoder_sources','outer_receipt':'decoder_outer','root_review':'decoder_root_readback'},'gamma':{'report':'gamma_report','source_archive':'gamma_sources'}}
    historical={family:{role:old[key]['sha256']for role,key in rows.items()}for family,rows in roles.items()}
    manifest={'schema':'qwen-full28-model-aggregate-v1','model_id':digest(content),'content':content,'catalog':catalog,'catalog_sha256':digest(catalog),'public_source_sha256':pins['PUBLIC_PINS'],'historical_provenance':historical,'application_binding':{'candidate_sha256':pins['PUBLIC_PINS']['inputs/candidate_manifest.json'],'full_owner_source_manifest_sha256':pins['PARENT_MANIFEST_SHA256'],'candidate_bridge_sha256':pins['PUBLIC_PINS']['inputs/candidate_bridge.json']}}
    for name,obj in (('unadmitted_content.json',content),('unadmitted_catalog.json',catalog),('unadmitted_aggregate_manifest.json',manifest)):(OUT/name).write_bytes(canonical(obj))
    status={'schema':'qwen-model-public-derivation-v1','scope':'Public metadata only; identities are unadmitted until actual current-store validation and independent terminal admission.','model_id':digest(content),'catalog_sha256':digest(catalog),'aggregate_manifest_sha256':digest(manifest),'physical_matrices':197,'logical_matrices':198,'gamma_vectors':113,'physical_matrix_raw_bytes':sum(math.prod(row['shape'])*8 for row in physical.values()),'gamma_raw_bytes':sum(math.prod(row['shape'])*8 for row in g['arrays'].values()),'parameter_body_reads':0,'runtime_imports':0,'tests_run':0,'actual_authorization':False,'body_integrity_admitted':False,'producer_source_and_process_pins_in_model_id':False}
    (OUT/'derivation.json').write_bytes(canonical(status));return status
if __name__=='__main__':print(json.dumps(derive(),sort_keys=True))
