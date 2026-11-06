"""Emit proposed closed JSON schemas; no validation implementation or execution."""
from pathlib import Path
import json
R=Path(__file__).resolve().parent
J=lambda n:json.loads((R/n).read_bytes())
def save(n,v):(R/n).write_text(json.dumps(v,sort_keys=True,indent=2)+'\n')
def obj(properties):return {'type':'object','properties':properties,'required':list(properties),'additionalProperties':False}
sha={'type':'string','pattern':'^[0-9a-f]{64}$'}
positive={'type':'integer','minimum':1};nonnegative={'type':'integer','minimum':0}
manifest=J('unadmitted_aggregate_manifest.json')
# This first fixed-profile schema deliberately admits exactly the public-derived
# manifest. Supporting another export/candidate requires a reviewed successor.
aggregate=obj({k:{'const':v}for k,v in manifest.items()})
aggregate.update({'$schema':'https://json-schema.org/draft/2020-12/schema','title':'Fixed public-derived Qwen aggregate manifest (unadmitted content)','description':'Canonical compact sorted JSON plus trailing newline is mandatory. Recompute model_id from content and catalog_sha256 from catalog; schema matching alone grants no body access.'})
save('aggregate_manifest.schema.json',aggregate)
source_map={'type':'object','minProperties':1,'maxProperties':512,'propertyNames':{'type':'string','minLength':1,'maxLength':512},'additionalProperties':sha}
path={'type':'string','pattern':'^/','minLength':2,'maxLength':4096}
root=obj({'path':path,'device':nonnegative,'inode':positive,'mode':{'type':'integer','minimum':0,'maximum':4095}})
state=obj({'device':nonnegative,'inode':positive,'size':positive,'mtime_ns':nonnegative,'ctime_ns':nonnegative,'mode':{'type':'integer','minimum':0,'maximum':4095}})
file=obj({'family':{'enum':['qkv','decoder','gamma']},'name':{'type':'string','minLength':1,'maxLength':255,'pattern':'^[^/\\\\\u0000]+$'},'state':state,'file_sha256':sha,'raw_sha256':{'anyOf':[sha,{'type':'null'}]}})
store=obj({'roots':obj({'qkv':root,'decoder':root,'gamma':root}),'files':{'type':'array','minItems':198,'maxItems':198,'items':file}})
report=obj({
 'schema':{'const':'qwen-model-store-validation-v1'},'mode':{'const':'reuse_existing_exports'},
 'passed':{'const':True},'complete':{'const':True},'error_type':{'type':'null'},
 'model_id':{'const':manifest['model_id']},'aggregate_manifest_sha256':sha,'catalog_sha256':{'const':manifest['catalog_sha256']},
 'execution_grant_sha256':sha,'source_manifest_sha256':sha,'source_archive_sha256':sha,'source_hashes_before':source_map,'source_hashes_after':source_map,
 'process':obj({'pid':positive,'pgid':positive}),
 'stores':store,
 'counts':obj({'logical_matrices':{'const':198},'physical_matrices':{'const':197},'gamma_vectors':{'const':113},'physical_matrix_raw_bytes':{'const':4767875072},'gamma_raw_bytes':{'const':524288}}),
 'checks':obj({k:{'const':True}for k in ('public_provenance_authenticated','complete_catalog_rederived','all_matrix_headers_extents_hashes_finite','all_gamma_members_headers_hashes_finite','original_held_identity_unchanged','readback_completed','source_postcheck_completed','cleanup_completed','signals_restored','borrowed_stores_retained')}),
 'scope':obj({'parameter_body_reads_performed':{'const':True},'new_export_performed':{'const':False},'model_forward':{'const':False},'backend_context_constructed':{'const':False},'keys_read':{'const':False},'ciphertexts_read':{'const':False},'token_inputs_read':{'const':False},'coefficient_selection':{'const':False},'scientific_calibration_pass_claimed':{'const':False}}),
 'audit_denials':{'const':[]}})
outer=obj({'schema':{'const':'qwen-model-store-validation-outer-v1'},'report_sha256':sha,'aggregate_manifest_sha256':sha,'model_id':{'const':manifest['model_id']},'source_manifest_sha256':sha,'source_archive_sha256':sha,'execution_grant_sha256':sha,'pid':positive,'pgid':positive,'exit_code':{'const':0},'normal_return':{'const':True},'reaped':{'const':True},'process_group_absent':{'const':True}})
review=obj({'schema':{'const':'qwen-model-stage-root-admission-v1'},'passed':{'const':True},'model_id':{'const':manifest['model_id']},'aggregate_manifest_sha256':sha,'catalog_sha256':{'const':manifest['catalog_sha256']},'validation_report_sha256':sha,'validation_outer_sha256':sha,'source_manifest_sha256':sha,'source_archive_sha256':sha,'execution_grant_sha256':sha,'current_store_binding_sha256':sha,'all_required_links_verified':{'const':True},'actual_server_execution_authorized':{'const':False}})
expected=obj({'schema':{'const':'qwen-model-expected-pins-v1'},'model_id':{'const':manifest['model_id']},'aggregate_manifest_sha256':sha,'catalog_sha256':{'const':manifest['catalog_sha256']},'producer_source_manifest_sha256':sha,'producer_source_archive_sha256':sha,'producer_source_map':source_map,'execution_grant_sha256':sha,'validation_report_sha256':sha,'validation_outer_sha256':sha,'root_review_sha256':sha,'consumer_source_manifest_sha256':sha})
admission=obj({'expected_pins':expected,'validation_report':report,'validation_outer':outer,'root_review':review})
admission.update({'$schema':'https://json-schema.org/draft/2020-12/schema','title':'Proposed positive Qwen model-stage admission contract','description':'expected_pins is a caller-supplied trusted argument, never read from untrusted evidence. This schema documents separately bounded byte arguments; an aggregate serialized bundle is not an authority. Cross-document digests/identities and canonical forms must be checked by the future pure implementation. Failed/incomplete reports are preserved but rejected by this positive admission schema.'})
save('admission.schema.json',admission)
