"""Fixed public Q5 metadata contract. No execution or admission authority.

Public bytes are authenticated before parsing. Actual source/grant/producer
admission must be supplied by a future independently reviewed role entry.
"""
import hashlib,json,math
from functools import lru_cache
from pathlib import Path
PUBLIC_PINS = {'profile.json': 'bee048004e3f62ed82e9c308841e9bed86dbc65b147374d4ec720db87492e916', 'test5.in': 'df67a2c4e59030d923b019c9d57a2eedd96d11ad1e4eb31bbbddee69d460c018', 'replay-report.json': '1207f2cd2328ddfc5a7e06f622579d956968382ffd581d59f769ef8909168cdd', 'checkpoint_plan.json': '5162049ffc45bf47917964ce76dcc04d8ef988737b598b7905348ee24f781acc'}
DEFAULT_DECODER_SHA256='bc978cb7412b55416f7ed634a689ae616a263b0f47fd9eef996ba244f2505db6'
KEYGEN_SOURCE_SHA256='742e6af9e1ce9025416fa2ef68ae037bca7e772d48fbbd6ca387cc79820119c8'
READY=False

def canonical(value):
    return (json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()

def parse(raw):
    if type(raw)is not bytes:raise TypeError('exact bytes required')
    def pairs(items):
        result={}
        for key,value in items:
            if key in result:raise ValueError('duplicate JSON key')
            result[key]=value
        return result
    return json.loads(raw,object_pairs_hook=pairs,parse_constant=lambda _:(_ for _ in ()).throw(ValueError('nonfinite JSON')))

@lru_cache(maxsize=1)
def _plan_bytes():
    root=Path(__file__).resolve().parent/'public';docs={}
    for name,pin in PUBLIC_PINS.items():
        path=root/name
        if path.is_symlink()or not path.is_file()or path.stat().st_size>512*1024:raise ValueError('fixed bounded public source required')
        raw=path.read_bytes()
        if hashlib.sha256(raw).hexdigest()!=pin:raise ValueError('fixed public source digest differs')
        docs[name]=parse(raw)
    profile=docs['profile.json'];case=docs['test5.in'];report=docs['replay-report.json']['diagnostic'];selection=docs['checkpoint_plan.json']['records']
    flat=dict(profile,**profile['ring']);flat.pop('ring')
    if(case['weight_seed']!=9105 or report['semantic_counts']!={'linear':22,'normalize':7,'normalize_heads':6,'rope':6,'attention':3,'polynomial':22}
       or len(selection)!=47 or len(report['trace'])!=216 or len(report['refresh_events'])!=19
       or report['primitive_counts']!={'add':270,'weighted':214,'multiply':226}
       or report['rotations']['union_count']!=100 or 262143 not in report['rotations']['union']):raise ValueError('fixed public Q5 inventory differs')
    return canonical(dict(profile=flat,raw_profile=profile,case=case,checkpoints=selection,full_trace=report['trace'],
                coefficients=report['coefficients'],required_rotation_indices=report['rotations']['union'],
                ideal_counts=report['primitive_counts'],semantic_counts=report['semantic_counts']))

def numerical_plan():return parse(_plan_bytes())

def positive(value):
    if type(value)is not float or not math.isfinite(value)or value<=0:raise ValueError('positive finite binary64 required')
    return value

def metadata_row(meta):
    if type(meta)is not dict or set(meta)!= {'level','scale','nsd'}:raise ValueError('closed actual metadata required')
    if type(meta['level'])is not int or not 0<=meta['level']<=49 or type(meta['nsd'])is not int or meta['nsd']!=1:raise ValueError('actual level/nsd required')
    positive(meta['scale']);return meta

def checkpoint_row(row,index):
    expected=numerical_plan()['checkpoints'][index]
    keys={'index','trace_index','kind','label','stack','level','scale','nsd'}
    if expected['kind'].startswith('refresh_'):keys.add('public_bound')
    if type(row)is not dict or set(row)!=keys:raise ValueError('closed checkpoint row required')
    required=dict(index=index,trace_index=expected['clear_trace_index'],kind=expected['kind'],label=expected['label'],stack=expected['stack'],level=expected['planned_level'],nsd=1)
    if expected['kind'].startswith('refresh_'):required['public_bound']=expected['public_bound']
    for key,value in required.items():
        if canonical(row[key])!=canonical(value)or type(row[key])is not type(value):raise ValueError('checkpoint contract differs: '+key)
    metadata_row({k:row[k]for k in ('level','scale','nsd')});return row

def all_checkpoints(rows):
    if type(rows)is not list or len(rows)!=47:raise ValueError('complete ordered47 required; partial server never admitted')
    for index,row in enumerate(rows):checkpoint_row(row,index)
    return rows
