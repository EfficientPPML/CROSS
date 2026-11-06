"""Pure Q5 role/terminal contracts; no constructor is execution authority.

Only setup-internal writer readback may use newly generated identities without
an independent terminal receipt. Server/client require an externally pinned
completed setup proof. The old checkpoint envelope remains an additional gate.
No proof instance or grant is supplied by this module.
"""
from dataclasses import dataclass
from pathlib import PurePosixPath
from q5_key_contract import (PublicProfile,ExpectedIds,authenticate_public,authenticate_private,
    canonical,parse,sha,hash_value,bindings,KEYGEN)
from q5_key_admission import bind_files
from q5_key_producer import same

LIMIT=1<<20
ROLES=('setup','server','client')

def object_bytes(raw,fields,limit=LIMIT):
    v=parse(raw,limit)
    if type(v)is not dict or set(v)!=set(fields)or canonical(v)!=raw:raise ValueError('closed canonical Q5 public envelope required')
    return v

def legacy_object_bytes(raw,fields,limit=LIMIT):
    # Original250 terminal envelopes use pretty canonical JSON. New intent and
    # phase schemas use compact JSON; never silently normalize pinned bytes.
    from q5_checkpoint_contract import canonical as original_canonical,parse as original_parse
    if type(raw)is not bytes or not 0<len(raw)<=limit:raise ValueError('bounded original terminal evidence required')
    value=original_parse(raw)
    if type(value)is not dict or set(value)!=set(fields)or original_canonical(value)!=raw:raise ValueError('original terminal envelope encoding differs')
    return value

def root_identity(value):
    if type(value)is not dict or set(value)!={'path','device','inode','mode','uid'}:raise ValueError('closed external original root required')
    p=value['path']
    if type(p)is not str or not p.startswith('/')or p.startswith('//')or str(PurePosixPath(p))!=p or '..'in PurePosixPath(p).parts or '\\'in p or '\x00'in p or p=='/':raise ValueError('canonical absolute root required')
    for n in ('device','inode','mode','uid'):
        if type(value[n])is not int or value[n]<(1 if n=='inode'else 0):raise ValueError('exact root identity integers required')
    if value['mode']>4095:raise ValueError('root mode out of range')
    return value

def output_identity(value):
    if type(value)is not dict or set(value)!={'path','parent'}:raise ValueError('fresh output and expected original parent required')
    parent=root_identity(value['parent']);p=PurePosixPath(value['path'])
    if type(value['path'])is not str or str(p)!=value['path']or not p.is_absolute()or p.parent!=PurePosixPath(parent['path'])or '..'in p.parts or '\\'in str(p)or '\x00'in str(p):raise ValueError('fresh child must match independently expected parent')
    return value

@dataclass(frozen=True)
class ExpectedSetupPins:
    raw:bytes
    def check(self):
        fields={'manifest','setup_report','setup_outer','setup_root_review','source_manifest','source_archive','grant',
            'context_id','model_id','public_manifest','private_manifest','assets_manifest','input_manifest',
            'setup_role_source','server_role_source','client_role_source'}
        v=object_bytes(self.raw,fields,1<<16)
        for x in v.values():hash_value(x)
        if v['setup_role_source']!=v['source_manifest']:raise ValueError('setup producer source differs')
        return v

@dataclass(frozen=True)
class SetupAdmission:
    profile:PublicProfile
    expected:ExpectedSetupPins
    manifest_bytes:bytes
    public_manifest_bytes:bytes
    assets_manifest_bytes:bytes
    input_manifest_bytes:bytes
    evidence:tuple
    def check(self):
        if type(self)is not SetupAdmission or type(self.profile)is not PublicProfile or type(self.expected)is not ExpectedSetupPins:
            raise TypeError('exact immutable completed setup admission required')
        self.profile.check();pins=self.expected.check()
        if type(self.evidence)is not tuple or tuple(n for n,_ in self.evidence)!=('setup_outer','setup_report','setup_root_review'):
            raise ValueError('complete ordered setup terminal evidence required')
        rows=dict(self.evidence);rows.update(manifest=self.manifest_bytes,public_manifest=self.public_manifest_bytes,assets_manifest=self.assets_manifest_bytes,input_manifest=self.input_manifest_bytes)
        for name,raw in rows.items():
            if type(raw)is not bytes or not 0<len(raw)<=LIMIT or sha(raw)!=pins[name]:raise ValueError('independent setup evidence pin differs')
        m=object_bytes(self.manifest_bytes,{'schema','bindings','context_id','model_id','roles','public_manifest_sha256','private_manifest_sha256','public_root','private_root','asset_root','assets_manifest_sha256','input_manifest_sha256','original_input','public_arrays','private_arrays','public_weights','input_ciphertexts','terminal_authority'})
        if m['schema']!='qwen-toy5-setup-manifest-v1'or m['terminal_authority']is not False:raise ValueError('setup manifest is not independent terminal authority')
        same(m['bindings'],bindings(),'fixed Q5 setup bindings differ')
        if m['context_id']!=pins['context_id']or m['model_id']!=pins['model_id']:raise ValueError('external setup context/model differs')
        same(m['roles'],{r:pins[r+'_role_source']for r in ROLES},'independently bound role sources differ')
        for name in ('public_manifest','private_manifest','assets_manifest','input_manifest'):
            if m[name+'_sha256']!=pins[name]:raise ValueError('setup declared file manifest differs')
        for name in ('public_root','private_root','asset_root'):root_identity(m[name])
        for name,count in (('public_arrays',203),('private_arrays',1),('public_weights',35),('input_ciphertexts',1)):
            if type(m[name])is not int or m[name]!=count:raise ValueError('fixed complete setup count differs')
        report=legacy_object_bytes(rows['setup_report'],{'schema','role','complete','passed','context_id','model_id','source_manifest_sha256','source_archive_sha256','grant_sha256','manifest_sha256','cleanup_completed','audit_denials'})
        outer=legacy_object_bytes(rows['setup_outer'],{'schema','role','report_sha256','source_manifest_sha256','source_archive_sha256','grant_sha256','exit_code','normal_return','reaped','process_group_absent','publication_completed'})
        if report['schema']!='qwen-toy5-role-report-v1'or outer['schema']!='qwen-toy5-role-outer-v1'or report['role']!='setup'or outer['role']!='setup':raise ValueError('exact setup report and outer schema required')
        if any(report[n]is not True for n in ('complete','passed','cleanup_completed'))or report['audit_denials']!=[]:
            raise ValueError('complete clean setup report required')
        if type(outer['exit_code'])is not int or outer['exit_code']!=0 or any(outer[n]is not True for n in ('normal_return','reaped','process_group_absent','publication_completed')):
            raise ValueError('independent normal0/reap/publication required')
        for record in(report,outer):
            for name,pin in (('source_manifest_sha256','source_manifest'),('source_archive_sha256','source_archive'),('grant_sha256','grant')):
                if record[name]!=pins[pin]:raise ValueError('setup source/archive/grant external join differs')
        if report['context_id']!=pins['context_id']or report['model_id']!=pins['model_id']or report['manifest_sha256']!=pins['manifest']or outer['report_sha256']!=pins['setup_report']:
            raise ValueError('setup terminal content joins differ')
        review=object_bytes(rows['setup_root_review'],{'schema','passed','manifest_sha256','setup_report_sha256','setup_outer_sha256','source_manifest_sha256','source_archive_sha256','grant_sha256','original_roots_verified','all_key_arrays_verified','original_input_verified','public_assets_verified','cleanup_and_exit_verified'})
        if review['schema']!='qwen-toy5-setup-root-review-v1'or any(review[n]is not True for n in ('passed','original_roots_verified','all_key_arrays_verified','original_input_verified','public_assets_verified','cleanup_and_exit_verified')):
            raise ValueError('independent completed setup root review required')
        for name,pin in (('manifest','manifest'),('setup_report','setup_report'),('setup_outer','setup_outer'),('source_manifest','source_manifest'),('source_archive','source_archive'),('grant','grant')):
            if review[name+'_sha256']!=pins[pin]:raise ValueError('independent setup root proof join differs')
        from q5_fixture_assets import AssetBinding,model_id
        from q5_checkpoint_contract import parse as legacy_parse,canonical as legacy_canonical,checkpoint_row
        if model_id(self.profile)!=m['model_id']:raise ValueError('original fixed public weight content identity differs')
        AssetBinding(self.profile,self.assets_manifest_bytes,pins['assets_manifest'],legacy_canonical(m['asset_root'])).check()
        inp=legacy_parse(self.input_manifest_bytes)
        if type(inp)is not dict or set(inp)!={'schema','bindings','original_input'}or inp['schema']!='qwen-toy5-input-manifest-v1'or inp['original_input']!=m['original_input']:
            raise ValueError('original input manifest join differs')
        root_identity(m['original_input']['root']);checkpoint_row(m['original_input']['artifact']['metadata'],0)
        expected_input=dict(context_id=m['context_id'],model_id=m['model_id'],setup_source_manifest_sha256=pins['source_manifest'],setup_grant_sha256=pins['grant'])
        same(inp['bindings'],expected_input,'original input producer bindings differ')
        self.key_files('server')
        return m
    def key_files(self,role,private_bytes=None):
        # No recursive full check: caller must complete check() before readers.
        pins=self.expected.check();expected=ExpectedIds(pins['context_id'],pins['public_manifest'],KEYGEN,pins['model_id'],pins['setup_role_source'],pins['client_role_source'],pins['private_manifest']if role=='client'else None)
        public=authenticate_public(self.public_manifest_bytes,self.profile,expected,role=role)
        private=authenticate_private(private_bytes,public)if role=='client'else None
        identity={n:getattr(expected,n)for n in ('context_id','model_id','producer_source_manifest_sha256','consumer_source_manifest_sha256')}
        return bind_files(public,canonical(identity),private)
    @property
    def execution_authorized(self):return False

@dataclass(frozen=True)
class ExpectedPhasePins:
    raw:bytes
    def check(self):
        fields={'role','grant_sha256','source_manifest_sha256','source_archive_sha256','dependency_policy_sha256','resource_policy_sha256','resource_review_sha256','setup_terminal_sha256','server_terminal_sha256'}
        v=object_bytes(self.raw,fields,1<<16)
        if v['role']not in ROLES:raise ValueError('one exact Q5 phase required')
        for name,value in v.items():
            if name=='role':continue
            if name=='setup_terminal_sha256'and v['role']=='setup':
                if value is not None:raise ValueError('setup cannot invent its post-generation terminal')
            elif name=='server_terminal_sha256'and v['role']!='client':
                if value is not None:raise ValueError('only client consumes a server terminal')
            else:hash_value(value)
        return v

@dataclass(frozen=True)
class PhaseAdmission:
    profile:PublicProfile
    expected:ExpectedPhasePins
    grant_bytes:bytes
    dependency_policy_bytes:bytes
    resource_policy_bytes:bytes
    resource_review_bytes:bytes
    setup:SetupAdmission|None=None
    server:object=None
    def check(self):
        if type(self)is not PhaseAdmission or type(self.profile)is not PublicProfile or type(self.expected)is not ExpectedPhasePins:raise TypeError('exact immutable Q5 phase required')
        self.profile.check();pins=self.expected.check()
        for raw,name in ((self.grant_bytes,'grant_sha256'),(self.dependency_policy_bytes,'dependency_policy_sha256'),(self.resource_policy_bytes,'resource_policy_sha256'),(self.resource_review_bytes,'resource_review_sha256')):
            if type(raw)is not bytes or not 0<len(raw)<=LIMIT or sha(raw)!=pins[name]:raise ValueError('independent phase policy/grant bytes differ')
        g=object_bytes(self.grant_bytes,{'schema','role','nonce','source_manifest_sha256','source_archive_sha256','bindings','model_id','roles','setup_terminal_sha256','server_terminal_sha256','dependency_policy_sha256','resource_policy_sha256','resource_review_sha256','borrowed_roots','outputs','actual_execution_enabled'})
        if g['schema']!='qwen-toy5-role-grant-v1'or g['role']!=pins['role']or g['actual_execution_enabled']is not False:
            raise PermissionError('only default-denied Q5 phase source preparation is defined')
        for name in ('source_manifest_sha256','source_archive_sha256','setup_terminal_sha256','server_terminal_sha256','dependency_policy_sha256','resource_policy_sha256','resource_review_sha256'):
            if g[name]!=pins[name]:raise ValueError('phase externally expected binding differs')
        same(g['bindings'],bindings(),'fixed Q5 phase profile/source differs');hash_value(g['model_id'])
        from q5_fixture_assets import model_id
        if g['model_id']!=model_id(self.profile):raise ValueError('original public fixed model content identity differs')
        if type(g['roles'])is not dict or set(g['roles'])!=set(ROLES):raise ValueError('three separate named role source bindings required')
        for value in g['roles'].values():hash_value(value)
        if g['roles'][g['role']]!=pins['source_manifest_sha256']:raise ValueError('current role source differs')
        if type(g['borrowed_roots'])is not dict or type(g['outputs'])is not dict:raise ValueError('closed role roots required')
        expected_borrowed=set()if g['role']=='setup'else {'public_keys','assets','input'}|({'private_keys','checkpoints'}if g['role']=='client'else set())
        expected_outputs={'public_keys','private_keys','assets','input','evidence'}if g['role']=='setup'else {'checkpoints','evidence'}if g['role']=='server'else {'evidence'}
        if set(g['borrowed_roots'])!=expected_borrowed or set(g['outputs'])!=expected_outputs:raise ValueError('role-specific borrowed/owned roots differ')
        for value in g['borrowed_roots'].values():root_identity(value)
        for value in g['outputs'].values():output_identity(value)
        paths=[PurePosixPath(v['path'])for v in [*g['borrowed_roots'].values(),*g['outputs'].values()]]
        if any(a==b or a in b.parents or b in a.parents for i,a in enumerate(paths)for b in paths[i+1:]):raise ValueError('role data roots must be disjoint')
        if g['role']=='setup':
            if self.setup is not None or self.server is not None:raise ValueError('setup cannot consume self-issued terminal readiness')
        else:
            if type(self.setup)is not SetupAdmission or sha(self.setup.manifest_bytes)!=pins['setup_terminal_sha256']:raise ValueError('independent completed setup required before key/body entry')
            m=self.setup.check();same(g['roles'],m['roles'],'same externally bound role sources required')
            if g['model_id']!=m['model_id']:raise ValueError('fixed setup model identity differs')
            for name,key in (('public_keys','public_root'),('assets','asset_root'),('input',None)):
                wanted=m['original_input']['root']if key is None else m[key]
                same(g['borrowed_roots'][name],wanted,'same admitted setup root required')
            if g['role']=='client':
                from q5_checkpoint_roles import PublicServer
                if type(self.server)is not PublicServer or sha(self.server.manifest_bytes)!=pins['server_terminal_sha256']:raise ValueError('independently completed server required before private entry')
                server=self.server.check()
                same(g['borrowed_roots']['private_keys'],m['private_root'],'original private key root differs')
                same(g['borrowed_roots']['checkpoints'],server['output_root'],'original completed checkpoint root differs')
                same(server['original_input'],m['original_input'],'same setup original input required')
                sb=server['bindings']
                if sb['context_id']!=m['context_id']or sb['model_id']!=m['model_id']or sb['source_manifest_sha256']!=g['roles']['server']:raise ValueError('separate server source/context/model differs')
                ep=self.setup.expected.check()
                if sb['setup_report_sha256']!=ep['setup_report']or sb['setup_outer_sha256']!=ep['setup_outer']:raise ValueError('server used a different setup terminal')
            elif self.server is not None:raise ValueError('server cannot self-issue its completion')
        return g
    @property
    def execution_authorized(self):return False
