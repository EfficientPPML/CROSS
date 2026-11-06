import ast,hashlib,json,pathlib
R=pathlib.Path('/tmp/qwen-server-snapshot-runtime-v2-20261104');O=pathlib.Path(__file__).parent
H=lambda b:hashlib.sha256(b).hexdigest()
class Surface(ast.NodeVisitor):
    def __init__(self):self.calls=[];self.numeric=[]
    def visit_FunctionDef(self,n):
        for d in n.decorator_list:self.visit(d)
        for d in n.args.defaults+n.args.kw_defaults:
            if d is not None:self.visit(d)
        for a in n.args.posonlyargs+n.args.args+n.args.kwonlyargs:
            if a.annotation:self.visit(a.annotation)
        if n.returns:self.visit(n.returns)
    visit_AsyncFunctionDef=visit_FunctionDef
    def visit_Lambda(self,n):
        for d in n.args.defaults+n.args.kw_defaults:
            if d is not None:self.visit(d)
    def visit_Call(self,n):self.calls.append(ast.unparse(n.func));self.generic_visit(n)
    def visit_Attribute(self,n):
        base=n
        while isinstance(base,ast.Attribute):base=base.value
        if isinstance(base,ast.Name)and base.id in('np','numpy','jax','jnp'):self.numeric.append(ast.unparse(n))
        self.generic_visit(n)
selected=['qwen_server_backend.py','qwen_key_readers.py','key_descriptor_io.py','qwen_public_parameters.py','qwen_backend_composition.py','reused_client_io.py','qwen_key_adapter.py','qwen_key_contract.py','qwen_file_contract.py','model_admission.py','qwen_finalizer.py','cleanup_origin.py','runtime/common/encrypted_ntt.py','runtime/common/encrypted_packed_qwen.py','runtime/jaxite_word/__init__.py']
refs={n:dict(path=str(R/n),bytes=len((R/n).read_bytes()),sha256=H((R/n).read_bytes()))for n in selected}
assert refs['qwen_server_backend.py']['sha256']=='b7a3ff42847a9daf985d02fd71c5c9a3cd9e23a7b07f434b030b1ccce4129cac'
surface={}
for n in('qwen_key_readers.py','key_descriptor_io.py','qwen_public_parameters.py','qwen_backend_composition.py','reused_client_io.py'):
    v=Surface();v.visit(ast.parse((R/n).read_bytes()));assert v.numeric==[]
    surface[n]=dict(definition_surface_calls=v.calls,numeric_attributes_evaluated_at_definition=v.numeric)
tree=ast.parse((R/'qwen_backend_composition.py').read_bytes());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='_runtime_dependencies')
imports=[n.module for n in fn.body if isinstance(n,ast.ImportFrom)]
assert imports==['encrypted_ntt','encrypted_packed_qwen','encrypted_chained_streaming_linear','encrypted_streaming_linear','encrypted_refresh','encrypted_qwen_full_model','full_contract','support']
report=dict(schema='qwen-server-constructor-import-source-review-v1',source_root=str(R),source_references=refs,definition_surface=surface,callgraph=[['ServerBackend.__init__','qwen_key_readers','numpy / key_descriptor_io / cleanup_origin / qwen_key_contract'],['ServerBackend.__init__','qwen_backend_composition','qwen_public_parameters','numpy / reused_client_io / pure model and file helpers'],['qwen_backend_composition._runtime_dependencies',*imports],['encrypted_ntt','jax / jax.numpy / numpy'],['encrypted_packed_qwen','jaxite_word.bsgs','jaxite_word.__init__','eager nn / packing and sys.path/alias mutation']],minimum_recording_seams=['Use actual selected module definitions with their true file/spec origins; an explicitly source-owned empty numpy ModuleType suffices for the inspected reader/parameter/composition definition surfaces.','Constructor-only ServerBackend retains jax but calls no jax API; a disclosed owned empty JAX namespace suffices for that scope.','Replace only qwen_key_readers.map_array and qwen_backend_composition._runtime_dependencies before capture. The latter returns the REAL Dependencies dataclass with ten explicit recording callback fields.','Keep ReadDirectory class original if only declaring GrantedPublicDirectory; its constructor must not be invoked against borrowed paths.','Recorded core.load returns objects for he_params/key_gen/ckks_ctx/polynomial, and retain returns declared recording leases, without installing or importing genuine core aliases.'],hazards=['Do not execute genuine _runtime_dependencies: it expands into numeric/native namespaces and the full common-owner import graph.','Do not call genuine _materialize: it imports jax.numpy and constructs arrays/Polynomial.','Real NumPy/JAX import is not authorized and may load shared libraries or discover plugins even without an explicit devices call.','Definition loading includes ordinary dataclass construction and composition threading.Lock; this is not a claim of zero Python execution.','Further parameter/request/publisher or process orchestration seams require a separate explicit table; no transfer of constructor-only coverage.'],earlier_static_observation='03994d',candidate_imports=False,tests=False,collection=False,backend_or_capacity_probes=False,borrowed_body_reads=False,actual_authority=False)
(O/'review.json').write_text(json.dumps(report,sort_keys=True,indent=2)+'\n');print(H((O/'review.json').read_bytes()))
