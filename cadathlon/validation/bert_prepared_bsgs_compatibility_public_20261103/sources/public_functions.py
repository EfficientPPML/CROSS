"""Execute only exact public planner definitions, never runtime/backend imports."""
import ast,dataclasses,math,numbers,typing
from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np
HERE=Path(__file__).parent
B=HERE/'inputs/bsgs'
def definition(path,name,namespace,*,owner=None):
 tree=ast.parse(path.read_text());body=tree.body
 if owner:body=next(n for n in body if isinstance(n,ast.ClassDef)and n.name==owner).body
 node=next(n for n in body if isinstance(n,(ast.FunctionDef,ast.ClassDef))and n.name==name)
 exported=name
 if isinstance(node,ast.FunctionDef):
  node.decorator_list=[]
  if owner:exported='_'+owner+'_'+name;node.name=exported
 exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),namespace)
 return namespace[exported]
def planners():
 ns=dict(math=math,Integral=numbers.Integral,Real=numbers.Real,Optional=typing.Optional,Tuple=typing.Tuple)
 compute=definition(B/'checkout/jaxite_word/bsgs.py','compute_bsgs_params',ns)
 ns=dict(np=np,compute_bsgs_params=compute)
 matrix=definition(B/'checkout/cadathlon/common/encrypted_packed_qwen.py','MatrixDiagonals',ns)
 ns=dict(LinearRotationPlan=lambda *v:NS(**dict(zip(('strategy','baby_indices','baby_steps','giant_indices','giant_steps','final_giant','required_keys','nonzero_rotation_calls'),v))))
 definition(B/'checkout/cadathlon/common/encrypted_chained_streaming_linear.py','validate_rotation_strategy',ns)
 planner=definition(B/'checkout/cadathlon/common/encrypted_chained_streaming_linear.py','linear_rotation_plan',ns)
 return matrix,planner

def fixed_keys():
 route=definition(B/'encrypted_bsgs_score_routing.py','required_rotation_indices',dict(SLOTS=65536))
 attention=definition(B/'encrypted_bert_bsgs_attention.py','required_rotation_indices',dict(required_rotation_indices=route,SLOTS=65536,FEATURES=4096,ROWS=16),owner='BertBSGSAttentionTile')
 class Layout:
  features=4096;slots=65536
 norm=definition(B/'checkout/cadathlon/common/encrypted_row_normalization.py','required_rotation_indices',dict(RowTileLayout=Layout),owner='RowLayerNorm')
 return dict(route=list(route()),attention=list(attention()),norm=list(norm(Layout())))

def pins():
 tree=ast.parse((B/'server_public_model.py').read_text());ns={}
 for n in tree.body:
  if isinstance(n,ast.Assign)and len(n.targets)==1 and isinstance(n.targets[0],ast.Name)and n.targets[0].id in ('CHECKPOINT_SHA256','CONFIG_SHA256','CANDIDATE_SHA256','PROFILE_SHA256','TENSOR_SHA256'):
   ns[n.targets[0].id]=ast.literal_eval(n.value)
 ns['shapes']=definition(B/'server_public_model.py','tensor_shapes',{})()
 return ns
