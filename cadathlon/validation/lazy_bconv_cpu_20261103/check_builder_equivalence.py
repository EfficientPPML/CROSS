"""Read-only check that moving BConv construction did not alter its arithmetic."""
import ast
import hashlib
import json
from pathlib import Path

LAB=Path(__file__).resolve().parent
old_path=LAB/'before/jaxite_word/he_params.py'
new_path=LAB/'checkout/jaxite_word/he_params.py'
old=ast.parse(old_path.read_text());new=ast.parse(new_path.read_text())
owner=next(n for n in old.body if isinstance(n,ast.ClassDef) and n.name=='HEParameterCache')
original=next(n for n in owner.body if isinstance(n,ast.FunctionDef) and n.name=='_build_bconv_params')
helper=next(n for n in new.body if isinstance(n,ast.FunctionDef) and n.name=='_build_level_bconv_params')
class PublicInputs(ast.NodeTransformer):
    def visit_Attribute(self,node):
        if isinstance(node.value,ast.Name) and node.value.id=='self':
            if node.attr in ('num_p','dnum','ff_context_cls','perf_test'):
                return ast.Name(id=node.attr,ctx=ast.Load())
            if node.attr=='p_towers':
                return ast.parse('qp_at_level[num_q_l:]',mode='eval').body
            raise AssertionError('Unexpected owner dependency: '+node.attr)
        return self.generic_visit(node)
expected=[PublicInputs().visit(node) for node in original.body]
actual=list(helper.body)
normalization=ast.parse('q_at_level, qp_at_level = list(q_at_level), list(qp_at_level)').body[0]
assert ast.dump(actual[1])==ast.dump(normalization)
del actual[1]
assert [ast.dump(n) for n in actual]==[ast.dump(n) for n in expected]
report={'scope':__doc__,'passed':True,
    'old_sha256':hashlib.sha256(old_path.read_bytes()).hexdigest(),
    'new_sha256':hashlib.sha256(new_path.read_bytes()).hexdigest(),
    'changes':'Owner attributes replaced by immutable public arguments; ordered Q/QP inputs normalized to lists.',
    'partition_and_arithmetic_ast_unchanged':True,
    'old_public_contract':'qp_at_level equals q_at_level followed by all P towers; unchanged by the caller.'}
(LAB/'builder_equivalence.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
