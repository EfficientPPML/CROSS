"""Verify public source-only proposals without importing candidate modules."""
from pathlib import Path
import argparse, ast, hashlib, json, math, re, tarfile

HERE=Path(__file__).resolve().parent
SOURCE=Path('/tmp/bert-scaled-crt-decoder-client-proposals-20261104')
PARENT=Path('/tmp/bert-client-clean-launch-source-proposals-20261104')
REPORT='2251a4bc580ba2f008a638fca74803d77a48fa666e12c25a1b93f0b4794b2e97'
OLD='bc978cb7412b55416f7ed634a689ae616a263b0f47fd9eef996ba244f2505db6'
NEW='9675b300d507fd9c6a3d03656c9df325d8da68a9331b4c92def0512784d0dc26'
REL='runtime/jaxite_word/ckks_ctx.py'
PINS={
 'original122':(122,'49ff2d27e11ce773d27baec448fc1bbb0672fbf2ff6e9e8b94ca14ceda126a56','6e3e30ad643921e9546be144e7da0dce485e9dd1be088f31f62f35cd2aa4c5f9','20fb4e7b818ad26fa0e69037c2558b017f995f7e5401c74390abca31fa84ec6f'),
 'combined125':(125,'f5fd949507f210a7b40b0b0cf508268f6a625911c7dcb6526cb0870a9dc269d7','6b2b1985c1f9e7298bf47cf8559fd8a2122b0ad854289fed8674c2c05e8475a0','6fd87ede77b605bac56c53f8b636d76aeacc1fc9160749e9c056a476e56d00fb')}
CHANGED={'README_CLIENT.md','client_source_guard.py',REL,'test_client_source_guard.py'}
sha=lambda b:hashlib.sha256(b).hexdigest()
dump=lambda n:ast.dump(n,include_attributes=False)

def read(path,limit=32<<20):
 assert not path.is_symlink() and path.is_file()
 raw=path.read_bytes();assert len(raw)<=limit;return raw

def assignment(raw,name):
 tree=ast.parse(raw)
 return ast.literal_eval(next(n.value for n in tree.body if isinstance(n,ast.Assign)
   and any(isinstance(t,ast.Name)and t.id==name for t in n.targets)))

def guard_without_overlay(raw):
 tree=ast.parse(raw)
 tree.body=[n for n in tree.body if not(isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='OVERLAYS'for t in n.targets))]
 return dump(tree)

class ReverseDecoder(ast.NodeTransformer):
 def __init__(self):self.calls=0;self.helpers=0
 def visit_FunctionDef(self,node):
  if node.name=='_scale_centered_crt_coefficient':self.helpers+=1;return None
  return self.generic_visit(node)
 def visit_Call(self,node):
  if isinstance(node.func,ast.Name)and node.func.id=='_scale_centered_crt_coefficient':
   assert len(node.args)==3 and not node.keywords
   assert dump(node.args[1])==dump(ast.Name(id='scale',ctx=ast.Load()))
   assert dump(node.args[2])==dump(ast.Name(id='scale_inv',ctx=ast.Load()))
   self.calls+=1
   return ast.BinOp(left=ast.Call(func=ast.Name(id='float',ctx=ast.Load()),args=[node.args[0]],keywords=[]),op=ast.Mult(),right=node.args[2])
  return self.generic_visit(node)

def reverse_patch(patch,candidate):
 """Reconstruct exact four parent files in memory from the recorded diff."""
 lines=patch.decode().splitlines(True);i=0;result={}
 while i<len(lines):
  assert lines[i].startswith('--- parent/');name=lines[i][11:].rstrip('\n');i+=1
  assert lines[i]=='+++ proposal/'+name+'\n';i+=1
  assert name in CHANGED and name not in result
  new=candidate[name].decode().splitlines(True);old=[];cursor=0
  while i<len(lines)and not lines[i].startswith('--- parent/'):
   match=re.fullmatch(r'@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@[^\n]*\n',lines[i]);assert match
   new_start=int(match[3]);new_count=int(match[4]or'1');old_count=int(match[2]or'1');i+=1
   start=new_start-1 if new_count else new_start
   assert start>=cursor;old.extend(new[cursor:start]);cursor=start;a=b=0
   while i<len(lines)and not lines[i].startswith(('@@ ','--- parent/')):
    line=lines[i];i+=1;kind=line[0];body=line[1:];assert kind in ' +-'
    if kind in ' +':assert cursor<len(new)and new[cursor]==body;cursor+=1;b+=1
    if kind in ' -':old.append(body);a+=1
   assert a==old_count and b==new_count
  old.extend(new[cursor:]);result[name]=''.join(old).encode()
 assert set(result)==CHANGED
 return result

def verify(current=False):
 index=json.loads(read(HERE/'SHA256SUMS.json'));assert type(index)is dict
 actual={p.relative_to(HERE).as_posix()for p in HERE.rglob('*')if p.is_file()}
 assert actual==set(index)|{'SHA256SUMS.json'}
 assert not any(p.is_symlink()for p in HERE.rglob('*'))
 for name,pin in index.items():
  assert not Path(name).is_absolute()and '..'not in Path(name).parts
  assert sha(read(HERE/name))==pin
 raw=read(HERE/'report.json');assert sha(raw)==REPORT;r=json.loads(raw)
 assert r['source_only']is True
 assert all(r[n]is False for n in ('actual_client','backend_imported','private_payload_read','full_preflight_executed','launch_authorized'))
 root=json.loads(read(HERE/'root-review/review.json'))
 assert root['passed']is True and root['proposal_report_sha256']==REPORT
 assert root['original_clients_remain_launch_defaults']is True
 assert all(root[n]is False for n in ('actual_BERT_result','new_client_preflight','new_launch_grant'))
 out={}
 for label,(count,parent_pin,map_pin,tar_pin)in PINS.items():
  base=HERE/label;map_raw=read(base/'source_manifest.json');parent_raw=read(base/'parent_source_manifest.json')
  assert sha(map_raw)==map_pin and sha(parent_raw)==parent_pin
  manifest=json.loads(map_raw);parent=json.loads(parent_raw)
  assert len(manifest)==count and set(manifest)==set(parent)
  assert sha(read(base/'sources.tar.gz'))==tar_pin
  with tarfile.open(base/'sources.tar.gz','r:gz')as t:
   members=t.getmembers();assert len(members)==count
   assert all(m.isfile()and not Path(m.name).is_absolute()and '..'not in Path(m.name).parts and m.size<=16<<20 for m in members)
   assert len({m.name for m in members})==count and sum(m.size for m in members)<=32<<20
   content={m.name:t.extractfile(m).read()for m in members}
  assert {n:sha(b)for n,b in content.items()}==manifest
  assert {n for n in parent if parent[n]!=manifest[n]}==CHANGED
  old_files=reverse_patch(read(base/'candidate.patch'),content)
  assert all(sha(b)==parent[n]for n,b in old_files.items())
  assert manifest[REL]==NEW and parent[REL]==OLD
  inverse=ReverseDecoder();reverted=inverse.visit(ast.parse(content[REL]))
  assert inverse.calls==2 and inverse.helpers==1 and dump(reverted)==dump(ast.parse(old_files[REL]))
  assert guard_without_overlay(content['client_source_guard.py'])==guard_without_overlay(old_files['client_source_guard.py'])
  old_overlays=assignment(old_files['client_source_guard.py'],'OVERLAYS');overlays=assignment(content['client_source_guard.py'],'OVERLAYS')
  assert overlays==old_overlays|{REL:NEW}
  assert set(assignment(content['client_bootstrap.py'],'EXPECTED_NAMES'))==set(manifest)
  p89=json.loads(content['client_89_sources.json']);assert len(p89)==89
  assert set(p89)|set(assignment(content['client_source_guard.py'],'NEW'))==set(manifest)
  assert all(manifest[n]==overlays.get(n,h)for n,h in p89.items())
  assert all(manifest[n]==h==parent[n]for n,h in r['clients'][label]['protected_unchanged'].items())
  q=json.loads(content['runtime/stage/p60_profile_and_refresh.json'])['profile']['q'][:6]
  product=math.prod(q);assert product.bit_length()==181 and(product//2).bit_length()==180
  assert q==r['final_BERT_response']['q']and str(product)==r['final_BERT_response']['product']
  assert math.isfinite(float(product//2))
  if current:
   for n,h in manifest.items():
    assert sha(read(SOURCE/label/'source'/n))==h
    assert sha(read(PARENT/label/'source'/n))==parent[n]
  out[label]=dict(source_count=count,unchanged_count=count-4,archive_exact=True,parent_patch_reconstruction_exact=True)
 return dict(passed=True,indexed_files=len(index),current_checked=current,clients=out,scope='public sources only; no imports/tests/preflight/launch',original_clients_remain_launch_defaults=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--current',action='store_true');args=p.parse_args();print(json.dumps(verify(args.current),sort_keys=True))
