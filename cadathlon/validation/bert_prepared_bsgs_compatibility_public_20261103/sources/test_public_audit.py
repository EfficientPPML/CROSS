from pathlib import Path
import json,os
import numpy as np
import pytest
import audit
from public_functions import planners,fixed_keys,pins

@pytest.mark.parametrize('shape,keys,n1,diagonals',[
 ((768,768),(1,64,64064),64,1535),((3072,768),(1,64,61760),64,3839),
 ((768,3072),(1,64,61760),64,3839),((2,768),(1,32,64768),32,769)])
def test_exact_source_public_full_support_planner(shape,keys,n1,diagonals):
 M,P=planners();m=M(np.ones(shape),65536,4096);plan=P(m,strategy='chained')
 assert m.n1==n1 and len(m.active)==diagonals and plan.required_keys==keys

@pytest.mark.parametrize('seed',range(5))
def test_sparse_gap_inventory_follows_actual_diagonals(seed):
 M,P=planners();rng=np.random.default_rng(seed);w=rng.integers(-1,2,(9,7));w[rng.random(w.shape)<.8]=0
 m=M(w,65536,4096);p=P(m,strategy='chained')
 active=tuple(sorted({(j-i)%65536 for i in range(9)for j in range(7)if w[i,j]!=0}))
 assert m.active==active
 babies=sorted({d%m.n1 for d in active}-{0});giants=sorted({d//m.n1*m.n1 for d in active},reverse=True)
 shifts=[b-a for a,b in zip([0]+babies,babies)]+[a-b for a,b in zip(giants,giants[1:])]+([min(giants)]if giants else [])
 assert p.required_keys==tuple(sorted({s%65536 for s in shifts if s%65536}))


def test_exact_fixed_nonmatrix_keys_and_parameter_pins():
 f=fixed_keys();assert len(f['route'])==36 and len(f['attention'])==59 and len(f['norm'])==22
 assert set(f['route'])<=set(f['attention'])
 p=pins();assert len(p['shapes'])==196 and sum(len(s)==2 for s in p['shapes'].values())==74
 assert set(p['TENSOR_SHA256'])==set(p['shapes'])


def test_bounded_public_reader_round_trip(tmp_path):
 p=tmp_path/'public';p.write_bytes(b'public');assert audit.read(p,6)==b'public'

@pytest.mark.parametrize('kind',['symlink','directory','oversize'])
def test_bounded_reader_rejects_wrong_public_object(tmp_path,kind):
 p=tmp_path/'public'
 if kind=='directory':p.mkdir()
 elif kind=='symlink':
  q=tmp_path/'actual';q.write_bytes(b'public');p.symlink_to(q)
 else:p.write_bytes(b'long')
 with pytest.raises((ValueError,OSError)):audit.read(p,3)


def test_reader_rejects_original_file_replacement_during_read(tmp_path,monkeypatch):
 p=tmp_path/'public';p.write_bytes(b'public');old=os.read;calls=[]
 def changed(fd,size):
  result=old(fd,size)
  if not calls:
   calls.append(True);p.rename(tmp_path/'old');p.write_bytes(b'public')
  return result
 monkeypatch.setattr(audit.os,'read',changed)
 with pytest.raises(ValueError):audit.read(p,6)
 assert p.read_bytes()==b'public'
