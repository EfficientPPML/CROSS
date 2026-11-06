"""Offline public-only readback; never imports a backend or executes the graph."""
from pathlib import Path
import hashlib,json,math,tarfile,xml.etree.ElementTree as ET
P=Path(__file__).resolve().parent
def sha(raw):return hashlib.sha256(raw).hexdigest()
def main():
 inventory=json.loads((P/'artifact_hashes.json').read_bytes())
 assert {n:sha((P/n).read_bytes())for n in inventory}==inventory
 report=json.loads((P/'preflight/report.json').read_bytes());plan=json.loads((P/'preflight/plan.json').read_bytes())
 for field in ('passed','complete','cleanup_completed','terminal_scope_completed','termination_scope_completed','source_files_unchanged','execution_files_unchanged','zero_audited_private_model_body_reads'):assert report[field]is True
 before=report['source_hashes_before'];assert len(before)==222 and before==report['source_hashes_after']==report['execution_source_hashes_before']==report['execution_source_hashes_after']
 archive=P/'preflight/sources.tar.gz';assert sha(archive.read_bytes())==report['source_archive_sha256']
 with tarfile.open(archive)as t:
  ms=t.getmembers();assert len(ms)==len(before) and len({m.name for m in ms})==len(ms) and all(m.isfile() for m in ms)
  raws={m.name:t.extractfile(m).read()for m in ms};assert {n:sha(b)for n,b in raws.items()}==before
 predecessor=json.loads(raws['predecessor.json']);assert len(predecessor['source_hashes'])==199
 assert all(sha(raws['predecessor/'+n])==h for n,h in predecessor['source_hashes'].items())
 for n in ('audit.json','plan-audit.json'):
  r=json.loads((P/'preflight'/n).read_bytes());assert r['passed']is True and not r['denied_attempts'] and not r['backend_modules']
 assert len(report['children'])==2 and all(c['direct_returncode']==0 and all(c[n]is True for n in ('direct_child_reaped','group_empty_verified','cleanup_completed'))for c in report['children'])
 tests=ET.parse(P/'preflight/tests.xml').getroot().findall('testsuite');assert {n:sum(int(s.get(n,0))for s in tests)for n in ('tests','errors','failures','skipped')}==dict(tests=61,errors=0,failures=0,skipped=0)==report['tests']
 assert sha((P/'preflight/plan.json').read_bytes())==report['plan_sha256']
 assert len(plan['checkpoints'])==plan['checkpoint_count']==96 and plan['all_slot_reference_comparisons']==96*65536
 assert all(x['index']==i and math.isfinite(x['scale']) and x['scale']>0 and x['nsd']==1 and math.isfinite(x['public_reference_error']) and x['public_reference_error']<=1e-10 for i,x in enumerate(plan['checkpoints']))
 assert plan['refresh_bounds']==[16.,8.,2.,4.,8.] and plan['checkpoints'][-1]['level']==11 and plan['ideal_final_toy_gate_passed']is True
 assert len(plan['polynomial_domains'])==11 and all(x['passed']is True for x in plan['polynomial_domains'])
 q=plan['profile']['q_towers'];p=plan['profile']['p_towers'];assert len(q)==100 and len(p)==24 and math.prod(q+p).bit_length()<=3523
 assert math.prod(p)>=max(math.prod(q[i:i+25])for i in range(0,100,25))
 assert len(plan['required_rotation_indices'])==199 and plan['required_rotation_indices']==sorted(set(plan['graph_rotation_indices'])|set(plan['bootstrap_rotation_indices']))
 assert plan['storage']['checkpoint_raw_payload_bytes']==sum(2*131072*2*(x['level']+1)*4 for x in plan['checkpoints'])==3758096384
 failed=json.loads(raws['failed_checkpoint/stdout.json']);assert failed['passed']is False and failed['error_type']=='ApproximationErrorTooHigh'
 assert plan['no_HE_or_TPU_execution']is True and report['HE_executed']is False
 print(json.dumps(dict(passed=True,sources=222,original_sources=199,tests=61,checkpoints=96,scope='Public plan/ideal references only; original encrypted failure preserved, actual diagnostic worker deferred.')))
if __name__=='__main__':main()
