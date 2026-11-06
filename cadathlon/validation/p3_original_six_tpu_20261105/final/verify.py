"""Read-only, standard-library verification of the completed P3 original six-case final capsule.

This checks archived evidence, not numerical execution. It never imports archived
code, extracts files, polls processes, reads key bodies or executes the archived root cleanup script.
"""
import argparse
import ast
import datetime
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import stat
import tarfile


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def read(path, cap=16 * 1024 * 1024):
    path = Path(path)
    before = path.lstat()
    require(stat.S_ISREG(before.st_mode), 'not a regular file: ' + str(path))
    fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        opened = os.fstat(fd)
        require((before.st_dev, before.st_ino) == (opened.st_dev, opened.st_ino), 'open identity')
        require(stat.S_ISREG(opened.st_mode) and opened.st_size <= cap, 'file size/type')
        parts = []
        total = 0
        while True:
            block = os.read(fd, min(1024 * 1024, cap - total + 1))
            if not block:
                break
            parts.append(block)
            total += len(block)
            require(total <= cap, 'read cap')
        raw = b''.join(parts)
        after = os.fstat(fd)
        require((opened.st_size, opened.st_mtime_ns, opened.st_ctime_ns) ==
                (after.st_size, after.st_mtime_ns, after.st_ctime_ns), 'file changed during read')
        require(len(raw) == opened.st_size, 'short read')
    finally:
        os.close(fd)
    named = path.lstat()
    require((named.st_dev, named.st_ino) == (opened.st_dev, opened.st_ino), 'name changed')
    return raw


def pairs(rows):
    result = {}
    for key, value in rows:
        require(key not in result, 'duplicate JSON key')
        result[key] = value
    return result


def decode(raw):
    def bad_constant(value):
        raise ValueError('nonfinite JSON constant: ' + value)
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=bad_constant)


def archive(raw, expected):
    result = {}
    with tarfile.open(fileobj=io.BytesIO(raw), mode='r:gz') as stream:
        members = stream.getmembers()
        require(len(members) == len(expected), 'archive member count')
        total = 0
        for member in members:
            name = member.name
            require(name in expected and name not in result, 'archive member name/duplicate')
            require(not Path(name).is_absolute() and '..' not in Path(name).parts, 'archive path')
            require(member.isfile() and not member.issym() and not member.islnk(), 'archive type')
            require(0 <= member.size <= 4 * 1024 * 1024, 'archive member bound')
            total += member.size
            require(total <= 8 * 1024 * 1024, 'archive total bound')
            raw_member = stream.extractfile(member).read(member.size + 1)
            require(len(raw_member) == member.size and sha(raw_member) == expected[name], 'archive bytes')
            result[name] = raw_member
    require(set(result) == set(expected), 'archive closure')
    return result


def finite(value, nonnegative=True):
    require(type(value) in (int, float) and math.isfinite(value), 'finite numeric field')
    require(not nonnegative or value >= 0, 'nonnegative numeric field')


def fn(tree, name):
    found = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name]
    require(len(found) == 1, 'unique function ' + name)
    return found[0]


def expression(text):
    return ast.dump(ast.parse(text, mode='eval').body, include_attributes=False)


def verify(root,current):
    root=Path(root);manifest_raw=read(root/'artifact_hashes.json');manifest=decode(manifest_raw)
    found=set()
    for directory,dirs,files in os.walk(root,followlinks=False):
        for n in dirs:require(stat.S_ISDIR((Path(directory)/n).lstat().st_mode),'directory alias')
        for n in files:
            p=Path(directory)/n;require(stat.S_ISREG(p.lstat().st_mode),'file alias');found.add(str(p.relative_to(root)))
    require(found==set(manifest)|{'artifact_hashes.json'},'closed capsule')
    raw={}
    for name,r in manifest.items():
        require(not Path(name).is_absolute() and '..' not in Path(name).parts,'safe name')
        b=read(root/name);require(set(r)=={'sha256','bytes'} and sha(b)==r['sha256'] and len(b)==r['bytes'],'indexed bytes '+name);raw[name]=b
    get=lambda n:decode(raw[n])
    fixed={
      'root/terminal-admission.json':'8724aaa62574c9346affd06797517569d018ee6a8a89332a29484e6d04f3ddd9',
      'root/terminal-receipt.json':'fa31b7920ddde5034b3c01a43fe499aca0a51434fe3efe9da96a2a80e7d1538d',
      'root/cleanup-pending.json':'09221e062dd11c9e9fab02f8a5e443de23f41bf15969f91d295a1c77d282f3a5',
      'raw/report.json':'f1a7b8a8841957b87d887cc755695df0a2884a1898559ec72296cf5037a10eb3',
      'preparation/preparation.json':'80e3b7fab9e439df7fe62a42ef98ef69288ff366f5a0b4ad07dcaedc3cd75a65',
      'preparation/sources.json':'7f1fa4a084c0a5a83216154a2d49ebdb0d7694c1fb9df67459fcad9ac615e748',
      'preparation/auxiliary.json':'95134510d4e0ad5b680679a3dcc93f5cdaae11c581c92c41682d2b68279b13b1',
      'sources/actual-128.tar.gz':'a5e8b41586e683bb6eb23d57266195bfcb0206acb94f2f1a4af9fef6636074eb',
      'sources/prepared-129.tar.gz':'07450d3d707c263616b4ffc8e1150ff0c45a821fa4b9b1f5a2cd867d6cbffec1',
      'root/launch.json':'f219750cc03acb90d05283d2076f53cb9623df62e35f0ef7b9e5021434b3d5d5',
      'root/launch-live-readback.json':'1f5bca1333ac7797bd94fd07982f6bb56f59b84c37fc28cb023bf850e245c0ad',
      'root/terminal-reader-source.py.txt':'1207affd6cdc2c6211126b2f1e170900544cf93a135086f5ce427d443bf5b4fd',
    }
    for n,h in fixed.items():require(sha(raw[n])==h,'fixed public pin '+n)
    prep=get('preparation/preparation.json');sources=get('preparation/sources.json');aux=get('preparation/auxiliary.json')
    actual=archive(raw['sources/actual-128.tar.gz'],sources);prepared=archive(raw['sources/prepared-129.tar.gz'],sources|aux)
    require(len(sources)==128 and len(aux)==1 and len(prepared)==129,'128/129 distinction')
    require(aux=={'repo/jaxite_word/API_REFERENCE.md':'ed1c589f0204ce878d26f0cac7ddf1d3446ae361214a99191df562ef6ee4961f'},'auxiliary source')
    require(all(actual[n]==prepared[n] for n in actual),'archive shared bytes')
    require(prep['source_map_sha256']==fixed['preparation/sources.json'] and prep['archive_sha256']==fixed['sources/prepared-129.tar.gz'],'prepared joins')
    require(prep['original_baseline_dynamic_constants_override'] is False,'original baseline selection')
    report=get('raw/report.json');admit=get('root/terminal-admission.json');terminal=get('root/terminal-receipt.json');pending=get('root/cleanup-pending.json');launch=get('root/launch.json')
    schedule=['test'+str(i)+'.in' for i in range(1,7)]
    require(report['complete'] is True and report['passed'] is True and 'active_case' not in report and 'error' not in report,'complete original report')
    require(report['schedule']==schedule and set(report['cases'])==set(schedule),'exact original schedule')
    require(report['collector_pid']==1598331 and report['cpu_affinity']==list(range(160,168)),'recorded parent identity')
    require(report['profile']=='original' and report['warmups']==0 and report['repeats']==1 and report['healthy_case_timeout_s'] is None,'cold original scope')
    require(sources==report['source_hashes_start']==report['source_hashes_end']==get('raw/source-start.json')==get('raw/source-end.json'),'start/end source joins')
    require(report['source_files_unchanged'] is True and report['source_archive_sha256']==fixed['sources/actual-128.tar.gz'],'report source closure')
    require(terminal['original_session']==81235 and terminal['terminal'] is True and terminal['exit_code']==0 and terminal['tool_chunk']=='84c601','original root-owned tool terminal')
    require(terminal['source']=='original tools.write_stdin result; recorded by root','terminal attribution')
    require(admit['passed'] is True and admit['complete'] is True and admit['original_session']==81235 and admit['exit_code']==0,'root admission')
    require(admit['terminal_tool_chunk']==terminal['tool_chunk'] and admit['terminal_receipt_sha256']==sha(raw['root/terminal-receipt.json']),'admission terminal join')
    require(admit['report_sha256']==pending['report_sha256']==sha(raw['raw/report.json']),'final report pin')
    require(admit['cleanup_pending_receipt_sha256']==sha(raw['root/cleanup-pending.json']),'cleanup receipt pin')
    require(admit['verifier_sha256']==sha(raw['root/terminal-reader-source.py.txt']),'archived root script pin; never executed here')
    require(admit['all_start_after_end_current_archive_sources_exact'] is True and admit['measured_sources']==128 and admit['prepared_sources']==129,'root source admission')
    require(admit['collector_absence']==pending['collector_absence']=={'1598331':'absent_ESRCH','-1598331':'absent_ESRCH'},'root collector absence receipt')
    parent='/dev/shm/cross-p3-original-six-key-parent-f383y30c'
    identity={'device':24,'inode':4064,'uid':2002,'gid':2002,'mode':448}
    require(pending['owned_key_parent']==admit['owned_key_parent_removed']==parent and pending['identity']==admit['owned_key_parent_original_identity']==identity,'root original key-parent identity')
    require(pending['empty_before_cleanup'] is True and pending['all_checks_before_parent_removal'] is True and pending['case_count']==6,'root cleanup ordering evidence')
    require(pending['terminal_receipt_sha256']==sha(raw['root/terminal-receipt.json']) and pending['source_count']==128,'cleanup source/terminal joins')
    require(launch['owned_public_key_parent']==parent and launch['owned_public_key_parent_identity']=={'mode':448,'st_dev':24,'st_gid':2002,'st_ino':4064,'st_uid':2002},'launch parent join')
    require(sha(raw['root/launcher-source.py.txt'])==launch['launcher_sha256'] and sha(raw['root/initial-launcher-source.py.txt'])==launch['initial_launcher_sha256'],'launch source/history pins')
    live=get('root/launch-live-readback.json');require(live['original_session']==81235 and live['terminal'] is False and live['launch_tool_chunk']=='58a623','historical initial live receipt')
    require(datetime.datetime.fromisoformat(terminal['observed_utc'])<=datetime.datetime.fromisoformat(pending['observed_utc'])<=datetime.datetime.fromisoformat(admit['observed_utc']),'terminal/cleanup chronology')
    he=ast.parse(actual['repo/jaxite_word/he_params.py']);runner=fn(ast.parse(actual['repo/cadathlon/non_agent_pset/p3_cnn/src/runner.py']),'run')
    table,=[n.value for n in he.body if isinstance(n,ast.Assign) and any(isinstance(x,ast.Name) and x.id=='_OPENFHE_CEILINGS_RAW' for x in n.targets)]
    ceiling_table,=[v for k,v in zip(table.keys,table.values) if ast.dump(k,include_attributes=False)==expression('(CostModel.CLASSICAL,128)')]
    ceiling=ast.literal_eval(ceiling_table)[32768];require(ceiling==881,'pinned security ceiling')
    require(ast.dump(fn(he,'modulus_product_is_secure').body[-1].value,include_attributes=False)==expression('openfhe_logq_bits(modulus_product) <= security_ceiling(degree, target)'),'inclusive security gate')
    require(ast.dump(fn(he,'openfhe_logq_bits').body[-1].value,include_attributes=False)==expression('modulus_product.bit_length()'),'integer security product rule')
    tests={ast.dump(n.test,include_attributes=False) for n in ast.walk(runner) if isinstance(n,ast.If) and any(isinstance(x,ast.Raise) for x in n.body)}
    require(expression('not np.all(np.abs(output - expected) <= 0.02 + 0.01 * np.abs(expected))') in tests and expression('errors["top1_agreement"] < 0.99') in tests,'original numerical gates')
    helper=fn(ast.parse(actual['collector/process_helpers.py']),'run_case')
    popens=[n for n in ast.walk(helper) if isinstance(n,ast.Call) and ast.unparse(n.func)=='subprocess.Popen']
    require(len(popens)==1 and any(k.arg=='start_new_session' and isinstance(k.value,ast.Constant) and k.value.value is True for k in popens[0].keywords),'source-bound wrapper PGID')
    release=[n.lineno for n in ast.walk(runner) if isinstance(n,ast.Call) and ast.unparse(n.func)=='mapping.release']
    totals=[n.lineno for n in runner.body if isinstance(n,ast.Assign) and any(ast.unparse(x)=="report['total_s']" for x in n.targets)]
    require(len(release)==len(totals)==1 and release[0]<totals[0],'mapping release included before total')
    audit=get('source-audit.json')
    for label,r in audit['source_functions'].items():
        name,function=label.rsplit(':',1);node=fn(ast.parse(actual[name]),function)
        require(r=={'line':node.lineno,'end_line':node.end_lineno,'AST_sha256':sha(ast.dump(node,include_attributes=False).encode())},'source definition pin')
    require(len(audit['source_functions'])==14,'source functions')
    rows=[];wrappers=set();raw_case_names=set();base='/tmp/p3-original-six-preparation-20261105'
    for i,(case,root_case) in enumerate(zip(schedule,admit['cases'],strict=True),1):
        folder='raw/test'+str(i)+'/';p=get(folder+'process.json');r=get(folder+'stdout.json')
        require(p==report['cases'][case] and p['result']==r and get(folder+'source-after.json')==sources,'case/result/source joins')
        for k in ['passed','source_files_unchanged','process_reaped','public_key_store_removed','owned_directory_handle_closed']:require(p[k] is True,'case cleanup/terminal '+k)
        require(type(p['returncode']) is int and p['returncode']==0 and 'error' not in p and p['timeout_s'] is None and p['cleanup_grace_s']==30.0,'case status/policy')
        names={name for name in raw if name.startswith(folder)};require(names=={folder+n for n in ['process.json','stdout.json','stderr.log','time.txt','source-after.json']},'five exact case files');raw_case_names|=names
        for n in names:require(sha(raw[n])==admit['case_file_hashes'][n.removeprefix('raw/')],'root case file pin')
        command=['/usr/bin/time','-v','-o',base+'/run-v1/test'+str(i)+'/time.txt','/home/jianming_gatech/miniconda3/envs/jaxite/bin/python',base+'/collector/check_p3_original_six.py','--repo',base+'/checkout','--child-runner',base+'/checkout/cadathlon/non_agent_pset/p3_cnn/src/runner.py','--','--case',base+'/checkout/cadathlon/non_agent_pset/p3_cnn/test/'+case,'--baseline','--warmups','0','--repeats','1']
        require(p['command']==command,'cold baseline argv')
        model='lenet' if i<=3 else 'alexnet_tiny';batch=(1,2,4)[(i-1)%3];fixture=decode(actual['repo/cadathlon/non_agent_pset/p3_cnn/test/'+case]);fp=prep['fixtures'][i-1]
        require(fixture['purpose']=='synthetic-arithmetic-not-trained-accuracy' and fixture['model']==model and fixture['batch']==batch,'public fixture')
        require(fp['case']==case and fp['model']==model and fp['batch']==batch,'fixture metadata')
        for suffix,key in [('in','input_sha256'),('out','output_sha256')]:require(sha(actual['repo/cadathlon/non_agent_pset/p3_cnn/test/test'+str(i)+'.'+suffix])==fp[key],'fixture pin')
        ref=decode(actual['repo/cadathlon/non_agent_pset/p3_cnn/test/test'+str(i)+'.out']);require(ref['kind']=='plaintext-numerical-reference' and len(ref['logits'])==batch and all(len(row)==10 for row in ref['logits']),'public reference shape')
        for row in ref['logits']:
            for value in row:finite(value,False)
        require(r['case']==case and r['batch']==batch and r['mode']=='HE-synthetic-smoke' and r['implementation']=='baseline.py' and r['device']=='TPU_0(process=0,(0,0,0,0))','recorded run scope')
        for k in ['passed','mapping_release_included','cold_start_included']:require(r[k] is True,'runner flag')
        require(r['trained_accuracy_evaluated'] is False and r['warmup_s']==r['warmup_server_s']==[],'no trained metric/warmups')
        require(len(r['end_to_end_s'])==len(r['server_s'])==1,'one cold request')
        for k in ['setup_s','median_s','total_s','max_abs_error','nrmse','top1_agreement']:finite(r[k])
        require(0.99<=r['top1_agreement']<=1 and r['median_s']==r['end_to_end_s'][0],'reported metrics')
        finite(p['process_elapsed_s']);require(0<r['server_s'][0]<=r['end_to_end_s'][0]<=r['total_s']<=p['process_elapsed_s'] and r['setup_s']+r['end_to_end_s'][0]<=r['total_s'],'time joins')
        rings=r['execution_rings'];require(len(rings)==1,'one reported ring');ring=rings[0]
        for k,v in {'degree':32768,'num_slots':16384,'num_q':16,'num_p':4,'dnum':4,'max_level':7,'security_bits':128}.items():require(type(ring[k]) is int and ring[k]==v,'ring field')
        require(ring['security_model']=='classical' and len(ring['q_towers'])==16 and len(ring['p_towers'])==4,'reported security profile')
        require(all(type(n) is int and n>1 for n in ring['q_towers']+ring['p_towers']),'integer moduli')
        product=math.prod(ring['q_towers']+ring['p_towers']);require(product.bit_length()==605<=ceiling,'QP integer security gate')
        require(abs(math.log2(product)-ring['log2_QP'])<1e-10 and abs(math.log2(math.prod(ring['q_towers']))-ring['log2_Q'])<1e-10,'reported log joins')
        time=raw[folder+'time.txt'].decode();require('Command being timed: "'+' '.join(command[4:])+'"' in time and re.search(r'Exit status: 0\s*$',time),'GNU time command/exit')
        rss=int(re.search(r'Maximum resident set size \(kbytes\): (\d+)',time).group(1));require(rss==p['peak_child_rss_kib'],'GNU RSS')
        wall=re.search(r'Elapsed \(wall clock\) time \(h:mm:ss or m:ss\): ([\d:.]+)',time).group(1);wall_s=sum(float(v)*60**power for power,v in enumerate(reversed(wall.split(':'))));require(abs(wall_s-p['process_elapsed_s'])<0.1,'rounded wall time')
        pid=p['pid'];require(type(pid) is int and pid>0 and pid not in wrappers,'unique wrapper');wrappers.add(pid)
        require(root_case['wrapper_pid']==pid and root_case['wrapper_and_source_bound_group_absence']=={str(pid):'absent_ESRCH',str(-pid):'absent_ESRCH'},'root wrapper/group absence receipt')
        require(root_case['inner_pid_live_identity_claim'] is False,'no invented inner PID')
        child=Path(p['owned_public_key_directory']);require(str(child.parent)==parent and child.name.startswith('cross-transformer-case-') and root_case['child_directory_absent']==str(child),'root child absence record')
        require(root_case['case']==case and root_case['model']==model and root_case['batch']==batch and root_case['passed'] is True,'root case scope')
        for target,source in [('max_abs_error','max_abs_error'),('nrmse','nrmse'),('top1_agreement','top1_agreement'),('runner_total_s','total_s'),('setup_s','setup_s')]:require(root_case[target]==r[source],'root exact metric join')
        require(root_case['external_s']==p['process_elapsed_s'] and root_case['request_s']==r['end_to_end_s'][0] and root_case['server_s']==r['server_s'][0] and root_case['GNU_time_max_RSS_KiB']==rss and root_case['reported_QP_integer_bits']==605 and root_case['security_ceiling']==881,'root timing/security join')
        rows.append(root_case)
    require(len(raw_case_names)==30 and len(admit['case_file_hashes'])==30,'all thirty case files')
    finite(report['elapsed_s']);require(report['elapsed_s']==admit['elapsed_s']==3369.480570556014 and report['elapsed_s']>=sum(r['external_s'] for r in rows),'parent elapsed join')
    summary=get('summary.json');require(summary['cases']==rows and summary['passed_cases']==6 and summary['failed_cases']==0 and summary['complete'] is True and summary['parent_elapsed_s']==report['elapsed_s'],'summary exact scope')
    require(summary['independent_decrypted_logit_recomputation'] is False and summary['new_numerical_execution'] is False and summary['trained_accuracy_evaluated'] is False,'summary limits')
    capture=get('capture.json');require(capture['packager_polled_original_session'] is False and capture['packager_direct_wait_authority'] is False and capture['key_body_read'] is False and capture['new_process_or_key_path_observation'] is False,'packager attribution')
    origins=get('origins.json')
    for name,r in origins.items():
        require(name in raw and sha(raw[name])==r['sha256'] and len(raw[name])==r['bytes'],'copy origin')
        if current:require(sha(read(r['path']))==r['sha256'],'current public origin')
    names=set()
    for r in audit['current_sources']:
        require(r['name'] not in names and (sources|aux)[r['name']]==r['sha256'],'current source record');names.add(r['name'])
        if current:require(sha(read(r['prepared_path']))==sha(read(r['original_path']))==r['sha256'],'current original/prepared bytes')
    require(names==set(prepared) and len(names)==129,'current public source closure')
    require(read(root/'artifact_hashes.json')==manifest_raw,'index stable')
    for name,b in raw.items():require(read(root/name)==b,'capsule stable')
    return {'passed':True,'scope':'Original six-case P3 synthetic encrypted TPU validation; recorded/source-bound evidence',
        'cases':6,'failures':0,'indexed_files':len(manifest),'index_sha256':sha(manifest_raw),'actual_sources':128,'prepared_sources':129,
        'case_files':30,'parent_elapsed_s':report['elapsed_s'],'root_session':81235,'root_terminal_chunk':'84c601',
        'root_cleanup_receipts_joined':True,'new_process_or_key_observation':False,'candidate_execution':False,
        'independent_decrypted_logit_recomputation':False,'current_public_origins_checked':current}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parent)
    parser.add_argument('--current',action='store_true')
    args=parser.parse_args()
    print(json.dumps(verify(args.root,args.current),sort_keys=True,allow_nan=False))


