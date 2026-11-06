"""Stdlib-only public artifact readback. Never imports or executes tested code."""
import argparse, ast, hashlib, json, pathlib, stat, tarfile
import xml.etree.ElementTree as ET

ROOT=pathlib.Path(__file__).resolve().parent
def require(ok, label):
    if not ok: raise ValueError(label)
def sha(raw): return hashlib.sha256(raw).hexdigest()
def safe_name(s):
    p=pathlib.PurePosixPath(s)
    require(type(s) is str and not p.is_absolute() and '..' not in p.parts and str(p)==s, 'unsafe name')
    return s
def read(path):
    a=path.lstat(); require(stat.S_ISREG(a.st_mode) and not path.is_symlink(),'nonregular file '+str(path))
    b=path.read_bytes(); z=path.lstat()
    require((a.st_dev,a.st_ino,a.st_size,a.st_mtime_ns,a.st_ctime_ns)==(z.st_dev,z.st_ino,z.st_size,z.st_mtime_ns,z.st_ctime_ns),'changed file')
    return b
def raw(name): return read(ROOT/safe_name(name))
def doc(name): return json.loads(raw(name))
def digest(name): return sha(raw(name))
def archive(name, mapping):
    values={}
    with tarfile.open(ROOT/name,'r:gz') as tar:
        for member in tar.getmembers():
            n=safe_name(member.name)
            require(member.isfile() and n not in values and n in mapping,'archive type/name')
            stream=tar.extractfile(member); require(stream is not None,'archive stream')
            with stream: b=stream.read()
            require(len(b)==member.size and sha(b)==mapping[n],'archive digest '+n)
            values[n]=b
    require(set(values)==set(mapping),'archive closure')
    return values

def main(current):
    names=doc('WHITELIST.json')['files']; require(names==sorted(set(names)),'whitelist')
    actual=[]
    for p in ROOT.rglob('*'):
        require(not p.is_symlink(),'capsule symlink')
        if p.is_file(): actual.append(p.relative_to(ROOT).as_posix())
        else: require(p.is_dir(),'capsule special file')
    require(sorted(actual)==names,'closed capsule inventory')
    rows={}
    for line in raw('SHA256SUMS').decode().splitlines():
        h,n=line.split('  ',1); require(n not in rows and len(h)==64,'index format'); rows[safe_name(n)]=h
    require(set(rows)==set(names)-{'SHA256SUMS'},'index closure')
    for n,h in rows.items(): require(digest(n)==h,'index '+n)
    origins=doc('copy_origins.json')['files']
    generated={'summary.json','README.md','HISTORY.md','copy_origins.json','WHITELIST.json','SHA256SUMS'}
    require(set(origins)|generated==set(names) and not set(origins)&generated,'origin closure')
    for n,row in origins.items():
        b=raw(n); require(len(b)==row['bytes'] and sha(b)==row['sha256'],'copy origin '+n)
        if current: require(read(pathlib.Path(row['source']))==b,'current copied public artifact '+n)
    pins={
      'source/parent250/sources.json':'71c960c8d82f7d617ccb8167ac5bfbc526d5cdf54c2458b954ebfdfbb84f1ed3',
      'source/v1/sources.json':'91a4bc8c1715f8900edb49e97cc9c2be696b30a5037b5efd4a136ac088664339',
      'source/v2/sources.json':'1136d1a47ed36c2ae2723bb8f3c05eca2f64a18b7ae9de3533c6ea1073b64d7a',
      'proof/v1/sources.tar.gz':'6d890c097757327e1f4de6a624a1496dd40e466703210e617975bbf64e26e21a',
      'proof/v2/sources.tar.gz':'442ab2df032b451c9e067a49e046d26c661fa967b70aa026e117d20b874f9463',
      'proof/v1/report.json':'3b8576c1f84298b752a0f3f61aeb43cee228ca6ceba161e347ff0f63e3677d3c',
      'proof/v2/report.json':'dd8990a3c98531f3488476e27e4a56769d20d8fa65b30abd6c94513eace7cb1f',
      'reviews/root-v1/terminal-review.json':'c04da9fe0ca28314435d1efd7755f5162ea6e1a31f625f880f7413685e036f5d',
      'reviews/root-v2/terminal-review.json':'bbfbae5276b120569fd331ca1097f0d3048496186da0fd56d7cea0ce1167b8cb',
      'reviews/transformer-terminal/review.json':'30928d705c51193a3abf41131c031ef4ff1414fd4f9b5a99d0f05c75c57e6de9'}
    for n,h in pins.items(): require(digest(n)==h,'fixed pin '+n)
    parent=doc('source/parent250/sources.json'); require(len(parent)==250,'parent count')
    require(doc('source/parent250/handoff.json')['source_manifest_sha256']==digest('source/parent250/sources.json'),'parent handoff')
    archives=[]; maps=[]
    for v,fail,code,session,chunk in [(1,6,1,35774,'07788f'),(2,0,0,32409,'d48ad9')]:
        p='proof/v%d/'%v; r='reviews/root-v%d/'%v; s='source/v%d/'%v
        mapping=doc(s+'sources.json'); require(len(mapping)==261,'source count')
        require(raw(s+'sources.json')==(json.dumps(mapping,sort_keys=True,separators=(',',':'))+'\n').encode(),'compact map')
        for n in ('sources-before.json','sources-after.json','execution-after.json'):
            require(raw(p+n)==raw(s+'sources.json'),'before/after map '+n)
        values=archive(p+'sources.tar.gz',mapping); archives.append(values); maps.append(mapping)
        require({n.removeprefix('runtime/'):h for n,h in mapping.items() if n.startswith('runtime/')}==parent,'parent250 exact')
        require(values['runtime_sources.json']==raw('source/parent250/sources.json'),'nested parent map')
        config=doc(p+'run_config.json'); report=doc(p+'report.json'); audit=doc(p+'worker-audit.json'); live=doc(p+'live-observation.json')
        select=json.loads(values['selected_tests.json']); nodes=config['selected_nodes']
        require(len(nodes)==48 and len(set(nodes))==48 and nodes==select['ordered_nodes']==report['selected_nodes']==audit['collected_nodes'],'ordered48')
        cases=list(ET.fromstring(raw(p+'tests.xml')).iter('testcase'))
        xml_cases=[{'classname':c.attrib['classname'],'name':c.attrib['name']} for c in cases]
        require(xml_cases==config['selected_xml_cases']==select['ordered_xml_cases']==report['selected_xml_cases'],'ordered XML')
        counts={'tests':len(cases),'failures':sum(c.find('failure') is not None for c in cases),'errors':sum(c.find('error') is not None for c in cases),'skipped':sum(c.find('skipped') is not None for c in cases)}
        require(counts=={'tests':48,'failures':fail,'errors':0,'skipped':0}==report['tests'],'test outcome')
        require(report['passed'] is (v==2) and report['child_returncode']==audit['pytest_exit_code']==code,'typed outcome')
        require(report['child_reaped'] is True and report['child_group_absent'] is True and report['sources_unchanged'] is True,'reported child retirement')
        for flag in ['actual_execution_authorized','actual_client_role_execution','actual_key_generation','backend_context_construction','capacity_measurement','encrypted_execution','model_execution','actual_encrypted_metadata_captured','first_encrypted_divergence_claim','prior140_PASS_inherited']:
            require(report[flag] is False,'scope '+flag)
        require(config['cpu_affinity']==audit['affinity']==[108,109,110,111],'affinity')
        require(audit['denied']==audit['loaded_backend']==[] and audit['numpy_imported'] is True,'backend audit')
        require(audit['forbidden_roots']==config['forbidden_roots'],'denial set')
        expect=config['runtime_source_expectations']; require(len(expect)==4 and audit['runtime_sources_before']==audit['runtime_sources_after']==expect,'NumPy4')
        require(set(audit['runtime_module_origins'])==set(config['runtime_module_origins'])=={'numpy'},'root-only NumPy origin')
        for mod,path in config['runtime_module_origins'].items(): require(audit['runtime_module_origins'][mod]=={'file':path,'spec':path},'NumPy origin')
        require(type(audit['pid']) is int and audit['pid']==live['worker_pid'] and audit['pgid']==live['worker_pgid']==audit['pid'],'worker PID/PGID')
        require(doc(p+'owned-pytest-metadata.json')=={} and report['owned_pytest_files']==0,'no retained payloads')
        for key,file in [('config_sha256','run_config.json'),('driver_sha256','executed_driver.py'),('worker_driver_sha256','worker_driver.py'),('source_archive_sha256','sources.tar.gz')]:
            require(report[key]==digest(p+file),'report link '+key)
        require(report['source_manifest_sha256']==digest(s+'sources.json') and report['source_count']==261,'report source binding')
        review=doc(r+'terminal-review.json'); owner=doc(r+'direct-owner.json'); pre=doc(r+'preexecution-review.json'); grant=doc(r+'run-grant.json')
        require(review['counts']==counts and review['passed'] is (v==2) and review['evidence_verified'] is True,'root outcome')
        require(review['source_current_execution_before_after_archive_equal'] is True and review['outer_and_worker_pid_and_group_absent'] is True,'root source/absence attestation')
        require(review['independent_live_observation'] is False,'no independent live claim')
        require(owner['session_id']==session and owner['terminal_tool_chunk']==chunk and owner['exit_code']==code and owner['direct_session_reaped'] is True,'direct owner')
        for key,file in [('direct_owner_receipt_sha256','direct-owner.json'),('root_preexecution_review_sha256','preexecution-review.json'),('root_run_grant_sha256','run-grant.json'),('script_sha256','terminal.py')]:
            require(review[key]==digest(r+file),'root receipt link '+key)
        require(pre['script_sha256']==digest(r+'preexecution.py'),'preexecution script')
        for n,h in review['evidence'].items(): require(digest(p+n)==h,'root evidence '+n)
        require(grant['root_preexecution_review_sha256']==digest(r+'preexecution-review.json'),'grant review')
        for n,h in grant['harness_pins'].items(): require(digest(p+n)==h,'grant harness '+n)
        if current:
            for base in (pathlib.Path(config['source_root']),pathlib.Path(config['proof_root'])/'execution'):
                for n,h in mapping.items(): require(sha(read(base/n))==h,'current public source '+str(base/n))
            for path,h in expect.items(): require(sha(read(pathlib.Path(path)))==h,'current NumPy source '+path)
    require(set(maps[0])==set(maps[1]) and [n for n in maps[0] if maps[0][n]!=maps[1][n]]==['q5_test_fixtures.py'],'one fixture overlay')
    correction=doc('source/v2/history/correction.json')
    require(raw('source/v2/history/before_q5_test_fixtures.py')==archives[0]['q5_test_fixtures.py'],'preserved old fixture')
    require(raw('source/v1/readable/q5_test_fixtures.py')==archives[0]['q5_test_fixtures.py'],'readable old fixture')
    for n in names:
        if n.startswith('source/parent250/readable/'):
            rel=n.removeprefix('source/parent250/readable/'); require(sha(raw(n))==parent[rel] and raw(n)==archives[1]['runtime/'+rel],'readable parent '+rel)
        if n.startswith('source/v2/readable/'):
            rel=n.removeprefix('source/v2/readable/'); require(raw(n)==archives[1][rel],'readable v2 '+rel)
    old,new=map(lambda b: ast.parse(b),[archives[0]['q5_test_fixtures.py'],archives[1]['q5_test_fixtures.py']])
    # Independently check literal-index assignments stay identical; never execute them.
    assignments=lambda t:{ast.dump(n.targets[0],include_attributes=False):ast.dump(n.value,include_attributes=False) for n in t.body if isinstance(n,ast.Assign) and len(n.targets)==1}
    require(assignments(old)==assignments(new),'module literal expectations unchanged')
    tree=ast.parse(raw('proof/v2/worker_driver.py')); pytest_calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and isinstance(n.func.value,ast.Name) and n.func.value.id=='pytest' and n.func.attr=='main']
    require(len(pytest_calls)==1,'pytest call count')
    require(doc('proof/v2/run_config.json')['pytest_config_path']=='/dev/null' and "'/dev/null'" in raw('proof/v2/worker_driver.py').decode(),'explicit pytest config')
    require(doc('summary.json')['prior140_PASS_inherited'] is False and doc('summary.json')['actual_role_authorized'] is False,'summary scope')
    return {'passed':True,'mode':'current-public-sources' if current else 'portable','files':len(names),'indexed_files':len(rows),'source_counts':[261,261],'parent250_exact':True,'v1':{'pass':42,'fail':6},'v2':{'pass':48,'fail':0},'new_test_execution':False,'borrowed_body_reads':False,'independent_live_observation':False}

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--current',action='store_true'); args=parser.parse_args()
    print(json.dumps(main(args.current),sort_keys=True))
