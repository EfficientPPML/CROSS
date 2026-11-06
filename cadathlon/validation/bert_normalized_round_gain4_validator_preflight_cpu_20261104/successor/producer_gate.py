"""Bind the immutable163-source producer separately from the new consumer.

Only public reports/source archives are read here. No key, ciphertext, fixture
array or private client path is opened; the unchanged terminal gate remains the
parent/worker/workspace authority. An unbound server pin always fails first.
"""
from pathlib import Path
import component_protocol as protocol
import component_key_gate as key_gate

HERE=Path(__file__).resolve().parent
FIXED={
 'schema':'gain4-fixed-producer-binding-v1',
 'producer_source_count':163,
 'producer_map_sha256':'d8533cc0c770683691fb01fb078977a0e248812696ca7548edc537af2d93cbec',
 'ready_report_sha256':'6fbf898fbcb8a884e2f2c7f1e2308dfd04171be990bfde82a20200d20f00e19c',
 'ready_archive_sha256':'382933c982162e6d83c7d5cc366e59b7abfd674ef408efcc529c9ba13db912ce',
 'fixture_report_sha256':'aca358a292ca8cc867241e09abec18889b20b31ccc4111b4fa66b2830ff271ad',
 'client_io_preflight_report_sha256':'c5b49a569c48c678aac2152ec576be12d3dd63973241a50b0f1819a0bb314e2f',
 'client_io_preflight_archive_sha256':'f9265bea1194dd4d9c35fede0e6aad279e30eb8d3a5b34da2feda9b8fe515bb6'}
PROTOCOL_SHA='17dc37314a939da70c9055e8d3e75404e23628d8c9dd603fb6aceafb6f12819e'
_read=key_gate.read

def bindings(root=HERE):
 root=Path(root);value=key_gate.parse(_read(root/'producer_bindings.json'))
 if type(value)is not dict or set(value)!=set(FIXED)|{'server_report_sha256','server_root_review_sha256'}:raise ValueError('strict fixed producer binding schema required')
 for n,wanted in FIXED.items():
  if type(value[n])is not type(wanted)or value[n]!=wanted:raise ValueError('fixed producer binding differs: '+n)
 for n in ('server_report_sha256','server_root_review_sha256'):
  if value[n]is not None:key_gate.sha_field(value[n])
 if (value['server_report_sha256']is None)!=(value['server_root_review_sha256']is None):raise ValueError('server terminal and independent review must be bound together')
 return value

def inspect(root=HERE):
 root=Path(root);bound=bindings(root);raw=_read(root/'producer_sources.json',1<<20)
 if protocol.sha(raw)!=FIXED['producer_map_sha256']:raise ValueError('immutable producer source map differs')
 sources=key_gate.parse(raw)
 if type(sources)is not dict or len(sources)!=163:raise ValueError('exact163 producer source inventory required')
 for n,h in sources.items():protocol.relative(n);key_gate.sha_field(h)
 if sources.get('component_protocol.py')!=PROTOCOL_SHA or protocol.sha(_read(root/'component_protocol.py'))!=PROTOCOL_SHA:raise ValueError('unchanged producer terminal gate required')
 raw=_read(root/'producer_ready/report.json')
 if protocol.sha(raw)!=FIXED['ready_report_sha256']:raise ValueError('externally pinned ready proof differs')
 report=key_gate.parse(raw)
 flags=('passed','all_direct_children_reaped','all_owned_groups_empty','process_cleanup_completed','source_files_unchanged','source_archive_verified','terminal_scope_completed','termination_scope_completed','no_encrypted_execution','no_private_or_ciphertext_payload_read')
 if (type(report)is not dict or report.get('schema')!='cross-normalized-gain4-preflight-v1' or report.get('execute')is not False
     or any(report.get(n)is not True for n in flags)or report.get('cleanup_errors')!=[]
     or report.get('source_hashes_start')!=sources or report.get('source_hashes_end')!=sources
     or report.get('source_archive_sha256')!=FIXED['ready_archive_sha256']):raise ValueError('completed exact immutable producer ready proof required')
 for name,count in (('mock_tests',415),('installed_denials',10)):
  row=report.get(name)
  if type(row)is not dict or any(type(row.get(n))is not int or row[n]!=v for n,v in dict(tests=count,errors=0,failures=0,skipped=0,exit_code=0).items()):raise ValueError('complete fixed producer preflight tests required')
 archive=_read(root/'producer_ready/sources.tar.gz',16<<20)
 if protocol.sha(archive)!=FIXED['ready_archive_sha256']:raise ValueError('producer archive differs')
 key_gate.verify_source_archive(archive,sources)
 return dict(producer_sources=dict(sources),bindings=dict(bound),ready_report_sha256=FIXED['ready_report_sha256'],ready_archive_sha256=FIXED['ready_archive_sha256'],scope='Old163 producer authentication only; consumer source closure remains separate.')

def _authenticate(evidence,report_sha,*,stage,workspace,workspace_identity,request,root):
 bound=bindings(root)
 name='fixture_report_sha256'if stage=='fixture'else'server_report_sha256'
 expected=bound[name]
 if expected is None or stage=='server'and bound['server_root_review_sha256']is None:raise ValueError('server terminal pin/review still pending')
 key_gate.sha_field(report_sha)
 if report_sha!=expected:raise ValueError('externally bound producer stage hash required')
 checked=inspect(root)
 return protocol.terminal_gate(evidence,expected,stage=stage,expected_sources=checked['producer_sources'],workspace=workspace,workspace_identity=workspace_identity,request=request)

def authenticate_fixture(evidence,report_sha,*,workspace,workspace_identity,request,root=HERE):
 return _authenticate(evidence,report_sha,stage='fixture',workspace=workspace,workspace_identity=workspace_identity,request=request,root=root)

def authenticate_server(evidence,report_sha,*,workspace,workspace_identity,request,root=HERE):
 return _authenticate(evidence,report_sha,stage='server',workspace=workspace,workspace_identity=workspace_identity,request=request,root=root)
