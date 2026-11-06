"""Owned evidence publication with explicit non-authority until outer reap/root review."""
from q5_checkpoint_transport import Exporter
from q5_checkpoint_contract import parse

class EvidenceOutput(Exporter):
    NAMES=frozenset(('setup_manifest.json','client_result.json','report.json','failure.json','failed_checkpoint.json'))
    def __init__(self,path,expected_parent,bindings):
        self.written=set();super().__init__(None,path,expected_parent,{},bindings)
    def write(self,name,raw):
        self.check()
        if name not in self.NAMES or name in self.written or type(raw)is not bytes or not 0<len(raw)<=1<<20:
            raise ValueError('one bounded declared evidence file required')
        value=parse(raw)
        if type(value)is not dict:raise ValueError('public evidence object required')
        record=self._write(name,lambda stream:stream.write(raw),1<<20);self.written.add(name);self.check();return record
