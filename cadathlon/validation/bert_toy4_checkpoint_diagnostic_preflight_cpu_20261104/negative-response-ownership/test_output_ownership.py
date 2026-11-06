"""Synthetic retained-output replacement: no backend, private or payload reads."""
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parent/'sources'))
import collect as collector
import component_worker as worker
import component_protocol as p

@pytest.mark.parametrize('module',[collector,worker])
def test_replaced_acquired_output_keeps_foreign_receipt_untouched(tmp_path,monkeypatch,module):
 out=tmp_path/'owned';retired=tmp_path/'retained';sentinel=b'{"foreign":true}\n';primary=RuntimeError('body failure after output replacement')
 def body(*args):
  out.rename(retired);out.mkdir(mode=0o700);(out/'report.json').write_bytes(sentinel)
  raise primary
 monkeypatch.setattr(module,'collect'if module is collector else 'run',body)
 args=['command','--output',str(out)]
 if module is worker:
  config=tmp_path/'config';config.write_text('{}');source=tmp_path/'source';source.write_text('{}')
  monkeypatch.setattr(p,'require_ready',lambda:None)
  args+=['--execute','--config',str(config),'--config-sha256',p.sha(config.read_bytes()),'--source-manifest',str(source)]
 monkeypatch.setattr(sys,'argv',args)
 with pytest.raises(RuntimeError)as caught:module.main()
 assert caught.value is primary
 assert(out/'report.json').read_bytes()==sentinel,'foreign replacement directory was republished into'
