import sys
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
import collect_component as c
def test_existing_report_must_not_change(tmp_path,monkeypatch):
 out=tmp_path/'existing';out.mkdir();report=out/'report.json';old=b'{"passed":true,"foreign":"retained"}';report.write_bytes(old)
 monkeypatch.setattr(sys,'argv',['collector','--output',str(out),'--key-evidence',str(tmp_path/'keys-proof'),'--key-workspace',str(tmp_path/'keys')])
 with pytest.raises(FileExistsError):c.main()
 assert report.read_bytes()==old
