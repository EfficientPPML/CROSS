"""Mock-only frozen-source reproduction of cleanup exception precedence."""
import builtins,json
from pathlib import Path
import boundaries
from test_boundaries import fake
import pytest

def test_later_cleanup_cancellation_currently_loses_precedence(fake,monkeypatch):
    restores=[];restoring=False
    def assignment(owner,name,value):
        builtins.setattr(owner,name,value)
        if restoring:
            restores.append(name)
            if len(restores)==1:raise ValueError('first ordinary restore error')
            if len(restores)==2:raise KeyboardInterrupt('second cancellation')
    monkeypatch.setattr(boundaries,'setattr',assignment,raising=False)
    with pytest.raises(ValueError) as caught:
        with boundaries.encryption_only([]):restoring=True
    assert len(restores)==21 and any('KeyboardInterrupt' in n for n in caught.value.__notes__)
    Path(__file__).with_name('result.json').write_text(json.dumps({'scope':'Mock cleanup only; no actual CKKS/model call.','observed_exception':type(caught.value).__name__,'all_restorations_attempted':len(restores),'cancellation_only_noted':True},indent=2)+'\n')
