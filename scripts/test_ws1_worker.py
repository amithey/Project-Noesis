import json
import sys
import types
import pytest
import ws1_worker
from noesis_gpu import controller


@pytest.mark.parametrize('outcome',['completed','delegation_failure','missing_import'])
def test_frozen_delegation_context_and_patch_restoration(tmp_path,monkeypatch,outcome):
    cp=tmp_path/'weights.npz';cp.write_bytes(b'unit test checkpoint')
    output=tmp_path/'probe'
    argv=['ws1_worker','--root',str(tmp_path),'--output',str(output),'--checkpoint',str(cp),
          '--seed','10006','--variant','plus']
    monkeypatch.setattr(sys,'argv',argv)
    original=controller.import_checkpoint
    monkeypatch.setattr(ws1_worker,'checked_import',lambda native,net,path: {'actual_import_exact':True})
    def delegate():
        assert sys.argv[1:7]==['--phase','frozen','--bank','candidate','--seed','10006']
        if outcome=='delegation_failure':raise RuntimeError('test failure')
        if outcome!='missing_import':controller.import_checkpoint(None,cp)
        output.mkdir()
        report=dict(learning=False,weights_unchanged=True,ps1_worker_completed=True,
                    checkpoint_sha256=ws1_worker.hashlib.sha256(cp.read_bytes()).hexdigest())
        (output/'report.json').write_text(json.dumps(report))
    monkeypatch.setitem(sys.modules,'ps1_worker',types.SimpleNamespace(main=delegate))
    if outcome=='completed':ws1_worker.main()
    else:
        with pytest.raises((AssertionError,RuntimeError)):ws1_worker.main()
    assert controller.import_checkpoint is original and sys.argv is argv
    context=json.loads((tmp_path/'probe-ws1-context.json').read_text())
    assert context['worker_completed']==(outcome=='completed')
    assert context['constructed_not_trained'] and not context['learning']
    if output.exists():
        report=json.loads((output/'report.json').read_text())
        assert report['ws1_context']==context
