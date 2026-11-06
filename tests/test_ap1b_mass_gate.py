import json
import numpy as np
import pytest
from run_anatomical_ap1b import anatomy,root


def test_full_native_replay_passes_existing_mass_tolerance_without_hiding_discrepancy():
    directory=root/'outputs/anatomical-feedback-ap1-20261106/train-control-1005'
    model=json.loads((directory/'anatomy-model.json').read_text())
    assert model['total_mass']!=model['model_total_mass']
    assert np.isclose(model['total_mass'],model['model_total_mass'],rtol=1e-12,atol=1e-15)
    assert len(anatomy(directory,'train')['time'])==5001


def test_meaningful_mass_loss_is_rejected_by_same_full_trace_gate(monkeypatch):
    import run_anatomical_ap1b as runner
    original=runner.read
    def altered(path):
        value=original(path)
        if path.name=='anatomy-model.json':value={**value,'total_mass':value['total_mass']*.999}
        return value
    monkeypatch.setattr(runner,'read',altered)
    with pytest.raises(AssertionError):runner.anatomy(root/'outputs/anatomical-feedback-ap1-20261106/train-control-1005','train')
