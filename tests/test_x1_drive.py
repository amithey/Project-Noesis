import numpy as np
import pytest
from x1_drive import current_intervention,transfer_interaction


def test_current_mask_preserves_targets_and_input():
    source={'current_idx':np.array([2,4]),'current_mv_per_ms':np.array([.3,-.8])}
    on=current_intervention(source,True);off=current_intervention(source,False)
    np.testing.assert_array_equal(on['current_mv_per_ms'],source['current_mv_per_ms'])
    np.testing.assert_array_equal(off['current_mv_per_ms'],[0.,0.])
    np.testing.assert_array_equal(off['current_idx'],source['current_idx'])
    np.testing.assert_array_equal(source['current_mv_per_ms'],[.3,-.8])
    assert current_intervention({},False)=={}


def test_interaction_requires_all_cells_and_uses_paired_effects():
    fields=('standing_fraction','mean_stance_slip_mm_s','heading_forward_distance_mm','net_forward_displacement_mm','total_leg_steps')
    cells={k:{n:v for n in fields} for k,v in [(('parent','on'),1.),(('parent','off'),2.),(('child','on'),3.),(('child','off'),1.)]}
    r=transfer_interaction(cells)
    assert all(v==3 for v in r['difference_of_differences_on_minus_off'].values())
    cells.pop(('parent','off'))
    with pytest.raises(ValueError):transfer_interaction(cells)
