import numpy as np
import pandas as pd
import pytest

from noesis.proprioception import TypedSensoryMap, make_proprioception_map
from noesis.sensory import LEGS, LegState, SensoryMap


def neurons():
    kinds=['claw','claw','hook','hook','hook','claw','hook','bristle']
    types=['SNpp50','SNpp51','SNpp39','SNpp41','SNpp51','SNpp39','unknown','other']
    return pd.DataFrame({'Flow':['afferent']*8,
        'Sub Class':[f'front_leg_{kind}_chordotonal_organ_neuron' if kind in ('claw','hook') else 'front_leg_bristle_neuron' for kind in kinds],
        'Nerve':['left_prothoracic_leg_nerve']*8,'Soma side':['left']*8,
        'Primary Cell Type':types},index=[5,8,12,16,21,24,30,37])


def states(angle=0.,velocity=0.,load=0.):
    return {leg:LegState(tibia=angle,tibia_vel=velocity,load=load) for leg in LEGS}


def rates(mapping,state):
    return dict(zip(mapping.idx,mapping.drive(state),strict=True))


def test_typed_identity_order_and_effective_organs_preserve_annotated_cells():
    frame=neurons();saved=frame.copy(deep=True)
    original=SensoryMap(frame);typed=TypedSensoryMap(frame)
    np.testing.assert_array_equal(typed.idx,original.idx)
    organs={int(idx):kind for kind,leg,ids in typed.groups for idx in ids}
    assert organs[21]=='claw' and organs[24]=='hook'
    assert typed.coverage()['typed_neurons']==6
    assert typed.coverage()['unresolved_hook_neurons']==1
    pd.testing.assert_frame_equal(frame,saved)


@pytest.mark.parametrize('angle',[-2.,0.,2.])
@pytest.mark.parametrize('encoding',['typed-resistance','typed-reversed'])
def test_known_hooks_are_quiet_during_holds_at_every_position(angle,encoding):
    result=rates(TypedSensoryMap(neurons(),encoding),states(angle,0.))
    assert [result[idx] for idx in (12,16,24)]==[2.,2.,2.]


@pytest.mark.parametrize('velocity',[-3.,3.])
def test_hook_direction_is_signed_velocity_and_reversal_is_an_explicit_control(velocity):
    resistance=rates(TypedSensoryMap(neurons()),states(.4,velocity))
    reversed_rates=rates(TypedSensoryMap(neurons(),'typed-reversed'),states(.4,velocity))
    assert (resistance[12]>2)==(velocity<0)
    assert (resistance[16]>2)==(velocity>0)
    assert resistance[12]==reversed_rates[16]
    assert resistance[16]==reversed_rates[12]
    assert resistance[30]==reversed_rates[30]  # unknown tuning is not invented


def test_claw_types_encode_opposite_positions_and_keep_tonic_hold_rates():
    mapping=TypedSensoryMap(neurons())
    flexed=rates(mapping,states(1.,0.));extended=rates(mapping,states(-1.,0.))
    assert flexed[5]>flexed[8] and extended[8]>extended[5]
    assert flexed[21]==flexed[8]  # primary type corrects conflicting subclass
    assert rates(mapping,states(1.,3.))[5]==flexed[5]
    assert rates(mapping,states(1.,-3.))[8]==flexed[8]


def test_unknown_types_remain_exactly_legacy_and_default_factory_is_legacy():
    frame=neurons();legacy=SensoryMap(frame);typed=TypedSensoryMap(frame)
    rng=np.random.default_rng(50)
    for _ in range(100):
        state=states(rng.uniform(-2,2),rng.uniform(-10,10),rng.uniform(0,60))
        old=rates(legacy,state);new=rates(typed,state)
        assert old[30]==new[30] and old[37]==new[37]
        np.testing.assert_array_equal(make_proprioception_map(frame).drive(state),legacy.drive(state))
        assert np.all((typed.drive(state)>=2)&(typed.drive(state)<=150))


def test_type_mapping_does_not_depend_on_dataframe_row_parity_or_randomness():
    frame=neurons();first=rates(TypedSensoryMap(frame),states(.6,-2.))
    shuffled=frame.sample(frac=1,random_state=7)
    assert rates(TypedSensoryMap(shuffled),states(.6,-2.))==first
    np.testing.assert_array_equal(TypedSensoryMap(frame).idx,TypedSensoryMap(frame).idx)


def test_missing_primary_labels_do_not_invent_preferences():
    frame=neurons().drop(columns='Primary Cell Type')
    legacy=SensoryMap(frame);typed=TypedSensoryMap(frame)
    np.testing.assert_array_equal(legacy.idx,typed.idx)
    np.testing.assert_array_equal(legacy.drive(states(-.4,3.)),typed.drive(states(-.4,3.)))
    assert typed.coverage()['typed_neurons']==0


def test_organ_ablation_uses_effective_primary_type_and_retains_order():
    full=TypedSensoryMap(neurons());hooks=TypedSensoryMap(neurons(),kinds=frozenset({'hook'}))
    assert set(hooks.idx)=={12,16,24,30}
    full_rates=rates(full,states(.2,3.))
    assert rates(hooks,states(.2,3.))=={idx:full_rates[idx] for idx in hooks.idx}


def test_unknown_encoding_rejected_and_returned_coverage_cannot_mutate_mapping():
    with pytest.raises(ValueError,match='encoding'):
        make_proprioception_map(neurons(),'unknown')
    mapping=TypedSensoryMap(neurons());coverage=mapping.coverage()
    coverage['types']['SNpp39']=500
    assert mapping.coverage()['types']['SNpp39']==2
