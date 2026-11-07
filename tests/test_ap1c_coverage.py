import json
from pathlib import Path
import numpy as np
import pytest
import support_descriptor_c0 as old
import support_descriptor_c1 as new


def test_explicit_native_compound_link_is_proximal_in_each_leg_and_conserves_force():
    for leg in ('lf','lm','lh','rf','rm','rh'):
        name=f'nmf/{leg}_trochanterfemur'
        assert new.anatomical_region(name)=='proximal_leg'
        with pytest.raises(ValueError):old.anatomical_region(name)
        r=new.describe_support([[0,0,2],[0,0,3]],[name,'nmf/lf_tarsus5'])
        assert r['contact_mode']=='proximal_leg_contact' and r['reaction_share_by_region']['proximal_leg']==.4
        assert sum(r['vertical_reaction_by_region'].values())==5
    with pytest.raises(ValueError):new.describe_support([[0,0,1]],['nmf/rh_unknown'])


def test_all_previously_supported_inputs_remain_exact():
    names=['nmf/c_abdomen6','nmf/lf_coxa','nmf/lh_tibia','nmf/rh_tarsus5','nmf/rm_tarsus1']
    rng=np.random.default_rng(0)
    for _ in range(50):
        forces=rng.uniform(0,10,(len(names),3))
        assert old.describe_support(forces,names)==new.describe_support(forces,names)


def test_actual_compiled_floor_collision_coverage_and_native_readonly_inputs():
    from noesis_gpu.backend import FlyGymBatch
    from ap1c_body import observing_body
    import mujoco
    a=FlyGymBatch(1,'cpu');expected=a.observe();instances=[]
    b=observing_body(FlyGymBatch,instances)(1,'cpu');actual=b.observe();m=b.sim.mj_model
    for k in expected:np.testing.assert_array_equal(actual[k],expected[k])
    assert len(b.ap1_records)==1 and b.ap1_records[0]['time']==0
    ground=set(b.sim._internal_ground_geom_ids.tolist());potential=set()
    for i in range(m.ngeom):
        if i not in ground and any((m.geom_contype[i]&m.geom_conaffinity[j]) or (m.geom_contype[j]&m.geom_conaffinity[i]) for j in ground):potential.add(i)
    implicit_count=len(potential);explicit=[]
    for i,(g1,g2) in enumerate(zip(m.pair_geom1,m.pair_geom2)):
        if g1 in ground:geom=int(g2)
        elif g2 in ground:geom=int(g1)
        else:continue
        potential.add(geom);explicit.append(i)
    rows=[]
    for geom in sorted(potential):
        body=int(m.geom_bodyid[geom]);name=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,body)
        rows.append(dict(geom_id=geom,body_id=body,body_name=name,region=new.anatomical_region(name)))
    assert implicit_count==0 and len(explicit)==m.npair==55 and len(rows)==55
    compound=[r for r in rows if r['body_name'].endswith('_trochanterfemur')]
    assert len(compound)==6 and all(r['region']=='proximal_leg' for r in compound)
    root=Path(__file__).resolve().parents[1]
    (root/'outputs/anatomical-coverage-ap1c-static-20261106.json').write_text(json.dumps(dict(
        physical_integration_steps=0,neural_steps=0,model_time=float(b.sim.mj_data.time),
        implicit_ground_geoms=implicit_count,explicit_ground_pairs=len(explicit),all_ground_geometry_classified=True,
        compound_proximal_geoms=6,production_collision_pairs_unchanged=True,native_inputs_exact=True,
        fixture_is_not_learned_fly_success=True,rows=rows),indent=2),encoding='utf-8')
