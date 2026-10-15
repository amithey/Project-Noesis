from types import SimpleNamespace

import numpy as np
import pytest

from noesis.ground_contact import classify_contacts, ground_contacts


def test_only_external_tarsal_ground_contacts_support_the_fly():
    loads, obstacle = classify_contacts([
        ('fly/LFTarsus1', 'ground', 2.), ('ground', 'fly/RMTarsus5', 3.),
        ('fly/Thorax', 'ground', 100.), ('fly/LFFemur', 'ground', 80.),
        ('fly/LFTarsus1', 'fly/RFTarsus1', 40.), ('fly/LHTarsus1', 'obstacle', 4.),
        ('fly/Head', 'obstacle', 6.), ('fly/RHTarsus1', 'ground', 0.),
    ], 'fly')
    np.testing.assert_array_equal(loads, [2., 0., 0., 0., 3., 0.])
    assert obstacle == 10.


def test_real_mujoco_contact_force_adapter():
    mj = pytest.importorskip('mujoco')
    model = mj.MjModel.from_xml_string("""<mujoco><worldbody>
    <geom name="ground" type="plane" size="3 3 .1"/>
    <body pos="0 0 .05"><freejoint/><geom name="fly/LFTarsus5" type="sphere" size=".1"/></body>
    </worldbody></mujoco>""")
    data = mj.MjData(model)
    mj.mj_forward(model, data)
    sim = SimpleNamespace(physics=SimpleNamespace(model=SimpleNamespace(ptr=model),
                                                 data=SimpleNamespace(ptr=data)))
    loads, obstacle = ground_contacts(sim, SimpleNamespace(name='fly'))
    assert loads[0] > 0 and loads[1:].sum() == 0 and obstacle == 0
