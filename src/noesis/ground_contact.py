"""Force-bearing tarsal ground contacts, separately from body and obstacle contacts."""
import numpy as np

from noesis.sensory import LEGS


def classify_contacts(records, fly_name):
    """Records are (geom1 name, geom2 name, normal force); self contacts never support."""
    loads = np.zeros(6)
    obstacle = 0.
    prefix = fly_name + '/'
    for first, second, force in records:
        if not np.isfinite(force) or force <= 0:
            continue
        if first.startswith(prefix) and not second.startswith(prefix):
            part, surface = first[len(prefix):], second
        elif second.startswith(prefix) and not first.startswith(prefix):
            part, surface = second[len(prefix):], first
        else:
            continue
        surface = surface.split('/')[-1]
        if surface == 'ground' and part[:2] in LEGS and part[2:].startswith('Tarsus'):
            loads[LEGS.index(part[:2])] += force
        if surface == 'obstacle' or surface.startswith('wall_'):
            obstacle += force
    return loads, obstacle


def ground_contacts(sim, fly):
    import mujoco
    model, data = sim.physics.model.ptr, sim.physics.data.ptr
    # Minimal test doubles contain no contacts. Real MuJoCo always has ncon.
    if not hasattr(data, 'ncon'):
        return np.zeros(6), 0.
    records = []
    force = np.zeros(6)
    for i in range(data.ncon):
        contact = data.contact[i]
        names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(g)) or ''
                 for g in (contact.geom1, contact.geom2)]
        mujoco.mj_contactForce(model, data, i, force)
        records.append((*names, float(force[0])))
    return classify_contacts(records, fly.name)
