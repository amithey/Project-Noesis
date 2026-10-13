"""Newborn fly on the muscle-driven body: the brain decides, the body only obeys physics.

    brain (158k LIF neurons) -> leg motor-neuron pools -> antagonistic muscles (activation
    dynamics) -> joint equilibrium angles -> tissue servos with joint limits -> gravity/contact

There is no joint-target formula, gain or slew limit in this chain: the only mapping is the
muscle model in ``noesis.body``. Nothing is learned. The body starts in the real standing
posture, dropped from a small height with its muscles relaxed; at ``--birth`` the brain wakes.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from noesis.balance import BalanceMap
from noesis.body import BodyParams, MuscleModel, make_fly
from noesis.connectome import load_cached
from noesis.lif import LIFNetwork, LIFParams
from noesis.motor import RateTracker, build_muscle_matrices
from noesis.sensory import LEGS, LegState, SensoryMap


def _world_kwargs() -> dict:
    """The shared physical world (floor, gravity, contact) when available."""
    try:
        from noesis import world
    except ImportError:
        return {}
    return {"arena": world.floor_arena(), "gravity": world.GRAVITY}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--seconds", type=float, default=3.0, help="simulated time")
    ap.add_argument("--birth", type=float, default=0.5, help="when the brain wakes (s)")
    ap.add_argument("--bg-hz", type=float, default=5.0, help="spontaneous events/s per neuron")
    ap.add_argument("--bg-weight", type=float, default=90.0, help="synapse-equivalents per event")
    ap.add_argument("--adapt", type=float, default=1.0, help="spike-frequency adaptation, mV/spike")
    ap.add_argument("--dt-ms", type=float, default=0.5, help="LIF time step")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-senses", action="store_true", help="no leg sensory feedback")
    ap.add_argument("--no-balance", action="store_true", help="no gravity/rotation senses")
    ap.add_argument("--viewer", action="store_true", help="live MuJoCo window")
    ap.add_argument("--video", type=Path, help="write an mp4 (slower)")
    a = ap.parse_args()

    from flygym import Camera, PhysicsError, SingleFlySimulation, preprogrammed

    cn = load_cached(a.data / "raw", a.data / "processed")
    net = LIFNetwork(cn.W, LIFParams(dt=a.dt_ms, da=a.adapt), seed=a.seed)

    params = BodyParams()
    fly = make_fly(params, spawn_pos=(0.0, 0.0, 1.2),
                   contact_sensor_placements=preprogrammed.get_collision_geometries("legs"))
    cam = None
    if a.video:
        cam = Camera(attachment_point=fly.model.worldbody, camera_name="camera_left",
                     targeted_fly_names=fly.name, play_speed=0.1, fps=30)
    sim = SingleFlySimulation(fly=fly, cameras=[cam] if cam else [], timestep=1e-4, **_world_kwargs())
    try:
        from noesis.world import configure_floor_contacts
        configure_floor_contacts(sim)
    except ImportError:
        pass
    obs, _ = sim.reset()

    names = list(fly.actuated_joints)
    g_pos, g_neg = build_muscle_matrices(cn.neurons, names)
    muscles = MuscleModel(g_pos, g_neg, names, params)
    rest = muscles.stand
    rates = RateTracker(cn.n, tau_ms=20.0)
    print(f"muscle pools: {g_pos.nnz} agonist + {g_neg.nnz} antagonist motor-neuron links, "
          f"{int((g_pos.getnnz(axis=1) > 0).sum())} joints with an agonist, "
          f"{int((g_neg.getnnz(axis=1) > 0).sum())} with an antagonist")

    senses = None if a.no_senses else SensoryMap(cn.neurons)
    balance = None if a.no_balance else BalanceMap(cn.neurons)
    j = {(leg, d): names.index(f"joint_{leg}{d}") for leg in LEGS for d in ("Coxa", "Femur", "Tibia")}
    leg_of_sensor = np.array([LEGS.index(n.split("/")[-1][:2]) for n in fly.contact_sensor_placements])

    def leg_states(obs):
        ang, vel = obs["joints"][0] - rest, obs["joints"][1]
        load = np.bincount(leg_of_sensor, weights=np.linalg.norm(obs["contact_forces"], axis=1), minlength=6)
        return {leg: LegState(ang[j[leg, "Tibia"]], vel[j[leg, "Tibia"]], ang[j[leg, "Coxa"]],
                              ang[j[leg, "Femur"]], load[LEGS.index(leg)]) for leg in LEGS}

    def sensory_drive(obs):
        idx, rate = [], []
        if senses is not None:
            idx.append(senses.idx)
            rate.append(senses.drive(leg_states(obs)))
        if balance is not None:
            cv = obs["cardinal_vectors"]
            idx.append(balance.idx)
            rate.append(balance.drive(np.array([cv[0][2], cv[1][2], cv[2][2]]), obs["fly"][3]))
        if not idx:
            return {}
        return {"drive_idx": np.concatenate(idx), "drive_rate_hz": np.concatenate(rate),
                "drive_weight": 400.0}

    viewer = None
    if a.viewer:
        import mujoco.viewer
        viewer = mujoco.viewer.launch_passive(sim.physics.model.ptr, sim.physics.data.ptr)

    chunk_ms = 5.0
    phys_per_chunk = round(chunk_ms * 1e-3 / sim.timestep)
    adhesion = np.ones(6, int)  # tarsal pads stick wherever they touch; not a decision of the brain
    command = rest.copy()
    t_wall, awake = time.time(), False
    motion, height, up = [], [], []
    for k in range(round(a.seconds * 1e3 / chunk_ms)):
        t = k * chunk_ms * 1e-3
        if t >= a.birth:
            if not awake:
                print(f"t={t:.2f}s  *** the brain wakes ***", flush=True)
                awake = True
            _, spikes = net.run(chunk_ms, background_hz=a.bg_hz, background_weight=a.bg_weight,
                                **sensory_drive(obs))
            command = muscles.step(rates.update(spikes, chunk_ms), chunk_ms)
        try:
            for _ in range(phys_per_chunk):
                obs, *_ = sim.step({"joints": command, "adhesion": adhesion})
                if cam:
                    sim.render()
        except PhysicsError as err:
            print(f"t={t:.2f}s  physics became unstable ({err}); stopping", flush=True)
            break
        if viewer is not None:
            viewer.sync()
            if not viewer.is_running():
                break
        if awake:
            motion.append(np.abs(obs["joints"][0] - rest).mean())
            height.append(obs["fly"][0][2])
            up.append(obs["cardinal_vectors"][2][2])
        if k % 40 == 0:
            pos = obs["fly"][0]
            print(f"t={t:5.2f}s wall={time.time() - t_wall:6.1f}s  fly xyz=({pos[0]:.2f},{pos[1]:.2f},"
                  f"{pos[2]:.2f}) mm  max joint offset={np.abs(obs['joints'][0] - rest).max():.2f} rad  "
                  f"muscle activation mean +{muscles.a_pos.mean():.2f}/-{muscles.a_neg.mean():.2f}  "
                  f"active neurons={int(net.nact[0])}", flush=True)
    if motion:
        print(f"SUMMARY awake: mean joint offset {np.mean(motion):.2f} rad, "
              f"body height {np.mean(height):.2f} mm (min {np.min(height):.2f}, max {np.max(height):.2f}), "
              f"uprightness mean {np.mean(up):+.2f} min {np.min(up):+.2f} (+1 upright, -1 on its back)")
    if cam:
        a.video.parent.mkdir(parents=True, exist_ok=True)
        cam.save_video(a.video)
        print("saved", a.video)


if __name__ == "__main__":
    main()
