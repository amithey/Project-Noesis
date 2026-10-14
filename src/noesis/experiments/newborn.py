"""The birth of a fly: the connectome's leg motor neurons drive a physical body. Nothing is learned.

Timeline: the body is dropped and settles with the brain *off*; at ``--birth`` the brain wakes
with spontaneous background activity and whatever the wiring produces moves the legs.

    brain (158k LIF neurons) --leg motor-neuron rates--> joint targets --> NeuroMechFly body

Use ``--viewer`` for a live MuJoCo window and/or ``--video out.mp4`` to record.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from noesis.balance import BalanceMap
from noesis.connectome import load_cached
from noesis.environment_senses import (
    EnvironmentSenses,
    contact_body_touch,
    contact_tastes,
    merge_drive,
    mouth_position,
)
from noesis.habitat import habitat_arena
from noesis.lif import LIFNetwork, LIFParams
from noesis.motor import RateTracker, build_motor_matrix
from noesis.sensory import LEGS, LegState, SensoryMap
from noesis.world import (
    CONTACT_SOLREF,
    FLOOR_COLLISIONS,
    GRAVITY,
    configure_floor_contacts,
    enable_wing_and_head_joints,
    floor_arena,
    launch_viewer,
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--seconds", type=float, default=3.0, help="simulated time")
    ap.add_argument("--birth", type=float, default=0.05, help="when the brain wakes (s)")
    ap.add_argument("--bg-hz", type=float, default=5.0, help="spontaneous events/s per neuron")
    ap.add_argument("--bg-weight", type=float, default=90.0, help="synapse-equivalents per event")
    ap.add_argument("--gain", type=float, default=0.8, help="joint swing (rad) at saturation")
    ap.add_argument("--r0", type=float, default=25.0, help="rate difference (Hz) giving ~76%% swing")
    ap.add_argument("--slew", type=float, default=12.0, help="max joint-target speed (rad/s)")
    ap.add_argument("--adapt", type=float, default=1.0,
                    help="spike-frequency adaptation, mV per spike (0 = off)")
    ap.add_argument("--dt-ms", type=float, default=0.5, help="LIF time step")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-senses", action="store_true", help="no leg sensory feedback")
    ap.add_argument("--leg-kinds", help="comma list of leg organ classes to keep (claw,hook,club,hairplate,campaniform,bristle)")
    ap.add_argument("--no-balance", action="store_true",
                    help="no gravity (Johnston) / rotation (haltere) senses")
    ap.add_argument("--viewer", action="store_true", help="live MuJoCo window")
    ap.add_argument("--video", type=Path, help="write an mp4 (slower)")
    ap.add_argument("--arena", choices=("empty", "habitat"), default="empty")
    ap.add_argument("--no-environment-senses", action="store_true")
    a = ap.parse_args()

    from flygym import (  # heavy import, keep local
        Camera,
        Fly,
        PhysicsError,
        SingleFlySimulation,
        preprogrammed,
    )

    cn = load_cached(a.data / "raw", a.data / "processed")
    net = LIFNetwork(cn.W, LIFParams(dt=a.dt_ms, da=a.adapt), seed=a.seed)
    enriched = a.arena == "habitat"
    environmental_input = enriched and not a.no_environment_senses
    arena = habitat_arena() if enriched else floor_arena()

    from flygym.state import KinematicPose
    from noesis.body import standing_pose

    # contact sensors on every leg segment (default is tarsi only, which do not touch down)
    fly = Fly(init_pose=KinematicPose(joint_pos=standing_pose()), control="position", enable_adhesion=True,
              floor_collisions=FLOOR_COLLISIONS,
              contact_solref=CONTACT_SOLREF,
              enable_olfaction=environmental_input, enable_vision=environmental_input,
              vision_refresh_rate=20,
              spawn_pos=(0.0, 0.0, 0.2),
              contact_sensor_placements=preprogrammed.get_collision_geometries("legs"))
    enable_wing_and_head_joints(fly)
    cam = None
    if a.video:
        cam = Camera(attachment_point=fly.model.worldbody, camera_name="camera_left",
                     targeted_fly_names=fly.name, play_speed=0.1, fps=30)
    sim = SingleFlySimulation(fly=fly, cameras=[cam] if cam else [], timestep=1e-4,
                              arena=arena, gravity=GRAVITY)
    configure_floor_contacts(sim)
    obs, _ = sim.reset()
    rest = obs["joints"][0].copy()
    G = build_motor_matrix(cn.neurons, list(fly.actuated_joints))
    print(f"{G.nnz} motor neurons mapped onto {G.shape[0]} joints")
    rates = RateTracker(cn.n, tau_ms=25.0)
    kinds = frozenset(a.leg_kinds.split(",")) if a.leg_kinds else None
    senses = None if a.no_senses else SensoryMap(cn.neurons, kinds=kinds)
    names = list(fly.actuated_joints)
    wing_mask = np.array([n.startswith(("joint_LWing", "joint_RWing")) for n in names])
    j = {(leg, d): names.index(f"joint_{leg}{d}") for leg in LEGS for d in ("Coxa", "Femur", "Tibia")}
    sensors = [n.split("/")[-1][:2] for n in fly.contact_sensor_placements]
    leg_of_sensor = np.array([LEGS.index(x) for x in sensors])
    balance = None if a.no_balance else BalanceMap(cn.neurons)
    environment = EnvironmentSenses(cn.neurons, fly.retina, seed=0) if environmental_input else None
    print("leg senses:", "OFF" if senses is None else f"{len(senses.idx)} neurons",
          "| balance senses:", "OFF" if balance is None else f"{len(balance.idx)} neurons")

    def sensory_drive(obs):
        parts = []
        if senses is not None:
            parts.append((senses.idx, senses.drive(leg_states(obs)), 400.0))
        if balance is not None:
            cv = obs["cardinal_vectors"]  # thorax x/y/z axes in world coordinates
            parts.append((balance.idx, balance.drive(
                np.array([cv[0][2], cv[1][2], cv[2][2]]), obs["fly"][3]), 400.0))
        if environment:
            obs['surface_taste'] = contact_tastes(sim, fly, arena.habitat)
            obs['surface_touch'] = contact_body_touch(sim, fly)
            rates, _ = environment.drive(obs, arena.habitat, mouth_position(sim, fly), sim.curr_time)
            parts.append((environment.idx, rates, 100.0))
        return merge_drive(parts)

    def leg_states(obs):
        ang, vel = obs["joints"][0] - rest, obs["joints"][1]
        force = np.linalg.norm(obs["contact_forces"], axis=1)
        load = np.bincount(leg_of_sensor, weights=force, minlength=6)
        return {
            leg: LegState(ang[j[leg, "Tibia"]], vel[j[leg, "Tibia"]], ang[j[leg, "Coxa"]],
                          ang[j[leg, "Femur"]], load[LEGS.index(leg)])
            for leg in LEGS
        }

    viewer = None
    if a.viewer:
        viewer = launch_viewer(sim, enriched=enriched)

    chunk_ms = 5.0
    phys_per_chunk = round(chunk_ms * 1e-3 / sim.timestep)
    adhesion = np.ones(6, int)
    target = rest.copy()
    t_wall = time.time()
    awake = False
    motion, height, up = [], [], []
    for k in range(round(a.seconds * 1e3 / chunk_ms)):
        t = k * chunk_ms * 1e-3
        if t >= a.birth:
            if not awake:
                print(f"t={t:.2f}s  *** the brain wakes ***", flush=True)
                awake = True
            _, spikes = net.run(chunk_ms, background_hz=a.bg_hz,
                                background_weight=a.bg_weight, **sensory_drive(obs))
            r = rates.update(spikes, chunk_ms)
            net_drive = G @ r
            u = np.tanh(net_drive / a.r0)
            desired = rest + a.gain * u
            if wing_mask.any():
                desired[wing_mask] = rest[wing_mask] + 0.9 * np.tanh(net_drive[wing_mask] / 12.0) * np.sin(2.0 * np.pi * 40.0 * t)
            step_max = np.where(wing_mask, 0.35, a.slew * chunk_ms * 1e-3)
            target = target + np.clip(desired - target, -step_max, step_max)
            loads = np.bincount(leg_of_sensor, weights=np.linalg.norm(obs["contact_forces"], axis=1), minlength=6)
            adhesion = np.ones(6, int)
            for i, leg in enumerate(LEGS):
                jf = j[leg, "Femur"]
                if desired[jf] < obs["joints"][0][jf] - 0.04 or loads[i] < 0.15:
                    adhesion[i] = 0
        try:
            for _ in range(phys_per_chunk):
                obs, *_ = sim.step({"joints": target, "adhesion": adhesion})
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
            up.append(obs["cardinal_vectors"][2][2])  # thorax z-axis . world up (+1 upright)
        if k % 40 == 0:
            moved = np.abs(obs["joints"][0] - rest).max()
            pos = obs["fly"][0]
            print(f"t={t:5.2f}s wall={time.time() - t_wall:6.1f}s  fly xyz=({pos[0]:.2f},"
                  f"{pos[1]:.2f},{pos[2]:.2f}) mm  max joint offset={moved:.2f} rad  "
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
