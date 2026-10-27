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
from noesis.body import BodyParams, CollisionGuard, MuscleModel, make_fly
from noesis.connectome import corrected_scale, load_all_pairs, load_cached
from noesis.internal_state import InternalState
from noesis.lif import LIFNetwork, LIFParams
from noesis.motor import RateTracker, build_muscle_matrices
from noesis.mushroom_body import (
    DopamineGatedPlasticity,
    GradedAPL,
    populations,
    with_weak_pathway_edges,
)
from noesis.sensory import LEGS, LegState, SensoryMap
from noesis.viz.brain_map import spike_histogram, write_activity


def _world_kwargs(arena=None) -> dict:
    """The shared physical world (floor, gravity, contact) when available."""
    try:
        from noesis import world
    except ImportError:
        return {} if arena is None else {"arena": arena}
    return {"arena": arena if arena is not None else world.floor_arena(), "gravity": world.GRAVITY}


def _merge(parts) -> dict:
    """Combine drive parts that share one event weight each into ``LIFNetwork.run`` arguments."""
    if not parts:
        return {}
    return {"drive_idx": np.concatenate([p[0] for p in parts]),
            "drive_rate_hz": np.concatenate([p[1] for p in parts]),
            "drive_weight": np.concatenate([np.full(len(p[0]), p[2]) for p in parts])}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--seconds", type=float, default=3.0, help="simulated time")
    ap.add_argument("--birth", type=float, default=0.5, help="when the brain wakes (s)")
    ap.add_argument("--bg-hz", type=float, default=5.0, help="spontaneous events/s per neuron")
    ap.add_argument("--bg-weight", type=float, default=None,
                    help="synapse-equivalents per event (default 90 raw, 60 with --scaled)")
    ap.add_argument("--scaled", action="store_true",
                    help="BANC synapses corrected per region: brain to FAFB, nerve cord to MANC counts")
    ap.add_argument("--brain-only", action="store_true", help="with --scaled, leave the nerve cord as detected")
    ap.add_argument("--adapt", type=float, default=1.0, help="spike-frequency adaptation, mV/spike")
    ap.add_argument("--dt-ms", type=float, default=0.5, help="LIF time step")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-senses", action="store_true", help="no leg sensory feedback")
    ap.add_argument("--no-balance", action="store_true", help="no gravity/rotation senses")
    ap.add_argument("--no-internal-state", action="store_true",
                    help="no hemolymph/interoceptive internal-state coupling")
    ap.add_argument("--arena", choices=("empty", "habitat"), default="empty",
                    help="habitat = the world with odours, food, light, obstacles (needs noesis.habitat)")
    ap.add_argument("--no-environment-senses", action="store_true", help="habitat visible but not sensed")
    ap.add_argument("--plasticity", action="store_true",
                    help="intrinsic dopamine-gated learning at the mushroom-body KC->MBON synapses")
    ap.add_argument("--apl-gain", type=float, default=1024.0,
                    help="graded (non-spiking) APL inhibition with --plasticity; 0 = spiking APL")
    ap.add_argument("--eta", type=float, default=5e-4, help="plasticity strength")
    ap.add_argument("--no-guard", action="store_true", help="do not keep muscle commands collision-free")
    ap.add_argument("--brain-map", type=Path,
                    help="publish live firing rates here for `python -m noesis.viz.brain_view3d --watch`")
    ap.add_argument("--viewer", action="store_true", help="live MuJoCo window")
    ap.add_argument("--video", type=Path, help="write an mp4 (slower)")
    a = ap.parse_args()

    from flygym import Camera, PhysicsError, SingleFlySimulation, preprogrammed

    if a.bg_weight is None:  # calibrated to a ~5 Hz awake brain: raw 5 Hz x 90, corrected 5 Hz x 60
        a.bg_weight = 60.0 if a.scaled else 90.0
    kw = {}
    if a.scaled:
        kw["region_scale"] = corrected_scale(a.data, nerve_cord=not a.brain_only)
    cn = load_cached(a.data / "raw", a.data / "processed", **kw)
    brain_label = (
        ("BANC corrected (brain FAFB" + ("" if a.brain_only else " + nerve cord MANC") + ")")
        if a.scaled else "BANC as detected"
    )
    print("brain:", brain_label, f"| background {a.bg_hz:g} Hz x {a.bg_weight:g}", flush=True)
    W, pops = cn.W, populations(cn.neurons)
    if a.plasticity:  # the learning substrate is made of sub-threshold (1-4 synapse) pairs
        pairs = load_all_pairs(a.data / "raw", a.data / "processed", cn.neurons)
        W = with_weak_pathway_edges(cn, pairs, pops)
    net = LIFNetwork(W, LIFParams(dt=a.dt_ms, da=a.adapt), seed=a.seed)
    apl = GradedAPL(net, pops, gain=a.apl_gain) if a.plasticity and a.apl_gain > 0 else None
    plast = DopamineGatedPlasticity(net, pops, eta=a.eta) if a.plasticity else None
    apl_current: dict = {}
    habitat_world = a.arena == "habitat"
    arena = None
    if habitat_world:
        from noesis.environment_senses import (
            EnvironmentSenses,
            contact_tastes,
            merge_drive,
            mouth_position,
        )
        from noesis.habitat import habitat_arena
        arena = habitat_arena()
    environmental_input = habitat_world and not a.no_environment_senses
    # FlyGym renders the eyes with a second GLFW window. That window is created
    # with VISIBLE=0 and DOUBLEBUFFER=0, and GLFW is not thread-safe, so doing
    # it while the MuJoCo viewer thread is alive leaves a white frozen fly.
    # Smell, taste, heat and humidity stay on. The eyes stay dark in the live view.
    render_eyes = environmental_input and not a.viewer

    params = BodyParams()
    fly = make_fly(params, spawn_pos=(0.0, 0.0, 1.2),
                   contact_sensor_placements=preprogrammed.get_collision_geometries("legs"),
                   enable_olfaction=environmental_input, enable_vision=render_eyes,
                   vision_refresh_rate=20)
    cam = None
    if a.video:
        cam = Camera(attachment_point=fly.model.worldbody, camera_name="camera_left",
                     targeted_fly_names=fly.name, play_speed=0.1, fps=30)
    sim = SingleFlySimulation(fly=fly, cameras=[cam] if cam else [], timestep=1e-4, **_world_kwargs(arena))
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

    guard = None if a.no_guard else CollisionGuard(sim.physics, fly, names)
    if environmental_input:
        if render_eyes:
            retina = fly.retina
        else:
            from flygym.vision import Retina
            retina = Retina()
        environment = EnvironmentSenses(cn.neurons, retina, seed=0)
    else:
        environment = None
    senses = None if a.no_senses else SensoryMap(cn.neurons)
    balance = None if a.no_balance else BalanceMap(cn.neurons)
    internal = None if a.no_internal_state else InternalState(cn.neurons)
    j = {(leg, d): names.index(f"joint_{leg}{d}") for leg in LEGS for d in ("Coxa", "Femur", "Tibia")}
    leg_of_sensor = np.array([LEGS.index(n.split("/")[-1][:2]) for n in fly.contact_sensor_placements])
    dn_idx = np.flatnonzero(cn.neurons["Super Class"].eq("descending").to_numpy())
    leg_mn_idx = np.flatnonzero(cn.neurons["Class"].eq("leg_motor_neuron").to_numpy())

    def leg_states(obs):
        ang, vel = obs["joints"][0] - rest, obs["joints"][1]
        load = np.bincount(leg_of_sensor, weights=np.linalg.norm(obs["contact_forces"], axis=1), minlength=6)
        return {leg: LegState(ang[j[leg, "Tibia"]], vel[j[leg, "Tibia"]], ang[j[leg, "Coxa"]],
                              ang[j[leg, "Femur"]], load[LEGS.index(leg)]) for leg in LEGS}

    triggers: dict = {}
    pending_spikes: list[np.ndarray] = []

    def sensory_drive(obs, dt_s: float):
        parts = []  # (neuron indices, rates in Hz, event weight)
        triggers.clear()
        if senses is not None:
            parts.append((senses.idx, senses.drive(leg_states(obs)), 400.0))
        if balance is not None:
            cv = obs["cardinal_vectors"]
            parts.append((balance.idx, balance.drive(
                np.array([cv[0][2], cv[1][2], cv[2][2]]), obs["fly"][3]), 400.0))
        if environment is not None:
            if "vision" not in obs:
                n_px = len(environment.retina.pale_type_mask)
                obs["vision"] = np.zeros((2, n_px, 2))
            obs["surface_taste"] = contact_tastes(sim, fly, arena.habitat)
            rates_env, telemetry = environment.drive(obs, arena.habitat, mouth_position(sim, fly),
                                                     sim.curr_time)
            triggers.update(telemetry)
            parts.append((environment.idx, rates_env, 100.0))
        if internal is not None:
            effort = float(0.5 * (muscles.a_pos.mean() + muscles.a_neg.mean()))
            rates_int, int_telemetry = internal.step(
                rates.rate,
                dt_s,
                muscle_effort=effort,
                taste_sugar=triggers.get("taste_sugar", 0.0),
                taste_water=triggers.get("taste_water", 0.0),
                humidity=triggers.get("humidity", 0.5),
            )
            triggers.update(int_telemetry)
            parts.append((internal.idx, rates_int, 150.0))
        triggers["body_z_mm"] = float(obs["fly"][0][2])
        triggers["uprightness"] = float(obs["cardinal_vectors"][2][2])
        triggers["muscle_pos_mean"] = float(muscles.a_pos.mean())
        triggers["muscle_neg_mean"] = float(muscles.a_neg.mean())
        return merge_drive(parts) if parts and habitat_world else _merge(parts)

    viewer = None
    if a.viewer:
        from noesis.world import launch_viewer
        viewer = launch_viewer(sim)
        if habitat_world and hasattr(viewer, "cam"):
            viewer.cam.lookat[:] = (0.0, 0.0, 0.8)
            viewer.cam.distance, viewer.cam.elevation = 16.0, -35.0
            viewer.sync()

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
                                **sensory_drive(obs, chunk_ms * 1e-3), **apl_current)
            pending_spikes.append(np.asarray(spikes))
            if apl is not None:
                apl_current = apl.current(spikes, chunk_ms)
            if plast is not None:
                plast.step(spikes, chunk_ms)
            command = muscles.step(rates.update(spikes, chunk_ms), chunk_ms)
            if guard is not None:
                command = guard.filter(obs["joints"][0], command)
        elif k % 4 == 0:
            sensory_drive(obs, chunk_ms * 1e-3)
        if a.brain_map is not None and k % 4 == 0:  # every 20 ms of simulated time, asleep too
            dn_hz = float(rates.rate[dn_idx].mean()) if len(dn_idx) else 0.0
            mn_hz = float(rates.rate[leg_mn_idx].mean()) if len(leg_mn_idx) else 0.0
            note = f"{brain_label} | DN {dn_hz:4.1f} Hz | leg MN {mn_hz:4.1f} Hz"
            write_activity(
                a.brain_map, rates.rate, t, awake=awake, note=note, triggers=triggers,
                spikes=spike_histogram(pending_spikes, cn.n),
            )
            pending_spikes.clear()
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
                  f"DAN {0.0 if plast is None else plast.dopamine:.1f} Hz, "
                  f"plastic weights {1.0 if plast is None else plast.mean_weight_fraction:.3f}  "
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
