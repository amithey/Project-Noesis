"""Experimental connectome plasticity in a physical fly, with recorded episodes.

The reward and learning rule are modelling assumptions. Movement is not evidence
of learned walking. Flight is not implemented by this experiment.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np

from noesis.balance import BalanceMap
from noesis.connectome import load_cached
from noesis.embodied_metrics import body_reward
from noesis.environment_senses import (
    EnvironmentSenses,
    contact_body_touch,
    contact_tastes,
    merge_drive,
    mouth_position,
)
from noesis.habitat import Habitat, habitat_arena
from noesis.lif import LIFNetwork, LIFParams
from noesis.motor import RateTracker, build_motor_matrix
from noesis.plasticity import MotorPlasticity
from noesis.sensory import LEGS, LegState, SensoryMap
from noesis.viz.brain_map import write_activity
from noesis.world import (
    CONTACT_SOLREF,
    FLOOR_COLLISIONS,
    FLOOR_CONTACT_MARGIN_MM,
    GRAVITY,
    configure_floor_contacts,
    enable_wing_and_head_joints,
    floor_arena,
    launch_viewer,
)


def _initial_pose(flygym):
    """Real walking mean posture when FlyGym's kinematic data is available, else 'stretch'."""
    state = getattr(flygym, 'state', None)
    if state is None or not hasattr(state, 'KinematicPose'):
        return 'stretch'
    try:
        from noesis.body import standing_pose
        return state.KinematicPose(joint_pos=standing_pose())
    except Exception:
        return 'stretch'


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data', type=Path, default=Path('data'))
    ap.add_argument('--output', type=Path, required=True, help='new directory for this run')
    ap.add_argument('--episodes', type=int, default=3)
    ap.add_argument('--seconds', type=float, default=3.0)
    ap.add_argument('--birth', type=float, default=0.05)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--learn', action='store_true', help='enable experimental synaptic updates')
    ap.add_argument('--checkpoint', type=Path, help='load learned weights for evaluation/training')
    ap.add_argument('--learning-rate', type=float, default=0.02)
    ap.add_argument('--viewer', action='store_true', help='open interactive MuJoCo window')
    ap.add_argument('--brain-map', type=Path, help='publish live firing rates for the brain-map window')
    ap.add_argument('--video', action='store_true', help='record each episode as MP4')
    ap.add_argument('--no-senses', action='store_true')
    ap.add_argument('--no-balance', action='store_true')
    ap.add_argument('--sensory-weight', type=float, default=400.0)
    ap.add_argument('--arena', choices=('habitat', 'empty'), default='habitat')
    ap.add_argument('--no-environment-senses', action='store_true', help='world remains visible, sensory input disabled')
    ap.add_argument('--environment-weight', type=float, default=100.0)
    ap.add_argument('--vision-hz', type=float, default=20.0)
    ap.add_argument('--sensory-map-seed', type=int, default=0, help='fixed assumed retinal mapping across episodes/seeds')
    return ap


def validate_args(ap, args):
    if args.episodes < 1:
        ap.error('--episodes must be positive')
    if not np.isfinite(args.seconds) or args.seconds <= 0:
        ap.error('--seconds must be finite and positive')
    if not np.isfinite(args.birth) or not 0 <= args.birth < args.seconds:
        ap.error('--birth must be between zero and --seconds')
    for name in ('learning_rate', 'sensory_weight', 'environment_weight'):
        if not np.isfinite(getattr(args, name)) or getattr(args, name) < 0:
            ap.error(f'--{name.replace("_", "-")} must be finite and nonnegative')
    if not np.isfinite(args.vision_hz) or not 0 < args.vision_hz <= 200:
        ap.error('--vision-hz must be finite and within (0, 200]')
    if args.seed < 0 or args.seed + args.episodes > 2**32:
        ap.error('episode seeds must fit unsigned 32-bit integers')
    if not 0 <= args.sensory_map_seed < 2**32:
        ap.error('--sensory-map-seed must fit an unsigned 32-bit integer')
    if args.output.exists() and (not args.output.is_dir() or any(args.output.iterdir())):
        ap.error('--output must be a new or empty directory')


def run_episode(cn, args, episode, checkpoint):
    import flygym
    from flygym import Camera, Fly, PhysicsError, SingleFlySimulation, preprogrammed

    chunk_ms = 5.0
    seed = args.seed + episode
    enriched = args.arena == 'habitat'
    environmental_input = enriched and not args.no_environment_senses
    arena = habitat_arena() if enriched else floor_arena()
    net = LIFNetwork(cn.W, LIFParams(dt=0.5, da=1.0), seed=seed)
    fly = Fly(init_pose=_initial_pose(flygym), control='position', enable_adhesion=True,
              floor_collisions=FLOOR_COLLISIONS,
              contact_solref=CONTACT_SOLREF,
              enable_olfaction=environmental_input, enable_vision=environmental_input,
              vision_refresh_rate=args.vision_hz,
              spawn_pos=(0.0, 0.0, 0.2),
              contact_sensor_placements=preprogrammed.get_collision_geometries('legs'))
    enable_wing_and_head_joints(fly)
    camera_args = {'camera_parameters': {'name': 'habitat_camera', 'mode': 'fixed',
                    'pos': (0.5, -5.0, 3.8), 'euler': (0.92, 0, 0), 'fovy': 45},
                   'window_size': (960, 640)} if enriched else {}
    camera = Camera(attachment_point=arena.root_element.worldbody if enriched else fly.model.worldbody,
                    camera_name='camera_left', targeted_fly_names=[] if enriched else fly.name,
                    play_speed=0.1, fps=30, **camera_args) if args.video else None
    sim = SingleFlySimulation(fly=fly, cameras=[camera] if camera else [], timestep=1e-4,
                              arena=arena, gravity=GRAVITY)
    configure_floor_contacts(sim)
    viewer = None
    rows = []
    status = 'completed'
    start = time.monotonic()
    prefix = args.output / f'episode_{episode:03d}'
    try:
        obs, _ = sim.reset(seed=seed)
        rest = obs['joints'][0].copy()
        target = rest.copy()
        names = list(fly.actuated_joints)
        G = build_motor_matrix(cn.neurons, names)
        wing_mask = np.array([n.startswith(('joint_LWing', 'joint_RWing')) for n in names])
        gain_vec = np.full(len(names), 0.35)
        lo_vec = rest - 0.8
        hi_vec = rest + 0.8
        try:
            from noesis.body import _kinematics
            kin = _kinematics()
            for idx_j, jname in enumerate(names):
                if jname in kin:
                    a = kin[jname]
                    span = float(a.max() - a.min())
                    gain_vec[idx_j] = max(0.25, 0.65 * span)
                    lo_vec[idx_j] = float(a.min() - 0.15 * span)
                    hi_vec[idx_j] = float(a.max() + 0.15 * span)
        except Exception:
            pass
        plasticity = MotorPlasticity(net, np.unique(G.indices),
                                    learning_rate=args.learning_rate)
        if checkpoint:
            plasticity.load(checkpoint)
        tracker = RateTracker(cn.n, tau_ms=25.0)
        senses = None if args.no_senses else SensoryMap(cn.neurons)
        balance = (None if (args.no_senses or args.no_balance)
                   else BalanceMap(cn.neurons, seed=args.sensory_map_seed))
        environment = EnvironmentSenses(cn.neurons, fly.retina, seed=args.sensory_map_seed) if environmental_input else None
        if environment:
            (prefix.with_suffix('.sensory-map.json')).write_text(
                json.dumps(environment.coverage(), indent=2), encoding='utf-8')
        joints = {(leg, dof): names.index(f'joint_{leg}{dof}')
                  for leg in LEGS for dof in ('Coxa', 'Femur', 'Tibia')}
        sensor_legs = np.array([LEGS.index(n.split('/')[-1][:2])
                               for n in fly.contact_sensor_placements])
        if args.viewer:
            viewer = launch_viewer(sim, enriched=enriched)
        print(f'Episode {episode}: {cn.n:,} entries, {G.nnz} motor mappings; '
              f'learning={args.learn}, seed={seed}', flush=True)
        awake_origin = None
        for k in range(int(np.ceil(args.seconds * 1000 / chunk_ms))):
            t = k * chunk_ms / 1000
            awake = t >= args.birth
            previous = obs['fly'][0].copy()
            spikes = np.empty(0, dtype=np.int32)
            env_telemetry = {}
            env_rates = None
            adhesion = np.ones(6, int)
            if environment:
                obs['surface_taste'] = contact_tastes(sim, fly, arena.habitat)
                obs['surface_touch'] = contact_body_touch(sim, fly)
                env_rates, env_telemetry = environment.drive(
                    obs, arena.habitat, mouth_position(sim, fly), t)
            if awake:
                if awake_origin is None:
                    awake_origin = previous.copy()
                drive_parts = []
                forces = np.linalg.norm(obs['contact_forces'], axis=1)
                loads = np.bincount(sensor_legs, weights=forces, minlength=6)
                if senses is not None:
                    angles = obs['joints'][0] - rest
                    velocity = obs['joints'][1]
                    states = {leg: LegState(angles[joints[leg, 'Tibia']],
                                          velocity[joints[leg, 'Tibia']],
                                          angles[joints[leg, 'Coxa']],
                                          angles[joints[leg, 'Femur']], loads[i])
                              for i, leg in enumerate(LEGS)}
                    drive_parts.append((senses.idx, senses.drive(states), args.sensory_weight))
                if balance is not None and len(balance.idx):
                    cv = obs['cardinal_vectors']
                    omega = obs['fly'][3] if len(obs['fly']) > 3 else np.zeros(3)
                    drive_parts.append((balance.idx, balance.drive(
                        np.array([cv[0][2], cv[1][2], cv[2][2]]), omega), args.sensory_weight))
                if environment:
                    drive_parts.append((environment.idx, env_rates, args.environment_weight))
                drive = merge_drive(drive_parts)
                _, spikes = net.run(chunk_ms, background_hz=5.0, background_weight=90.0,
                                    **drive)
                rates_hz = tracker.update(spikes, chunk_ms)
                net_drive = G @ rates_hz
                desired = np.clip(rest + gain_vec * np.tanh(net_drive / 25.0), lo_vec, hi_vec)
                if wing_mask.any():
                    # Asynchronous flight power & steering muscles elevate and stroke the wings
                    stroke = 0.5 + 0.5 * np.sin(2.0 * np.pi * 25.0 * t)
                    for side, sgn in (('L', 1.0), ('R', -1.0)):
                        iy = names.index(f'joint_{side}Wing_yaw')
                        ip = names.index(f'joint_{side}Wing')
                        ir = names.index(f'joint_{side}Wing_roll')
                        amp = np.tanh((abs(net_drive[iy]) + abs(net_drive[ip]) + abs(net_drive[ir])) / 20.0)
                        desired[iy] = sgn * (0.25 + 0.65 * stroke) * amp
                        desired[ip] = -(0.15 + 0.45 * stroke) * amp
                        desired[ir] = sgn * (0.15 + 0.40 * stroke) * amp
                slew = np.where(wing_mask, 0.45, 0.06)
                target += np.clip(desired - target, -slew, slew)
                lift = np.array([desired[joints[leg, 'Femur']] - rest[joints[leg, 'Femur']]
                                 for leg in LEGS])
                # Release at most two feet, the ones the flexors are actually lifting. Letting every
                # pad go at once drops the body into a contact the solver cannot resolve.
                swinging = np.flatnonzero(lift > 0.08)
                if len(swinging) > 2:
                    swinging = swinging[np.argpartition(lift[swinging], -2)[-2:]]
                adhesion[swinging] = 0
            try:
                for _ in range(round(chunk_ms / 1000 / sim.timestep)):
                    obs, *_ = sim.step({'joints': target, 'adhesion': adhesion})
                    if camera:
                        sim.render()
            except PhysicsError as exc:
                status = f'physics_error: {exc}'
                break
            score = body_reward(previous, obs['fly'][0],
                                float(np.clip(obs['cardinal_vectors'][2][2], -1, 1)),
                                obs['joints'][1], chunk_ms / 1000)
            learning = plasticity.update(spikes, chunk_ms, score['reward']) \
                if awake and args.learn else plasticity.metrics()
            pos = obs['fly'][0]
            rows.append(dict(time_s=t + chunk_ms / 1000, awake=int(awake),
                             x_mm=float(pos[0]), y_mm=float(pos[1]), z_mm=float(pos[2]),
                             spikes=len(spikes), **score,
                             **env_telemetry,
                             weight_change=learning['mean_relative_change']))
            if args.brain_map is not None and k % 4 == 0:
                write_activity(args.brain_map, tracker.rate, t, awake=awake,
                               triggers=env_telemetry or None)
            if k % 40 == 0:
                smell = env_telemetry.get('odor_food', 0)
                light = env_telemetry.get('vision_mean', 0)
                temp = env_telemetry.get('temperature_c', 0)
                print(f'  t={t:.2f}s upright={score["uprightness"]:.2f} '
                      f'reward={score["reward"]:+.3f} spikes={len(spikes)} '
                      f'smell={smell:.2f} light={light:.2f} temp={temp:.1f}', flush=True)
            if viewer:
                viewer.sync()
                if not viewer.is_running():
                    status = 'viewer_closed'
                    break
        plasticity.save(prefix.with_suffix('.npz'))
        if camera:
            camera.save_video(prefix.with_suffix('.mp4'))
        if environment:
            np.savez_compressed(prefix.with_suffix('.senses.npz'),
                                neuron_indices=environment.idx, rates_hz=environment.last_rates,
                                vision=obs['vision'], odors=obs['odor_intensity'],
                                surface_taste=contact_tastes(sim, fly, arena.habitat),
                                surface_touch=contact_body_touch(sim, fly))
            import imageio.v3 as iio
            eyes = [fly.retina.hex_pxls_to_human_readable(eye, color_8bit=True)
                    for eye in obs['vision']]
            rgb = [np.concatenate([np.zeros((*eye.shape[:2], 1), np.uint8), eye], axis=2)
                   for eye in eyes]
            iio.imwrite(prefix.with_suffix('.eyes.png'), np.concatenate(rgb, axis=1))
    finally:
        if viewer:
            viewer.close()
        sim.close()

    if rows:
        with prefix.with_suffix('.csv').open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    awake_rows = [r for r in rows if r['awake']]
    summary = {'episode': episode, 'seed': seed, 'status': status, 'learning': args.learn,
               'wall_seconds': time.monotonic() - start,
               'simulated_seconds': rows[-1]['time_s'] if rows else 0,
               'awake_samples': len(awake_rows),
               'mean_reward': float(np.mean([r['reward'] for r in awake_rows]))
               if awake_rows else None,
               'mean_uprightness': float(np.mean([r['uprightness'] for r in awake_rows]))
               if awake_rows else None,
               'forward_displacement_mm': rows[-1]['x_mm'] - float(awake_origin[0])
               if awake_rows else None,
               'checkpoint': str(prefix.with_suffix('.npz'))}
    prefix.with_suffix('.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    return summary, prefix.with_suffix('.npz')


def main():
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    ap = parser()
    args = ap.parse_args()
    validate_args(ap, args)
    try:
        import flygym  # noqa: F401
    except ImportError as exc:
        ap.error(f'Body simulator unavailable ({exc}). Install: python -m pip install -e ".[body]"')
    raw = args.data / 'raw'
    if not (raw / 'neurons.csv').exists() and Path('downloaded file/neurons.csv').exists():
        raw = Path('downloaded file')
    cn = load_cached(raw, args.data / 'processed')
    args.output.mkdir(parents=True, exist_ok=True)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(neuron_table_entries=cn.n, synaptic_pairs=cn.W.nnz,
                  model='LIF + assumed motor/sensory mappings + experimental motor plasticity',
                  flight=False, reward='artificial upright forward-motion objective',
                  gravity_mm_s2=GRAVITY, floor_z_mm=0.0, floor_collisions=FLOOR_COLLISIONS,
                  contact_solref=CONTACT_SOLREF,
                  floor_contact_margin_mm=FLOOR_CONTACT_MARGIN_MM,
                  habitat=Habitat().manifest() if args.arena == 'habitat' else None,
                  versions={name: version(name) for name in
                            ('flygym', 'mujoco', 'numpy', 'scipy', 'numba', 'pandas')},
                  neural_dt_ms=0.5, body_dt_seconds=0.0001, control_dt_ms=5.0,
                  adaptation_mv_per_spike=1.0, background_hz=5.0, background_weight=90.0,
                  motor_gain=0.8, motor_rate_scale_hz=100.0, motor_slew_rad_s=8.0)
    (args.output / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    checkpoint = args.checkpoint
    summaries = []
    for episode in range(args.episodes):
        summary, saved = run_episode(cn, args, episode, checkpoint)
        print(json.dumps(summary, ensure_ascii=True), flush=True)
        summaries.append(summary)
        (args.output / 'summary.json').write_text(json.dumps(summaries, indent=2), encoding='utf-8')
        if args.learn:
            checkpoint = saved
        if summary['status'] != 'completed':
            break
    print(f'Results: {args.output.resolve()}', flush=True)
    print('Weight changes are experimental plasticity, not proof of learned walking.')
    if summaries[-1]['status'].startswith('physics_error:'):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
