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

from noesis.adhesion import FootAdhesion
from noesis.balance import BalanceMap
from noesis.behavior_metrics import behavior_summary
from noesis.body import BodyParams, MuscleModel, make_fly
from noesis.embodied_metrics import GROUNDED_OBJECTIVES, body_reward, locomotion_reward
from noesis.embodiment import BrainReadout, MotorExploration, gait_report, load_training_brain
from noesis.environment_senses import (
    EnvironmentSenses,
    contact_body_touch,
    contact_tastes,
    merge_drive,
    mouth_position,
)
from noesis.ground_contact import ground_contacts
from noesis.habitat import Habitat, habitat_arena
from noesis.lif import LIFNetwork, LIFParams
from noesis.motor import RateTracker, build_motor_matrix, build_muscle_matrices
from noesis.plasticity import MotorPlasticity
from noesis.proprioception import ENCODINGS, make_proprioception_map
from noesis.sensory import LEGS, LegState
from noesis.training_scene import training_arena
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
    except (ImportError, OSError, KeyError, AttributeError):
        return 'stretch'


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data', type=Path, default=Path('data'))
    ap.add_argument('--output', type=Path, required=True, help='new directory for this run')
    ap.add_argument('--episodes', type=int, default=3)
    ap.add_argument('--seconds', type=float, default=3.0)
    ap.add_argument('--birth', type=float, default=0.05)
    ap.add_argument('--pool-gains', type=Path, default=None,
                    help='JSON list of 2 x 42 log-gains for the leg muscle pools (frozen evaluation only; default none)')
    ap.add_argument('--warmup-ms', type=float, default=0.0,
                    help='run the brain on background activity alone for this long before it is first coupled to the body (default 0)')
    ap.add_argument('--tonic-tau-ms', type=float, default=0.0,
                    help='online tonic adaptation time constant in ms for MuscleModel (0 disables; default 0)')
    ap.add_argument('--tonic-balance', type=float, default=0.85,
                    help='fraction of tonic agonist/antagonist bias subtracted when --tonic-tau-ms > 0 (default 0.85)')
    ap.add_argument('--femur-stance-bias', type=float, default=0.15,
                    help='downward femur extensor stance bias when --tonic-tau-ms > 0 (default 0.15)')
    ap.add_argument('--training-fall-seconds', type=float, default=0.,
                    help='training-only terminal sustained fall duration; 0 keeps full horizon')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--brain', choices=('corrected', 'raw'), default='corrected')
    ap.add_argument('--body-model', choices=('muscles', 'legacy'), default='muscles')
    ap.add_argument('--bg-hz', type=float, default=3.0)
    ap.add_argument('--bg-weight', type=float, default=60.0)
    ap.add_argument('--motor-exploration', type=float, default=None,
                    help='bounded motor current mV/ms; default 1.5 training, 0 frozen')
    ap.add_argument('--learn', action='store_true', help='enable experimental synaptic updates')
    ap.add_argument('--checkpoint', type=Path, help='load learned weights for evaluation/training')
    ap.add_argument('--import-weights', action='store_true',
                    help='explicitly transfer compatible synaptic weights across body model versions')
    ap.add_argument('--normalize-reward-error', action='store_true')
    ap.add_argument('--bias-correct-reward-error', action='store_true',
                    help='remove startup bias from reward-normalization moments')
    ap.add_argument('--learning-rate', type=float, default=0.2)
    ap.add_argument('--learning-rule', choices=('perturbation', 'hebbian'), default='perturbation')
    ap.add_argument('--reward-mode', choices=('grounded', 'legacy'), default='grounded')
    ap.add_argument('--grounded-objective', choices=GROUNDED_OBJECTIVES, default=GROUNDED_OBJECTIVES[0])
    ap.add_argument('--wing-mode', choices=('neural', 'legacy'), default='neural')
    ap.add_argument('--adhesion-mode', choices=('geometry', 'legacy'), default='geometry')
    ap.add_argument('--viewer', action='store_true', help='open interactive MuJoCo window')
    ap.add_argument('--brain-map', type=Path, help='publish live firing rates for the brain-map window')
    ap.add_argument('--video', action='store_true', help='record each episode as MP4')
    ap.add_argument('--no-senses', action='store_true')
    ap.add_argument('--no-balance', action='store_true')
    ap.add_argument('--proprioceptive-encoding', choices=ENCODINGS, default='legacy',
                    help='typed directional tuning is an inferred hypothesis; reversed is its control')
    ap.add_argument('--sensory-weight', type=float, default=400.0)
    ap.add_argument('--arena', choices=('habitat', 'empty', 'quiet', 'obstacle'), default='habitat')
    ap.add_argument('--depression-u', type=float, default=0.)
    ap.add_argument('--no-environment-senses', action='store_true', help='world remains visible, sensory input disabled')
    ap.add_argument('--environment-weight', type=float, default=100.0)
    ap.add_argument('--vision-hz', type=float, default=20.0)
    ap.add_argument('--sensory-map-seed', type=int, default=0, help='fixed assumed retinal mapping across episodes/seeds')
    return ap


def validate_args(ap, args):
    if args.reward_mode != "grounded" and args.grounded_objective != GROUNDED_OBJECTIVES[0]:
        ap.error("nondefault grounded objective requires grounded reward mode")
    if getattr(args, 'bias_correct_reward_error', False) and not args.normalize_reward_error:
        ap.error('bias correction requires reward normalization')
    if args.normalize_reward_error and args.learning_rule != 'perturbation':
        ap.error('reward normalization requires perturbation learning')
    if args.depression_u not in (0., .03, .1):
        ap.error('--depression-u must be 0, .03 or .1')
    if not args.learn and args.motor_exploration not in (None, 0.):
        ap.error('evaluation requires exploration disabled')
    if args.episodes < 1:
        ap.error('--episodes must be positive')
    if not np.isfinite(args.seconds) or args.seconds <= 0:
        ap.error('--seconds must be finite and positive')
    if (not np.isfinite(args.training_fall_seconds)
            or not 0 <= args.training_fall_seconds <= args.seconds):
        ap.error('--training-fall-seconds must be finite and within [0, --seconds]')
    if not args.learn and args.training_fall_seconds:
        ap.error('evaluation requires training fall termination disabled')
    if (getattr(args, 'pool_gains', None) is not None or getattr(args, 'pool_genome', None) is not None) and args.learn:
        ap.error('pool gains are for frozen evaluation only (do not combine with --learn)')
    if not np.isfinite(args.warmup_ms) or args.warmup_ms < 0:
        ap.error('--warmup-ms must be finite and nonnegative')
    if not np.isfinite(getattr(args, 'tonic_tau_ms', 0.0)) or getattr(args, 'tonic_tau_ms', 0.0) < 0:
        ap.error('--tonic-tau-ms must be finite and nonnegative')
    if not np.isfinite(getattr(args, 'tonic_balance', 0.85)) or not 0 <= getattr(args, 'tonic_balance', 0.85) <= 1:
        ap.error('--tonic-balance must be finite and in [0, 1]')
    if not np.isfinite(getattr(args, 'femur_stance_bias', 0.15)) or not -1 <= getattr(args, 'femur_stance_bias', 0.15) <= 1:
        ap.error('--femur-stance-bias must be finite and in [-1, 1]')
    if not np.isfinite(args.birth) or not 0 <= args.birth < args.seconds:
        ap.error('--birth must be between zero and --seconds')
    for name in ('learning_rate', 'sensory_weight', 'environment_weight', 'bg_hz', 'bg_weight'):
        if not np.isfinite(getattr(args, name)) or getattr(args, name) < 0:
            ap.error(f'--{name.replace("_", "-")} must be finite and nonnegative')
    if args.motor_exploration is not None and (not np.isfinite(args.motor_exploration) or args.motor_exploration < 0):
        ap.error('--motor-exploration must be finite and nonnegative')
    if not np.isfinite(args.vision_hz) or not 0 < args.vision_hz <= 200:
        ap.error('--vision-hz must be finite and within (0, 200]')
    if args.seed < 0 or args.seed + args.episodes > 2**32:
        ap.error('episode seeds must fit unsigned 32-bit integers')
    if not 0 <= args.sensory_map_seed < 2**32:
        ap.error('--sensory-map-seed must fit an unsigned 32-bit integer')
    if args.output.exists() and (not args.output.is_dir() or any(args.output.iterdir())):
        ap.error('--output must be a new or empty directory')


def run_episode(cn, args, episode, checkpoint, *, deadline=None):
    terminal_after = args.training_fall_seconds
    if (not np.isfinite(terminal_after) or not 0 <= terminal_after <= args.seconds):
        raise ValueError('training fall duration must be finite and within episode horizon')
    if not args.learn and terminal_after:
        raise ValueError('evaluation requires training fall termination disabled')
    import flygym
    from flygym import Camera, Fly, PhysicsError, SingleFlySimulation, preprogrammed

    chunk_ms = 5.0
    seed = args.seed + episode
    enriched = args.arena != 'empty'
    environmental_input = enriched and not args.no_environment_senses
    arena = (training_arena(args.arena == 'obstacle') if args.arena in ('quiet', 'obstacle')
             else habitat_arena() if enriched else floor_arena())
    net = LIFNetwork(cn.W, LIFParams(dt=0.5, da=1.0, depression_u=args.depression_u), seed=seed)
    body_options = { 'control': 'position', 'enable_adhesion': True,
              'floor_collisions': FLOOR_COLLISIONS,
              'contact_solref': CONTACT_SOLREF,
              'enable_olfaction': environmental_input, 'enable_vision': environmental_input,
              'vision_refresh_rate': args.vision_hz,
              'spawn_pos': (0.0, 0.0, 1.2),
              'contact_sensor_placements': preprogrammed.get_collision_geometries('legs')}
    if args.body_model == 'muscles':
        body_options.pop('init_pose', None)
        body_options.pop('control', None)
        fly = make_fly(BodyParams(), body_proxies=False, **body_options)
    else:
        body_options['spawn_pos'] = (0.0, 0.0, 0.2)
        fly = Fly(init_pose=_initial_pose(flygym), **body_options)
    enable_wing_and_head_joints(fly)
    camera = None
    if args.video:
        if enriched:
            from noesis.recording import make_follow_camera
            camera = make_follow_camera(fly, arena, output_path=args.output / f'episode_{episode:03d}.mp4')
        else:
            camera = Camera(attachment_point=fly.model.worldbody,
                            camera_name='camera_left', targeted_fly_names=[fly.name],
                            play_speed=.1, fps=30, timestamp_text=True)
    sim = SingleFlySimulation(fly=fly, cameras=[camera] if camera else [], timestep=1e-4,
                              arena=arena, gravity=GRAVITY)
    configure_floor_contacts(sim)
    viewer = None
    rows = []
    motor_times, motor_spikes = [], []
    status = 'completed'
    start = time.monotonic()
    prefix = args.output / f'episode_{episode:03d}'
    try:
        obs, _ = sim.reset(seed=seed)
        rest = obs['joints'][0].copy()
        target = rest.copy()
        names = list(fly.actuated_joints)
        G = build_motor_matrix(cn.neurons, names)
        foot_adhesion = FootAdhesion(sim.physics, fly, names) if args.adhesion_mode == 'geometry' else None
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
        except (ImportError, OSError, KeyError, AttributeError) as exc:
            print(f'Kinematic range fallback: {exc}', flush=True)
        leg_rows = np.array([i for i, name in enumerate(names)
                             if name.startswith(tuple(f'joint_{leg}' for leg in LEGS))])
        muscles = None
        if args.body_model == 'muscles':
            gp, gn = build_muscle_matrices(cn.neurons, names)
            gp_leg, gn_leg = gp[leg_rows], gn[leg_rows]
            pool_genome = getattr(args, 'pool_genome', None)
            if pool_genome is None and getattr(args, 'pool_gains', None) is not None:
                pool_genome = json.loads(Path(args.pool_gains).read_text(encoding='utf-8'))
            if pool_genome is not None:
                from noesis.pool_genome import apply as apply_pool_genome
                gp_leg, gn_leg = apply_pool_genome(gp_leg, gn_leg, pool_genome)
            tonic_kw = ({
                'tonic_tau_ms': args.tonic_tau_ms,
                'tonic_balance': getattr(args, 'tonic_balance', 0.85),
                'femur_stance_bias': getattr(args, 'femur_stance_bias', 0.15),
            } if getattr(args, 'tonic_tau_ms', 0.0) > 0 else {})
            muscles = MuscleModel(gp_leg, gn_leg, [names[i] for i in leg_rows], **tonic_kw)
        amplitude = args.motor_exploration
        if amplitude is None:
            amplitude = 1.5 if args.learn else 0.0
        if not args.learn and amplitude != 0 and not getattr(args, 'diagnostic_frozen_exploration', False):
            raise ValueError('evaluation requires motor exploration disabled')
        exploration = MotorExploration(cn.neurons, amplitude, seed=seed)
        readout = BrainReadout(cn.neurons, np.unique(G.indices))
        rates_hz = np.zeros(cn.n)
        leg_motor_ids = np.flatnonzero(cn.neurons['Class'].eq('leg_motor_neuron')) if 'Class' in cn.neurons else np.unique(G.indices)
        plasticity = MotorPlasticity(net, np.unique(G.indices),
                                    learning_rate=args.learning_rate, rule=args.learning_rule,
                                    normalize_reward_error=args.normalize_reward_error,
                                    bias_correct_reward_error=getattr(args, 'bias_correct_reward_error', False),
                                    tau_eligibility_ms=100 if args.learning_rule == "perturbation" else 500)
        context = {name: getattr(args, name) for name in (
            'brain', 'body_model', 'bg_hz', 'bg_weight', 'learning_rule', 'learning_rate',
            'reward_mode', 'wing_mode', 'adhesion_mode', 'sensory_map_seed',
            'sensory_weight', 'environment_weight', 'vision_hz', 'depression_u', 'proprioceptive_encoding')}
        context.update(schema=7, reward_objective=args.grounded_objective if args.reward_mode == 'grounded' else 'legacy-body-v1',
                       normalize_reward_error=args.normalize_reward_error, body_interface='isolated-joint-lists-v2',
                       motor_interface='anatomical-coxa-flexion-v3',
                       body_params=vars(BodyParams()) if args.body_model == 'muscles' else None,
                       actuated_joints=names,
                       contact='tarsal-ground-only', neural_dt_ms=.5,
                       eligibility_ms=100 if args.learning_rule == 'perturbation' else 500)
        if getattr(args, 'bias_correct_reward_error', False):
            context.update(schema=8, bias_correct_reward_error=True)
        if checkpoint:
            metadata = Path(checkpoint).with_suffix('.model.json')
            if (not args.import_weights and metadata.exists()
                    and json.loads(metadata.read_text(encoding='utf-8')) != context):
                raise ValueError('checkpoint model context mismatch')
            plasticity.load(checkpoint, weights_only=args.import_weights)
        tracker = RateTracker(cn.n, tau_ms=25.0)
        senses = None if args.no_senses else make_proprioception_map(cn.neurons, args.proprioceptive_encoding)
        if senses is not None and hasattr(senses, 'coverage'):
            prefix.with_suffix('.proprioception-map.json').write_text(
                json.dumps(senses.coverage(), indent=2), encoding='utf-8')
        balance = (None if (args.no_senses or args.no_balance)
                   else BalanceMap(cn.neurons, seed=args.sensory_map_seed))
        environment = EnvironmentSenses(cn.neurons, fly.retina, seed=args.sensory_map_seed) if environmental_input else None
        if environment:
            (prefix.with_suffix('.sensory-map.json')).write_text(
                json.dumps(environment.coverage(), indent=2), encoding='utf-8')
        joints = {(leg, dof): names.index(f'joint_{leg}{dof}')
                  for leg in LEGS for dof in ('Coxa', 'Femur', 'Tibia')}
        if args.viewer:
            viewer = launch_viewer(sim, enriched=enriched)
        print(f'Episode {episode}: {cn.n:,} entries, {G.nnz} motor mappings; '
              f'learning={args.learn}, seed={seed}', flush=True)
        awake_origin = None
        obstacle_duration = 0.
        fall_duration = 0.
        training_fall_terminated = False
        for k in range(int(np.ceil(args.seconds * 1000 / chunk_ms))):
            if deadline is not None and time.monotonic() >= deadline:
                status = 'wall_budget'
                break
            t = k * chunk_ms / 1000
            awake = t >= args.birth
            previous = obs['fly'][0].copy()
            previous_feet = obs['end_effectors'].copy()
            heading = obs['cardinal_vectors'][0].copy()
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
                    if args.warmup_ms > 0:
                        net.run(args.warmup_ms, background_hz=args.bg_hz, background_weight=args.bg_weight)
                drive_parts = []
                loads, _ = ground_contacts(sim, fly)
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
                times_ms, spikes = net.run(chunk_ms, background_hz=args.bg_hz, background_weight=args.bg_weight,
                                    **drive, **exploration.step(chunk_ms))
                selected = np.isin(spikes, leg_motor_ids)
                motor_times.append(times_ms[selected])
                motor_spikes.append(spikes[selected])
                rates_hz = tracker.update(spikes, chunk_ms)
                net_drive = G @ rates_hz
                desired = np.clip(rest + gain_vec * np.tanh(net_drive / 25.0), lo_vec, hi_vec)
                if muscles is not None:
                    desired[leg_rows] = muscles.step(rates_hz, chunk_ms)
                if wing_mask.any() and args.wing_mode == 'legacy':
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
                if foot_adhesion is not None:
                    adhesion = foot_adhesion.command(target)
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
            loads, obstacle_force = ground_contacts(sim, fly)
            obstacle_duration = obstacle_duration + chunk_ms / 1000 if obstacle_force > 1e-5 else 0.
            if args.reward_mode == 'grounded':
                foot_velocity = (obs['end_effectors'] - previous_feet) / (chunk_ms / 1000)
                score = locomotion_reward(previous, obs['fly'][0],
                    float(np.clip(obs['cardinal_vectors'][2][2], -1, 1)),
                    obs['joints'][1][leg_rows], chunk_ms / 1000, heading, loads, foot_velocity, obstacle_duration,
                    objective=args.grounded_objective)
            credit = {'perturbation': exploration.credit(plasticity.motor_indices)} \
                if args.learning_rule == 'perturbation' else {}
            learning = plasticity.update(spikes, chunk_ms, score['reward'], **credit) \
                if awake and args.learn else plasticity.metrics()
            pos = obs['fly'][0]
            neural = readout.rates(rates_hz)
            motion = {
                'obstacle_force': obstacle_force,
                'yaw_rad': float(np.arctan2(obs['cardinal_vectors'][0][1], obs['cardinal_vectors'][0][0])),
                'joint_speed_rad_s': float(np.mean(np.abs(obs['joints'][1][leg_rows]))),
                'joint_offset_rad': float(np.mean(np.abs(obs['joints'][0][leg_rows] - rest[leg_rows]))),
                'command_offset_rad': float(np.mean(np.abs(target[leg_rows] - rest[leg_rows]))),
                'released_feet': int(np.count_nonzero(adhesion == 0)),
                'muscle_pos': float(muscles.a_pos.mean()) if muscles else 0.0,
                'muscle_neg': float(muscles.a_neg.mean()) if muscles else 0.0,
            }
            for li, leg in enumerate(LEGS):
                motion[f'{leg}_ground_load'] = float(loads[li])
                motion[f'{leg}_foot_z_mm'] = float(obs['end_effectors'][li, 2])
                mine = [i for i in leg_rows if names[i].startswith(f'joint_{leg}')]
                motion[f'{leg}_speed_rad_s'] = float(np.mean(np.abs(obs['joints'][1][mine])))
            rows.append(dict(time_s=t + chunk_ms / 1000, awake=int(awake),
                             x_mm=float(pos[0]), y_mm=float(pos[1]), z_mm=float(pos[2]),
                             spikes=len(spikes), **score, **neural, **motion,
                             **env_telemetry,
                             weight_change=learning['mean_relative_change'],
                             reward_prediction_error=learning.get('prediction_error', 0.),
                             learning_signal=learning.get('learning_signal', 0.),
                             reward_error_scale=learning.get('reward_error_scale', 1.)))
            if args.brain_map is not None and k % 4 == 0:
                write_activity(args.brain_map, tracker.rate, t, awake=awake,
                               triggers={**env_telemetry, **neural, **motion,
                                         'learning': args.learn, 'brain': args.brain})
            if k % 40 == 0:
                smell = env_telemetry.get('odor_food', 0)
                light = env_telemetry.get('vision_mean', 0)
                temp = env_telemetry.get('temperature_c', 0)
                print(f'  t={t:.2f}s upright={score["uprightness"]:.2f} '
                      f'reward={score["reward"]:+.3f} spikes={len(spikes)} '
                      f'motor={neural["motor_hz"]:.1f}Hz legs={motion["joint_speed_rad_s"]:.2f}rad/s '
                      f'wall={time.monotonic()-start:.1f}s smell={smell:.2f} light={light:.2f} temp={temp:.1f}', flush=True)
            if viewer:
                viewer.sync()
                if not viewer.is_running():
                    status = 'viewer_closed'
                    break
            # Training environment termination only: reward/eligibility above already
            # includes the actual failure transition. No posture or motor correction.
            if terminal_after and awake:
                fallen = score['uprightness'] < .5 or score['height_mm'] < .3
                fall_duration = fall_duration + chunk_ms / 1000 if fallen else 0.
                if fall_duration + 1e-12 >= terminal_after:
                    training_fall_terminated = True
                    break
        plasticity.save(prefix.with_suffix('.npz'))
        prefix.with_suffix('.model.json').write_text(json.dumps(context, indent=2), encoding='utf-8')
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
        try:
            if camera is not None and hasattr(camera, 'close'):
                camera.close()
        finally:
            sim.close()

    if rows:
        with prefix.with_suffix('.csv').open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    times_ms = np.concatenate(motor_times) if motor_times else np.empty(0)
    spike_idx = np.concatenate(motor_spikes) if motor_spikes else np.empty(0, dtype=int)
    np.savez_compressed(prefix.with_suffix('.motor-spikes.npz'),
                        times_ms=times_ms, neuron_idx=spike_idx, duration_ms=net.time_ms)
    gait = gait_report(times_ms, spike_idx, cn.neurons, net.time_ms, seed)
    prefix.with_suffix('.gait.json').write_text(json.dumps(gait, indent=2), encoding='utf-8')
    awake_rows = [r for r in rows if r['awake']]
    summary = {'episode': episode, 'seed': seed, 'status': status, 'learning': args.learn,
               'termination_reason': 'training_fall' if training_fall_terminated else (
                   'horizon' if status == 'completed' else status.split(':', 1)[0]),
               'episode_horizon_seconds': args.seconds, 'training_fall_seconds': terminal_after,
               **behavior_summary(rows), 'arena': args.arena,
               'imported_weights_from': str(checkpoint) if checkpoint and args.import_weights else None,
               'external_senses': environmental_input,
               'wall_seconds': time.monotonic() - start,
               'simulated_seconds': rows[-1]['time_s'] if rows else 0,
               'awake_samples': len(awake_rows),
               'mean_reward': float(np.mean([r['reward'] for r in awake_rows]))
               if awake_rows else None,
               'mean_uprightness': float(np.mean([r['uprightness'] for r in awake_rows]))
               if awake_rows else None,
               'forward_displacement_mm': rows[-1]['x_mm'] - float(awake_origin[0])
               if awake_rows else None,
               'mean_motor_hz': float(np.mean([r['motor_hz'] for r in awake_rows])) if awake_rows else None,
               'mean_joint_speed_rad_s': float(np.mean([r['joint_speed_rad_s'] for r in awake_rows])) if awake_rows else None,
               'per_leg_mean_speed_rad_s': {leg: float(np.mean([r[f'{leg}_speed_rad_s'] for r in awake_rows])) for leg in LEGS} if awake_rows else {},
               'motor_exploration_mv_ms': amplitude,
               'body_model': args.body_model, 'brain': args.brain,
               'learning_rule': args.learning_rule, 'reward_mode': args.reward_mode,
               'proprioceptive_encoding': args.proprioceptive_encoding,
               'normalize_reward_error': args.normalize_reward_error,
               'reward_objective': context['reward_objective'],
               'wing_mode': args.wing_mode, 'adhesion_mode': args.adhesion_mode,
               'heading_forward_distance_mm': float(sum(r['forward_speed_mm_s'] * chunk_ms / 1000 for r in awake_rows)),
               'mean_supporting_feet': float(np.mean([r['supporting_feet'] for r in awake_rows])) if awake_rows and args.reward_mode == 'grounded' else None,
               'mean_stance_slip_mm_s': float(np.mean([r['stance_slip_mm_s'] for r in awake_rows])) if awake_rows and args.reward_mode == 'grounded' else None,
               'gait_metrics': str(prefix.with_suffix('.gait.json')),
               'tripod_score': gait.get('tibia_flexor_phase', {}).get('tripod_score'),
               'checkpoint': str(prefix.with_suffix('.npz'))}
    if getattr(args, 'bias_correct_reward_error', False):
        summary['bias_correct_reward_error'] = True
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
    try:
        cn, scale = load_training_brain(args.data, args.output / 'cache', args.brain)
    except (ValueError, FileNotFoundError) as exc:
        ap.error(str(exc))
    args.output.mkdir(parents=True, exist_ok=True)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(neuron_table_entries=cn.n, synaptic_pairs=cn.W.nnz,
                  model='LIF + assumed motor/sensory mappings + experimental motor plasticity',
                  flight=False, reward='contact-gated forward motion and stance slip' if args.reward_mode == 'grounded' else 'legacy upright forward-motion objective',
                  gravity_mm_s2=GRAVITY, floor_z_mm=0.0, floor_collisions=FLOOR_COLLISIONS,
                  contact_solref=CONTACT_SOLREF,
                  floor_contact_margin_mm=FLOOR_CONTACT_MARGIN_MM,
                  habitat=Habitat().manifest() if args.arena == 'habitat' else None,
                  versions={name: version(name) for name in
                            ('flygym', 'mujoco', 'numpy', 'scipy', 'numba', 'pandas')},
                  neural_dt_ms=0.5, body_dt_seconds=0.0001, control_dt_ms=5.0,
                  adaptation_mv_per_spike=1.0, background_hz=args.bg_hz,
                  background_weight=args.bg_weight, region_scale=scale,
                  motor_exploration_mv_ms=args.motor_exploration if args.motor_exploration is not None else (1.5 if args.learn else 0.0),
                  motor_control='antagonistic activation -> bounded equilibrium' if args.body_model == 'muscles' else 'legacy tanh targets',
                  motor_slew_rad_s=12.0, body_params=vars(BodyParams()) if args.body_model == 'muscles' else None)
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
