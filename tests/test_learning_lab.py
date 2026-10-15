"""Episode integration against a tiny brain and deterministic fake body API."""

import csv
import json
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from noesis.connectome import Connectome
from noesis.experiments import learning_lab as lab
from noesis.habitat import Habitat
from noesis.lif import LIFNetwork
from noesis.plasticity import MotorPlasticity
from noesis.sensory import LEGS


@pytest.fixture
def lab_environment(monkeypatch, tmp_path):
    state = SimpleNamespace(simulations=[], networks=[], learners=[], fail_step=None)
    names = [f"joint_{leg}{dof}" for leg in LEGS for dof in ("Coxa", "Femur", "Tibia")]

    class PhysicsError(Exception):
        pass

    class Fly:
        def __init__(self, **kwargs):
            self.actuated_joints = names
            self.contact_sensor_placements = [f"{leg}Tarsus1" for leg in LEGS]
            self.model = SimpleNamespace(worldbody=object())
            self.name = "test_fly"
            self.retina = SimpleNamespace(
                pale_type_mask=np.array([0, 1, 0, 1]),
                hex_pxls_to_human_readable=lambda eye, **kw: (eye[None] * 255).astype(np.uint8))

    class Simulation:
        def __init__(self, timestep, **kwargs):
            self.timestep = timestep
            self.steps = 0
            self.closed = False
            self.physics = SimpleNamespace(model=SimpleNamespace(ptr=object()),
                                           data=SimpleNamespace(ptr=object()))
            state.simulations.append(self)

        def observation(self):
            return {"joints": np.zeros((2, len(names))),
                    "fly": np.array([[self.steps * self.timestep, 0., 0.8]]),
                    "cardinal_vectors": np.eye(3), "contact_forces": np.zeros((6, 3)),
                    "end_effectors": np.zeros((6, 3)), "vision": np.full((2, 4, 2), 0.5),
                    "odor_intensity": np.full((6, 4), 0.5)}

        def reset(self, seed):
            self.seed = seed
            return self.observation(), {}

        def step(self, action):
            assert action["joints"].shape == (len(names),)
            self.steps += 1
            if state.fail_step == self.steps:
                raise PhysicsError("injected instability")
            return self.observation(), 0, False, False, {}

        def close(self):
            self.closed = True

    class Network(LIFNetwork):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            state.networks.append(self)

        def run(self, ms, **kwargs):
            self.last_drive = kwargs
            # Deterministic coactivity isolates episode plumbing from firing calibration.
            self.step += round(ms / self.p.dt)
            return np.array([self.time_ms, self.time_ms]), np.array([0, 1])

    class Learner(MotorPlasticity):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            state.learners.append(self)

        def load(self, *args, **kwargs):
            super().load(*args, **kwargs)
            assert not self.activity.any()
            assert not self.eligibility.any()
            assert self.reward_baseline == 0

    flygym = ModuleType("flygym")
    flygym.Fly, flygym.PhysicsError = Fly, PhysicsError
    flygym.SingleFlySimulation = Simulation
    flygym.Camera = object
    flygym.preprogrammed = SimpleNamespace(get_collision_geometries=lambda _: [])
    monkeypatch.setitem(sys.modules, "flygym", flygym)
    monkeypatch.setattr(lab, "floor_arena", lambda: object())
    monkeypatch.setattr(lab, "configure_floor_contacts", lambda sim: None)
    monkeypatch.setattr(lab, "LIFNetwork", Network)
    monkeypatch.setattr(lab, "MotorPlasticity", Learner)
    monkeypatch.setattr(lab, "build_motor_matrix", lambda neurons, joints: sparse.csr_matrix(
        ([1.0], ([0], [1])), shape=(len(joints), 2)))
    state.cn = Connectome(pd.DataFrame({"root_id": [1, 2]}),
                         sparse.csr_matrix([[0., 50.], [0., 0.]]))
    state.args = lab.parser().parse_args([
        "--output", str(tmp_path), "--seconds", "0.025", "--birth", "0.005",
        "--no-senses", "--learn", "--seed", "7", "--arena", "empty",
        "--body-model", "legacy", "--motor-exploration", "0",
        "--learning-rule", "hebbian", "--learning-rate", "0.02", "--reward-mode", "legacy", "--adhesion-mode", "legacy",
    ])
    return state


def test_train_two_episodes_then_frozen_evaluation(lab_environment):
    env = lab_environment
    first, checkpoint1 = lab.run_episode(env.cn, env.args, 0, None)
    second, checkpoint2 = lab.run_episode(env.cn, env.args, 1, checkpoint1)
    env.args.learn = False
    evaluation, checkpoint3 = lab.run_episode(env.cn, env.args, 2, checkpoint2)
    with np.load(checkpoint1) as one, np.load(checkpoint2) as two, np.load(checkpoint3) as three:
        assert one["weights"][0] > 50
        assert two["weights"][0] > one["weights"][0]
        np.testing.assert_array_equal(three["weights"], two["weights"])
        assert not three["eligibility"].any()
        assert float(three["reward_baseline"]) == 0
        learned_change = float(two["weights"][0] / 50 - 1)
    for episode, summary in enumerate((first, second, evaluation)):
        assert summary["status"] == "completed"
        assert summary["seed"] == 7 + episode
        assert summary["awake_samples"] == 4
        assert summary["simulated_seconds"] == pytest.approx(0.025)
        assert summary["forward_displacement_mm"] == pytest.approx(0.02)
        assert summary["learning"] is (episode < 2)
        prefix = env.args.output / f"episode_{episode:03d}"
        saved = json.loads(prefix.with_suffix(".json").read_text())
        assert saved == summary
        with prefix.with_suffix(".csv").open(newline="") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 5
        assert [int(row["awake"]) for row in rows] == [0, 1, 1, 1, 1]
        assert all(-1 <= float(row["reward"]) <= 1 for row in rows)
        if episode == 2:
            assert all(float(row["weight_change"]) == pytest.approx(learned_change) for row in rows)
    assert all(sim.closed for sim in env.simulations)
    assert [sim.seed for sim in env.simulations] == [7, 8, 9]
    assert len({id(net) for net in env.networks}) == 3
    np.testing.assert_array_equal(env.cn.W.data, [50.])


def test_physics_error_records_partial_episode_and_closes_body(lab_environment):
    env = lab_environment
    env.fail_step = 55
    summary, checkpoint = lab.run_episode(env.cn, env.args, 0, None)
    assert summary["status"] == "physics_error: injected instability"
    assert summary["simulated_seconds"] == pytest.approx(0.005)
    assert summary["awake_samples"] == 0
    assert summary["mean_reward"] is None
    assert checkpoint.exists()
    assert env.simulations[0].closed


def test_habitat_observations_reach_brain_automatically_and_are_saved(lab_environment, monkeypatch):
    env = lab_environment
    env.args.arena = 'habitat'
    neurons = pd.DataFrame({
        'root_id': [1, 2], 'Class': ['olfactory_receptor_neuron', 'leg_motor_neuron'],
        'Sub Class': ['antenna_olfactory_receptor_neuron', 'front_leg_motor_neuron'],
        'Primary Cell Type': ['ORN', 'MN'], 'Soma side': ['left', 'left'],
        'Nerve': ['left_antennal_nerve', 'left_prothoracic_leg_nerve'],
        'Function': ['olfactory,fruity_volatile', ''], 'Flow': ['afferent', 'efferent'],
    })
    env.cn = Connectome(neurons, env.cn.W)
    monkeypatch.setattr(lab, 'habitat_arena', lambda: SimpleNamespace(habitat=Habitat()))
    monkeypatch.setattr(lab, 'mouth_position', lambda sim, fly: np.array([0, 0, 1]))
    monkeypatch.setattr(lab, 'contact_tastes', lambda sim, fly, habitat: np.zeros((7, 3)))
    summary, checkpoint = lab.run_episode(env.cn, env.args, 0, None)
    assert summary['status'] == 'completed' and checkpoint.exists()
    drive = env.networks[0].last_drive
    np.testing.assert_array_equal(drive['drive_idx'], [0])
    np.testing.assert_array_equal(drive['drive_rate_hz'], [61])
    np.testing.assert_array_equal(drive['drive_weight'], [100])
    prefix = env.args.output / 'episode_000'
    with np.load(prefix.with_suffix('.senses.npz')) as saved:
        np.testing.assert_array_equal(saved['neuron_indices'], [0])
        assert saved['vision'].shape == (2, 4, 2)
    assert prefix.with_suffix('.eyes.png').exists()
    with prefix.with_suffix('.csv').open(newline='') as f:
        rows = list(csv.DictReader(f))
    assert all(float(row['odor_food']) == 0.5 for row in rows)
    assert env.simulations[0].closed


def test_viewer_close_records_interruption_and_releases_resources(lab_environment, monkeypatch):
    env = lab_environment
    env.args.viewer = True
    env.args.arena = "habitat"
    env.args.no_environment_senses = True
    monkeypatch.setattr(lab, "habitat_arena", lambda: SimpleNamespace(habitat=Habitat()))
    glfw = ModuleType("glfw")
    glfw.hints_reset = 0
    glfw.init = lambda: True
    glfw.default_window_hints = lambda: setattr(glfw, "hints_reset", glfw.hints_reset + 1)
    viewer = SimpleNamespace(
        syncs=0, closed=False,
        cam=SimpleNamespace(lookat=np.zeros(3), distance=0.0, elevation=0.0),
    )
    viewer.sync = lambda: setattr(viewer, "syncs", viewer.syncs + 1)
    viewer.is_running = lambda: False
    viewer.close = lambda: setattr(viewer, "closed", True)
    module = ModuleType("mujoco")
    module.viewer = ModuleType("mujoco.viewer")
    module.viewer.launch_passive = lambda *args: viewer
    monkeypatch.setitem(sys.modules, "glfw", glfw)
    monkeypatch.setitem(sys.modules, "mujoco", module)
    monkeypatch.setitem(sys.modules, "mujoco.viewer", module.viewer)
    summary, _ = lab.run_episode(env.cn, env.args, 0, None)
    assert summary["status"] == "viewer_closed"
    assert summary["simulated_seconds"] == pytest.approx(0.005)
    assert glfw.hints_reset == 1
    assert viewer.syncs == 2 and viewer.closed
    assert viewer.cam.distance == 22 and viewer.cam.elevation == -50
    assert env.simulations[0].closed


def test_muscle_path_routes_neural_rates_and_records_actual_motion(lab_environment, monkeypatch):
    env = lab_environment
    env.args.body_model = 'muscles'
    env.args.motor_exploration = .5
    env.args.brain_map = env.args.output / 'live.npz'
    env.cn = Connectome(pd.DataFrame({
        'root_id': [1, 2], 'Class': ['other', 'leg_motor_neuron'],
        'Sub Class': ['other', 'front_leg_motor_neuron'],
        'Primary Cell Type': ['PAM_test', 'tibia_flexor'],
        'Soma side': ['left', 'left'], 'Nerve': ['', 'left_leg_nerve'],
    }), env.cn.W)
    calls = []

    class Muscles:
        def __init__(self, gp, gn, names):
            self.a_pos, self.a_neg = np.zeros(len(names)), np.zeros(len(names))
            calls.append(self)

        def step(self, rates, dt_ms):
            assert rates[1] > 0 and dt_ms == 5
            self.a_pos[:] = .5
            return np.full(len(self.a_pos), .25)

    monkeypatch.setattr(lab, 'make_fly', lambda params, **kw: sys.modules['flygym'].Fly(**kw))
    monkeypatch.setattr(lab, 'MuscleModel', Muscles)
    monkeypatch.setattr(lab, 'build_muscle_matrices', lambda neurons, names: (
        sparse.csr_matrix(np.tile([0, 1], (len(names), 1))),
        sparse.csr_matrix((len(names), 2))))
    summary, _ = lab.run_episode(env.cn, env.args, 0, None)
    assert calls and summary['body_model'] == 'muscles'
    assert summary['motor_exploration_mv_ms'] == .5
    np.testing.assert_array_equal(env.networks[0].last_drive['current_idx'], [1])
    assert abs(env.networks[0].last_drive['current_mv_per_ms'][0]) < .5
    with (env.args.output / 'episode_000.csv').open(newline='') as f:
        rows = list(csv.DictReader(f))
    # Fake physical joints deliberately stay still despite nonzero muscle commands.
    # Diagnostics must expose the distinction instead of claiming movement.
    assert float(rows[-1]['command_offset_rad']) > 0
    assert float(rows[-1]['joint_speed_rad_s']) == 0
    assert float(rows[-1]['pam_hz']) > 0
    assert float(rows[-1]['muscle_pos']) == .5
    from noesis.viz.brain_map import read_activity
    _, _, awake, status = read_activity(env.args.brain_map)
    assert awake and status['learning']
    assert status['brain'] == 'corrected'
    assert status['motor_hz'] > 0 and status['joint_speed_rad_s'] == 0


def test_new_learning_and_grounded_reward_path_runs_without_phantom_exploration(lab_environment):
    env = lab_environment
    env.args.learning_rule = 'perturbation'
    env.args.reward_mode = 'grounded'
    env.args.wing_mode = 'neural'
    # No exploration: this must freeze weights even with positive body reward.
    summary, checkpoint = lab.run_episode(env.cn, env.args, 0, None)
    assert summary['learning_rule'] == 'perturbation'
    assert summary['mean_supporting_feet'] == 0
    with np.load(checkpoint) as stored:
        np.testing.assert_array_equal(stored['weights'], stored['initial_weights'])


def test_deadline_episode_is_partial_and_cannot_claim_completion(lab_environment):
    env = lab_environment
    summary, checkpoint = lab.run_episode(env.cn, env.args, 0, None, deadline=0.)
    assert summary['status'] == 'wall_budget'
    assert summary['simulated_seconds'] == 0
    assert checkpoint.exists() and env.simulations[0].closed


def test_frozen_episode_rejects_accidental_exploration(lab_environment):
    env = lab_environment
    env.args.learn = False
    env.args.motor_exploration = 1.5
    with pytest.raises(ValueError, match='exploration disabled'):
        lab.run_episode(env.cn, env.args, 0, None)
    assert env.simulations[0].closed


def test_checkpoint_rejects_changed_dynamics_before_loading_weights(lab_environment):
    env = lab_environment
    _, checkpoint = lab.run_episode(env.cn, env.args, 0, None)
    env.args.depression_u = .03
    with pytest.raises(ValueError, match='context mismatch'):
        lab.run_episode(env.cn, env.args, 1, checkpoint)


def test_explicit_weight_transfer_preserves_experience_across_body_interface_fix(lab_environment):
    env = lab_environment
    _, checkpoint = lab.run_episode(env.cn, env.args, 0, None)
    metadata = checkpoint.with_suffix('.model.json')
    context = json.loads(metadata.read_text())
    context['body_interface'] = 'older-body-model'
    metadata.write_text(json.dumps(context))
    env.args.import_weights = True
    summary, saved = lab.run_episode(env.cn, env.args, 1, checkpoint)
    assert summary['imported_weights_from'] == str(checkpoint)
    with np.load(checkpoint) as old, np.load(saved) as new:
        assert new['weights'][0] >= old['weights'][0]
    assert json.loads(saved.with_suffix('.model.json').read_text())['body_interface'] == 'isolated-joint-lists-v2'
