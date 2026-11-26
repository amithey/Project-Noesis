"""Frozen constant thermo input versus a zero-event-weight RNG-matched sham."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', required=True, choices=('plain', 'sham', 'stimulus'))
    parser.add_argument('--seed', required=True, type=int)
    for name in ('core', 'data', 'checkpoint', 'output'):
        parser.add_argument('--' + name, required=True, type=Path)
    args = parser.parse_args(argv)
    if args.seed in (10, 11, 200, 201, 202):
        parser.error('Final validation seeds prohibited')
    sys.path.insert(0, str(args.core.resolve()))
    from . import controller, run
    from .sensed_brain import sensed_class
    from .senses_bridge import select_modality
    from .tp2_rng import rng_state

    original = controller.FrozenBrain
    chosen = original if args.mode == 'plain' else sensed_class(original)
    created = []

    def forbidden_field(_):
        raise RuntimeError('Constant input must never evaluate a position-dependent field')

    class Observed(chosen):
        def __init__(self, connectome, rest, lo, hi, **options):
            if args.mode != 'plain':
                options.update(field=forbidden_field, gain=50., mode='constant', level=.5)
            super().__init__(connectome, rest, lo, hi, **options)
            taken = np.union1d(self.senses.idx, self.balance.idx)
            self.warm = select_modality(connectome.neurons, 'thermo_warm', taken=taken)
            self.cool = select_modality(connectome.neurons, 'thermo_cool', taken=np.union1d(taken, self.warm))
            if len(self.warm) != 6 or len(self.cool) != 6:
                raise ValueError('Require six warm and six cool annotation-selected targets')
            self.recorded = {key: [] for key in ('warm_spikes', 'cool_spikes', 'total_spikes', 'rng_state',
                             'extra_idx', 'extra_rate_hz', 'extra_weight', 'drive_length', 'drive_indices_sha256',
                             'commands', 'adhesion')}
            created.append(self)

        def _extra_drive_parts(self, observation):
            parts = super()._extra_drive_parts(observation)
            if args.mode == 'plain':
                assert not parts
                idx, rates, weight = np.full(6, -1, dtype=np.int64), np.zeros(6), 0.
                full_idx = np.concatenate([self.senses.idx, self.balance.idx])
            else:
                assert len(parts) == 1
                idx, rates, weight = parts[0]
                idx, rates = np.asarray(idx, dtype=np.int64), np.asarray(rates)
                np.testing.assert_array_equal(idx, np.sort(self.warm))
                np.testing.assert_array_equal(rates, np.full(6, 25.))
                assert weight == 400.
                weight = 0. if args.mode == 'sham' else 400.
                parts = [(idx, rates, weight)]
                full_idx = np.concatenate([self.senses.idx, self.balance.idx, idx])
            self.extra = (idx.copy(), rates.copy(), np.full(6, weight))
            self.drive_length = len(full_idx)
            self.drive_signature = np.frombuffer(hashlib.sha256(full_idx.astype('<i8').tobytes()).digest(), np.uint8).copy()
            return parts

        def step(self, observation, dt_ms):
            commands, pads = super().step(observation, dt_ms)
            # Read-only MT state capture; native B2 replay independently gates
            # observational invariance through the actual brain/body trajectory.
            values = dict(warm_spikes=(self.last_spikes[:, None] == self.warm).sum(axis=0),
                          cool_spikes=(self.last_spikes[:, None] == self.cool).sum(axis=0),
                          total_spikes=len(self.last_spikes), rng_state=rng_state(),
                          extra_idx=self.extra[0], extra_rate_hz=self.extra[1], extra_weight=self.extra[2],
                          drive_length=self.drive_length, drive_indices_sha256=self.drive_signature,
                          commands=commands.copy(), adhesion=pads.copy())
            for key, value in values.items():
                self.recorded[key].append(value)
            return commands, pads

    command = ['--backend', 'cpu', '--worlds', '1', '--seconds', '5', '--wall-seconds', '300',
               '--seed', str(args.seed), '--noesis-source', str(args.core.resolve()),
               '--data', str(args.data.resolve()), '--checkpoint', str(args.checkpoint.resolve()),
               '--output', str(args.output)]
    controller.FrozenBrain = Observed
    try:
        report = run.run(run.parser().parse_args(command))
    finally:
        controller.FrozenBrain = original
    if report['status'] != 'completed_transport_pilot' or len(created) != 1:
        raise RuntimeError('Incomplete probe or wrong population')
    brain = created[0]
    arrays = {key: np.asarray(value) for key, value in brain.recorded.items()}
    assert all(len(value) == 1000 and np.isfinite(value).all() for value in arrays.values())
    np.savez_compressed(args.output / 'tp2-observations.npz', **arrays)
    afferents = dict(warm=brain.warm.tolist(), cool=brain.cool.tolist(), index_context='SHA-verified transferred dataset')
    (args.output / 'afferents.json').write_text(json.dumps(afferents, indent=2), encoding='utf-8')
    report.update(tp2_mode=args.mode, warm_afferent_count=6, cool_afferent_count=6,
                  thermo_requested_hz=0. if args.mode == 'plain' else 25.,
                  thermo_event_weight=400. if args.mode == 'stimulus' else 0.,
                  observer_module_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    report['senses'] = [*report['senses'], 'TP2-mode:' + args.mode]
    (args.output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({key: report[key] for key in ('status', 'tp2_mode', 'completed_control_chunks', 'weights_unchanged')}))


if __name__ == '__main__':
    main()
