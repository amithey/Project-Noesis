"""Frozen 5 s evaluation of one checkpoint under the locomotion protocol's exact settings.

Rebuilds the case exactly as ``train_validate`` does for its frozen jobs (quiet arena, learning off, no fall cutoff),
using the V24 registration (standing-gate-linear-slip-v5, bg weight 40, legacy proprioception, normalised and
bias-corrected reward error). ``--exploration`` > 0 is a DIAGNOSTIC that injects the training-time motor current
into an otherwise frozen run (weights never change); it is not a protocol condition. Output must be a new directory.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from noesis.embodiment import load_training_brain
from noesis.experiments import learning_lab as lab


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--checkpoint', type=Path, help='omit for the original weights')
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--exploration', type=float, default=0.0)
    ap.add_argument('--seconds', type=float, default=5.0)
    a = ap.parse_args()
    if a.output.exists() and any(a.output.iterdir()):
        ap.error('output must be new or empty')
    template = lab.parser().parse_args(['--data', str(a.data), '--output', str(a.output / 'probe'), '--seconds', '5',
                                        '--learning-rate', '0.2', '--arena', 'quiet', '--depression-u', '0', '--learn'])
    template.grounded_objective = 'standing-gate-linear-slip-v5'
    template.training_fall_seconds = 0.1
    template.proprioceptive_encoding = 'legacy'
    template.normalize_reward_error = True
    template.bias_correct_reward_error = True
    template.bg_weight = 40.0
    lab.validate_args(lab.parser(), template)
    case = argparse.Namespace(**vars(template))
    case.seed, case.arena, case.learn = a.seed, 'quiet', False
    case.seconds, case.training_fall_seconds = a.seconds, 0.0
    case.motor_exploration = a.exploration
    case.diagnostic_frozen_exploration = a.exploration > 0
    case.import_weights, case.no_environment_senses, case.video = False, False, False
    case.output = a.output
    case.output.mkdir(parents=True, exist_ok=True)
    cn, _ = load_training_brain(a.data, a.output / 'cache')
    summary, _ = lab.run_episode(cn, case, 0, a.checkpoint)
    keys = ('status', 'simulated_seconds', 'standing_fraction', 'mean_stance_slip_mm_s', 'heading_forward_distance_mm',
            'mean_supporting_feet', 'mean_reward', 'mean_uprightness', 'mean_motor_hz', 'leg_step_cycles')
    print('RESULT ' + json.dumps({k: summary[k] for k in keys if k in summary}))


if __name__ == '__main__':
    main()
