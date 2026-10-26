# Brief for Gemini 4 Argon: locomotion robustness of the Project Noesis fly (2026-10-26)

You are being asked to act as an **independent reviewer and, if a scope is agreed, a bounded implementer** on one problem. Read this
whole file first. Do not claim results you have not measured with the tools listed in section 6.

## 1. The system in one paragraph
A leaky-integrate-and-fire network of 158,262 neurons built from a Drosophila connectome (BANC, rescaled with FAFB/MANC synapse
counts) drives a MuJoCo/NeuroMechFly body. Leg motor neurons are pooled per joint into agonist/antagonist pools; a muscle model
turns pooled firing into joint equilibrium angles (`a = r / (r + r_half)`, r_half = 25 Hz); leg proprioceptive afferents and balance
afferents feed back. Optional learning is a perturbation rule on synapses into the motor circuit. The behaviour we care about is
**standing and then walking, measured with the network frozen** (no learning, no exploration noise), 5 simulated seconds, quiet arena.

## 2. The problem
The outcome of a frozen episode depends mostly on the **random seed**, not on training. Standing fraction (share of time upright and
supported) of the *original untrained weights* over development seeds 10-19 is `0.44 0.93 0.74 0.86 0.44 0.86 0.90 0.95 0.33 0.93`
(mean 0.74, sd 0.24). Falls start in the first 1-2 s: the body spins about the vertical axis by ~2 rad with fewer than three feet on
the floor, then ends on its back. Every training attempt so far produced candidates that stand on some seeds and fall on others, with
no better average than the original.

## 3. What has been tried (all measured; no walking is established)
| Attempt | Result |
|---|---|
| Locomotion track, ~13 training cycles (reward: standing-gate-linear-slip-v5; exploration current 1.5 mV/ms during training) | training births stand 85-98%; frozen candidates fail the protocol; one cycle (V23) passed standing but failed stance slip |
| V24 candidate, frozen with vs without the training exploration current | stands 0.93 with the current, 0.08-0.19 without: the policy depended on exploration noise |
| V25: six births with decaying exploration | seed 10 stands 0.934 frozen, but over seeds 10-14 candidate mean 0.63 vs original 0.68 |
| 12 further births on new seeds (exploration 1.0 -> 0.1), evaluated on 10 dev seeds, paired | candidate 0.64 vs original 0.74, difference -0.10, 95% bootstrap CI [-0.39, +0.21]; no evidence of improvement |
| Soft start (body settles 0.5 s; brain warmed up 0.5 s; both) | no reduction of the seed spread (sd 0.30, 0.29, 0.39 vs 0.24) |
| Inhibition scaling g_I = 4 (all inhibitory weights x4, actuator re-matched) | stands unreliably and slides backward (-3 to -51 mm vs +7 to +11 mm for g_I = 1) |
| Evolution of 84 muscle-pool gains (sep-CMA-ES, 8 generations, population 8, fitness = mean standing over 3 fresh seeds, 2 s episodes) | per-generation difference between distribution mean and original averages about -0.04; final 10-seed evaluation of the evolved mean: standing 0.75 vs 0.74 (difference +0.01, 95% CI [-0.28, +0.29], sign test 6/10, p = 0.38); stands >= 0.9 on 7 of 10 seeds vs 4 of 10 but falls on seeds 11, 12 and 17: no evidence of improvement |
Stable across all models and seeds: forward heading progress of +7 to +12 mm in 5 s.

## 4. What I would like from you (ranked)
1. **Root cause of the early falls.** Read the body, contact, adhesion and muscle-model code and the failing episodes' logic, and propose
   the most likely mechanism(s) for the initial yaw spin / loss of support, with a falsifiable prediction for each.
   Likely places: spawn pose and height (1.2 mm drop), contact softness (`CONTACT_SOLREF`), foot adhesion release rule (at most two feet
   released by femur lift), `CollisionGuard`, the 0.06 rad/step slew limit, joint ranges and damping, tonic motor drive.
2. **Smallest cheap tests.** For each hypothesis, a minimal experiment (a few lines or a flag, original weights only, 10 dev seeds) with
   the outcome that would confirm or kill it.
3. **Critique of our method.** Is the evaluation (paired bootstrap on 10 seeds, standing fraction as the main metric, the protocol's
   stance-slip gate relative to the original weights) sound? What would you change?
4. **Learning-rule review.** Is a single scalar reward per 5 s episode enough for a perturbation rule to assign credit across ~73,000
   motor-circuit synapses? What smaller or better-conditioned parameterisation would you try?
5. Only after agreement: **implement one proposed change** on your own branch with unit tests.

## 5. Rules (the project's, not optional)
- Work on your **own branch and worktree** (suggested `gemini/locomotion-review`). Do **not** write to the worktrees or outputs of the
  other agents (`C:/Users/heyma/.codex/worktrees/...`, `C:/dev/noesis-claude-*`).
- **No heavy simulation runs without telling the user first**: the machine has 8 cores but only ~2-3 GB of free RAM; each episode needs
  ~1 GB, and two other agents may be running. One frozen episode of 5 s takes 5-14 minutes.
- **Never read, copy or transmit** the raw connectome tables (`data/`, `FABA/`, `MANC/`; non-commercial licence) or any key/credential.
- **Seeds:** development seeds are 10-19. Seeds 200, 201, 202 are held out and must not be touched. Seeds 80-111 were used for training and
  1000-1999 for the evolution; do not reuse them for evaluation.
- **No claims of walking, learning or improvement** without the paired 10-seed evaluation below, reported with the interval, and
  with a pre-stated criterion. A single-seed result is not evidence here; seed 10 once looked like a success and was not.
- Report negative results plainly. Preserve all existing uncommitted work.

## 6. Tools and how to run them (Windows, Python 3.11 venv `C:/dev/venvs/noesis`, `PYTHONPATH=src`)
- One frozen episode: `python -u scripts/frozen_probe.py --data <data dir> --output <new dir> --seed N [--checkpoint X.npz] [--pool-gains J.json] [--birth S] [--warmup-ms M] [--seconds T]`
  (prints a `RESULT {...}` line with standing fraction, stance slip, heading-forward distance, cycles).
- Paired evaluation over development seeds: `python -u scripts/dev_eval.py --data <data dir> --out-dir <new dir> --original-dir outputs/dev-eval-orig --models candidate,original --seeds 10 11 12 13 14 15 16 17 18 19 --workers 2 [--candidate-args "..."] [--checkpoint X.npz]`
  (cached results for the original weights exist; it prints the per-seed table, the bootstrap interval, the sign test and the verdict
  of the rule "improved only if the interval's lower bound > 0 and the candidate mean >= 0.75").
- Tests and lint: `python -m pytest -q tests` (about 350 pass), `python -m ruff check src tests scripts`.
- Code to read first: `src/noesis/experiments/learning_lab.py` (episode loop), `src/noesis/body.py` (body, `MuscleModel`),
  `src/noesis/embodiment.py`, `src/noesis/ground_contact.py`, `src/noesis/adhesion.py`, `src/noesis/motor.py`, `src/noesis/sensory.py`,
  `src/noesis/embodied_metrics.py` (reward), `src/noesis/pool_genome.py`, `src/noesis/sep_cma.py`.
- Documents to read first (branch `claude/locomotion-v25`, folder `docs/`): `startup-study.md`, `training-progress-claude-dev-seeds.md`,
  `training-progress-v25.md`, `findings-locomotion-v24-diagnostic.md`, `findings-body-ab-g-inh.md`, `survival-drive-summary.md`,
  and the locomotion track's `training-progress-v12.md` ... `slip-credit-v24.md`.

## 7. Deliverable
A markdown report in `docs/gemini-review-locomotion.md` on your branch: ranked hypotheses for the early falls with evidence and
predictions, the cheapest decisive tests, comments on the method, and (if agreed) one implemented change with its paired evaluation.
Say explicitly what you did not verify.
