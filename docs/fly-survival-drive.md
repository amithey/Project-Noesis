# Survival drive: senses → internal need → behaviour (without legs)

Status 2026-10-16: **stage 1 partly passed (odour specificity at PN/KC); descending-neuron test pending.** Nothing here shows the fly behaves like a fly.

## Why this track exists
Locomotion training (standing, then walking) runs on `codex/brain-body-feedback` and has not
produced measurable walking. This branch (`feature/fly-survival-drive`, based on that branch) works on
the part that does not need legs: sensory input, a homeostatic need, dopamine-gated learning, and a
leg-free virtual agent. It never edits synapses onto leg motor neurons, which belong to the locomotion
track. Merging with a walking body is a later step, planned only after results below.

## Method (staged; each stage has a gate)
| Stage | Content | State |
|---|---|---|
| 0 | Own worktree `C:/dev/noesis-claude-drive` | done |
| 1 | `experiments/sense_audit.py`, `operating_point.py`: does each odour channel reach the mushroom body, dopamine cells and descending neurons? | run; gate partly passed: PN/KC odour specificity found (modest), steering output untested; see [findings-sense-audit.md](findings-sense-audit.md) |
| 2 | `drives.py`: energy/water state, hunger-scaled sensory gain, reward = net intake | done, unit-tested |
| 3 | `experiments/mb_foraging_conditioning.py`: odour A + sugar, fed vs hungry, with no-reward control | script done and unit-tested; **not yet run** |
| 4 | Leg-free virtual agent steered by descending neurons | blocked until a per-neuron descending-neuron test beats shuffle nulls |
| 5 | Evolution of a small genome (sensory gains, hunger coupling, plasticity rates) | not started |
| 6 | Reconnect to the full MuJoCo body | out of scope until walking is measurable |

Hunger and dopamine rates are measured variables, not feelings.

## Stage 1 audit: what it measures
For each olfactory channel in the neuron table (six used by the habitat plus `animal_matter_volatile`,
`carbon_dioxide_volatile`, `pheromone_volatile`, which the habitat does not use), drive its receptor
neurons (left, right, both; 80 Hz, weight 400; 0.3 s rest, 0.7 s stimulus; 3 seeds) on the corrected
brain. Controls: no stimulus, and the same number of receptors drawn at random from other channels.
A descending neuron counts as responsive only if it beats both controls by at least 2 Hz.

### Result
Population-mean responses match the random-receptor control at every operating point tried, but per-glomerulus / per-Kenyon-cell tests with shuffle nulls do find modest odour specificity; see [findings-sense-audit.md](findings-sense-audit.md).

### Note on "bad smell"
The habitat has no faecal or carrion odour. Its only aversive odour is `aversive_volatile`, emitted by
the purple (bitter) patch. The neuron table has an `animal_matter_volatile` receptor channel (26
neurons), the closest label to carrion; it is audited but not yet part of the habitat.

## Stage 3 design (not yet run)
Four conditions: {fed, hungry} × {sugar pairing, no reward}; hunger scales sugar-receptor drive only.
Per seed, learning index = (ΔA − ΔB) / baseline A for MBON rate with plasticity frozen at test.
A conditioned effect requires the sugar index to be below the no-reward index, paired by seed.
Caveat: earlier work in this repo found only tiny weight changes (≈0.004%) in world-driven
learning, so a null result is plausible and will be reported as such.

## Reproduce
```powershell
$env:PYTHONPATH='src'
$data='C:\Users\heyma\OneDrive\Desktop\מדעי המחשב\פרוייקטים\Project Noesis\data'
python -m noesis.experiments.sense_audit --data $data --seeds 3
python -m noesis.experiments.mb_foraging_conditioning --data $data --seeds 3
```
Each full run takes tens of minutes (158,262 neurons). Output directories are git-ignored.

## Known open issues
- 3 tests in `tests/test_brain_map.py` fail in a full-suite run with
  `module 'matplotlib._api' has no attribute 'check_getitem'` but pass alone (order-dependent;
  seen on the earlier branch, unrelated to this work, not yet investigated).
- The stage 4 mapping from descending neurons to speed/turning is an assumption to be tested
  against random-neuron and shuffled-brain controls before any behavioural claim.
