# V29: retain directional credit during partial slip

V28's full frozen candidate10 falls:12.22% standing versus required90%, despite
lower mean slip while collapsed. Its failure and exact interrupted-prefix replay
are retained in premotor-credit-v28.md and immutable boundary002 artifacts.
Latest training97weights remain the parent, not a validated deployment model.

Training-only diagnostics on all six V28births find positive progress credit in
1.05–3.55% of awake frames.87.17–95.39% of upright, height-valid forward-moving
samples with three loaded feet receive zero progress because slip exceeds body
speed. This is sparse scalar feedback, not proof of the cause of failed gait.
A fixed-data counterfactual retains signed progress in those eligible frames;
it cannot predict the behavior of a newly trained fly.

The opt-in objective standing-gate-smooth-grip-v6 changes only grip to
1/(1+stance_slip/(abs(heading_speed)+.1)). It retains the same posture/height,
three-foot and speed gates, standing/support/effort/obstacle terms, linear
.03per-mm/s slip cost and final reward clipping. The default v3 and old v4/v5
outputs stay exact. Physical standing, forward distance, lift cycles, slip
reduction and obstacle acceptance are unchanged. No motor action, timing,
external gait clock, prescribed leg order, topology or sign is introduced.

Full staged source digest:
b190fadaadd43ff2bf7945819125e0b65ed2d7fa0794ddfa087e812e48f747e8.
Only embodied_metrics.py differs from CF8515; installed bytes match tested
source.384full tests pass286.86s, Ruff passes. Tests include real MuJoCo body,
eye, contact and collision checks, plus small synthetic/fake-body pipelines.
The suite runs alongside frozen V28without a second full-connectome body worker;
its wall duration is not a performance benchmark. Full log SHA256:
ca5cc3a18667c29c237eda342eafdec8d95136e57026b716c588a3cd517c34b4.
Thirty installed targeted tests also pass5.52s. Coverage includes360random
old-objective golden outputs, physical/speed/support guards, reward monotonicity
outside clipping, V5-toV6weight-only circuit import, no frozen injected currents,
all25controls, retained proprio/balance in ablation and incompatible resume
rejected before report mutation. The v5choice remains in its original third
position; its test now permits later opt-in choices to be appended.

Actual full-connectome preflight passes without neural/body integration:
158262neurons,562plasticity targets including64existing CNS targets,
85390existing plastic edges. All parent weights, original references, target
selection, topology/signs and limits are exact; maximum deviation25.480914%
from original. All learning traces/baselines/normalization reset to zero;
full-network save/load is exact and frozen exploration is completely off.
Imported-reset checkpoint has SHA256f28efa79a9c80da2acc0ebeda0350d4d648daeed21e1d89f761fd589e2ccaf72;
its new model context records only the explicit scalar objective change.

Protocol registered before body results: six serial quiet births98..103from
retained V28training97/A2E5, lr.2, eligibility100ms, bounded OU1.5, circuit scope,
original sign/50% bounds, BG40,U0. Five-second training horizon, training-only
sustained-fall cutoff.1s. Validation10/11and untouched heldout200/201/202retain
all25full5s frozen cases with learning/exploration/fall cutoff off. Matched
original/trained/externally-ablated controls retain proprioception and balance.
A failed quiet candidate is rejected early without claiming skipped controls.
Each invocation is limited to3600wall seconds; partial episodes replay from
their original start. Latest training remains separate from best validated.

This is a serial continuation with new training seeds, not a causal comparison
isolating reward from every historical protocol difference. It is a learning
hypothesis. No walking result is claimed. No normal-startup update is allowed
without all behavioral gates, actual frozen video evidence and startup check.
The pilot omits recording to reduce overhead; a passing bank still needs video.
