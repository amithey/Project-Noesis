# V23: optional startup correction for motor-action reward credit

V22 completed six training births but failed frozen seed10: standing89.89899%
and slip16.084272mm/s against original9.366594. No model was promoted.
Its final training weights are retained; all source/artifacts and the stopped
ledger remain archived. Twenty-two required frozen cases were not run after
predeclared early rejection; they are not treated as successful controls.

The legacy two-second baseline/RMS start at zero at every birth. With fixed V22
recorded rewards,91-100% of the first up-to-one awake second saturates at±1.
The V23 hypothesis corrects initialization bias in the weighted reward moments:
EMA mass m += alpha*(1-m), effective alpha = alpha/m. The first observation
establishes the baseline with zero prediction error; subsequent errors use the
previous corrected mean. The same effective alpha updates squared-error RMS;
the0.01floor and±1signal clipping remain. This optional action-credit heuristic
is not an established biological mechanism or proof of walking improvement.

--bias-correct-reward-error requires --normalize-reward-error and perturbation
learning. The flag defaults off. Legacy update results and checkpoint bytes
match the preserved v3 reference exactly in normalized and unnormalized modes.
Opt-in checkpoints use v4 and model-context schema8, saving normalization mass;
ordinary loads require matching learning/context configuration. Explicit
weights-only import crosses versions while still checking topology, motor IDs,
original reference weights, signs and±50%bounds. Every fresh birth/frozen load
resets baseline, RMS, mass and traces; exact trace restoration is separately
supported/tested. Checkpoints do not save body/membrane/RNG state.

The full staged suite passed317tests plus Ruff. After making the legacy fixture
self-contained, all17new learner tests passed again. Full-connectome preflight
imports the exact V22 bank79ed5b2c3392ca082f8289ad8e43abbedd0162fcd01cb843e6f4987df471f418:
158262neurons,73594plastic edges into498readouts, all selected weights/reference
arrays/topology/motor IDs unchanged; max original-weight change22.071025%.
No body or neural integration occurs in this preflight.

Prospective V23 training: six quiet births74..79,5seconds each, training-only
terminal fall0.1seconds; background3Hz x40, lr0.2, eligibility100ms, bounded
motor-current exploration1.5mV/ms. Only reward normalization changes. Original
connectome, LIF dynamics (depressionU0), muscle geometry/adhesion, sensory and
reward code are unchanged. No clock, prescribed stepping or external walking
controller is added. The existing scalar standing/slip reward is unchanged.

All comparisons remain full5seconds/no learning/no exploration/no fall cutoff:
validation10/11, heldout200/201/202, matched original/candidate quiet and obstacle
trials, plus external-sense ablation retaining joint/balance feedback. All gates
must pass before promotion; early quiet rejection retains the last training bank
without replacing best validated. Each worker has a3600-second wall limit and
--resume; incomplete episodes restart from their beginning, not from a partial
body state. Normal startup stays unchanged. Artifacts/negative videos and
completed-case observations will be preserved without opening a live ledger.

The installed full suite also passed317tests in72.52seconds, with Ruff and
git diff --check passing. Physical outcomes are pending at this registration point. The user goal of validated basic walking remains open.
