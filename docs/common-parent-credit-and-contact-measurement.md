# Common-parent credit pilot and contact measurement

V24 sequential training and V25 frozen screen motivate diagnosing transfer from
exploratory training to a frozen newborn. No model is validated or promoted.
All changes in this note are research tools in outputs; core source remains
cd36906d477df98817acab5e62cdb5d2b025e27903af674ec41bf24f5a9bc865.

## Registered common-parent pilot (not yet trained)

Six independent births86..91 will begin from the same V23 bank2f3400da...
with explicit weights-only import into objectivev5. Learning remains rate.2,
eligibility100ms, bounded OU1.5mV/ms, normalization+bias correction, background
3Hz*40, no depression. All completed births, including sustained falls, have
equal weight in an arithmetic mean of synaptic increments; no best-child
selection. Incomplete births restart from the registered common parent.
Original topology, neuron order, signs and +/-50%original bounds remain fixed.
All learning traces/statistics are zeroed in the mean bank before frozen testing.
No joint trajectory, leg order, clock, direct motor controller or changed sensor
mapping is introduced. This is a hypothesis, not evidence of improved walking.

Sequential V24 increments have pairwise cosine from approximately-.167to.192,
with net-to-path update coherence.41254. Because start banks and trajectories
vary, these figures are NOT independent gradient-variance estimates. Equal-parent
births allow a cleaner examination of update consistency. A smaller update alone
would not prove useful learning. Physical frozen controls remain decisive.

The pilot driver/aggregator and fifteen focused tests are under
outputs/batch-credit-v26-prototype. Actual full-connectome preflight verifies
158262neurons/73594plastic edges/498readouts, exact imported weights and original
references, all newborn statistics reset. Ruff and fifteen tests pass. A mean
bank is latest training only; best validated stays null. Even a successful
seed10screen cannot promote it. The remaining validation/heldout/obstacle and
sensory-ablation cohort and frozen video remain necessary.

## Read-only contact measurement before further credit training

Current stance slip is a5ms distal-endpoint velocity, gated by any positive
force-bearing tarsal contact in that leg. A rigid segment rotating about a
stationary proximal point can move its distal endpoint. This possibility alone
does not show that the actual metric is wrong, or explain prior failed walking.
The previous onset association did not support a dominant onset-only artifact.

outputs/contact-slip-v27-prototype registers a one-second frozen original10
measurement diagnostic. An observer calls the existing ground-contact function
unchanged, then computes actual contact-point global Jacobians times qvel for
both bodies. Horizontal relative contact velocity is force-weighted within each
leg, averaged over the same supported legs; raw contact geometries, forces and
points are retained. Existing reward, dynamics and success gates are unchanged.
Three MuJoCo tests verify analytic translation/rotation, a stationary-point /
moving-endpoint example, and actual force-bearing ground-contact observation
without mutating physics state or returned support. Ruff and all three pass.

The actual diagnostic must reproduce every first200CSV rows, motor-spike prefix
and original checkpoint arrays of the existing five-second original10control.
It is NOT a behavioral trial and cannot establish standing/walking success.
Only after V25 is terminal and matching Python/encoder absence verified may
another physical worker start. No concurrent simulation has been launched.


## V25 terminal rejection

Frozen seed10 of training83 completed5s with zero learning/exploration and no
fall cutoff. Standing93/990=9.393939%, mean uprightness−.474094, mean slip
4.698131mm/s and reward−.765151. Standing and reward checks failed; low slip
accompanied collapse. Recorded support/lift cycles are not walking during this
collapse. Frozen weight arrays exactly equal the selected bank and all learner
statistics are zero. Body CSV measures independently recomputed.
Worker18585 exited0 after680.735s; OS inventory confirmed no matching worker
or encoder before stopped report read. All completed non-video artifacts copied
with hash checks. No promotion; latest training85 is retained independently.
Read-only original10 contact measurement now takes precedence over the prepared
common-parent pilot. No additional learning has yet been performed.


## Actual contact diagnostic: sliding is present

The one-second original10 diagnostic completed with exact first200CSV rows,
exact950ms motor-spike prefix and every original checkpoint array unchanged.
190awake samples: endpoint proxy13.416658mm/s, actual contact-point horizontal
velocity15.729462mm/s, mean absolute difference5.163442mm/s. This slice does not
support an inflated endpoint proxy as the explanation for failed slip reduction;
force-bearing contact points actually slide. Instantaneous and finite-difference
measures differ, and this is only a one-second diagnostic, not a walking pass.
Existing objective, body, neural dynamics and acceptance metrics remain unchanged.
Worker33980 exited0 after170.969s; OS inventory confirmed no matching processes
before terminal ledger read. Raw contacts/frames, CSV/motor/gait/sensory and
checkpoints copied with hashes to primary. Seventeen common-parent tests now
pass, including actual driver interrupted-birth replay and frozen separation
with controlled tiny checkpoints. Full-connectome preflight previously passed.
The registered six-birth common-parent learning pilot is the next physical run.
