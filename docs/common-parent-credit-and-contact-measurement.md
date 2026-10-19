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


## Common-parent worker live; immutable-case audit prepared

Worker46529 is actually training86, learningTrue, observed through2.4body
seconds. All six branches use the same original V23parent. No committed birth
yet and no V26frozen outcome. Source/plan/driver remain fixed during the batch.
The added preserve_completed_birth.py accepts only stdout-committed summaries,
then checks source/context, parent identity, topology/signs/weight bounds, CSV
body measures, independently reconstructed v5reward and corrected RPE/EMA, and
first sustained-fall timing. It preserves child artifacts separately from an
as-yet-unformed mean bank; actual ledger start pointers remain a terminal audit.
Ruff and compile pass; real-case execution awaits a committed birth.

Hash-verified previous V24training trajectories show lower reward clipping in
3.33–15.95% of awake samples. Clipping exists but does not dominate all learning
frames. Negative reward in valid posture is possible during genuine slip and
is not itself evidence of wrong credit. No reward/gate/dynamics changes follow
from this retrospective diagnostic, and it does not describe the active birth.


## Contact-point distribution limits the proxy-artifact hypothesis

The stopped one-second original10 diagnostic has611awake Tarsus5contact
records and only one Tarsus3 and one Tarsus4record. Instantaneous contact slip
median10.816648mm/s, mean15.729462mm/s;189/190awake frames exceed1mm/s.
Thus proximal-contact / moving-distal-endpoint geometry is not a dominant
explanation in this measured slice. This does not prove all longer trajectories
or seeds equivalent. The diagnostic plot shows different instantaneous and5ms
finite-difference measurements; no metric or reward is replaced. Actual control
prefix and checkpoint arrays remain exact. PNG and distribution JSON are copied
to the primary checkout. The common-parent worker continues unchanged.


## First independent birth: failure retained, no mean bank yet

Seed86 ended with sustained training fall at4.595s/909awake samples;
standing91.969197% only in this shortened window, slip15.791781mm/s, mean
reward.185450699, heading advance5.807274mm, support/lift counts12/13/14/6/10/4.
The counts and advance do not constitute walking after a terminal fall.
The immutable-case audit passes original topology/motor IDs/references/signs
and +/-50%bounds (max original deviation22.671389%), actual v5formula
(max error2.22e-16), independently reconstructed bias-corrected learning credit,
CSV/body measures and first sustained-fall termination exactly at the final
awake sample. Raw checkpoint, motor/gait, CSV and sensory artifacts are copied
with hashes to young-fly-v26-progress. Actual attempt start-pointer ledger
audit remains pending until worker termination; live ledger was not opened.
The branch is retained and will enter the equal-weight mean including failures.
No mean/best validated bank exists. Worker46529 has begun independent87 from
the same registered V23parent, observed learningTrue through.2s. Core source,
plan, reward and physical gates remain unchanged during the batch.


## Second independent birth: full horizon, slip remains high

Seed87 completed5s/990awake samples, standing93.737374%, stance slip
14.710625mm/s, mean v5reward.236853803, heading advance10.270391mm and
cycles16/15/11/8/14/5. This is exploration-on training, not frozen walking.
The raw case audit passes parent/model context, exact original topology/motor
IDs/references, signs/bounds (max original deviation22.408073%), CSV body/gait
measurements, independently reconstructed v5reward (max error2.22e-16) and
normalized bias-corrected learning moments. No sustained fall occurred. All
raw artifacts are copied/hash-verified to young-fly-v26-progress. The ledger
start-pointer audit remains pending until the worker stops.

Two independent births now total9.595body seconds: one full and one fall.
Neither lower training slip across different seeds nor a shortened fall window
is treated as a learned improvement. The branch remains separate from an
unformed mean and best validatednull. Worker46529 has started88, learningTrue.
The long1480.609s wall time for87includes the previously observed host/tool
pause; no additional simulated progress is inferred from elapsed time.


## Third independent birth: early fall retained

Seed88 ended at2.040s/398awake samples with training_fall, standing91.708543%
in that shortened window, slip14.096214mm/s and mean reward.271306211.
Heading advance1.689936mm and cycles4/7/5/2/2/3 do not convert a terminal fall
into walking success. Lower slip and higher average reward in the shortened
window are not treated as an improvement over full-horizon trajectories.
The actual immutable-case audit passes scalar reward/corrected credit moments
(max scalar error2.22e-16), exact first sustained-fall terminal timing, body/CSV
measures, source/model/parent, topology/signs/original references and bounds
(max original deviation23.378739%). All raw child artifacts are copied with
hashes. Each independent branch remains retained, no mean or best bank yet.
Three completed births total11.635body seconds, one full and two falls.
The same worker46529 has begun89, learningTrue, observed through.2s. No live
ledger read, source/plan/reward/gate change or model promotion occurs.


## Fourth independent birth: another terminal fall

Seed89 ended at1.855s/361awake samples with training_fall, standing80.886427%,
slip11.598113mm/s, reward.145610452, heading advance1.251662mm and cycles
3/5/6/2/4/3. These short-window values do not establish walking or an improvement
over full trials. The independent artifact audit passes original topology/motor
IDs/references/signs/bounds (max deviation22.797694%), model/common-parent
context, CSV/body measures, v5scalar/corrected learning moments (max scalar
error2.22e-16), and first sustained fall exactly at the last awake sample.
All raw child artifacts are copied with hashes. Four completed births now
total13.490body seconds: one full and three falls, all retained for the planned
equal-weight mean. No mean/best validated bank exists yet. The same worker46529
has begun90 with learningTrue, observed through.2s. No live ledger read, reward,
source, physical-gate or normal-startup change occurs.
