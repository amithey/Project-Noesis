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


## Launch observed

Worker session44661 starts the registered3600-second cycle under source
6eb9405de05f11a4741ab269cabf0b8fafe1d667a978700f5a97c0529843909d.
Producer stdout confirms158262entries,556motor mappings, learning=True,
seed74 and physical t=0.00seconds. The launcher/interpreter process pair was
verified in the OS inventory. No completed episode or new behavior result is
claimed from this startup observation. The live ledger remains unopened.


## First complete physical training birth (seed74)

The first birth completed5seconds (990awake samples) with learning and motor
exploration active. Standing89.191919%, stance slip15.846478mm/s, projected
heading advance10.451245mm, leg cycles16/17/16/9/13/4; tripod score unavailable.
These are training observations, not a matched frozen comparison or validated
walking. Standing remains below90%, and all required frozen controls are pending.

Independent immutable-artifact audit passes topology, signs and±50%bounds,
actual expected parent, full horizon, body/CSV metrics and explicit weighted
reward-moment reconstruction. Scalar reward max error1.11e-16;36118selected
edges differ from the previous bank, max original-weight deviation22.533541%.
The first awake prediction error and learning signal are exactly0, as designed.
The observed unsaturated fraction is0.855000over the first
awake second and0.854545over the full awake window. This single
training seed does not establish behavioral improvement. Its complete checkpoint,
CSV, neural spike artifacts and MP4 were copied and hash-verified in the user
checkout, outputs/young-fly-v23-progress/cycle_000/train-0/attempt_0000.
Worker session44661 continues the next birth (seed75), learning=True. No live
ledger was opened, best validated remains unset and normal startup unchanged.


## Second complete training birth and physical limitation

Seed75 completed5seconds with no terminal fall: standing93.434343%, slip
17.354199mm/s, projected heading advance5.614072mm; cycles15/20/15/10/12/4.
Independent checkpoint/CSV/weighted-reward audits pass.35430selected edges
changed from the preceding bank; max original deviation23.336195%, signs and
bounds intact. Checkpoint/CSV/motor artifacts and the full MP4 are copied with
hash checks. This passes neither a matched frozen comparison nor the slip gate;
seed74/75 differences cannot be attributed to learning because seeds differ
and both have active motor exploration.

Supplementary analysis uses only the two immutable audited copies. Seed74/75
horizontal paths66.369/65.674mm have net displacements10.313/2.621mm. Projected
heading changes during adjacent valid-posture intervals average346.520/340.337
degrees per second (descriptive heading, not independent rigid-body rotation).
Posture-conditional slip15.742486/17.158546mm/s remains high; falling alone does
not explain the all-window slip. Right-hind ground-load fractions10.30%/4.75%
and right-front20.91%/31.41% contrast with greater loading of other legs. These
are observations, not evidence of an anatomical error or causal mechanism.
Both tibia-flexor phase estimates are invalid, so tripod_score stays null.
Spectral significance against a stationary Poisson null alone is not gait.

The running source/reward/gates are unchanged. Worker44661 is confirmed in
seed76; no live ledger read or duplicate worker started. Two of six training
births are audited; all25frozen cases remain pending. The existing experiment
continues before choosing another change. Detailed immutable-file hashes and
metric definitions accompany completed-motion-diagnostics.json.


## Third training birth: sustained-fall failure retained

Seed76 ended with training_fall at4.895body seconds (969awake samples),
standing86.996904% and stance slip17.495973mm/s. Cycles12/16/14/8/15/5 and
projected heading advance7.828961mm do not establish walking. The independent
artifact auditor confirms the first sustained0.1-second fall triggers exactly
at the final awake sample, with negative terminal reward. This is a completed
training attempt, not a completed5-second behavioral test.

The complete failure MP4/CSV/neural artifacts and checkpoint are copied with
hash checks. Checkpoint SHA25629efb902f4040bbaf41a4d01108edcc815cd60bde505d8b07413b67b315e6339.
Topology, original references, signs and bounds remain intact;36216selected
edges change from the preceding bank, max original deviation23.734955%.
Reward reconstruction max error1.11e-16, with normalization state independently
recomputed using weighted accumulators. Experiences are retained for subsequent
training; this candidate is not promoted or described as validated.

Three of six births now contribute14.895body seconds, two full windows and one
terminal fall. All25frozen cases remain pending. Producer stdout confirms the
same live session44661 has begun seed77, learning=True. Neither the live ledger
nor running-episode files were read. All original physical/sensory/heldout gates
remain unchanged and the user goal stays unachieved.


## Fourth training birth complete

Seed77 completed5seconds: standing92.424242%, stance slip16.309770mm/s,
mean reward0.550237, projected heading advance8.808548mm and cycles17/13/15/9/11/3.
The experiment remains learning=True with motor exploration1.5; these are not
frozen controls or evidence of matched improvement. Tripod score stays null.

Independent immutable-artifact audit passes full horizon/no sustained fall,
all CSV/body metrics, reward/normalization reconstruction, original topology,
signs and bounds.35320selected edges differ from the actual previous bank;
max original-weight deviation23.062409% (weight deviation is not a progress
criterion). Checkpoint SHA256c6b112074aad01e7a213862bd050669289fcb6c65c456de4d51e2a7236d826e3.
Complete artifacts including MP4 are copied and hash-checked in the user
checkout. Four births now total19.895body seconds: three full windows and one
terminal fall. All25frozen controls remain pending, and no model is promoted.

Producer stdout confirms the same worker44661 has started seed78, learning=True.
Source, reward, controls and gates are unchanged. No live ledger was opened.
