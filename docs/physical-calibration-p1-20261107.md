# P1 prospective implementation: physical pulse with exact neural baselines

New opt-in Codex files only, branchcodex/physical-calibration-p1-20261107,
base0ad50ddeb9779a50d86d85c376d2e7edf4109df0, issue18claim6709766137 before
implementation. Existing core/native/controller/learning/reward/evaluator/C1/
Claude/Gemini files unchanged. Current coordination and comments read; no peer
response, activity or review approval inferred. This is calibration, not training.

AP2's0/3verdicts stay unchanged. FA2found that old height misses may include
tarsal-only states while old positives may include central/proximal support.
P1asks how two frozen neural-controlled bodies respond to a fixed environmental
input, relative to their own baseline. No new acceptance/recovery gate or ranking.
Neither mechanical return nor changed motor commands qualifies standing/gait.

Models: common parent9fd versus AP2candidate1004CP068b49a0d44f4dca6fd53bd0950048a4db1a3ebd891ac0e0727ff0a39ff33c10.
One known diagnostic seed10011,5s/1000control chunks,50physics steps/chunk,
dt.0001,zero learning or motor exploration. Every joint/pad command is still
computed by the existing neural controller. No altered afferent adapter, desired
leg sequence, action override, weight merge or topology/sign/reference change.

FOURworkers, in order:parent-baseline,candidate-baseline,parent-pulse,candidate-
pulse. Both new baseline fullbody/pad/spikes and all prior anatomy fields must
EXACTLY reproduce the signed existing5s traces before any pulse interpretation.
No data-selected checkpoint upgrade. Full checkpoint/source/input hashes guarded.

Pulse:world+Xforce at compiled thorax COM, .1times whole-fly weight, active ONLY
in control chunks393..396(20ms), starting modeltime1.965s. Selected diagnostically
from existing candidate high/upright/tarsal-only/>=3loaded-claw run after1s;
the selected moving4claw state is NOTa positive standing/balance reference. All
old/finalheldout seeds/gates and AP2scores stay unchanged. No parameter search.

Source inspection confirms native actuator/pad setters write `ctrl` only and
Simulation.step calls installed `mj_step` without clearing `xfrc_applied`. The
P1wrapper sets that external-force field, then verifies its ENTIREarray immediately
before and after EACHof50000native physics steps. It records actual input vectors
and pre-step times; exactly200microsteps must have the fixed positive force, all
other components/steps zero. Impulse is computed from verified inputs and dt.

Force epoch matters: native observe calls `mj_forward`. Each endpoint's force
field is the input held during the PRECEDINGcontrol chunk, until the next
advance updates it. At the right pulse endpoint, the last integrated force is
still recorded; no subsequent physics step uses it after the declared chunk
window. This convention is explicit in micro/endpoint telemetry and tests.

Telemetry:1001read-only snapshots(fullqpos/qvel/qacc/COM/body poses/forces), actual
ground contact locations/wrenches, all native observations/commands/pads, and
50000force/time inputs. Correct compiled mass/fused-head layout reused from H2.
Separate checks: ground forces close root TRANSLATIONALCONSTRAINTforce; the
external pulse is not added to that equality. Installed `mj_applyFT` maps
Cartesian COMforce to generalized coordinates. With dense mass from installed
`mj_fullM`, verify:

    smooth = passive - bias + actuator + qfrc_applied + mapped_xfrc
    M*qacc = smooth + qfrc_constraint

Checks are read-only after native forward, relative tolerance1e-9plus1e-10global
scale and independent component comparisons. State/time/sensory/control/force
arrays must remain exact across measurement. Each source/seed/CP/body context and
full horizon/readonly weights is checked. Prefix requirements versus own baseline:
393native body rows,394initial+post-step telemetry/anatomy rows,394neural
command/pad rows. Commands for chunk393are computed before the pulse acts.

Report actual force/impulse, COM/velocity/up/contact deviations, late own-baseline
deviation, command/adhesion changes and their timing. A command change is separate
from passive mechanical return and is not a gait/retention qualification. One
known context cannot establish general neural feedback or biological gust response.

Budget1200s aggregate/300s perworker,max4jobs. FreshRAM>=2GiB/no competingPython,
atomic canonical lease, Windows JobObject containment/exit/tree receipts and full
remaining-worker budget before each. Stop hash/context/mass/force/horizon/prefix/
resource/timeout failures, no automatic retry/retune or partial interpretation.
Parents and every old result/partial remain immutable; no model/default promotion.

Preparation:41focused tests passed, including3new P1tests for exact schedule,
installed force/mass mapping, one primitive impulse step and zero-time real-native
observer/input preservation. The primitive fixture step is a synthetic technical
control, NOTlearned-fly success. Compile/prepare-only signed-input/selected-case
checks passed; source and compact protocol published before allocation.

Only compact scientific metadata/code are public. Extensive private integrity
maps remain local after prior automatic publication rejection. Raw connectome,
cache,environment,private lease excluded from Git; Release upload still pending.
