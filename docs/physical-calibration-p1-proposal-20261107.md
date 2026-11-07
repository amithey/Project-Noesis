# P1 proposal: matched frozen physical-disturbance calibration

PROPOSED, NOT IMPLEMENTED/ALLOCATED. AP2's0/3original verdicts remain unchanged.
FA2shows both body-supported old positives and high tarsal-supported old misses;
neither descriptor certifies stability. Do not select a new training objective or
relax the old gate before testing the actual physical behavior independently.

Diagnostic case (not representative validation): common parent9fd versus AP2
candidate1004checkpoint068b49a0d44f4dca6fd53bd0950048a4db1a3ebd891ac0e0727ff0a39ff33c10,
probe seed10011,one CPUworld5s, frozen weights/no learning/no exploration, all
motor commands still from the original neural controller. Existing raw baseline
traces and private integrity signatures are available for both models.

Four prospective workers: parent-unperturbed, candidate-unperturbed, parent-
pulse, candidate-pulse. Unperturbed full observations/pads/spikes must EXACTLY
reproduce the signed prior traces before pulse results are interpreted. New
read-only full qpos/qvel/COM/root/contact/command telemetry must not affect the
native inputs. Retain actual contact locations/forces and full compiled mass;
do not substitute anatomical-name aliases or the old force-norm proxy.

Fixed environmental pulse, not a command/gait intervention: world+Xforce on the
compiled thorax body, amplitude.1times whole-fly weight, duration20ms(4control
chunks), starting controlchunk393/modeltime1.965s. Set and clear the external
force only at those declared boundaries; assert zero elsewhere. Selection of
time came from the first known high/upright/tarsal-only>=3claw run>=50ms after
1s in the existing candidate trace, not from responses to a pulse. That state
has COMvz3.4532mm/sand is explicitly NOTassumed to be standing or balanced.

Amplitude is an engineering calibration constant, not a biological gust estimate.
With current mass.00102431andgravity9810nativeunits, weight10.0484811,
force1.00484811nativeunits; free-body20msvelocity increment would be19.62mm/s.
This calculation does not predict the constrained/adhesive fly response. No
amplitude/duration/timing search after outcomes; a weak or uninformative probe
is reported as such, not a stability pass or an automatic retune.

Require exact neural-command/body/telemetry prefix before the force starts, same
CP/seed/source/body/sensory/servo/pad/RNG settings and full5s/1000chunks/zero
weight changes. There is no new afferent adapter or external leg sequence. The
world pulse acts through physics and existing body senses. Measure responses
relative to each model's OWNmatched unperturbed trajectory, not just absolute
differences between two different starting behaviors.

Report force actually applied/impulse, full COMpose/velocity deviations, upright-
and support-mode changes, contact topology, root force closure including external
terms, command/adhesion changes and late deviation from the matched baseline.
Do not infer neural feedback from mechanical recovery alone; command changes
are separately measured. No new binary recovery/standing/gait acceptance gate,
no ranking/adoption/promotion or claim that one known context generalizes. This
tests physical sensitivity and measurement interpretation, not retained learning
or external vision/chemical sensing, which remain separate outstanding work.

Proposed budget1200s aggregate/300s perworker, max4full probes, zero training.
FreshRAM>=2GiB/no competing Python, atomic canonical host lease, Windows
containment, immutable source/input/CP/force-context checks before every worker.
Stop hash/context/horizon/prefix/mass/force/resource/timeout/remaining-worker-
budget failure, no automatic retry/retune/partial interpretation. No reserved
final seeds used. Source/tests/compact protocol publication and issue18ownership
claim precede any allocation. Existing Claude/Gemini/controller/core files stay
untouched; request independent physical review without inferring approval.
