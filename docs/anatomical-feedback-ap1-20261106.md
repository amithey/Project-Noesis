# AP1: prospective anatomical feedback pilot

Codex owns the new opt-in scripts only, claimed in issue18 comment6691386897,
branch `codex/anatomical-feedback-ap1-20261106`, base
`f5b6a5b7d71fbcb5d934436e36d5a60e9ec35640`. Existing Claude learning/reward,
Gemini evaluation, native body/controller/core and acceptance gates are unchanged.
Peer review has been requested; no peer response or approval is inferred.

H2B/C0 found central-body support inside the legacy standing mask and tarsal-only
support both inside and outside it. This motivates a feedback hypothesis, not
proof of balance or an excuse to relax the original height threshold.

Let q=0 when total upward ground reaction <=1e-5; otherwise q is the fraction
carried by central body plus proximal leg segments (coxa/trochanter/femur/tibia).
All tarsal segments are excluded from the penalty. Candidate reward is
`clip(native_v7_reward - 0.5*q,-1,1)` AFTER native clipping. Lambda .5 is fixed
before outcomes, an engineering hypothesis without biological calibration.
The control returns the native score object unchanged. Candidate changes only
its scalar reward field; no leg sequence, action override, extra afferent or RNG.

The complete immutable protocol is `anatomical-feedback-ap1-20261106-protocol.json`.
Parent9fd7d41177bf2c745809401c3b142a832661d1a18d0b117b30cb02aea63fea3c,
training seed1005, legacy pad mapping, v7 base, 25s/4990updates, rate.2,
eligibility100ms, exploration1.5, original-reference bound.5. Candidate actual
objective ID explicitly names AP1 in final report and checkpoint sidecar; a
prewritten context file declares the correction before native episode creation.
Weights-only continuation is not exact mid-episode state restoration.

Fixed schedule, at most15new jobs under ONE canonical host lease:

1. Disabled observer-control25s birth. Require EXACT SR1seed1005checkpoint
   46a6975dc7a1479f794034258abae67ff178d31772ac961d54617519a2b00853 and every
   training-observation/pad/exploration array before proceeding.
2. Parent frozen5s at1005/10006/10010/10011; reuse signed H2Bparent10007.
3. Existing v7child frozen5s at same four seeds; reuse H2Bchild1005/10007.
   Require every new baseline body/pad array and spikes exact to the old trace.
4. Candidate25s birth, same parent/seed. Require unchanged drive/target/current
   schedule, exact initial11body/pad rows before learning affects the next step,
   full horizon, scalar reward accounting and original checkpoint invariants.
5. Candidate frozen5s at all1005/10006/10007/10010/10011. No probe-time learning,
   motor exploration or reward shaping. All cells reported, no adaptive extras.

Aggregate3000s; contained worker600s per birth/300s per probe. Fresh RAM>=2GiB
and no competing Python before every worker; Windows JobObject exit/tree cleanup
receipt, immutable data/source/prior/checkpoint hashes and remaining full-worker
budget required. Technical/hash/anatomy/force/context/horizon/parity/time/resource
failure stops the bank; no automatic retry, retuning or partial scientific verdict.
A scientific failure still completes the fixed bank.

The force observer uses corrected whole-compiled-subtree mass/COM and root-force
closure from H2B. C0 classifies FORCE-BEARING bodies only; unloaded eyes/antennae
still contribute to mass, and loaded unknown anatomy is rejected. During code
preparation a zero-step test caught eager classification of the unloaded eye;
this was corrected before allocation. Native physics/state/sensory inputs are
unchanged at zero time; the full disabled replay is the decisive dynamic control.

Report old standing/slip/heading/net/cycles/tripod strict retention versus parent
and v7child, plus identical C0contact fractions/reaction shares/COM motion. The
primary anatomical hypothesis requires increased tarsal-only sample fraction
AND reduced mean non-tarsal reaction fraction on EACH development seed versus
BOTH references. Native frozen warmup is retained: 990measured chunks, excluding
initial state and first10chunks. Own seed1005 is descriptive, not held-out.
Mixed anatomical gains/capability losses are not a pass. One birth is only a
pilot; independent birth confirmation is required before method qualification.
Contact modes, COM motion and leg cycles do not certify stable balance/walking.
Reserved final seeds10/11/200/201/202 are unused. No automatic promotion/merge.

Before allocation:23focused tests passed (AP1scalar/objective/observer/warmup,
C0sixknown-role static fixtures and invariants, unchanged S2contracts); compile
and prepare-only signed-source/input/reused-bridge checks passed. Static fixtures
are measurement controls, not learned-fly successes. No AP1simulation result yet.

Artifacts stay local with SHA256 manifests pending Release upload authentication;
do not claim that local signed evidence has already been uploaded. Raw connectome
tables, derived graph caches, environments and private lease metadata stay out of Git.
