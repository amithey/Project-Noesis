# AP1 review proposal: non-tarsal reaction penalty, no gate or learner change

PROPOSED, NOTIMPLEMENTED/ALLOCATED. Informed by H2B/C0; a new learning-feedback
hypothesis, not evidence that the existing learner is fixed. Codex new opt-in
curriculum wrapper/runner scope must be claimed before implementation. Claude
retains existing learning/reward file ownership; request review, do not edit those
files or infer active participation from this document. Gemini physical review.

For flat ground, use the calibrated world-force anatomical partition AFTER the
normal native body observation. Let F be total upward vertical ground reaction;
F_nonfoot is central-body(including fused head)+proximal-leg reaction. Define
q=0whenF<=1e-5, otherwise clip(F_nonfoot/F,0,1). Fixed candidate feedback:

    r_AP1 = clip(r_native_v7 - 0.5*q, -1, 1)

This is a correction AFTER native v7reward clipping, not an undocumented rewrite
of its internal formula. Lambda.5is a prespecified engineering hypothesis, not a
biologically calibrated constant. No sweep/retune after outcomes. Tarsal-only
states(q0) and unsupported states retain exact native reward; body/proximal support
never increases reward. Mixed support receives graded feedback, not a new
binary standing label. Existing v7height/up/contact/effort/slip terms, native
perturbation rule/rate.2/tau100ms/normalization/bias correction, exploration1.5,
50msbirth/25shorizon and original-reference.5bound unchanged. No extra sensory
afferents/RNG draws, external gait, joint-command policy or selected-weight merge.

Rationale: C0found tarsal-only states inside current height/up gates(383parent,
447child1005samples). Targeting those anatomical states avoids assuming that a
higher pose is automatically stable or that a height threshold should be relaxed.
This could still fail because of noisy/late credit, counterproductive clipping,
contact-force sensitivity to adhesion or new proximal/body tradeoffs. Actual
frozen outcomes, not shaped training reward, decide usefulness.

Proposed FIRSTtechnical/scientific pilot, trainseed1005chosen prospectively from
the diagnosed case, not claimed to represent unseen births:

1. Control25s birth from common9fdparent with new force observer but penalty
   disabled. Require exact SR1seed1005checkpoint46a6975dc7a1479f794034258abae67ff178d31772ac961d54617519a2b00853,
   every training/body/pad/exploration array and4990updates BEFORE treatment.
2. Candidate25s birth from the SAMEparent/seed, only lambda0->.5. Require complete
   horizon/logs and exact exploration streams/target/seed schedule to control,
   original topology/sign/reference/bound/sidecar checks. New objective identity
   must explicitly name AP1/base-v7/lambda/source in report/checkpoint sidecar;
   partial banks never become candidates.
3. Instrumented frozen parent/control-child/candidate-child at own1005and all
   4development seeds10006/10007/10010/10011. Reuse H2Bparent10007 and control-
   child1005/10007only after exact hash/context checks. Missing parent4+control-
   child4+candidate5frozen jobs, plus2births =15newjobs maximum. Existing raw
   v7child/parent traces gate every replay. No conditional extra probes/selection.

Report all cells: old standing/slip/heading/net/cycle/tripod retention AND C0mode
fractions/reaction shares/COMmotion. Primary anatomical question is whether
candidate improves tarsal-only support versus both parent and matched v7child
across EACHshared development seed, with reduced non-tarsal support. Also apply
unchanged original strict retention screen. Anatomical gains with old-score or
other capability losses are mixed evidence, not a pass; neither a contact-mode
pass nor old-mask pass certifies balance/walking. One birth is only a pilot;
require preregistered independent-birth confirmation before any model/method
qualification. Reserved final seeds10,11,200,201,202remain unused. No promotion.

Before allocation: full code/protocol tests including q-boundary/conservation/
unknown-anatomy handling, scalar-only change, exact disabled control replay,
read-only force observation, exact reward accounting and complete sidecar identity.
Record fresh budget(e.g.3000s aggregate600s train300s probe,15jobs) and canonical
lease/Windows containment/resource checks. Stop technical/hash/context/horizon/
mass/force/RNG/mismatch/timeout/remaining-budget failure; no retry/retune/partial
interpretation. A scientific failure completes the declared bank and is published.
These are proposal constraints, not a current heavy allocation or instruction to
replace validated-best/defaults. Full project goal remains unchanged.
