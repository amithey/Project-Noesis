# FK1/A1 decision and next learning hypothesis — 2026-11-06

Full user goal remains active and unachieved. No stable walking, balance-recovery validation, retained-learning bank, model/stage promotion or shared source-pin change has been established.

## Closed questions and measured limitations

FK1 source96cc271 used only forward kinematics: all126sampled leg/configuration cases lower the observed foot origin for positiveFemur, the old release direction. The neutral negative direction raises the foot while old adhesion stays1. Analytic Jacobian/two-scale/translation/cross-leg checks pass, zero model time/neural steps, no source/parent changes. Core flexor signs agree with apparent neutral bending directions; the finding concerns the pad interface, not a justified synapse/muscle sign flip. Root/passive-neutral/partial-derivative limits remain explicit.

Upstream pinned FlyGym source independently confirms pad0 is released,1 applies adhesion gain, and native leg ordering is LF/LM/LH/RF/RM/RH. The convention observation is not an accidental inversion of that API's values/order.

A1 sourcedd61cd9 isolated ONLY desiredFemur pad sign, same.08rad magnitude and learned parent9fd7d411/nativee0cbd10/core42c2076. Eightfull5s jobs in498.969s; legacy arrays exactly replay ES1parent on all4reused development seeds, all1000pad formulas and one-factor context gates pass. Ordinary compare rejects differing body.adhesion_mapping. Every worker exits0 with entire job empty; all source/data/parent hashes unchanged and canonical lease released.

Corrected standing92.6263/97.0707/94.3434/89.0909% versus legacy96.2626/94.4444/89.2929/91.9192%. Two improve/two regress; each passes absolute standing3/4, with different failing seed. Slip improves3/4 but worsens10010. Paired median heading-.225470mm/net-forward-.169712mm and tripod median lower; cycle median retained. Corrected regimes all STEPPING_IN_PLACE. The registered physical improvement screen FAILS and stays failed. No fresh/final probes, adoption, training or weight update. The sign-consistent helper is opt-in diagnostic code only.

Thus static interface consistency and performance of a bank learned under the old interface are separate. Do not claim this sign explains all earlier learning failures, that fixing it alone restores gait, or that its frozen mixed outcome erases the measured geometry. No threshold/gain tuning follows from this result.

## Proposed next learning experiment S2 — not allocated

Test the specific hypothesis that the geometry-consistent interface changes reacquisition/retention under the existing learner. This returns to real fly training with a matched control, rather than generalizing CL1's tiny-plant firing floor to Noesis (real ES1weights already changed frozen behavior).

Prospective design to finalize before implementation/execution: same completed parent9fd7d411, same development training seed1004, two independent full25s births at standing-v7 under legacy versus sign-consistent mapping. Preserve nativeLR.2, exploration1.5mV/ms, eligibility100ms, bounds/signs/topology/reward/neural code; only pad mapping differs. No merging children. Each child is evaluated against its OWN mapping-matched frozen parent on reused dev10006/10007/10010/10011; A1 supplies those parent observations. A parameterized frozen worker must first exactly replay BOTH A1modes on10006 as a bridge before spending the training budget.

A prospective standing/retention-stage gate should require all four frozen probes standing>=.90 AND>=own parent, slip<=own parent; median heading/net/cycles/tripod not reduced. It does not replace the broader walking/balance/environment/retained-learning goal, and passing one birth does not advance or validate the model. Report the legacy child under the same gate and compare learning effects within each body; do not silently compare different body policies as matched controls. If both fail, preserve failures; do not extend duration or change rate until a new hypothesis is registered.

Indicative budget2400s aggregate,600s per full25s training worker and300s per5s frozen worker under one canonical lease,120s estimated bridge cost. Require sufficient remaining budget for each job, fresh resources, hard descendant containment, complete5000chunks/4990updates and original-reference checkpoint audits. Partial births never become training-last or frozen candidates. No heavy allocation follows from this proposal: finalize exact code, budget/stop rules, ownership, source/checkpoint context and screen before launch.

Codex owns new orchestration/metadata wrappers, not Claude's learning rule. Claude retains credit/retention mechanism implementation, Gemini retains independent sensory/anatomical/metrics review. No peer branch edits or presumed live acknowledgment. Native shared runtime stays unchanged; any future production opt-in needs complete body-policy/source metadata and independent qualification.

## Evidence

PR33 contains FK1; PR34 contains A1 source/protocol/raw compact results and audit. FK1four-file archive41884bytes SHA2569ec25a9dc94fee22445d294db6aac4d4f21f996c9c1f0062f5b45a70f3d2c904. A1thirty-eight-file archive11330873bytes SHA256cee924bff17e1a72a2cb379c8f44e51aa5fab06fcaff6015a71f9df5ef9f92bf. All CRC/every manifest hash verified; named body/checkpoint evidence only, no licensed raw graph/cache/private lease/assets. Archives remain local; Release upload is pending the previously verified invalid CLI authentication/logged-out browser. Compact evidence is already public.
