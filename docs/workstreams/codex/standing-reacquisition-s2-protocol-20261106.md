# S2: paired standing reacquisition after pad-mapping intervention

Prospective protocol, 2026-11-06. Codex issue #18 owns these new scripts/tests/docs;
Claude retains learner/reward ownership, Gemini retains independent physical review.
The user's resumed goal supersedes the historical November 1 pause. No automatic merge.

FK1 found that positive Femur commands lower the foot in sampled static poses,
while the legacy pad mapping releases adhesion on positive Femur commands. A1
changed only that mapping; it did not improve the strict frozen retention screen.
S2 tests whether the unchanged learner can reacquire standing under each mapping.
This is a hypothesis, not evidence that the mapping caused prior learning failure.

Base branch: codex/adhesion-sign-a1-20261106, commit
447c8f876cbc226514db1d49577c01b22ce72ae3. New branch:
codex/standing-reacquisition-s2-20261106. Native runtime e0cbd109bb6e018bd253c9f1b5e1698e0421b276,
core 42c20769376bda5694b4c1c66ca75e52cbd9f1f4: all 20 pilot and 40 core Python
files remain unchanged, checked against the TP2 byte manifest. The inherited
adhesion_mapping.py is unchanged. No shared peer modules or default settings edited.

Common completed experimental parent SHA-256:
9fd7d41177bf2c745809401c3b142a832661d1a18d0b117b30cb02aea63fea3c.
It is not validated-best. Legacy releases for Femur delta > .08; corrected
releases for delta < -.08. Pads depend solely on neural desired angles. No
external timing, leg order, gait generator, muscle sign or joint-command change.

1. Two full 5 s parent bridges, seed10006 in each mapping, must reproduce every
   native observation and pad-observation array exactly against A1 before training.
2. Two separate full 25 s standing-v7 births, both seed1004 and the same parent:
   5000 control chunks, 4990 learning updates after the unchanged 50 ms birth.
   Native learning rate .2, eligibility100 ms, exploration1.5 mV/ms, normalized
   bias-corrected perturbation rule and original-reference magnitude bound .5.
   Preserve topology/signs/nonplastic edges; audit each completed checkpoint.
3. Both complete exploration logs (drive, credit, indices, time) must match
   exactly. Neural and physical trajectories may differ. Final checkpoint
   sidecars must match the enriched report and explicitly label the body policy.
4. Eight full 5 s child frozen probes:10006,10007,10010,10011 in each mapping,
   each compared with its own A1 parent under that same mapping. These are reused
   development seeds. Final seeds10,11,200,201,202 remain unused.

Standing-stage screen: EACH child seed standing >=90% and >= its own parent;
stance slip <= its own parent; median paired heading/net displacement/leg-cycle
gains and median paired tripod-score gain >=0. This does not prove walking or
generalization. One birth per mapping is one training pair, not four independent
training replications. Training reward/standing is not frozen retained ability.
No conditional extra probes, candidate merging, model promotion or gate relaxation.

Budget:2400 s aggregate from canonical lease acquisition,600 s per training
worker,300 s per frozen worker;12 jobs maximum. One atomic canonical host lease,
Windows Job Object per worker, fresh no-competing-Python and >=2 GiB available RAM
checks before each job. Check remaining full worker budget after preflight.
Stop on timeout, incomplete horizon, hash/source/data/context/log/invariant failure,
resource failure or insufficient remaining budget. No automatic retry, retune,
duration extension or partial-checkpoint probes. Preserve partial evidence with
explicit context; never call it completed or resume mid-episode.

Run in the prepared portable layout:
`work/noesis-env/Scripts/python.exe work/run_standing_s2.py`.
Checks:14 focused tests passed (0.09 s) and workers compile. The runner records
public protocol, source/data/parent hashes, process receipts, bridge/stream gates,
child checkpoint identities, paired results and lease cleanup. Keep raw data and
connectome caches out of Git. Publish compact results in Git and a signed artifact
manifest; Release upload must be reported pending until actually authenticated.
