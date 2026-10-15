# Coxa direction audit and corrected-body training progress

The goal remains unverified. The active v8 protocol uses the corrected proximal
flexion/extension interface and is now measuring its frozen original controls.
Its source remains fixed at `e781c5f9e9c00d318594f939922679b36f75897b3d53cdb574aaea9638957a66`.

## Completed training, not acceptance

Training seed 2 completed ten simulated seconds with exploration active. Standing
fraction was 0.977387; mean supporting feet 2.839196; all six legs had repeated
physical cycles (LF=30, LM=30, LH=34, RF=25, RM=28, RH=26). Heading-relative forward
motion was only 0.272940 mm and mean stance slip 12.337649 mm/s. These measurements
show substantially more support than the preceding training episode, but they
are different seeds/body versions with exploration and do not establish a causal
learning gain, frozen walking, or acceptance. All completed weights and traces remain
in `outputs/birth-cycle-v8/cycle_000/train-0/attempt_0000`.

Slip remains high even without new foot contacts. New-contact frames account for
30.8% of frames and 31.6% of the slip total: mean 12.640627 mm/s on touchdown frames
versus 12.202772 otherwise. Touchdown transitions do not explain away this failure.
The slip reward and 20% improvement criterion are unchanged.

## Additional physical direction error

At the model's standing pose, small activation of sternal anterior rotator and
tergopleural promotor moves the distal coxa backward along the thorax's forward
axis in all six legs. Sternal posterior rotator and pleural remotor/abductor move
it forward. This conflicts with their annotated anterior/protraction and
posterior/retraction functions in the
[primary anatomical appendix, Table A1](https://faculty.washington.edu/tuthill/docs/azevedo24_appendix.pdf).
The proposed change reverses ONLY the `Coxa` component of these four groups.
All existing roll/yaw couplings and active degrees of freedom are retained.
It preserves graph topology, neural weight signs/bounds, learning, sensors, adhesion,
reward and acceptance requirements. The motor-interface tag becomes
`anatomical-coxa-flexion-v3` so normal loading cannot silently mix body contexts.

For the tarsus, unsigned interior angle alone is insufficient to identify its
functional direction at the rest pose. Direct physical foot displacement confirms
that the present depressor lowers the foot and levator raises it, in all six legs.
Those signs remain unchanged. The long tendon controls a claw in the anatomical
model ([primary motor-circuit paper](https://www.biorxiv.org/content/10.1101/2022.12.15.520299v2.full));
its existing single-joint proxy and the other multi-axis couplings remain modeling
assumptions, not newly validated anatomy. No new claw model or coupling is introduced.

## Tested, reviewable candidate

The new regression activates each named pool, evaluates both motor matrix APIs and
the muscle model, and measures physical coxa displacement in the thorax frame.
Before correction, four coxa cases fail while both tarsal direction controls pass.
The separately corrected snapshot passes **203 tests** and Ruff, including the
preceding proximal flexion, bilateral reflection, reward, sensing, checkpoint and
resumable training tests. Its source fingerprint is
`81e68f7945a434fb994097fc33709d0f3a9ad9822f4823c30595823ce04f2913`.
These are interface tests, not behavioral evidence.

The portable patch is `docs/patches/coxa-direction.patch`; `git apply --check
--ignore-space-change` passed. It includes only two source files and the six new
physical regression cases. The exact tested source is preserved separately in
`outputs/protocol-v9-candidate/src/noesis`, and current v8 source is archived in
`outputs/protocol-v8-source/src/noesis`. The patch is not applied to the live run.
After the active bounded protocol is terminal, apply the tested correction,
explicitly import the last complete training weights with reset traces, and use a
new output directory. Previous-body controls cannot validate the new interface.
Original and candidate frozen tests remain five seconds, with the same reserved
validation and heldout seeds and all existing promotion gates.

Machine-readable evidence: `docs/coxa-direction-audit.json`. Full regression log,
physical audit script and raw diagnostics: `outputs/anatomical-audit-v8`.
