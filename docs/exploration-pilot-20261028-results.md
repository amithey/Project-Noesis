# Exploration pilot results — 2026-10-28

Status: both training births and all six frozen probes completed successfully.
Protocol: [exploration-pilot-20261028.md](exploration-pilot-20261028.md).
Source used for the run: `0398b0346a8b0e0e701e85228fb419592cb0a19c`.
Core remains `41b5fe6534ed4c8f0c21a2e123c8d518b3bf5953`.

Both births completed 990 learning updates from the same provisional parent,
training seed 2100, v6, five seconds. Parent topology, original reference weights,
selected targets, signs and ±50% bounds passed a direct checkpoint audit for both
children. Neither child replaces validated-best. The parent remains unchanged.

Frozen outcomes (corrected horizontal displacement; 50ms warmup excluded):

| Bank | Probe seed | Standing | Forward mm |
| --- | ---: | ---: | ---: |
| Parent | 10008 | 96.06% | 0.218 |
| Parent | 10009 | 95.05% | 0.445 |
| Control, exploration 1.5 | 10008 | 85.45% | -0.576 |
| Control, exploration 1.5 | 10009 | 86.36% | 1.380 |
| Lower exploration 1.0 | 10008 | 88.89% | 1.487 |
| Lower exploration 1.0 | 10009 | 92.22% | 0.752 |

The lower-exploration arm fails the preregistered requirement of >=90% standing
on both probes, and regresses progress against control on seed10009. It improves
standing and slip against control on both probes and moves forward on both, but
sacrifices standing compared with the parent. No curriculum or model promotion.
With one birth per arm, these descriptive results do not establish a reliable
learning trend. The next method requires a new hypothesis and registered controls.

Stance slip (10008 / 10009, mm/s): parent 1.917 / 1.878; control 2.917 / 2.401;
lower exploration 2.776 / 2.229. Neither child improves slip over the parent.
Full per-leg counts and all matched comparisons are included in the artifact audit.

Both completed banks passed direct checkpoint audits. Control SHA-256:
`7ab309cc5101bfa38a6639680d78a2e840d445f75860d10d25695f5a5c221a2f`;
lower exploration SHA-256:
`224b1e797c9add1d2ff3e9f37a9d2eb9d4f515f22615fc521f721835cea98070`.
The parent remained unchanged, pilot per-file source hashes matched, all child
processes exited and the host lease was released.

Actual Python 3.12/FlyGym 2 verification before training: 82 passed, 1 NVIDIA-only
skip. Ruff passed for native CLI changes and the bounded runner. CPU-only pilot;
no CUDA validation, final held-out controls or generalization evidence.

Provenance correction: the protocol's pilot_source_sha256 field used the
core-only hash helper and contains an empty-tree hash. Preserve that original
field as an audit trail. Independent per-file pilot source hashes were captured
while the clean committed pilot source remained unchanged; the final audit
verified these hashes. The core source digest and parent digest were checked
between jobs. The runner helper was corrected after execution for future runs.
This limitation does not change the trained weights or outcomes.

Artifacts: https://github.com/amithey/Project-Noesis/releases/tag/exploration-pilot-2026-10-28
