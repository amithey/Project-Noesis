# AP1 full-bank result: mixed changes, original criteria failed

Decision: do not adopt or promote AP1. The fixed non-tarsal reaction penalty did
not satisfy the original strict retention rule against either reference, nor the
prespecified anatomical rule across every development seed. No standing/balance/
walking ability or method qualification is established. The complete checkpoint
is preserved as an experimental candidate only; the common parent is unchanged.

Evaluated training sourcea05b0daa3c4416f1ffc0a2c59f8770f77069bd8b (PR42),
coverage correction sourcea1fbcd3240c839e13e665bf7fdebfde7dc223cd0 (PR43).
Candidate SHAaced093428ea850e6f091f36fa13caa86afad0948ca0deb6cc5f3d85321b265c,
parent9fd7d41177bf2c745809401c3b142a832661d1a18d0b117b30cb02aea63fea3c,
matched native-v7control46a6975dc7a1479f794034258abae67ff178d31772ac961d54617519a2b00853.
One training seed1005,25s/4990updates, native learning rule/rate.2/tau100ms/
exploration1.5/original-reference bound.5 unchanged. Only training scalar reward
was `clip(v7 - .5*q_non_tarsal,-1,1)`. Frozen tests have no learning, motor
exploration or shaping. No external gait/actions, topology/sign changes or merges.

## Technical completion

AP1A0full25s disabled observer-control exactly reproduced the SR1checkpoint and
all body/pad/exploration arrays. A new exact floating-mass equality check then
stopped orchestration. AP1B used the existing H2numerical mass tolerance and
reused that full control; it completed candidate training,8new references and
2candidate probes before seed10007stopped at1.83s on missing native compound
leg coverage. Both original failure banks remain immutable, not successful banks.

The C1correction explicitly recognizes six native trochanterfemur proximal
links; all55compiled explicit ground pairs are covered, with no physics/collision
change.28focused tests passed. AP1C ran only the3remaining frozen probes, not
another birth, after the resource gate recovered. All3completed full5s/1000chunks
in183.657s aggregate, exit0/empty trees/canonical lease released. Seed10007
replayed all366old body rows and367anatomy rows exactly; the first newly covered
compound reaction was4.62781native force units at1.835s.

Independent terminal audit recomputed5000native reward transitions in EACH
control/candidate birth and every scalar correction, checked identical drive/
target/initial-body schedule, source/data/prior/checkpoint invariants and all
15frozen bank positions. Original failed partial remained preserved. Across the
correction chain:15complete new workers and1partial failed worker;2additional
H2Bfrozen reference cells were reused. Allocation wall time1428.376s in total
(A0289.235+B955.484+C183.657), not just the final correction time. No active job.

## All development seeds

Percentages below use990post-warmup frozen samples. S is the ORIGINAL standing
proxy; T is tarsal-only force-bearing anatomy, NOT a balance/standing validator.

| Seed | Parent S% | v7 S% | AP1 S% | Parent T% | v7 T% | AP1 T% |
|---|---:|---:|---:|---:|---:|---:|
|10006|96.26|97.68|91.92|42.32|55.96|61.82|
|10007|94.44|87.68|91.52|39.90|56.26|59.90|
|10010|89.29|94.24|91.82|67.47|63.33|49.80|
|10011|91.92|90.81|95.35|43.84|37.78|58.28|

AP1exceeded90%legacy standing on4/4development seeds versus3/4for each reference.
This is an absolute proxy result, not retained physical ability. Mean standing
was92.65%versus92.98%parent/92.60%v7. Mean tarsal-only fraction was57.45%versus
48.38%parent/53.33%v7. Mean non-tarsal fraction was.19786versus.24819/.21335.
These averages do not override seed-specific failures.

Both anatomical improvements (more tarsal-only samples AND less mean non-tarsal
reaction versus BOTH references) held only at10006and10011:2/4. At10007tarsal-only
time increased, but non-tarsal fraction worsened versus v7by.01357. At10010
tarsal-only time fell13.54percentage points versus v7and17.68versus parent,
while non-tarsal fraction increased. Thus the prespecified anatomical rule failed.

The strict retention rule also failed versus BOTH references: standing and slip
were not retained on every seed. Median paired heading/net changes versus parent
were-.12380/-.14110mm; tripod-score change-.05311. Versus v7, heading-.08359mm,
net+.10973mm and tripod+.01202, with0median leg-cycle gain. More cycles or mixed
progress do not establish walking. Candidate regimes were3uncoordinated-stepping
and2stepping-in-place across own+development seeds; no walking label.

Own training seed1005standing97.27%,tarsal-only51.11%is reported separately in
the compact JSON and is not held-out evidence. The penalty applied on3228/5000
training rows, mean effective penalty.15424(native reward.55458,shaped.40034).
Different objectives' reward means are not comparable learning-success metrics.

## Consequence and limits

This pilot changes the next action: the new feedback does not simply repair
retention, and its average/minimum improvements coexist with losses. No lambda
sweep, candidate selection, gate relaxation, continued training from the mixed
child, default replacement or merge is authorized by this result. One selected
birth cannot establish a general method effect or absence of learning.

Next proposed AP2 tests the SAMEfixed method across the two other already-paired
training seeds1004/1006, with exact disabled controls and the original criteria,
to distinguish reproducibility from the selected1005case. It is not allocated
until its own code/tests/protocol are published; all3births must be reported,
including the current failure. No reserved final seeds were used.

Selected local evidence:184files,72972023bytes,
SHA2562f60d305c87dcf08640476fbe541b6aaf3eb66aa69cfd563553cdcd52a676540,
all manifest hashes/CRC verified. Release upload is NOT complete. The extensive
private integrity manifest remains local after automatic publication rejection;
only compact scientific statistics/code/protocol and one artifact receipt are
published. No raw connectome/cache/environment/private lease in Git.
