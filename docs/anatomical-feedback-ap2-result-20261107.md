# AP2 result: fixed anatomical feedback did not retain all required capabilities

Decision: no adoption, checkpoint selection, promotion, default change or merge.
All three declared training births1004/1005/1006 FAILthe original combined rule.
Every birth fails strict standing/slip/proxy retention versus BOTHparent and
matched native-v7control; anatomical improvement versus both references holds
at only2of4development seeds in each birth. This is not proof of no learning or
statistical equivalence. It is evidence that this fixed method, under the tested
budget/conditions, does not meet the project's retention criteria.

Evaluated runtime source7692865891a05a939c983ce8ee1946b9f768fff4, PR44.
Common parent9fd7d41177bf2c745809401c3b142a832661d1a18d0b117b30cb02aea63fea3c,
fixedclip(v7-.5*q_non_tarsal), C1coverage/native learner/rate.2/tau100ms/
exploration1.5/original-reference bound.5 and body/pad/gates unchanged. Each new
candidate trained25s/4990updates from that SAMEparent, not from the mixed1005
child. No pooling/averaging of learned weights, external gait or action override.

## Technical reliability

All25prespecified new workers completed full horizon in4603.484s aggregate
(76.7minutes, within5400s);4births(2disabled/2candidate),21frozen5s probes.
Both disabled25s controls exactly reproduced their old native checkpoint and
every body/pad/exploration array. Candidate currentANDinnovation-credit streams
and target/chunk schedules exactly match their controls; initial11nonreward
body/pad rows match. Actual objective/parent/birth metadata and sidecars agree.
Original topology/sign/reference bounds/normalization flags are preserved.

Independent terminal audit recomputed all20000new native reward transitions and
scalar corrections from raw observations, and recomputed all45bank positions
representing37UNIQUEfrozen cells (shared parent-development data are not separate
replications). The original failed1005bank remains included and unchanged.
Source/data/prior/child signatures, all full-horizon reports, exit0/empty process
trees and canonical lease release passed.36focused tests passed before allocation.
No active simulation remains. Source and original parents/partials are unchanged.

## All candidate development results

S is the ORIGINALstanding proxy, T is tarsal-only force-bearing anatomy. T is
NOTa balance/standing validator. Each fraction uses990post-warmup frozen samples.

| Training seed | Probe seed | S% | T% | Mean non-tarsal fraction |
|---|---|---:|---:|---:|
|1004|10006|94.95|75.25|.11|
|1004|10007|91.92|61.92|.18|
|1004|10010|89.39|54.24|.22|
|1004|10011|85.05|58.48|.20|
|1005(original)|10006|91.92|61.82|.15|
|1005(original)|10007|91.52|59.90|.20|
|1005(original)|10010|91.82|49.80|.25|
|1005(original)|10011|95.35|58.28|.20|
|1006|10006|94.34|52.63|.22|
|1006|10007|91.52|45.35|.25|
|1006|10010|94.75|45.25|.26|
|1006|10011|91.31|50.51|.23|

Absolute legacy-standing>=90%counts are2/4,4/4,4/4; minima85.05%,91.52%,91.31%.
Minimum tarsal-only fractions are54.24%,49.80%,45.25%. Minima/averages are
descriptive and do not replace the original per-seed retention rule. The known
limitations of the old proxy, including acceptance of body support and rejection
of some higher foot-supported states, still apply. No retrospective rescore.

Seed1004median heading/net gains versus parent are+.78034/+.81018mm and
versus v7+.34839/+.51938mm, but standing/slip retention and tripod changes fail.
Seed1006heading/net gains versus parent are+.20906/+.24920mm; versus v7,
heading-.08857mm/net+.05360mm. Its tripod gains are negative versus BOTHrefs.
Seed1005retains its previously reported mixed failure. More leg cycles or net
movement is not walking. Neither new candidate meets all anatomical conditions
on each development seed. All per-cell numbers and reference comparisons are in
the compact resultJSON; own-seed values remain descriptive rather than held out.

## Consequence

Do not continue unchanged single-seed births or sweep lambda on the assumption
that more attempts will fix retention. The repeatability test answered its
declared question: the combined qualification did not repeat in either new birth,
and the original failure was not discarded. What causes the remaining misses is
not fully established. Before selecting the next training method, audit the
existing raw37cell bank for upright failure, missing claw support, high/low-height
proxy misses and anatomical support during those intervals. This is descriptive
diagnosis, not a new gate; it must preserve all current0/3verdicts. In particular,
do not assume that a high-height miss is a fall or certified stable standing.

Longer retained capability across subsequent training, physical balance/gait and
environment response remain UNACHIEVED. No reserved final validation seeds were
used. Future method changes require their own hypothesis/controls/budget/stop
conditions and source ownership, not an automatic retry of this allocation.

Local selected evidence:358files,181327024bytes,
SHA25642e20e37d1ee95c550d65d2cb491d69895ae75777b4f279788110d1468d1373d,
all manifest hashes/CRC verified. Release upload remains incomplete. Extensive
private integrity manifests remain local after the prior automatic publication
rejection; public files contain compact scientific statistics and one artifact
receipt. No raw connectome/cache/environment/private lease is committed.
