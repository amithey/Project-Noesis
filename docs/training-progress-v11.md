# V11 short quiet-floor training progress

Status: all three five-second training births completed; both original seeds 10/11
frozen quiet-floor controls are complete; frozen candidate seed 10 is live. Candidate/obstacle/heldout evaluation is pending. No validated model exists or has been deployed.
The run uses source fc03104befece72e24bcabc7f49c751d63f2f9363557c1c8211e31aeb2b6a7fd,
imports complete v10 training weights and preserves all 25 frozen test cases.

Seed 5 completed 5.0 simulated seconds in 818.0 wall seconds. It stood for 93.13%
of the awake window. Measured stance/lift cycles were LF15 LM16 LH13 RF5 RM12 RH4;
forward heading-integrated displacement was 10.55 mm, mean stance slip 15.69 mm/s,
and mean reward -0.54991. The common-frequency tibia-flexor tripod score was
undefined. This is learning-on data with 1.5 mV/ms exploration, not a frozen
walking result. It cannot be compared causally with seed-10 original controls.

The complete checkpoint changed 35,968 of 73,594 eligible incoming motor edges
since v10; mean absolute update was 0.001157 of original magnitude. Total mean
relative deviation is 0.003940 and maximum 0.072424. Original topology, motor
identities, initial weights and all signs match; every weight remains within the
50% bound. The runner loaded this complete checkpoint for seed 6. Best-validated
weights remain null. The source fingerprint was rechecked and is unchanged.

The saved full-body follow video and a frame at 3.02 simulated seconds were
inspected; body and visible legs remain in frame, including posture changes.
The full video retains the weaker end-of-episode posture. The recording is
evidence of simulated movement, not proof of learning or validated gait.

Additional read-only v10 reward audit: 25.19% of original upright samples and
25.21% of candidate upright samples received the -1 reward floor. In the
20-second learning episode the upright floor fraction was 24.63%. This shows
substantial clipping of the scalar credit signal; it does not establish a causal
failure or authorize modifying the current experiment. Complete v11 frozen
results must guide the next training change.

Artifacts: outputs/birth-cycle-v11/report.json, cycle_000/train-0/attempt_0000,
and outputs/motor-balance-audit-v11/first-completed-training.json. A preserved
copy with video, checkpoint, CSV and source lives in primary-checkout
outputs/young-fly-v11-progress. Training has not met the goal until quiet standing,
walking/slip, obstacle response and held-out gates all pass without exploration.

## Measured foot support

A contact-state raster was generated from every saved 5 ms sample and visually
checked. It shows force-bearing contacts (green), unloaded clearance >0.05 mm
(blue), uprightness with every out-of-standing interval shaded, heading-integrated
progress and raw stance-slip bursts. The complete plot covers the full awake
window, including poor posture and high-slip bursts.

Per-leg loaded-foot fractions in the first training episode were LF64.7%,
LM37.8%, LH41.7%, RF18.6%, RM62.5%, RH6.5%. Thus the measured cycle count does
not mean the legs share support evenly. In particular, the right hind leg is
mostly unloaded. This describes one learning-on seed; it does not establish a
causal mapping defect or a learned improvement. No target timing or support
sequence was fed back into the controller.

Reproducible driver: outputs/motor-balance-audit-v11/plot_first_contacts.py;
plot: first-training-contacts.png; data: first-training-contact-statistics.json.

## Second completed birth: seed 6

Seed 6 completed five seconds in 791.4 wall seconds. Standing fraction was 58.59%,
heading-integrated motion 8.77 mm, mean stance slip 15.05 mm/s and mean reward
-0.68911. It fell late in the episode and did not recover before the endpoint.
The full failure remains in its saved video and CSV; a frame around four simulated
seconds is preserved. This demonstrates that the first birth's high standing
fraction is not consistent stability across training seeds. These trials retain
learning/exploration and are not matched frozen evidence of improvement.

The complete second checkpoint passed sign and 50% magnitude-bound checks; it
is the last-training checkpoint used by seed 7. Best-validated weights remain null.
All frozen validation, obstacle controls and held-out tests remain pending.

## Third completed birth: seed 7

Seed 7 completed five seconds in 820.1 wall seconds. Standing fraction was 14.55%,
heading-integrated motion 8.38 mm, mean stance slip 7.88 mm/s and mean reward
-0.78829. It fell early and did not recover; the lower all-window slip is not
evidence of better walking. The full failure is retained in the CSV and video.
A frame at approximately two simulated seconds was exported for review.

Across the three different training seeds, standing was 93.13%, 58.59% and 14.55%.
This demonstrates inconsistent standing in the short curriculum. Because both
seed and weights change, it does not isolate a causal degradation from learning.
The subsequent matched frozen original/candidate controls are still necessary.

The final complete training checkpoint passed sign and 50% bound checks. Source
fingerprint was reverified unchanged. Best-validated weights remain null. The
runner has started original seed-10 quiet-floor control with learning=False and
exploration off. It must finish required controls across bounded resume cycles
before promotion. The separately tested standing-first reward remains unapplied.

## Completed frozen original seed 10

The original five-second control completed with learning and exploration off:
92.22% standing, 8.319 mm heading-integrated displacement and 15.839 mm/s mean
stance slip. Every field of all 1,000 CSV rows and all recorded motor-spike arrays
exactly matches the completed v10 original seed-10 control. The short training
job schedule therefore did not contaminate this original model or change the
recorded conditions. This reproducibility result establishes a baseline, not
trained-model improvement. Original seed 11 and candidate evaluations remain
unfinished; no promotion is allowed.

## First wall budget and exact protocol resume

The first wall cycle closed normally after 3,596.42 seconds (process exit 0).
Original seed 11 reached 3.13 simulated seconds and was marked wall_budget. It
was not added to completed jobs and cannot count as a five-second control. Its
partial CSV, checkpoint, gait report and video remain preserved in attempt_0004,
along with the terminal first-hour report.

After confirming the old process was terminal and the source fingerprint was
unchanged, --resume started original seed 11 again from time zero in attempt_0005
with no checkpoint, learning or exploration. Exact settings, completed-job records
and last-training weights match the preceding terminal report; no source/config
checks were bypassed. Training episodes are not repeated. Best-validated remains
null and the standing-first reward is still unapplied.

## Complete resumed original seed 11

The replay completed all five simulated seconds with learning/exploration off:
94.95% standing, 10.151 mm heading-integrated displacement and 16.222 mm/s mean
stance slip. All 626 rows of the earlier interrupted prefix match the full replay
exactly. All 1,000 rows and all motor-spike arrays match the completed v10 original
control. No partial test was counted complete. The frozen candidate seed-10 test
has now started from the final complete trained weights, with learning and motor
exploration disabled. Original-control reproducibility does not demonstrate
trained improvement; candidate, obstacle and held-out gates remain pending.
