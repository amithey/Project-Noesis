# V11 short quiet-floor training progress

Status: first five-second training birth completed; seed 6 is live, seed 7 and
frozen evaluations remain pending. No validated model exists or has been deployed.
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
