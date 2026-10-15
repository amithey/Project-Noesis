# Separating upright leg use from falling motion

This read-only audit was performed while v11 training was live. It changes no
brain, body, sensory, learning, reward or acceptance source. It uses completed
five-second frozen seed-10 recordings; interrupted outputs are excluded.

| Measurement | Original | V9 candidate | V10 candidate |
| --- | ---: | ---: | ---: |
| Standing fraction | 0.9222 | 0.8273 | 0.4808 |
| All-window forward distance, mm | 8.3190 | 8.9638 | 11.5347 |
| Forward distance while upright, mm | 7.1736 | 8.0424 | 5.0207 |
| Upright with >=3 supports, forward mm | 0.2162 | 1.8118 | 1.0493 |
| All-window stance slip, mm/s | 15.8391 | 17.8385 | 10.2572 |
| Slip while upright, mm/s | 15.6961 | 17.9804 | 14.7526 |
| Slip conditional on any loaded foot, mm/s | 16.1824 | 18.1501 | 12.4750 |

The v10 all-window slip improvement is 35.2%, but upright-only improvement is
6.0%. Conditional on at least one foot carrying load it is 22.9%. Thus these
conditionings answer different questions. Unloaded fallen intervals contribute
zero to the official average. This does not change the rejection: v10 already
failed the mandatory standing criterion and was never promoted. Its full-window
11.53 mm translation is not proof of walking.

A conservative cycle diagnostic resets the stance/lift state whenever posture or
height leaves the standing range. Original cycles are LF8 LM4 LH6 RF4 RM7 RH1;
v9 candidate LF6 LM8 LH4 RF2 RM6 RH0; v10 candidate LF4 LM6 LH4 RF3 RM4 RH2.
Because every brief posture break resets the state, these counts can undercount
leg cycles; they are supplementary evidence, not replacement acceptance results.

The paired motor-rate audit finds no valid common-frequency tibia-flexor tripod
score in original, v9 candidate or v10 candidate. Among 34 annotated paired joint
pools, both sides exceed 1 Hz in 24, 23 and 27 pools respectively. The left-front
femur remains extremely asymmetric: original positive/negative rates 94.04/0.073
Hz; v10 85.40/0.018 Hz. Rates alone establish neither muscle torque nor a causal
explanation for falling. No externally imposed target firing rates, leg sequence
or controller are inferred from these descriptions.

Next decisions must use the completed v11 frozen evaluation, not its noisy live
training frames. If stability still fails, investigate the support/posture reward
ranking recorded in outputs/curriculum-audit-v10/reward-posture-ranking.json: an
explicit supported-upright 16 mm/s sliding state scores -0.71 while a motionless
unsupported fall scores -0.6. That local inversion is a concrete credit-assignment
concern; it does not imply the whole observed dataset globally rewards falling
(its standing mean reward was higher). Any reward change needs a new source
fingerprint, new run, matched frozen controls and preserved prior failures.

Evidence and reproducible drivers: outputs/motor-balance-audit-v11/compare.py,
report.json, posture_conditioned.py and posture-conditioned.json. Completed
original v9/v10 controls are byte-equivalent in every CSV field and spike array.
