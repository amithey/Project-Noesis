# Legs fix (`--legs real`): realistic rest pose and joint ranges

Requested by Amit 2026-11-26: give every leg a real fly's range of motion, then keep training on the fixed legs.
Code `6b40aa5` (`gpu-pilot/src/noesis_gpu/real_legs.py`). Rest = step-cycle mean of each joint in NeuroMechFly walking
kinematics; range = observed min..max widened by 25% of the span + 0.35 rad.

## Step 1 — kinematics file
Found `single_steps_untethered.pkl` in the legacy FlyGym 1.x env `C:\dev\venvs\noesis\Lib\site-packages\flygym\data\behavior\`
(18,024 bytes, sha256 `1E5B28BB…FAF8C`, 44 keys). Copied to `<project>\data\kinematics\` (not committed).

## Step 2 — sanity check: **FAILED, stopped here**
`leg_usage_audit.py --legs real --kinematics <pkl> --seconds 3` → `max_abs_shift_from_native_rad` = **4.80 rad** (limit 1.2).
With the uncorrected mapping the fly collapses: claw-load share 0.00–0.16 on every leg, 0–2 steps
(file: `audit-legs/leg-usage-parent-reallegs-UNCORRECTED.json`).

Native v2 rest/limits vs v1 kinematics (largest shifts):

| joint | v2 joint | native rest [lo, hi] | kinematics mean (min..max) | shift |
|---|---|---|---|---|
| RHCoxa_roll | c_thorax-rh_coxa-roll | +2.339 [+1.639, +3.039] | −2.459 (−2.639..−2.203) | 4.80 |
| LHCoxa_roll | c_thorax-lh_coxa-roll | +2.339 [+1.639, +3.039] | +2.459 (+2.203..+2.639) | 0.12 |
| RMCoxa_roll | c_thorax-rm_coxa-roll | +1.518 [+0.818, +2.218] | −1.839 (−1.915..−1.749) | 3.36 |
| LMCoxa_roll | c_thorax-lm_coxa-roll | +1.518 [+0.818, +2.218] | +1.839 (+1.749..+1.915) | 0.32 |
| RFCoxa_roll | c_thorax-rf_coxa-roll | +0.576 [−0.124, +1.276] | −1.007 (−1.321..−0.646) | 1.58 |
| LFCoxa_roll | c_thorax-lf_coxa-roll | +0.576 [−0.124, +1.276] | +1.007 (+0.646..+1.321) | 0.43 |
| LFFemur_roll | lf_coxa-lf_trochanterfemur-roll | −0.262 [−0.962, +0.438] | +0.813 (+0.583..+1.027) | 1.08 |
| RFFemur_roll | rf_coxa-rf_trochanterfemur-roll | −0.262 [−0.962, +0.438] | −0.813 (−1.027..−0.583) | 0.55 |
| LMCoxa_yaw | c_thorax-lm_coxa-yaw | −0.175 [−0.875, +0.525] | +0.791 (+0.654..+0.920) | 0.97 |
| RMCoxa_yaw | c_thorax-rm_coxa-yaw | −0.175 [−0.875, +0.525] | −0.791 (−0.920..−0.654) | 0.62 |

Pattern: FlyGym 2 uses the **same sign on both sides** for roll/yaw joints (native L and R values are identical),
while the v1 kinematics are **mirrored** (the right side is the negative of the left). Pitch joints (Coxa, Femur, Tibia, Tarsus1)
agree on both sides (shifts ≤ 0.94 rad, mostly < 0.4). The likely correction is to negate the v1 kinematics for right-side
`*_roll` and `*_yaw` joints. After that, the left-side shifts would remain (LFFemur_roll 1.08, LMCoxa_yaw 0.97, LM/RM Femur 0.94).
Femur_roll may also have an opposite overall sign convention. Waiting for the coordinator's decision before changing anything.

## Related
- R4 (4 seeds) was interrupted for this work: [R4-es-stand-4seeds.md](R4-es-stand-4seeds.md).
