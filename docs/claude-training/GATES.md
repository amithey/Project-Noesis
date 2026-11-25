# Acceptance gates (fixed 2026-11-25, before any run)

Evaluation: frozen brain (no exploration current, no plasticity), held-out seeds
**20001, 20002, 20003** (never used for training; training seeds are 30000+),
5 simulated seconds each, paired with the starting parent on the same seeds.
Implemented in `gpu-pilot/es_evaluate.py`.

## Standing gate (all must hold)
1. No physics errors.
2. Standing fraction >= 0.97 on every seed (claw-to-floor support, uprightness >= 0.7, height 0.3-1.5 mm).
3. Mean uprightness >= 0.85 on every seed.
4. Mean stance slip not worse than the parent (mean over the 3 seeds).
5. Mean curriculum-standing-v7 reward better than the parent on every seed.

## Walking gate (all must hold)
1. No physics errors.
2. Net forward displacement >= 2 mm in 5 s on every seed.
3. Standing fraction >= 0.90 on every seed.
4. At least 4 legs with >= 3 claw touchdowns (swing->stance) each, on every seed.
5. Forward displacement beats the parent by > 0.5 mm on every seed.

Passing a gate is recorded as "gate passed", not as "walking proven" or a
promoted model; promotion would still need independent validation.

## Addendum 2026-11-25 (after R0, tightening only)
R0 showed 12-73 raw "touchdowns" per leg on the untrained parent: claw contact
flicker, not stepping. Walking gate criterion 4 is kept, and a sixth criterion is
added: **at least 4 legs with >= 2 debounced steps** on every seed (a swing of
>= 40 ms without claw load followed by >= 40 ms of stance). No criterion was relaxed.

## Addendum 2 2026-11-25 (Amit's goal statement, tightening only)
Goal: stable standing and stable walking **using all the legs**. Walking gate
criterion 7: **every one of the six legs makes >= 2 debounced steps** on every
held-out seed. The walking fitness also rewards the share of legs that step.
Training continues round after round until both gates pass.
