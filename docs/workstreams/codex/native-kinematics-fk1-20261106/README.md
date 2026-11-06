# FK1: native pad-release sign versus geometry

The existing positive-Femur pad-release direction LOWERS the observed foot origin in all126leg/configuration cases. The negative-Femur direction raises it. This is a measured mismatch with the intended lift-release convention in the sampled geometry, not proof of the cause of locomotion failure.

|Leg|Neutral dz/dFemur mm/rad|Lowering cases|
|---|---:|---:|
|LF|-1.376462|21/21|
|LM|-1.326808|21/21|
|LH|-1.356628|21/21|
|RF|-1.376462|21/21|
|RM|-1.326808|21/21|
|RH|-1.356628|21/21|

Max finite-difference/Jacobian error 7.71e-08mm/rad; step-halving error 2.31e-07; translation error 1.33e-15mm; cross-leg derivative0. No limit crossings. Clock stayed0; no dynamics or neural step. Parent/source unchanged, worker exit0 and lease released.

At native neutral, +.1rad lowers the foot by .130-.135mm and releases the pad; -.1rad raises it by .135-.140mm while retaining adhesion. Apparent hip-angle/tibia-angle derivatives also agree with the encoded flexor bending directions at neutral. Do not flip neural synapse signs or muscle mappings based on this pad-interface observation.

Limits: active-joint samples with root/passive joints fixed neutral; isolated derivatives hold other joints fixed; commanded and actual angles can differ under lag; biological torque/force is unvalidated. No walking/balance/learning benefit measured and no runtime change applied. A separate preregistered one-factor frozen dynamic comparison is required before adopting a revised release convention.
