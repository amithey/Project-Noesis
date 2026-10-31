# D3 full training reproducibility and exploration logging parity

Codex issue18. Review/copy Claude2db4d11 learning/profile modules; native train
flags add learning_rate default.2 and opt-in exploration logging, unchanged rule.
This test follows D2's exact physical prefix and resource telemetry. It is not a
new reward/LR/sensory method or a candidate-selection experiment.

Two sequential5s/1000chunk training births, same complete9fd7d411 parent,
core42c2076, seed1000, gripped-cycle-v9 weight.1, exploration1.5, LR.2. First
logging disabled, then enabled; require BOTH reproduce historical complete
2074b1e2c12061a03e13d60daf93e07a6517d832bc1f31fe914a797784d70214 checkpoint.
Raw topology/indices/reference/parameters/rule/sign/original50percent bound audit
before accepting each. Parent/source hashes frozen, exclusive existing host lock,
live Python inventory and>=2GiB available memory before each birth.

Budget720seconds aggregate,300seconds inner per birth,330seconds process watchdog,
reject late output after335seconds. Stop at first failure, incomplete report,
reproduction mismatch, mutation, resource failure, invalid log, or time shortage.
No automatic retry/budget increase. Output primary:
gpu-pilot/outputs/logging-parity-d3-20261030.

Require990updates and exact equality of every plain/logged observation array,
including rewards/learning flags, plus exact checkpoint hash. Exploration logs
must contain990 finite rows aligned to zero-based control chunks10..999; drive
and credit columns use explicit drive_idx/motor_indices and original amplitude
bounds. Warmup is not silently matched to learning row0. Native profile wrapper
must preserve these alignment fields instead of overwriting the exported log.

Default-compatible synthetic small-brain/fixed-body checks passed before launch;
those do not establish physical parity. Full D3 completion establishes local
full training reproducibility and logging transparency only. No promotion or
walking/retention/generalization claim. Held-out seeds remain unused.

## Observed outcome

Both physical births completed1000chunks/990updates. Plain inner117.153s,
process120.828s; logged inner123.518s,process127.141s; total248.438s.
Available memory at starts3.434/3.382GiB. Both checkpoint hashes exactly equal
the preregistered2074b1e2... value. All19observation/reward/command arrays are
exactly equal. Logged990rows align to chunks10..999 with finite bounded values.
Original-reference/topology/sign/parameters/rule audits passed. Lease released.

This establishes full training reproducibility and read-only logging parity for
this parent/seed under the observed resource envelope. It does not establish a
new learned ability, biological correctness or reliability for every host state.
See logging-parity-d3-result.json and the artifact SHA256 manifest.
