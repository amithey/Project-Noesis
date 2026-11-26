# TP1 — frozen thermosensory integration pilot (registered 2026-11-06)

Purpose: integrate Claude's existing optional thermo input into the native brain/body route and test source correctness, disabled-path invariance, neural observability and standing/slip retention. This does not train, prescribe motion, prove walking or select a model.

## Source and ownership

Integration branch codex/sensory-hook-20261104 from0e560229 (runtime75cdc34). Core42c2076 remains unchanged; parent9fd7d411 remains read-only. Data are the four SHA-verified privately restored BANC/FAFB/MANC sources.

Reviewed Claude source d2160a9: sensed_brain.py and senses_bridge.py. Existing external_senses.py and its tests match that delivery and remain unchanged. Codex integrates the default-empty controller hook and isolated probe, without editing Claude/Gemini checkouts. Review amendment: inactive/constant modes lazily avoid the field callback; regression added. Native Python/Numba seed schedule, body, weight signs/reference bounds and rewards are unchanged. The worker installs a per-process observed brain factory only for this diagnostic and restores it afterwards.

44focused sensory/controller/bridge tests passed in5.69s; probe/runner compile. Synthetic checks are not physical evidence.

## Fixed protocol before data

- Parent9fd7d411; two already-used development seeds10006/10007. Final10/11/200/201/202 forbidden by probe.
- Ten sequential jobs: for each seed plain,off,zero,constant,active; CPU1world; full5s/1000control chunks; first50ms excluded from metrics; no learning/exploration.
- Six warm and six cool annotation-selected afferents, disjoint from wired leg/balance input. Gain50Hz per normalized signal unit; input event weight remains sensory_weight400.
- The normalized environmental field is u=clip((x+1)/2,-1,1), with x in the native position units (mm as used by the body model). Warm=max(u,0),cool=max(-u,0). This is an explicit analytic stimulus assumption, not calibrated thermal physics or degrees Celsius.
- Constant control u=.5 independent of position. It matches active at x=0 only, not its full time-varying exposure; active/constant differences alone cannot isolate temporal pattern from stimulus magnitude.
- Log actual requested channel rates, input body position/u, per-channel afferent spike counts, total spikes, commands, adhesion and native body observations. Metadata identify mode and source; ordinary compare() must not silently treat different sensory modes as matched unchanged conditions.

## Gates and interpretation

1. Before either stimulus arm: plain native observations must exactly reproduce B2 for that seed; off and zero must exactly match plain in every native AND sensory observer array. Driver checks completed horizon, finite arrays, unchanged weights, parent/source context and stimulus formula. Any failure stops remaining jobs.
2. Primary neural observability rule: both constant and active warm-afferent mean spike rate must exceed matched off by>=5Hz per neuron on BOTH seeds. The threshold is a pilot assumption fixed here; report negative outcomes without increasing gain.
3. Retention rule: all constant/active probes standing>=.90, standing>=matched off AND slip<=matched off. Report each delta, gait regime and cycle metrics. This is a strict descriptive pilot screen, not a retained-learning or generalization claim.
4. No weight updates, new checkpoint, stage/model promotion or default activation regardless of outcome. Successful disabled integration may be retained separately from an active input that fails retention. No biological thermotaxis/obstacle response inference.

## Resource and failure controls

Canonical lease at PRIMARY checkout gpu-pilot/outputs/codex-training-host.lock; one heavy worker on this host. Private lease identity/host/PID never included in public protocol. Fresh available RAM>=2GiB and no unrelated Python process before each job; parent/source hashes checked throughout.

Budget1800s aggregate;300s hard subprocess timeout per job including imports/setup, plus native300s cooperative check. Do not start a job with<300s remaining. Stop on timeout/error/incomplete/nonfinite/invariance/source/parent/resource failure. Preserve partial evidence separately, release only the owned lease, no retries/parameter substitutions/extra seeds. No remote or peer simulation allocation.

Output: outputs/thermo-pilot-tp1-20261106. Code/public protocol/compact results go to Git; numerical artifact archive excludes licensed graph data/caches, environments and private lease metadata. Release upload status must be reported honestly; CLI authentication was invalid at last check.

The broad goal remains stable fly-like walking, balance, environmental response and improvements retained through training. TP1 addresses the sensory transport prerequisite only.
