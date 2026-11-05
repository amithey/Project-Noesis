# Optional afferent interface

The default controller now exposes Claude's proposed hook for additional afferent drive. The default returns an empty list. Subclasses supply afferent parts through the existing merge path, which checks for overlap before invoking the LIF network. This candidate does not select or activate a new sensor.

## Reproduce

From the candidate checkout, use the core source at commit 42c20769376bda5694b4c1c66ca75e52cbd9f1f4. Put the candidate's gpu-pilot/src, gpu-pilot/tests and that core's src on PYTHONPATH. Run:

```text
python docs/workstreams/codex/sensory-interface-20261105/verify.py
```

The verifier reads the pristine controller from the exact Git base 24b69a1. It uses an existing synthetic checkpoint fixture and deterministic observations with changing joint angles, velocities and load. Across two seeds and 120 steps per seed, all commands, adhesion and per-step spike arrays are exactly equal, with 717 spikes total across the candidate trials. Topology and weights are equal and read-only. The checkpoint file hash is unchanged. The existing merge rejects a duplicate afferent supplied by a subclass.

This is a synthetic interface check. Before a full FlyGym trial, the sensory caller must isolate its off, zero and constant modes from active-field evaluation, and the protocol must register the field, exposure controls and target-afferent measurements. The physical pilot source pin and parent remain recorded in their existing branches.

See issue #20 for the registered scope and the review of Claude's caller. No heavy run was allocated.
