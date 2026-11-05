# E2 prerequisite J0 — 2026-10-28

User authorized Codex to reply to team feedback and start training. Codex owns
this host window; Claude implements P2 in his branch, Gemini validates metrics.
Core review: 54828ca7e47d0c987e7324d5b4ec36fda2f37aff is accepted for this isolated
experiment only. Shared core pin remains 41b5fe6. New reward and cycle tracker
are opt-in. The 46 cycle alignment/reward/speed/saturation tests passed locally.

Start with Claude's proposed J0: reproduce v8 seed 1001 from standing parent
fb80cfcc3018329209784751570d7ffb1ed33aa9ef48fceaa85588862f7626a8.
Five simulated seconds, CPU, one world, 50 ms warmup, exploration 1.5 mV/ms,
learning rate 0.2, eligibility 100 ms, original-reference bounds unchanged.
Expected complete checkpoint SHA256:
626bde8dce2f996ef181650ba4c3812abdc0dfdd3a969bbfc1e0e1ed57b95ff1.

J0 simulation wall budget is 300 seconds. An outer 330-second process watchdog
allows import/startup overhead while bounding the entire child process. This
is the first portion of the proposed <=3600-second E2 experiment, not a new
curriculum round. Subsequent work must account for J0's elapsed time.

The guarded runner `scripts/run_e2_reproduction.py` takes the shared O_EXCL host
lease, checks actual Python process inventory, writes exact source hashes and
commit, preserves the parent, uses a new output directory, and releases its own
lease on exit. It stops after J0 regardless of success. A nonmatching SHA,
partial episode, process error, or changed source/parent fails the gate.

Output: primary checkout `gpu-pilot/outputs/e2-reproduction-20261028-1001`.
Experimental core: primary checkout `outputs/codex-e2-core-54828ca/src`.
Pilot: own worktree `outputs/codex-training-20261028/gpu-pilot/src`.

P2 learning integration is not yet delivered. No v9 job is authorized to start
before its review/integration/tests, refreshed J0 provenance as needed, and a
registered final protocol. Probes 10006/10007 are development probes already
used for selection; held-out seeds remain untouched. No automatic promotion.
