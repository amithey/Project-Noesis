# Project continuation and parallel work

Updated 2026-10-27. Repository: https://github.com/amithey/Project-Noesis

## Shared starting point

The project hub is the branch `codex/github-project-sync`. It preserves the
previously uncommitted local integration, the entire `gpu-pilot` source and tests,
training plans and status. This is a preservation snapshot, not a merge of all
experiments and not a claim that every component passes validation.

`docs/coordination.json` lists the actual worktrees, source commits and separate
backup branches for uncommitted work observed during this backup. Ownership and
activity must be confirmed in issues; the inventory does not launch workers.
The original branches and worktrees remain available. Do not combine all branches
blindly: several contain alternative approaches to the same core files.

## Current training state

The active route is FlyGym 2.1.0 plus the existing Noesis brain, with one CPU
world. Read `gpu-pilot/training-plan.md`, `gpu-pilot/training.md` and
`gpu-pilot/continuation-status.json`. Historical training remains on the other
branches. The required core is `codex/brain-body-feedback`, commit
`41b5fe6534ed4c8f0c21a2e123c8d518b3bf5953`, which includes `circuit_learning.py`.
It is deliberately kept separate from this hub's older core.

- Completed experimental birth: seed 11000, five seconds, 990 learning updates.
- Candidate: `gpu-pilot/outputs/standing-001-train-11000/weights-last.npz`.
  SHA-256: `fb80cfcc3018329209784751570d7ffb1ed33aa9ef48fceaa85588862f7626a8`.
- Parent: core checkout's
  `outputs/birth-curriculum-v31/round_002/train-5/attempt_0026/episode_000.npz`.
  SHA-256: `078cfe3ce59964ca655d39287f9f60249dde878dc13c119a7ee01749d8b4afe8`.
- Two frozen training probes passed the intermediate standing threshold. No
  validated best checkpoint, final curriculum pass, learned walking proof or
  verified CUDA execution has been recorded.
- Next: interruption-safe curriculum orchestration and physical leg-cycle
  metrics, then gripped-step training under the existing criteria.
- The current pilot does not restore mid-episode neural/physics state. Start a
  new birth from a completed checkpoint; do not call it an exact episode resume.

## Restore code and training experience

1. Clone `codex/github-project-sync` into a hub checkout. Clone
   `codex/brain-body-feedback` into a separate core checkout and verify the exact
   commit above for reproduction. Use Python 3.12 and the pinned installation in
   `gpu-pilot/README.md`, plus the core's dependencies. Keep these environments
   separate from the legacy FlyGym environment.
2. Download the assets from the GitHub Release `training-backup-2026-10-27`:
   https://github.com/amithey/Project-Noesis/releases/tag/training-backup-2026-10-27
   `manifest.json` records each archive, its SHA-256 and every included file.
3. Extract `feature-habitat-in-view.zip` into the hub checkout and
   `codex-brain-body-feedback.zip` into the core checkout. Other archives contain
   historical worktree training evidence and should be extracted into their
   matching checkouts. ZIP paths are relative to the original checkout.
4. Obtain the licensed raw connectome data separately and place it in `data/`
   following the existing data documentation. Raw datasets and derived graph
   caches are excluded from this backup. Environments/dependency caches are
   recreated from package definitions.
5. Verify checkpoint SHA-256 values before use. Pass the core's absolute `src/`
   directory to `--noesis-source`, an explicit data directory to `--data`, and a
   new output directory. Historical reports preserve their original absolute
   paths as provenance; those paths need remapping on another machine.

The release contains training NPZ checkpoints/observations, JSON reports, MP4
videos and PNG previews from registered worktrees. It excludes raw data,
connectome caches, runtime packages, authentication logs and temporary files.
It is a checkpoint/evidence backup, not a complete memory snapshot of a running
simulation. `scripts/archive_training.py` creates the same manifest-based layout
for future releases and rejects files modified while being copied.

## How agents stay coordinated

Each agent reads `AGENTS.md` and this guide, checks open issues and PRs, then
claims a bounded task in an issue with its branch, files, starting commit and
acceptance criteria. Use separate branches/checkouts and avoid shared output
directories. Record progress in that task's issue/PR and push coherent commits.
At handoff include the exact commit, artifact release/hash, checks and next step.

Suggested task boundaries, **currently unassigned**:

| Task | Likely files | Completion evidence |
| --- | --- | --- |
| Curriculum manager | new pilot orchestration module/tests | interruption and restart checks; unchanged gates |
| Physical gait metrics | pilot metric module/tests | stance/lift cycles and slip fixtures |
| External senses/obstacles | pilot adapters and sensory tests | matched frozen controls |
| NVIDIA validation | hardware reports, separate run folders | real CUDA execution and measured parity |

Agree ownership before editing shared bridge/controller/core files. There is no
automatic messaging or synchronization between bots: commits, issues, PRs and
release manifests are the common record. Do not treat an old status file as a
live worker monitor.
