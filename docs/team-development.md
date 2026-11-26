# Concurrent capability development

`scripts/team_development.py` runs real file-editing provider sessions in two
separate Git worktrees. It does not use the analysis-only review adapter.

| Worker | Checkout / branch | Assignment |
|---|---|---|
| Claude | outputs/team-dev-claude / codex/team-claude-retention | Offline checkpoint retention and invariant audit |
| Gemini 3.8 Flash Medium | outputs/team-dev-gemini / codex/team-gemini-senses | Opt-in external sensory mapping adapter |
| Codex | outputs/codex-training-20261028 / codex/training-pilot-20261028 | Locomotion experiments, tests, coordination and integration |

State is primary `outputs/team-development-20261029`. Open report.md or run
`Team-Development.cmd status` from the primary checkout. `run` starts only queued
tasks, skips finished work, and does not silently retry interrupted or failed jobs.
No scheduler or unlimited development loop is enabled. These are new managed
provider sessions; existing manual agent sessions and branches remain separate.

Each worker receives WORK_ORDER.md with exactly three owned deliverables. Tools
allow file inspection and edits. Claude uses safe-mode, file tools and acceptEdits;
Antigravity uses a custom file-tools agent, accept-edits and session-only rules
denying commands, browser/network actions and MCP. No permission bypass or global
permission change is used. This is provider permission enforcement plus instructions,
not a claim of complete OS isolation. A post-run allowlist rejects unexpected
changes; no output is automatically merged or executed as a training policy.

The coordinator runs the worker's synthetic pytest file independently in the
actual Python runtime after checking ownership and hashing files. It rejects
missing or empty deliverables, failed tests and deliverables modified during tests.
Status ready_for_review means files and tests exist and passed; it does not prove
biological realism, sensory hookup, retention of behavior or successful walking.
Test execution uses generated project code, authorized by this development task.

Watchdogs: 480s Antigravity provider / 510s worker / 120s synthetic tests. At most
two provider jobs run concurrently per round. A SQLite queue and exclusive OS
coordinator lock preserve state. Inspect logs and changes before any explicit
repair attempt. Provider logs remain local; code and concise results go to GitHub.
Codex snapshots completed worker deliverables on their dedicated new branches and
reviews integration before connecting either module to the active controller.

All shared full-connectome simulations continue to use a single host lease.
No worker may launch training; no held-out seeds, scientific gates or default core
pin change in this round. Tests and file development proceed concurrently.

Provider reference: https://antigravity.google/docs/cli/headless/ and
https://antigravity.google/docs/permissions describe accept-edits and fine-grained
session permissions; installed Claude --help defines its file tools/safe-mode.
