# Shared project instructions

Read `docs/coordination.md`, `docs/coordination.json`, and the current
`gpu-pilot/continuation-status.json` before changing training behavior.
For active team ownership, read `docs/team-training-plan.md` and
`docs/team-assignments.json`. These assignments supersede the historical
"unassigned" ownership entries in the backup inventory. Team tracker: issue #17;
Codex #18, Claude/Anthropic #19, Gemini #20. Acknowledge your assigned issue before
implementation; do not claim a second overlapping task.

- GitHub is the shared record. Use a separate branch and checkout per agent.
  Never commit, reset, merge, or rewrite another agent's working branch.
- Before starting, record your task, branch, owned files, starting commit and
  acceptance checks in a GitHub issue. Check existing issues/PRs for overlapping
  ownership. Ask the overlapping owner to resolve ownership before editing the
  same files. A branch name alone does not establish that an agent is active.
- Commit coherent changes and push progress regularly, including on handoff.
  Link a PR to the issue; record completed work, tests, remaining work and exact
  commits/checkpoint hashes. Resolve conflicting edits through review.
- Do not automatically merge experimental branches or relax scientific gates.
  Preserve topology, signs, original-reference weight bounds and held-out seeds.
  An exploratory result is not validation or proof of walking.
- Keep `training-last` separate from `validated-best`. Never overwrite a parent
  checkpoint. Partial episodes do not become completed training checkpoints.
- Store training artifacts in GitHub Releases with SHA-256 manifests. Keep code,
  plans, settings and compact results in Git. Do not commit data, derived
  connectome caches, credentials, local environments or dependency downloads.
- The FlyGym 2 pilot currently needs the core on `codex/brain-body-feedback`;
  the hub checkout's older core is not a substitute. See the continuation guide.
- At every handoff, update the issue/PR and relevant status documents with
  observed facts. Do not infer another agent's activity from old documentation.

These instructions coordinate work; they do not authorize sending messages to
other users or other chats, deploying models, or automatically starting training.
