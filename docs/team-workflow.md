# Automated E2 team review

## Connection update — 2026-10-29

The default Google provider is now **Antigravity (`agy`)**, replacing Gemini CLI.
The new queue is `outputs/team-workflow-e2-antigravity`; the old reviewed queue
and its failed attempts remain untouched. The root launcher selects the new
queue automatically. Antigravity uses a workspace-scoped `noesis-reviewer` agent
with `tools: []`, plan mode, and a210s provider timeout within the240s watchdog.
The installed CLI warns that disabling slash commands disables plan mode, so
that conflicting flag is intentionally omitted. Terminal responses must have
status `SUCCESS`; partial/error responses cannot complete a task.

A live OK-only connection check reached Antigravity but returned HTTP429
`RESOURCE_EXHAUSTED`: individual quota reached, reset in approximately93h15m
at the time of checking. No project evidence was sent in this check. Claude's
auth status still reported loggedIn=false; its official subscription sign-in
flow was opened for the user. These are account blockers, not successful reviews.
Quota errors are separate from authentication failures and are never retried
automatically. No subscriptions, billing routes or global permissions changed.

After signing in and resolving/waiting for quota, use:

```powershell
.\Team-Workflow.cmd retry claude-1
.\Team-Workflow.cmd retry antigravity-1
.\Team-Workflow.cmd run
```

The older Gemini notes below describe the initial deployment, not the new default.

The local coordinator dispatches two independent reviews to Claude and Gemini,
then automatically passes both reviews to each provider for cross-review.
It collects all four responses into one report and HTML status page. It does
not rely on another chat noticing a message. These are new CLI sessions.

Implemented in `scripts/team_workflow.py`, using only Python's standard library.
The current installed state is in the primary checkout:
`outputs/team-workflow-e2-reviewed`. Open its `status.html` or `report.md`.
The root `Team-Workflow.cmd` launcher supports `status`, `run`, and
`retry claude-1` / `retry gemini-1`. Default action is status.

Portable setup from this branch:

```powershell
python scripts/team_workflow.py --state C:/local/noesis-review init --root C:/path/to/primary-checkout
python scripts/team_workflow.py --state C:/local/noesis-review run
python scripts/team_workflow.py --state C:/local/noesis-review status
```

Use a local disk for SQLite state, one computer and coordinator per state folder;
do not share a live database across computers or use synced copies concurrently.
An OS lock prevents a second coordinator. SQLite retains task state. `run` skips
completed tasks; after an interrupted coordinator it marks unfinished requests
interrupted instead of silently repeating a possibly billed request. Inspect
surviving provider processes before explicitly retrying an interrupted request.

Each provider has a separate working directory. Claude runs safe mode with
tools disabled. Gemini runs plan mode with a deny-all tool policy, no extensions
or MCP servers and hooks disabled. The coordinator passes an allowlisted E2
snapshot on stdin: numeric metrics, experimental labels and provenance hashes.
It excludes credentials, user paths, package configuration, raw connectome data
and weights. The snapshot hash is checked before each coordinator run. Responses
are untrusted analysis, never executed. HTML output is escaped.

There are at most four requests on a normal run, two at a time, with a 240-second
timeout per request. Explicit retry can incur additional provider usage. Model
and authentication follow the existing CLI configuration; this is not a dollar
cost cap. No secret is stored by this coordinator. Request prompts and provider
stdout/stderr are local in `attempts/` for diagnosis, not committed to Git.

## Observed acceptance run — 2026-10-28

Eight local tests cover exclusive coordination, persisted state, phase ordering,
no repeated successful requests, crash recovery, modified-evidence rejection,
provider errors, output escaping and payload minimization.

The real headless dispatch reached both installed clients, but the accounts
blocked completion:

- Claude returned `Not logged in · Please run /login` (zero API tokens reported).
- Gemini returned `IneligibleTierError` / `UNSUPPORTED_CLIENT` for the current
  Code Assist individual account route. This is an observed response from this
  installed client, not a statement that all Gemini accounts are unsupported.

Authenticate Claude through its local CLI. For Gemini, configure a supported
account/API route through the CLI's official authentication flow. Never paste
tokens or API keys into the shared chat or this report. After resolving accounts:

```powershell
.\Team-Workflow.cmd retry claude-1
.\Team-Workflow.cmd retry gemini-1
.\Team-Workflow.cmd run
```

The coordinator then performs the first reviews and cross-reviews automatically.
No successful end-to-end model review has been claimed while authentication is
blocked. Existing sessions are not imported or awakened. This v1 has no boot
service, recurring scheduler, training dispatcher, code-edit worker or automatic
GitHub publisher. Code remains in the existing PR; model outputs stay local
until reviewed for publication. Automatic training is a separate next phase.

References: [Claude headless](https://code.claude.com/docs/en/headless),
[Gemini headless](https://geminicli.com/docs/cli/headless/),
[Gemini policy engine](https://geminicli.com/docs/reference/policy-engine/).
