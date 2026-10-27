# Gemini collaborator

Run `./Start-Gemini.ps1` from PowerShell to open Antigravity CLI. Use `/model`
to select Gemini 4 Argon explicitly before submitting the startup investigation
printed by the launcher. It does not automatically submit work to the default
model. The session stays interactive for authentication and tool approvals.
It does not change global Gemini settings.

The legacy Gemini CLI rejected individual-account authentication with
`UNSUPPORTED_CLIENT` on 2026-10-14 and directed migration to Antigravity.
Antigravity authenticated the local Google AI Pro account successfully, but its
model selector did not include Argon. The update command reported the installed
client was current. Argon has therefore not been activated for this project.

`./Start-Gemini.ps1 -Client GeminiApi` retains the explicit-model Gemini CLI
route for separately configured API/enterprise access.

Sign in with a Google account that has access to Argon when prompted. Installed
CLI software and cached credentials alone do not establish model access. An
HTTP 401 means authentication failed; it does not establish model availability.
Never put API keys in the repository or paste them into the chat.

Assign separate files to concurrent agents and preserve existing uncommitted
changes. Gemini operates through its own CLI session; this launcher does not
provide automatic messaging between Gemini and Codex.

The fly simulation runs independently. Its normal live launcher is
`./Start-Fly.ps1 -Learn -Video`; see `learning-lab.md` for dependencies and
scientific limitations.

Official references:
- https://ai.google.dev/gemini-api/docs/models/gemini-4-argon
- https://geminicli.com/docs/get-started/authentication/
