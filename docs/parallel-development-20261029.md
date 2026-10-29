# Parallel development domains

User clarified on 2026-10-29 that three agents should develop different capabilities.

| Owner | Capability | First concrete deliverable |
|---|---|---|
| Codex | Standing and walking | Finish E3 interpretation and define next bounded locomotion experiment |
| Gemini / Antigravity | External senses and environmental reaction | Inventory available real sensory signals, implement a separate opt-in adapter and synthetic signal/no-signal tests before controller integration |
| Claude / Anthropic | Learning and memory | Audit what checkpoints retain versus reset, measure forgetting/retention on saved traces, propose a bounded retention experiment |

The third domain is Codex's proposed assignment based on the user's request for
another capability; it does not imply biological memory is already implemented.
Current pilot already has leg/joint/load/balance feedback. External vision and
chemical senses remain disabled. Do not claim these are connected or trained.
Gemini starts with new sensory adapter modules and a proposal for controller/
backend interfaces; it may not overwrite the running integration or core.
Claude retains learning/reward ownership, with separate proposals for any core
changes. Codex coordinates interface handoffs before edits overlap.

Code, analysis and synthetic tests may proceed in parallel in separate branches.
Full simulations on this shared host remain sequential under the exclusive lease.
E3 completed nine jobs; no model promotion. Its reviewers do not count as active
implementation workers in these newly assigned development domains.

Assignments are recorded, not evidence that provider implementation sessions
have been launched. Existing managed sessions have tools disabled and only
produce analysis. A development-capable dispatch must be configured separately.
