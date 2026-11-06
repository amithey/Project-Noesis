# ES1 — single-generation episodic synaptic search

Preregistered2026-11-06. Full user goal remains standing/balance/stable walking/environment response retained through learning; it is unachieved. This is one feasibility trial of a different training strategy after repeated within-episode credit/retention failures, not a replacement definition of success.

## Source, ownership and candidate construction

Codex owns new proposer/runner/tests/docs on codex/episodic-search-es1-20261106, from isolated revieweeea7d4. Native pilot runtimee0cbd10/core42c2076 stays byte-identical to the verified TP2 source manifest. No Claude/Gemini files, within-episode update rule, controller, body, rewards, external sensory activation or common pins change. The existing empty sensory hook remains empty. Parent9fd7d411 is a completed experimental anchor, not validated-best; prior four-probe standing qualification failed3/4.

Use the existing562plasticity targets and85390selected incoming edges. Construct ONE Rademacher direction per postsynaptic target with seed60061, independently of motor anatomy or time. Fixed step2% of ORIGINAL magnitude:
w_plus/minus = w_parent +/- .02*abs(w_original)*direction[post_target].
This adjusts effective signed input, preserves every edge/topology/sign and original.5–1.5magnitude bound. Refuse any violation instead of clipping or reducing step. Actual parent ratios.776779–1.242496 leave room. Both candidates come from the SAME completed parent, with no mixing of separately trained banks. No joint command or leg ordering is generated.

Create new weights-proposed-{plus,minus}.npz exclusively; retain original references/topology/target/configuration fields and reset learner activity/eligibility/reward moments to zero. The native weights-only loader verifies each proposal. Nonplastic graph weights must exactly equal original values. Sidecars explicitly label untrained proposals/external episodic selection, not a completed native training birth. No parent overwrite, training-last/validated-best pointer update or automatic model promotion.

## Prospective physical protocol and selection

Twelve full native frozen5s/1000chunk jobs: for each reused development seed10006/10007/10010/10011 run parent,plus,minus. CPU one world, no within-episode learning/exploration/current injection/external senses. Parent arrays must exactly replay the historical S1 parent before either proposal on each seed. The native evaluator enforces matched body/core/packages/senses/seeds/horizon and unchanged within-episode weights. Source/parent/proposal hashes are checked before every job.

Fixed development screen, all conditions required:
- EACH seed: candidate standing>=.90 AND>=parent; candidate slip<=parent.
- Median paired gains in heading-forward AND net-forward distance>=.25mm; both candidate absolute forward medians must be positive. Becoming less backward is not a forward-progress pass.
- Median total leg cycles and tripod-coordination score must not decrease.
All conditions remain mandatory even if another metric improves. Report every case, regime and failed condition. This screen is NOT proof of stable walking, balance recovery, biological learning or environmental skill.

If neither proposal passes, reject this generation and run no fresh followup. If one/both pass, choose the highest min(median heading gain,median net gain), fixed tie priorityplus. ONLY that selection receives parent/selected pairs on fresh DEVELOPMENT10020/10021, with exactly the same absolute/paired/median gates. No fallback candidate or changed scale after seeing fresh outcomes. No reserved final10/11/200/201/202 is used. At most16physics jobs. A candidate passing both screens is provisional for review; no deployment/curriculum/model promotion.

This is ONE random direction and one antithetic pair, not an estimate of the optimizer's general efficacy. Selecting against reused development probes can overfit; two fresh probes do not supply final validation. No other directions/scales are authorized by a failure.

## Resource, provenance and stop contract

One canonical lease in PRIMARY gpu-pilot/outputs/codex-training-host.lock, no parallel heavy job. Fresh RAM>=2GiB and no unrelated Python before each worker.1800s aggregate INCLUDING graph/proposal setup;300s hard per worker including startup, no launch with<300s remaining. The already-tested Windows Job Object assigns a suspended worker before resume, contains descendants and verifies the entire job empty before releasing the owned lease. Retain lease on unverified cleanup.

Stop immediately on timeout/error/incomplete/nonfinite/mismatched context/source/bank/replay/resource failure. Preserve all partial evidence separately, no retries/retuning/additional directions or seeds. A scientific screen failure remains an ordinary completed rejection, not a technical failure. Data are the four SHA-verified licensed files; caches remain private and excluded from Git/evidence export.

Nine final focused checks passed in.73s: antithetic/sign/reference/zero preservation, rejection of boundary crossing instead of clipping, native checkpoint import and trace reset, exclusive output/parent preservation, hard per-seed retention gates, missing/nonfinite failure, and rejection of backward progress. Existing watchdog checks already verify parent-exits/descendant cleanup. Runner compiles. These are software checks, not physical outcomes.

Prepared-layout reproduction: restore scripts/es1_proposal.py and scripts/run_episodic_search_es1.py into ROOT/work, using the existing ROOT/work/Noesis-Sensory exact runtime, Noesis-Core, NoesisData, B2/S1 evidence and isolated environment. Invoke python ROOT/work/run_episodic_search_es1.py --root ROOT once with a new output directory outputs/episodic-search-es1-20261106. Publish final source commit and fresh allocation separately before launch.

Code/protocol/compact results in Git; named checkpoints/body arrays/reports/receipts in SHA-manifest artifact archives, excluding licensed raw tables/derived graph caches/private lease/environment. Release upload remains pending invalid local CLI auth/logged-out browser; never call a local archive a published asset.
