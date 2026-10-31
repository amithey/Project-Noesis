# Learning & memory retention audit — contract (#19)

Status: first increment. Owner: Claude (learning & memory retention). Coordinator: Codex.

## What this is

`noesis_gpu.retention_audit.compare_checkpoints(parent_path, child_path)` is an
**offline, read-only** audit that answers a narrow question:

> Did the `child` NPZ checkpoint *retain* the learned motor synapses of the
> `parent` checkpoint, on top of the same frozen reference network?

It never mutates a weight bank, never loads a connectome, never builds an LIF
network and never runs physics. It is an **audit, not a resume**: it does not —
and cannot from the checkpoint alone — prove that the child reproduces the
parent's behaviour. It makes **no success-threshold and no walking/locomotion
claim**.

## Checkpoint layout audited

The audit targets the layout written by `MotorPlasticity.save`
(`src/noesis/plasticity.py`):

| key | kind | role |
| --- | --- | --- |
| `topology` | scalar string | hash of `(n, indptr, indices)`; identifies the frozen graph |
| `motor_indices` | 1-D int | the selected motor/readout neurons |
| `initial_weights` | 1-D float | **original reference** signed edge weights (the frozen connectome values the learner was built from) |
| `parameters` | float[3] | `[learning_rate, tau_eligibility_ms, max_change]` |
| `weights` | 1-D float | **learned** edge weights (same shape as `initial_weights`) |
| `activity`, `eligibility`, `reward_baseline` | — | learning scratch |
| `version`, `rule`, `normalize_reward_error`, `bias_correct_reward_error` | — | metadata (optional for this audit) |

### Retained vs. reset — the key distinction

`MotorPlasticity.load(..., restore_traces=False)` (the default) **resets**
`activity`, `eligibility` and `reward_baseline` for a fresh brain/body episode.
They are therefore *not* retained synaptic memory and the audit judges
retention **independently of them** — two checkpoints with identical `weights`
but different eligibility/baseline are reported as fully retained.

- **Retained reference** (must match between parent and child): `topology`,
  `motor_indices`, `initial_weights`.
- **Retained learned state** (may legitimately differ): `weights`, bounded to
  `|initial_weights|` × (1 ± `max_change`) with signs preserved.
- **Reset state** (ignored): `activity`, `eligibility`, `reward_baseline`, plus
  any neural/membrane or physics/body state — which this checkpoint format
  does not even store.

## Mandatory compatibility metadata — exact contract

The audit **requires** `topology`, `motor_indices`, `initial_weights`,
`parameters` and `weights` to be present in **both** checkpoints. This is
mandatory, and here is exactly why: a "retained" learned weight only has
meaning *relative to a shared frozen reference*. Without `initial_weights` we
cannot tell a retained edge from an edge of a different network; without
`topology`/`motor_indices` we cannot tell the two checkpoints describe the same
graph and the same selected synapses; without `parameters` we do not know the
`max_change` that defines the legal magnitude band. A missing one is a hard
`ValueError`, not a silent pass.

`rule` is optional: when absent it defaults to `"hebbian"` (matching
`noesis_gpu.controller.import_checkpoint`). The audit reports `rule` for parent
and child and whether they match, but a mismatch is informational, not fatal —
it does not change which weights are retained.

## API

```python
report = compare_checkpoints(parent_path, child_path)
```

- Loads both files with `np.load(..., allow_pickle=False)`.
- Returns a **JSON-serializable** `dict` (plain `str`/`int`/`float`/`bool`/
  `list`/`dict`; verified by a `json.dumps` round-trip test).
- Raises a `ValueError` naming the offending checkpoint and reason for any
  rejection (see below).

### Rejections (clear `ValueError`)

- Unreadable / non-NPZ / pickled payload (`allow_pickle=False`).
- Missing required array, wrong rank/dtype, or `weights`/`initial_weights`
  shape mismatch.
- Non-finite values in `initial_weights` or `weights`.
- `max_change` outside `0 ≤ max_change < 1`.
- A learned weight whose **sign differs** from its reference (explicit-zero
  reference edges must stay exactly zero).
- A learned weight **outside** the `±max_change` (±50% for a pilot checkpoint)
  magnitude band around its reference.
- **Incompatible topology** (different graphs).
- **Incompatible reference** (`initial_weights` differ) or **incompatible
  `motor_indices`** (different selected synapses).

These mirror the invariants `MotorPlasticity.load` enforces, evaluated offline
without mutation.

### Report contents

- `keys`: the key set of each checkpoint, the required-retained keys, and which
  reset-state keys are present (and ignored).
- `shapes`, `topology`, `motor_indices`, `initial_weights`: compatibility facts.
- `parameters`: both triples, named fields, and a `match` flag.
- `rule`: both values and a `match` flag.
- `weights`:
  - `unchanged_fraction` — fraction of edges identical between parent and child,
  - `changed_count` — number of edges that differ,
  - finite norms: `parent_norm`, `child_norm`, `difference_norm`
    (‖child − parent‖), and deviation-from-initial norms,
  - `relative_change` — per-edge `|Δ| / |initial|` summaries (max vs. initial
    for each, mean/max child-vs-parent),
  - `sign_preserved` / `within_reference_bounds` flags (always `True` on a
    returned report, since violations raise instead).

There is deliberately **no** "pass/fail retention score", no threshold, and no
behavioural claim. A caller decides what unchanged fraction or norm difference
is meaningful for its experiment.

## Assumptions & limitations

- Reads only the two checkpoint files; no other repo paths, no network.
- `topology` is an opaque string compared for equality only; the audit does not
  re-derive it from a connectome.
- Bounds use the **child's/parent's own** `max_change` from `parameters` and a
  `1e-9` relative slack so a weight sitting exactly on the ±`max_change` edge is
  accepted. A pilot checkpoint carries `max_change = 0.5` (±50%).
- `weights`/`initial_weights` are treated as aligned 1-D arrays in the same edge
  order; the audit asserts equal shape and equal `initial_weights`, but cannot
  detect a reordering that preserves both.
- Checkpoint equality of learned weights does **not** imply equal post-resume
  behaviour, because membrane/body/RNG/simulation-time state is neither stored
  nor restored. This audit intentionally makes no exact-resume assertion.
- First increment: a practical structural/statistical audit, not a claim of
  biological memory fidelity.

## Rollout-summary retention comparison (round two)

`noesis_gpu.retention_audit.compare_rollout_summaries(before, after)` complements
the checkpoint audit. Where `compare_checkpoints` inspects *stored weights*, this
compares two *rollout evaluation summaries* — a `before` **parent** rollout and
an `after` **candidate** rollout — to describe how locomotion behaviour moved.
It never runs physics, never loads a connectome and never mutates its inputs; it
reads already-computed summary dicts (of the shape produced by
`noesis_gpu.evaluate.summarize`, with the context/provenance fields below
attached by the caller).

### Mandatory matched context — exact contract

Both summaries MUST carry, and MUST agree on, this evaluation context by
**strict equality**; a missing key raises `missing comparison metadata`, a
differing value raises `mismatched comparison metadata`. Two rollouts run under
different timing are not comparable, so we refuse rather than silently compare.

| key | kind | match rule |
| --- | --- | --- |
| `seed` | int ≥ 0 (no bool) | strict equality |
| `duration_s` | float > 0, finite | strict equality |
| `body` | str | strict equality (body setup) |
| `control_ms` | float > 0, finite | strict equality (control setup) |

### Mandatory frozen evaluation metadata — exact contract

A matched behavioural comparison is only meaningful across a **frozen**
evaluation (weights held fixed, no learning/exploration during the rollout).
Both summaries MUST carry these keys with exactly these values, or a
`ValueError` is raised (`missing frozen evaluation metadata` /
`frozen evaluation metadata ...`):

| key | required value |
| --- | --- |
| `status` | `"completed_transport_pilot"` |
| `learning` | `False` |
| `motor_exploration` | `0` (integer; a bool is rejected) |
| `weights_unchanged` | `True` |

### Provenance — source must match, weight may differ

Both summaries must **supply** `source_sha256` and `weight_sha256`; a missing
one raises `missing provenance`. These are treated **differently**:

- `source_sha256` identifies the evaluation source (fixed body/control/eval code
  path). A matched comparison **requires the same source**, so a differing
  `source_sha256` raises `mismatched provenance` (reported `source_sha256_match`
  is therefore always `True` on a returned report).
- `weight_sha256` is **expected to differ** between parent and candidate (that is
  the point of the comparison); the report records both values and
  `weight_sha256_match` but never requires equality.

### Required metrics

Both summaries must provide finite `standing_fraction` (in `[0, 1]`),
`heading_forward_distance_mm`, `net_forward_displacement_mm`, non-negative
`mean_stance_slip_mm_s`, and a `leg_step_counts` mapping. The mapping must hold
**exactly the six canonical legs** `LF LM LH RF RM RH`, each with an integer,
non-negative count (bools and fractional counts are rejected) so a fabricated or
partial leg set cannot inflate the `>= 4`-legs gate. A missing metric key raises
`missing rollout metric data`; a non-finite value raises.

### Report contents

- `context`, `frozen_evaluation`, `provenance`: the matched context, the frozen
  evaluation metadata snapshot, and the hashes with match flags (`source` always
  matching, `weight` possibly differing).
- `standing`, `heading`, `net_forward_displacement`, `stance_slip`: each a
  `{before, after, delta}` block (delta = candidate − parent).
- `thresholds` and `threshold_checks`: per-rollout pass/fail for each
  engineering gate, reported for **both** parent and candidate. Each check
  carries a `role` of `"primary"` or `"sensitivity"`.

  **Primary gates** (decide the headline verdict):
  - `standing_fraction >= 0.90`
  - `heading_forward_distance_mm >= 0.25` — **preregistered primary** forward
    metric (distance along the commanded heading)
  - `mean_stance_slip_mm_s <= 10.0`
  - `>= 2` step cycles on `>= 4` legs

  **Sensitivity gate** (reported, never decides the primary verdict):
  - `net_forward_displacement_mm >= 0.25` — net straight-line displacement

  The two forward metrics are **not interchangeable** and **both fields are
  required**, so the primary heading gate can never be silently replaced by the
  net displacement metric. `primary_gate_names` and `sensitivity_gate_names`
  list the roles explicitly.
- `candidate_meets_all_thresholds` / `parent_met_all_thresholds` are decided by
  the **primary** gates only; `candidate_meets_sensitivity_thresholds` /
  `parent_met_sensitivity_thresholds` report the sensitivity gate separately.
- `regressed_gates` (any gate the parent passed but the candidate failed) and
  `regressed_primary_gates` (that subset restricted to primary gates).

### What it is **not**

Threshold pass/fail is a per-rollout engineering gate, **not** a retention
score. The comparison makes **no** claim of memory retention, **no** claim of
forgetting, and **no** claim of statistical significance — a single matched pair
proves none of those. A differing `weight_sha256` does not by itself imply a
behavioural change; it is reported for traceability only (`source_sha256` must
match or the comparison is refused).

## Tests

`gpu-pilot/tests/test_retention_audit.py` builds tiny synthetic NPZ files (no
connectome/physics) covering: full retention, bounded learned change (with
quantities), reset-state differences ignored, the ±50% boundary, and rejection
of missing/non-finite/sign-flipped/out-of-bounds weights and incompatible
topology/reference/motor-indices, plus a non-mutation (byte-identical files)
check and JSON serializability. For `compare_rollout_summaries` it adds
synthetic stable, regressed, incompatible-context, missing-context,
missing-provenance, missing-metric and non-finite cases, plus two cases
isolating the primary/sensitivity split (heading passes while net fails, and
net passes while heading fails), a missing-heading case, JSON serializability
and an input non-mutation check. Focused repair regressions additionally cover:
a differing `source_sha256` rejected while a differing `weight_sha256` is
permitted and reported; fabricated/partial leg sets, bool and fractional leg
counts rejected; missing/wrong frozen metadata (including `motor_exploration`
rejecting a bool) rejected; strict-equality timing (sub-`isclose` `control_ms`
drift now fatal); and negative-seed / non-positive-duration rejection.
