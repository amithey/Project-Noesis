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

## Tests

`gpu-pilot/tests/test_retention_audit.py` builds tiny synthetic NPZ files (no
connectome/physics) covering: full retention, bounded learned change (with
quantities), reset-state differences ignored, the ±50% boundary, and rejection
of missing/non-finite/sign-flipped/out-of-bounds weights and incompatible
topology/reference/motor-indices, plus a non-mutation (byte-identical files)
check and JSON serializability.
