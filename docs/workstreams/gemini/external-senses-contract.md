# External Senses Drive Adapter & Integration Contract

**Workstream**: External Senses and Environmental Response  
**Owner / Role**: Gemini via Antigravity (#20)  
**Deliverables**:
- Implementation: [`gpu-pilot/src/noesis_gpu/external_senses.py`](file:///C:/Users/heyma/OneDrive/Desktop/מדעי%20המחשב/פרוייקטים/Project%20Noesis/outputs/team-dev-gemini-r2/gpu-pilot/src/noesis_gpu/external_senses.py)
- Synthetic Tests: [`gpu-pilot/tests/test_external_senses.py`](file:///C:/Users/heyma/OneDrive/Desktop/מדעי%20המחשב/פרוייקטים/Project%20Noesis/outputs/team-dev-gemini-r2/gpu-pilot/tests/test_external_senses.py)
- Specification Contract: [`docs/workstreams/gemini/external-senses-contract.md`](file:///C:/Users/heyma/OneDrive/Desktop/מדעי%20המחשב/פרוייקטים/Project%20Noesis/outputs/team-dev-gemini-r2/docs/workstreams/gemini/external-senses-contract.md)

---

## 1. Scope and Architectural Purpose

This increment provides an **opt-in neural sensory-drive adapter** that bridges measured external scalar physical signals into the Poisson drive interface required by Noesis's Leaky Integrate-and-Fire network (`LIFNetwork`) and the core `merge_drive` contract.

### Explicit Non-Claims & Scientific Boundaries
- **Not an already-connected visual/chemical system**: This module provides the transport adapter and mathematical contract. It does not hardcode unverified biological mappings from fly anatomy to connectome IDs.
- **No invented connectome labels or target neurons**: All channel-to-neuron target mappings are caller-provided (`ChannelSpec`). Neuron indices must be verified against connectome metadata by the experiment caller.
- **No motor policy, reflexes, or gait generators**: This component operates strictly as an afferent sensory-drive generator. It does not compute motor commands or locomotion trajectories.
- **No artificial sensors claimed real**: Signals are treated as explicit measured scalar values (or simulation-provided sensor readings) with user-defined units and ranges.

---

## 2. Core Drive Contract Matching `merge_drive`

In the core Noesis architecture ([`src/noesis/environment_senses.py`](file:///C:/Users/heyma/OneDrive/Desktop/מדעי%20המחשב/פרוייקטים/Project%20Noesis/outputs/team-dev-gemini-r2/src/noesis/environment_senses.py#L242-L251)), sensory pathways are merged into network Poisson stimulation parameters via `merge_drive(parts)`:

```python
def merge_drive(parts):
    """Combine (indices, rates, weight) without duplicating overlapping neurons."""
    if not parts:
        return {}
    idx = np.concatenate([np.asarray(p[0], np.int64) for p in parts])
    if len(np.unique(idx)) != len(idx):
        raise ValueError('sensory pathways overlap')
    return {
        'drive_idx': idx,
        'drive_rate_hz': np.concatenate([p[1] for p in parts]),
        'drive_weight': np.concatenate([np.full(len(p[0]), p[2]) for p in parts]),
    }
```

The adapter provides three direct representations matching this contract:

1. **Drive Part Tuple** (`adapter.as_drive_part(signals, weight=None)`):
   - Returns `(drive_idx, drive_rate_hz, weight)`:
     - `drive_idx`: `np.ndarray` of `np.int64`, strictly unique target neuron indices in canonical ascending order.
     - `drive_rate_hz`: `np.ndarray` of `np.float64`, non-negative Poisson event rates in Hz.
     - `weight`: `float`, synaptic conductance weight (dimensionless).
   - Can be passed directly into `merge_drive([..., adapter.as_drive_part(signals)])`.

2. **Drive Dictionary** (`adapter.as_drive_dict(signals, weight=None)`):
   - Returns `{ 'drive_idx': ..., 'drive_rate_hz': ..., 'drive_weight': ... }`.
   - Matches the dictionary format expected by `net.run(..., **drive)`.

3. **Drive Dict Augmentation** (`adapter.augment_drive(base_drive, signals=None, weight=None)`):
   - Directly augments an existing `base_drive` dictionary (such as returned by `merge_drive`) with external sensory drive.
   - **Validation of `base_drive`**:
     - Must be a `Mapping` with required keys `drive_idx`, `drive_rate_hz`, `drive_weight`.
     - Validates shapes (all 1D arrays matching in length).
     - Validates indices (strictly integers, bounded by `0 <= idx < n_neurons`, non-negative, unique).
     - Validates rates (finite numeric values, non-negative rates >= 0.0 Hz).
     - Validates weights (finite numeric values, non-negative conductance weights >= 0.0).
   - **Overlap Enforcement**: Fails with `ValueError('sensory pathways overlap: ...')` if external channels target neurons already present in `base_drive`.
   - **Deterministic Disjoint Composition**: Disjoint neuron drives are composed deterministically via concatenation.
   - **Preservation of Caller Arrays**: Input arrays and dictionaries are never mutated.
   - **Zero-Signal / Disabled Guarantee**: When adapter is disabled or external signals evoke no drive, `base_drive` values are preserved exactly with NO spontaneous bias, random activity, or baseline drive.

---

## 3. Units, Bounds, and Limits

| Parameter / Field | Type | Units | Valid Range | Description |
| :--- | :--- | :--- | :--- | :--- |
| `n_neurons` | `int` | Count | `n_neurons > 0` | Total number of neurons in connectome graph. |
| `neuron_indices` | `tuple[int, ...]` | Index | `0 <= index < n_neurons` | Connectome target neuron IDs. Strictly integers (supports Python `int` and `np.integer`; rejects floats/booleans including `np.bool_` without truncation; arrays must be 1D). |
| `gain` | `float` | Hz / unit | Finite, `gain >= 0.0` | Transfer gain converting measured scalar to firing rate. |
| `min_signal` | `float` | Signal unit | Finite | Lower bound of expected sensor measurement. |
| `max_signal` | `float` | Signal unit | Finite, `>= min_signal` | Upper bound of expected sensor measurement. |
| `clamp` | `bool` | Flag | `True` or `False` | When `True`, clamps measurement to `[min_signal, max_signal]`. When `False`, out-of-bounds raises `ValueError`. |
| `threshold` | `float` | Signal unit | Finite | Signal at or below threshold produces `0.0 Hz`. |
| `default_weight` | `float` | Dimensionless | Finite, `>= 0.0` | Synaptic drive weight (default 400.0, matching `sensory_weight`). |
| `drive_rate_hz` | `np.float64` | Hz (1/s) | `rate >= 0.0` | Poisson stimulus rate injected into target neurons. |

---

## 4. Invariants and Behavioral Guarantees

### Zero Spontaneous Drive Guarantee
When the adapter is disabled (`enabled=False`) or when a signal is missing or at/below threshold:
- `drive_idx` and `drive_rate_hz` are **empty arrays** (length 0).
- No background bias, baseline firing, or random activity is injected into the network.
- When concatenated by `merge_drive`, empty arrays add 0 elements and cannot cause collisions.

### Missing and Unmapped Signal Policies
- **`missing_policy`**:
  - `'skip'` *(default)*: Missing channels produce NO drive for their target neurons (opt-in safety).
  - `'error'`: Raises `KeyError` if any configured channel is missing from the input mapping.
  - `'zero'`: Evaluates missing channels with scalar value `0.0`.
- **`unmapped_policy`**:
  - `'error'` *(default)*: Raises `KeyError` if unexpected channel keys are passed.
  - `'ignore'`: Extra signal keys in the input dictionary are ignored.

### Deterministic Multi-Channel Aggregation
When multiple channels target the same neuron (or a channel specifies redundant indices):
- Target indices are consolidated into unique entries.
- Rates are combined deterministically using `aggregation_mode`:
  - `'sum'` *(default)*: Rates add linearly.
  - `'max'`: Takes maximum rate across channels.
  - `'mean'`: Averages rates across active channels.
- Output arrays are sorted in strictly ascending neuron index order. This guarantees **direction-independent canonical output** regardless of channel registration order or dictionary key ordering.

### Input Immutability
Caller-provided dictionaries and numpy arrays are never modified in-place.

---

## 5. Integration Proposal with `FrozenBrain`

In [`gpu-pilot/src/noesis_gpu/controller.py`](file:///C:/Users/heyma/OneDrive/Desktop/מדעי%20המחשב/פרוייקטים/Project%20Noesis/outputs/team-dev-gemini-r2/gpu-pilot/src/noesis_gpu/controller.py), `FrozenBrain` manages sensory merging and network advancement:

```python
class FrozenBrain:
    def __init__(self, connectome, rest, lo, hi, seed=0, checkpoint=None,
                 bg_hz=3.0, bg_weight=40.0, sensory_weight=400.0, shared_weights=None,
                 external_adapter=None):
        ...
        self.external_adapter = external_adapter  # Optional ExternalSensoryAdapter

    def _current_drive(self, dt_ms):
        # Existing hook returns {}
        return {}

    def step(self, observation, dt_ms):
        from noesis.environment_senses import merge_drive
        ...
        # Option A: Merge via parts tuple
        parts = [
            (self.senses.idx, self.senses.drive(states), self.sensory_weight),
            (self.balance.idx, self.balance.drive(observation["up_body"], observation["omega"]),
             self.sensory_weight),
        ]
        if self.external_adapter is not None and self.external_adapter.enabled:
            ext_part = self.external_adapter.as_drive_part(observation.get("external_senses"))
            if len(ext_part[0]) > 0:
                parts.append(ext_part)
        drive = merge_drive(parts)

        # Option B: Direct drive augmentation on base_drive dictionary
        # base_drive = merge_drive(parts)
        # if self.external_adapter is not None:
        #     drive = self.external_adapter.augment_drive(base_drive, observation.get("external_senses"))
        ...
```

Per current work order boundaries, `controller.py` remains unmodified. When the coordinator schedules controller integration, either pattern enables clean plug-and-play wiring.

---

## 6. Assumptions and Limitations

1. **Neuron Target Allocation**: Caller must assign neuron indices that do not overlap with existing `SensoryMap` (proprioception) or `BalanceMap` pathways, or `merge_drive` will intentionally raise `ValueError('sensory pathways overlap')`.
2. **Poisson Drive Semantics**: Firing rates are converted to Poisson spike probabilities inside `LIFNetwork.run` using `prob = rate * dt * 1e-3`. Continuous graded current injection is not used for this adapter.
3. **No Dynamic Plasticity**: External sensory drive participates in network spiking dynamics; checkpoint weights remain frozen in accordance with pilot constraints.
