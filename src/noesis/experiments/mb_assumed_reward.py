"""ASSUMED reward: does the mushroom-body learning machinery work when reward is given to it directly? (SHORTCUT, not a connectome result)

``mb_foraging_conditioning`` showed that sugar-receptor drive does not reach the dopamine cells through the connectome (also on the FAFB
mushroom body), so hunger and sugar had nothing to act on. Here the missing route is replaced by an explicit assumption:

    appetitive-reward dopamine cells = the PAM types ``PAM01``-``PAM08`` (an assumed subset, chosen by name, not found by the model);
    they are driven at ``pam_hz * contact * hunger_gain`` Hz, contact = 1 while sugar is touched, hunger_gain = ``sensory_gain("sugar")``
    (fed 1.0, hungry 1.8 — hunger scaling the reward is itself an assumption).

Training is four 500 ms blocks (``*_gap`` schedules: eight 250 ms blocks). Odour A is on in blocks 0 and 2 for every condition (same exposure); the reward is on in
  paired    blocks 0, 2 (with odour A)
  unpaired  blocks 1, 3 (same reward amount, odour A absent: the shuffled-contingency control)
  none      never
Odour B is never presented in training. MBON responses to A and B (plasticity frozen) before and after give the learning index
(dA - dB) / before_A (negative = A weakened relative to B). Conditioned learning needs paired < none and paired < unpaired, paired by seed.
Hungry/none equals fed/none by construction (hunger only scales the reward), so it is not run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from noesis.drives import HomeostaticState
from noesis.experiments.mb_conditioning import _run
from noesis.experiments.mb_foraging_conditioning import (
    add_brain_arguments,
    build_brain,
    learning_index,
    paired_effect,
)
from noesis.lif import LIFNetwork, LIFParams
from noesis.mushroom_body import DopamineGatedPlasticity, GradedAPL

STATES = {"fed": 1.0, "hungry": 0.2}
ASSUMED_PAM_TYPES = tuple(f"PAM0{i}" for i in range(1, 9))
SCHEDULES = {  # per 500 ms block: (odour A on, reward on)
    "paired": [(1, 1), (0, 0), (1, 1), (0, 0)],
    "unpaired": [(1, 0), (0, 1), (1, 0), (0, 1)],
    "none": [(1, 0), (0, 0), (1, 0), (0, 0)],
    # eight 250 ms blocks; the unpaired reward comes >= 1.25 s after the odour, beyond the 500 ms Kenyon-cell trace
    "paired_gap": [(1, 1), (1, 1)] + [(0, 0)] * 6,
    "none_gap": [(1, 0), (1, 0)] + [(0, 0)] * 6,
    "unpaired_gap": [(1, 0), (1, 0)] + [(0, 0)] * 4 + [(0, 1), (0, 1)],
}


def assumed_reward_cells(neurons, dan: np.ndarray) -> np.ndarray:
    """Indices (within the dopamine population ``dan``) of the assumed appetitive PAM types."""
    ctype = neurons["Primary Cell Type"].fillna("").to_numpy()[dan]
    return dan[np.isin(ctype, ASSUMED_PAM_TYPES)]


def run_one(W, pops, odor, reward_idx, state: str, schedule: str, seed: int, a) -> dict:
    net = LIFNetwork(W, LIFParams(dt=0.5, da=1.0), seed=seed)
    plast = DopamineGatedPlasticity(net, pops, eta=a.eta)
    apl = GradedAPL(net, pops, gain=a.apl_gain) if a.apl_gain > 0 else None

    def test():
        out = {}
        for name, idx in odor.items():
            (_, mbon, _), _ = _run(net, plast, pops, a.test_ms, idx, a.odor_hz, 400.0, learn=False, apl=apl)
            out[name] = mbon
            _run(net, plast, pops, 300.0, None, 0.0, 0.0, learn=False, apl=apl)
        return out

    before = test()
    pam_hz = a.pam_hz * HomeostaticState(energy=STATES[state]).sensory_gain("sugar")
    dan_reward, dan_other = [], []
    for odour_on, reward_on in SCHEDULES[schedule]:
        idx = [odor["A"]] * odour_on + [reward_idx] * reward_on
        rate = [np.full(len(odor["A"]), a.odor_hz)] * odour_on + [np.full(len(reward_idx), pam_hz)] * reward_on
        (_, _, dan), _ = _run(net, plast, pops, a.train_ms / len(SCHEDULES[schedule]), np.concatenate(idx) if idx else None,
                              np.concatenate(rate) if rate else 0.0, 400.0, learn=True, apl=apl)
        (dan_reward if reward_on else dan_other).append(dan)
    after = test()
    return {"before": before, "after": after, "index": learning_index(before, after), "pam_hz": pam_hz,
            "dan_hz_reward_blocks": float(np.mean(dan_reward)) if dan_reward else None,
            "dan_hz_other_blocks": float(np.mean(dan_other)) if dan_other else None,
            "weight_fraction": float(plast.mean_weight_fraction)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("outputs/mb_assumed_reward.json"))
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--first-seed", type=int, default=0)
    ap.add_argument("--train-ms", type=float, default=2000.0)
    ap.add_argument("--test-ms", type=float, default=600.0)
    ap.add_argument("--eta", type=float, default=5e-4)
    ap.add_argument("--odor-hz", type=float, default=80.0)
    ap.add_argument("--pam-hz", type=float, default=20.0, help="drive rate of the assumed reward cells when fed")
    ap.add_argument("--glomeruli", type=int, default=5)
    ap.add_argument("--conditions", default="fed/paired,fed/unpaired,fed/none,hungry/paired,hungry/unpaired")
    add_brain_arguments(ap)
    a = ap.parse_args()

    cn, pops, W = build_brain(a)
    reward_idx = assumed_reward_cells(cn.neurons, pops.dan)
    print(f"assumed reward cells: {len(reward_idx)} of {len(pops.dan)} dopamine cells ({', '.join(ASSUMED_PAM_TYPES)})", flush=True)
    glom = cn.neurons.iloc[pops.orn]["Primary Cell Type"].fillna("?").to_numpy()
    results = {}
    for seed in range(a.first_seed, a.first_seed + a.seeds):
        types = np.random.default_rng(seed).permutation(np.unique(glom[glom != "?"]))
        odor = {n: pops.orn[np.isin(glom, types[i * a.glomeruli:(i + 1) * a.glomeruli])] for i, n in enumerate("AB")}
        for key in a.conditions.split(","):
            state, schedule = key.split("/")
            r = run_one(W, pops, odor, reward_idx, state, schedule, seed, a)
            results.setdefault(key, []).append(r)
            print(f"seed {seed} {key:15s} index {r['index']:+.3f} | MBON A {r['before']['A']:.2f}->{r['after']['A']:.2f} "
                  f"B {r['before']['B']:.2f}->{r['after']['B']:.2f} | DAN in reward blocks {r['dan_hz_reward_blocks']} "
                  f"| weights {r['weight_fraction']:.4f}", flush=True)
    idx_of = lambda key: [r["index"] for r in results[key]]
    summary = {}
    for suffix in ("", "_gap"):
        for state in STATES:
            for ctrl in ("none", "unpaired"):
                paired, other = f"{state}/paired{suffix}", f"{'fed' if ctrl == 'none' else state}/{ctrl}{suffix}"
                if paired in results and other in results:
                    summary[f"{paired} - {other}"] = paired_effect(idx_of(paired), idx_of(other))
        if f"hungry/paired{suffix}" in results and f"fed/paired{suffix}" in results:
            summary[f"hungry/paired{suffix} - fed/paired{suffix}"] = paired_effect(idx_of(f"hungry/paired{suffix}"),
                                                                                   idx_of(f"fed/paired{suffix}"))
    print("paired-by-seed learning-index differences (negative = learning):", json.dumps(summary))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({"runs": results, "summary": summary, "assumed": list(ASSUMED_PAM_TYPES)}, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
