"""Does an odour reach the brain, and does it reach the cells that could steer the fly?

Brain only (no body). For every olfactory channel in the neuron table, drive its receptor neurons on
the left side, the right side, or both, for 0.7 s after 0.3 s of rest, over several seeds, and measure
the response of the mushroom body, the dopaminergic cells and the descending neurons (the brain's
output to the nerve cord). Controls: no stimulus, and the same number of receptor neurons chosen at
random from other channels. A descending neuron counts as responsive only if its mean rate exceeds
the matched no-stimulus mean by ``--min-delta-hz`` AND exceeds the random-channel control.

This audits wiring and sign. It does not show the fly would walk toward or away from anything.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from noesis.embodiment import load_training_brain
from noesis.lif import LIFNetwork, LIFParams

HABITAT_CHANNELS = ("fruity_volatile", "yeasty_volatile", "alcoholic_fermentation_volatile",
                    "decaying_fruit_volatile", "aversive_volatile", "plant_matter_volatile")
# Present in the neuron table but not in the habitat; animal_matter is the closest label to carrion.
EXTRA_CHANNELS = ("animal_matter_volatile", "carbon_dioxide_volatile", "pheromone_volatile")
CHANNELS = HABITAT_CHANNELS + EXTRA_CHANNELS


def asymmetry_index(left: float, right: float) -> float | None:
    """(left - right) / (left + right) of two non-negative responses; None if both are ~0."""
    if left < 0 or right < 0:
        raise ValueError("responses must be non-negative")
    total = left + right
    return None if total < 1e-9 else float((left - right) / total)


def responsive(stim_hz: np.ndarray, base_hz: np.ndarray, control_hz: np.ndarray,
               min_delta_hz: float) -> np.ndarray:
    """Per-neuron: stimulus mean beats both the no-stimulus mean (by min_delta) and the control mean."""
    return (stim_hz - base_hz >= min_delta_hz) & (stim_hz > control_hz)


def receptor_sides(neurons: pd.DataFrame) -> pd.Series:
    side = neurons["Nerve"].fillna("").str.extract(r"^(left|right)_")[0]
    return side.fillna(neurons["Soma side"])


def channel_groups(neurons: pd.DataFrame) -> dict[str, dict[str, np.ndarray]]:
    """channel -> {'left','right'} afferent olfactory receptor indices."""
    function = neurons["Function"].fillna("")
    orn = neurons["Flow"].eq("afferent") & neurons["Class"].eq("olfactory_receptor_neuron")
    side = receptor_sides(neurons)
    out = {}
    for channel in CHANNELS:
        has = function.str.contains(rf"(?:^|,){channel}(?:,|$)", regex=True)
        out[channel] = {s: np.flatnonzero((orn & has & side.eq(s)).to_numpy()) for s in ("left", "right")}
    return out


def readout_groups(neurons: pd.DataFrame) -> dict[str, np.ndarray]:
    ctype = neurons["Primary Cell Type"].fillna("")
    where = lambda m: np.flatnonzero(np.asarray(m))
    side = neurons["Soma side"]
    dn = neurons["Super Class"].eq("descending")
    return {"kc": where(ctype.str.startswith("KC")), "mbon": where(ctype.str.startswith("MBON")),
            "pam": where(ctype.str.startswith("PAM")), "ppl1": where(ctype.str.startswith("PPL1")),
            "dn": where(dn), "dn_left": where(dn & side.eq("left")), "dn_right": where(dn & side.eq("right")),
            "pn": where(neurons["Class"].fillna("").eq("antennal_lobe_projection_neuron"))}


def trial(W, drive: np.ndarray, seed: int, rest_ms: float, stim_ms: float, rate_hz: float, weight: float):
    """Return per-neuron stimulus-window rate (Hz) after a rest window on a fresh network."""
    net = LIFNetwork(W, LIFParams(dt=0.5, da=1.0), seed=seed)
    net.run(rest_ms)
    if len(drive):
        _, spikes = net.run(stim_ms, drive, rate_hz, weight)
    else:
        _, spikes = net.run(stim_ms)
    return np.bincount(spikes, minlength=W.shape[0]) / (stim_ms * 1e-3)


def _mean_over_seeds(W, drive, seeds, **kw) -> np.ndarray:
    return np.mean([trial(W, drive, s, **kw) for s in seeds], axis=0)


def audit(cn, seeds: list[int], min_delta_hz: float, rate_hz: float, weight: float,
          rest_ms: float, stim_ms: float, channels=CHANNELS) -> dict:
    groups, read = channel_groups(cn.neurons), readout_groups(cn.neurons)
    kw = {"rest_ms": rest_ms, "stim_ms": stim_ms, "rate_hz": rate_hz, "weight": weight}
    base = _mean_over_seeds(cn.W, np.zeros(0, np.int64), seeds, **kw)
    all_orn = np.unique(np.concatenate([g[s] for g in groups.values() for s in g]))
    rng = np.random.default_rng(0)
    report = {"seeds": seeds, "rate_hz": rate_hz, "weight": weight,
              "baseline": {k: float(base[v].mean()) for k, v in read.items()}, "channels": {}}
    for channel in channels:
        entry = {"n_left": len(groups[channel]["left"]), "n_right": len(groups[channel]["right"])}
        resp = {}
        for label, drive in (("left", groups[channel]["left"]), ("right", groups[channel]["right"]),
                             ("both", np.concatenate([groups[channel]["left"], groups[channel]["right"]]))):
            if not len(drive):
                resp[label] = None
                continue
            others = np.setdiff1d(all_orn, drive)
            ctrl_drive = rng.choice(others, size=min(len(drive), len(others)), replace=False)
            stim = _mean_over_seeds(cn.W, drive, seeds, **kw)
            ctrl = _mean_over_seeds(cn.W, ctrl_drive, seeds, **kw)
            dn = read["dn"]
            mask = responsive(stim[dn], base[dn], ctrl[dn], min_delta_hz)
            dl, dr = read["dn_left"], read["dn_right"]
            gain = lambda idx, s=stim: float(max(s[idx].mean() - base[idx].mean(), 0.0))
            resp[label] = {
                "delta_hz": {k: float(stim[v].mean() - base[v].mean()) for k, v in read.items()},
                "control_delta_hz": {k: float(ctrl[v].mean() - base[v].mean()) for k, v in read.items()},
                "dn_responsive": int(mask.sum()), "dn_total": len(dn),
                "dn_asymmetry_left_vs_right": asymmetry_index(gain(dl), gain(dr)),
            }
        entry["responses"] = resp
        report["channels"][channel] = entry
        both = resp.get("both")
        print(f"{channel:34s} ORN L/R {entry['n_left']:3d}/{entry['n_right']:3d} | "
              + (f"DN responsive {both['dn_responsive']:4d}/{both['dn_total']} "
                 f"PAM {both['delta_hz']['pam']:+5.2f} PPL1 {both['delta_hz']['ppl1']:+5.2f} Hz"
                 if both else "no receptors"), flush=True)
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--cache", type=Path, default=Path("outputs/sense-audit-cache"))
    ap.add_argument("--out", type=Path, default=Path("outputs/sense_audit.json"))
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--rate", type=float, default=80.0)
    ap.add_argument("--weight", type=float, default=400.0)
    ap.add_argument("--rest-ms", type=float, default=300.0)
    ap.add_argument("--stim-ms", type=float, default=700.0)
    ap.add_argument("--min-delta-hz", type=float, default=2.0)
    ap.add_argument("--brain", choices=("raw", "corrected"), default="corrected")
    ap.add_argument("--channels", default=",".join(CHANNELS))
    a = ap.parse_args()
    cn, _ = load_training_brain(a.data, a.cache, a.brain)
    print(f"brain={a.brain} neurons={cn.n} synaptic pairs={cn.W.nnz}", flush=True)
    report = audit(cn, list(range(a.seeds)), a.min_delta_hz, a.rate, a.weight, a.rest_ms, a.stim_ms,
                   tuple(a.channels.split(",")))
    report["brain"] = a.brain
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print("saved", a.out)


if __name__ == "__main__":
    main()
