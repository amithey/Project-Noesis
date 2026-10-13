"""Does the brain really respond? A stimulus battery with predictions fixed before running.

Each trial: 0.5 s rest, 1 s stimulus, 1 s recovery; repeated over seeds, on the raw and the corrected
(FAFB-scaled) BANC brain. Predictions (from Shiu et al. 2024 and fly taste/olfaction biology):

  P1  sugar raises MN9 (the feeding motor neuron) above the no-stimulus control
  P2  water raises MN9
  P3  bitter does not drive MN9 as sugar does (bitter MN9 < half of sugar MN9)
  P4  bitter added to sugar lowers MN9 (sugar+bitter < sugar)
  P5  sugar raises the reward dopamine neurons (PAM) above control
  P6  bitter raises the punishment dopamine neurons (PPL1) more than sugar does
  P7  an odour drives the projection neurons of its own glomeruli more than the others
  P8  after the stimulus stops, activity returns toward rest (recovery <= 2x rest + 50 neurons)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from noesis.connectome import DEFAULT_NT_SIGN, corrected_scale, load_cached
from noesis.lif import LIFNetwork, LIFParams


def _groups(cn, seed: int) -> dict:
    n = cn.neurons
    fn = n["Function"].fillna("")
    ct = n["Primary Cell Type"].fillna("")
    sub = n["Sub Class"].fillna("")
    where = lambda m: np.flatnonzero(np.asarray(m))
    orn = where(sub.str.contains("olfactory_receptor"))
    glom = ct.iloc[orn].to_numpy()
    types = np.random.default_rng(seed).permutation(np.unique(glom[glom != ""]))[:5]
    names = [t.replace("ORN_", "") for t in types]
    pn = where(n["Class"].fillna("").eq("antennal_lobe_projection_neuron"))
    on = np.array([any(t.startswith(g + "_") for g in names) for t in ct.iloc[pn]])
    return {
        "sugar": where(fn.str.contains("sugar")), "water": where(fn.str.contains("water")),
        "bitter": where(fn.str.contains("bitter")), "odour": orn[np.isin(glom, types)],
        "mn9": where(ct.eq("MN9")), "pam": where(ct.str.startswith("PAM")),
        "ppl1": where(ct.str.startswith("PPL1")), "pn_on": pn[on], "pn_off": pn[~on],
    }


def _trial(W, stim, seed: int, rate: float):
    net = LIFNetwork(W, LIFParams(dt=0.5, da=1.0), seed=seed)
    windows = {}
    for name, ms, drive in (("rest", 500.0, None), ("stim", 1000.0, stim), ("recovery", 1000.0, None)):
        _, i = net.run(ms, drive, rate, 400.0) if drive is not None and len(drive) else net.run(ms)
        windows[name] = (np.bincount(i, minlength=W.shape[0]) / (ms / 1000.0), len(np.unique(i)))
    return windows


def run_brain(cn, seeds: list[int]) -> dict:
    out = {}
    for seed in seeds:
        g = _groups(cn, seed=0)  # same odour across seeds; seeds vary the network noise
        conditions = {"none": np.zeros(0, int), "sugar": g["sugar"], "water": g["water"],
                      "bitter": g["bitter"], "sugar+bitter": np.concatenate([g["sugar"], g["bitter"]]),
                      "odour": g["odour"]}
        for cond, stim in conditions.items():
            w = _trial(cn.W, stim, seed, 80.0 if cond == "odour" else 100.0)
            hz, active = w["stim"]
            rec = out.setdefault(cond, {k: [] for k in ("mn9", "pam", "ppl1", "pn_on", "pn_off", "rest_active",
                                                        "stim_active", "recovery_active")})
            for k in ("mn9", "pam", "ppl1", "pn_on", "pn_off"):
                rec[k].append(float(hz[g[k]].mean()))
            rec["rest_active"].append(w["rest"][1])
            rec["stim_active"].append(active)
            rec["recovery_active"].append(w["recovery"][1])
            print(f"    seed {seed} {cond:13s} MN9 {hz[g['mn9']].mean():6.1f} | PAM {hz[g['pam']].mean():5.2f} | "
                  f"PPL1 {hz[g['ppl1']].mean():5.2f} | PN on/off {hz[g['pn_on']].mean():5.1f}/"
                  f"{hz[g['pn_off']].mean():5.1f} | active rest/stim/recovery {w['rest'][1]}/{active}/"
                  f"{w['recovery'][1]}", flush=True)
    return out


def judge(r: dict) -> list[tuple[str, bool, str]]:
    m = lambda c, k: float(np.mean(r[c][k]))
    res = [
        ("P1 sugar -> MN9", m("sugar", "mn9") > m("none", "mn9") + 2, f"{m('none', 'mn9'):.1f} -> {m('sugar', 'mn9'):.1f} Hz"),
        ("P2 water -> MN9", m("water", "mn9") > m("none", "mn9") + 2, f"{m('none', 'mn9'):.1f} -> {m('water', 'mn9'):.1f} Hz"),
        ("P3 bitter < half of sugar at MN9", m("bitter", "mn9") < 0.5 * max(m("sugar", "mn9"), 1e-9),
         f"bitter {m('bitter', 'mn9'):.1f} vs sugar {m('sugar', 'mn9'):.1f} Hz"),
        ("P4 sugar+bitter < sugar at MN9", m("sugar+bitter", "mn9") < m("sugar", "mn9"),
         f"{m('sugar+bitter', 'mn9'):.1f} vs {m('sugar', 'mn9'):.1f} Hz"),
        ("P5 sugar -> PAM", m("sugar", "pam") > m("none", "pam") + 0.2, f"{m('none', 'pam'):.2f} -> {m('sugar', 'pam'):.2f} Hz"),
        ("P6 bitter -> PPL1 more than sugar", m("bitter", "ppl1") > m("sugar", "ppl1") + 0.2,
         f"bitter {m('bitter', 'ppl1'):.2f} vs sugar {m('sugar', 'ppl1'):.2f} Hz"),
        ("P7 odour: own glomeruli > others", m("odour", "pn_on") > 2 * m("odour", "pn_off"),
         f"on {m('odour', 'pn_on'):.1f} vs off {m('odour', 'pn_off'):.1f} Hz"),
    ]
    worst = max((np.mean(r[c]["recovery_active"]) - 2 * np.mean(r[c]["rest_active"]) - 50, c) for c in r)
    rec_ok = worst[0] <= 0
    res.append(("P8 recovery after stimulus", rec_ok,
                (f"worst: {worst[1]} {np.mean(r[worst[1]]['recovery_active']):.0f} active vs rest "
                f"{np.mean(r[worst[1]]['rest_active']):.0f}")))
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--out", type=Path, default=Path("outputs/response_battery.json"))
    ap.add_argument("--brains", default="raw,corrected",
                    help="comma list of: raw, corrected (brain FAFB + nerve cord MANC), "
                         "corrected-silent-monoamines (DA/5-HT/OA/TA neither excite nor inhibit)")
    a = ap.parse_args()
    raw, cache = a.data / "raw", a.data / "processed"
    factors = corrected_scale(a.data)
    silent = dict(DEFAULT_NT_SIGN, DA=0, SER=0, OCT=0, TYR=0)
    options = {"raw": ("raw BANC", {}),
               "corrected": ("corrected BANC", {"region_scale": factors}),
               "corrected-silent-monoamines": ("corrected BANC, monoamines silent",
                                               {"region_scale": factors, "nt_sign": silent})}
    seeds = list(range(a.seeds))
    report = {}
    for label, kw in (options[b] for b in a.brains.split(",")):
        print(f"=== {label}", flush=True)
        r = run_brain(load_cached(raw, cache, **kw), seeds)
        verdicts = judge(r)
        for name, ok, detail in verdicts:
            print(f"  {'PASS' if ok else 'FAIL'}  {name:36s} {detail}", flush=True)
        report[label] = {"trials": r, "verdicts": [(n, bool(o), d) for n, o, d in verdicts]}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print("saved", a.out)


if __name__ == "__main__":
    main()
