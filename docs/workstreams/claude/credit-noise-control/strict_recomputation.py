"""Recompute the three pair verdicts from the stored tables with the explicit rule (gpu-pilot/src/noesis_gpu/pair_decision.py).
Does not rewrite control-result.json / replication-result.json. usage: strict_recomputation.py <out.json>"""
import json
import sys

sys.path.insert(0, "gpu-pilot/src")
from noesis_gpu.pair_decision import classify_pair

D = "docs/workstreams/claude/credit-noise-control/"
rep = json.load(open(D + "replication-result.json", encoding="utf-8"))
out = {"note": "explicit same-probe/symmetric-tie rule applied to the stored tables; the registered first-use verdicts are unchanged", "pairs": {}}
for seed, pair in rep["pairs"].items():
    new = classify_pair(pair["table"]["real"], pair["table"]["control"])
    out["pairs"][seed] = {"registered_verdict": pair["verdict"], "strict_verdict": new["verdict"], **{k: new[k] for k in (
        "median_dheading", "median_dnet", "real_noninferior_probes", "control_noninferior_probes")}}
out["all_verdicts_unchanged"] = all(v["registered_verdict"] == v["strict_verdict"] for v in out["pairs"].values())
json.dump(out, open(sys.argv[1], "w"), indent=1)
print(json.dumps(out, indent=1))
