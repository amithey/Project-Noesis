"""usage: run_mechanism_control.py <core src> <out.json>   (registered in protocol.md; 16 seeds 0..15)"""
import json
import sys

sys.path[:0] = [sys.argv[1], "gpu-pilot/src"]
from noesis_gpu import mechanism_control as mc

result = mc.run_control(range(16), kappa=0.2, lr=0.2)
json.dump(result, open(sys.argv[2], "w"), indent=1)
print(json.dumps({"summary": result["summary"], "verdict": result["verdict"]}, indent=1))
