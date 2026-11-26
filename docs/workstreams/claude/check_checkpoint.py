import sys, hashlib, numpy as np
core, pilot, data, ckpt, cache = sys.argv[1:6]
sys.path[:0] = [core, pilot]
from noesis.embodiment import load_training_brain
from noesis_gpu.learning import LearningBrain
sha = hashlib.sha256(open(ckpt,'rb').read()).hexdigest()
print("sha256", sha)
from pathlib import Path
cn, _ = load_training_brain(Path(data), Path(cache))
rest, lo, hi = np.zeros(42), np.full(42,-.7), np.full(42,.7)
b = LearningBrain(cn, rest, lo, hi, checkpoint=ckpt, seed=0)
ref = b.learner.initial_weights
idx = b.learner.edge_indices
cur = b.net.data[idx]
print("targets", len(b.learner.motor_indices), "edges", len(idx))
print("sign preserved", bool(np.array_equal(np.sign(cur), np.sign(ref))), "nonzero", bool((cur!=0).all()))
r = np.abs(cur)/np.abs(ref)
print("ratio min/max", float(r.min()), float(r.max()), "changed", int((r!=1).sum()))
print("outside-bounds", int(((r<.5-1e-6)|(r>1.5+1e-6)).sum()))
print("learning_active", b.learning_active, "updates", b.updates)
