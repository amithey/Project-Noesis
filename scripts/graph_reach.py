"""Excitatory-edge reach (hop count) from taste or random afferents to dopamine/MN9/MBON targets.

Ad hoc graph check used in docs/findings-foraging-conditioning.md; counts neurons reachable, ignores weights
and inhibition. Run from the repo root with PYTHONPATH=src and an edited data path.
"""
import numpy as np, pandas as pd
from pathlib import Path
from scipy import sparse
from noesis.embodiment import load_training_brain
from noesis.mushroom_body import populations
D=Path("C:/Users/heyma/OneDrive/Desktop/מדעי המחשב/פרוייקטים/Project Noesis/data")
cn,_=load_training_brain(D,Path("outputs/selectivity-cache"),"corrected")
pops=populations(cn.neurons); n=cn.n
A=(cn.W>0).astype(np.float32).tocsr()   # excitatory edges, pre->post
f=cn.neurons["Function"].fillna(""); aff=cn.neurons["Flow"].eq("afferent")
pick=lambda t: np.flatnonzero((f.str.contains(t)&aff).to_numpy())
ct=cn.neurons["Primary Cell Type"].fillna("").to_numpy()
targets={"PAM":np.flatnonzero(np.char.startswith(ct.astype(str),"PAM")),"PPL1":np.flatnonzero(np.char.startswith(ct.astype(str),"PPL1")),"MN9":np.flatnonzero(ct=="MN9"),"MBON":pops.mbon}
rng=np.random.default_rng(0)
srcs={"sugar":pick("sugar"),"bitter":pick("bitter"),"water":pick("water")}
srcs["random"]=rng.choice(np.setdiff1d(np.flatnonzero(aff.to_numpy()),srcs["sugar"]),size=len(srcs["sugar"]),replace=False)
for name,idx in srcs.items():
    v=np.zeros(n,np.float32); v[idx]=1; seen=v.copy(); line=[]
    cur=v
    for hop in range(1,6):
        cur=(A.T@cur>0).astype(np.float32); seen=np.maximum(seen,cur)
        line.append(f"h{hop}: "+" ".join(f"{t}={int(seen[ix].sum())}/{len(ix)}" for t,ix in targets.items()))
    print(name,len(idx)); print("  "+"\n  ".join(line))
