"""Recompute forward progress with a horizontal projection and compare with the 3-D projection."""
import numpy as np

OUT = r"C:/Users/heyma/OneDrive/Desktop/מדעי המחשב/פרוייקטים/Project Noesis/gpu-pilot/outputs"
cases = {
    "parent": "standing-001-candidate-{s}",
    "1000": "gripped-v8-002-candidate-{s}",
    "1001": "gripped-v8-003-candidate-{s}",
    "1002": "gripped-v8-004-candidate-{s}",
    "1003(chain)": "gripped-v8-005-candidate-{s}",
}


def forward(path):
    z = np.load(f"{OUT}/{path}/observations.npz")
    pos, head = z["position"], z["heading"]
    if pos.ndim == 3:  # (T, worlds, 3)
        pos, head = pos[:, 0], head[:, 0]
    pos, head = pos[10:], head[10:]  # drop the 50 ms warmup like the evaluation does
    d = np.diff(pos, axis=0)
    h3 = head[:-1] / np.linalg.norm(head[:-1], axis=1, keepdims=True)
    h2 = head[:-1, :2] / np.maximum(np.linalg.norm(head[:-1, :2], axis=1, keepdims=True), 1e-12)
    three_d = float(np.sum(np.sum(d * h3, axis=1)))
    horiz = float(np.sum(np.sum(d[:, :2] * h2, axis=1)))
    vert = float(pos[-1, 2] - pos[0, 2])
    return three_d, horiz, vert, float(np.abs(head[:, 2]).max())


print(f"{'bank':12s} {'seed':6s} {'3D proj mm':>11s} {'horizontal mm':>14s} {'dz mm':>8s} {'max|heading z|':>15s}  >=0.25 horiz?")
for name, pattern in cases.items():
    for seed in (10006, 10007):
        t, h, v, hz = forward(pattern.format(s=seed))
        print(f"{name:12s} {seed:<6d} {t:11.3f} {h:14.3f} {v:8.3f} {hz:15.3f}  {'yes' if h >= .25 else 'NO'}")
