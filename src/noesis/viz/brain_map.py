"""Rotatable 3-D map of every downloaded neuron, with the real synapses drawn as wires.

The BANC tables carry no 3-D neuron positions, so each neuron sits in a schematic cloud for its
neuropil (``Top in/out region``), mirrored by side, brain above and nerve cord below. The wires are
the strongest signed synapses from the downloaded connectome: gold excitatory, blue inhibitory.
Drag the window to rotate. ``--flat`` keeps the older 2-D schematic. A coordinates CSV
(``root_id,x,y,z``) replaces the layout with measured positions.

Live use: the simulation writes recent firing rates with ``write_activity``; in another terminal

    python -m noesis.viz.brain_map --watch outputs/live_activity.npz

opens a window that redraws whenever the file changes. ``--snapshot`` renders one frame to a PNG.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Schematic front-view anchor (|x|, y) of each region; x is mirrored by soma side. Units are arbitrary.
REGION_XY: dict[str, tuple[float, float]] = {
    # optic lobes
    "ME": (5.2, 0.7), "LO": (4.2, 0.5), "LOP": (3.7, 1.0), "AME": (3.5, -0.1),
    # lateral and superior central brain
    "AVLP": (2.7, -0.4), "PVLP": (3.0, 0.2), "PLP": (2.7, 1.0), "LH": (2.5, 1.7), "SLP": (2.0, 2.1),
    "SMP": (0.7, 2.3), "SIP": (1.3, 2.4), "SCL": (1.7, 1.5), "ICL": (1.4, 1.0), "IB": (0.9, 0.7),
    "ATL": (0.6, 1.7), "CRE": (1.0, 1.8), "LAL": (1.3, 0.1), "VES": (1.2, -0.5), "SPS": (1.8, 0.5),
    "IPS": (2.0, -0.1), "WED": (2.3, -0.5), "AMMC": (1.6, -1.0), "SAD": (1.1, -1.1), "GOR": (1.5, 0.5),
    "EPA": (1.7, -0.2), "AOTU": (2.1, 1.3), "BU": (1.2, 1.2), "GA": (1.0, 1.4),
    # olfactory and learning centres
    "AL": (0.9, -0.6), "MB_CA": (1.7, 2.7), "MB_PED": (1.4, 1.9), "MB_VL": (1.0, 2.5), "MB_ML": (0.4, 2.0),
    # central complex (midline)
    "FB": (0.15, 1.6), "EB": (0.15, 1.0), "PB": (0.3, 2.7), "NO": (0.4, 1.1),
    # suboesophageal zone
    "GNG": (0.35, -1.7), "PRW": (0.1, -1.3), "FLA": (0.3, -1.1), "CAN": (0.5, -0.9),
    # ventral nerve cord, below the neck
    "NTCT": (0.15, -2.8), "NTCT_UTCT_T1": (0.2, -2.9), "T1_PRONM": (0.7, -3.3), "LEGNP_T1": (0.9, -3.3),
    "AMNP": (0.5, -3.7), "WTCT": (0.2, -3.8), "WTCT_UTCT_T2": (0.25, -3.9), "T2_MVAC": (0.35, -4.1),
    "T2_MESONM": (0.7, -4.3), "INTTCT": (0.1, -4.4), "LTCT": (0.25, -4.5), "HTCT": (0.2, -4.8),
    "HTCT_UTCT_T3": (0.25, -4.9), "T3_METANM": (0.7, -5.3), "ABDNM": (0.3, -6.2),
    # neurons without connections, kept out of the way
    "NO_CONS": (6.3, -5.5),
}
OPTIC = {"ME", "LO", "LOP", "AME"}
LABELS = {  # region -> text shown on the schematic map (right side only, except midline/VNC)
    "ME": "optic lobe", "AL": "antennal lobe", "MB_CA": "mushroom body", "LH": "lateral horn",
    "FB": "central complex", "GNG": "GNG (taste)", "T1_PRONM": "front legs (T1)",
    "T2_MESONM": "middle legs (T2)", "T3_METANM": "hind legs (T3)", "ABDNM": "abdomen",
    "NO_CONS": "no synapses (noise only)",
}


def neuron_positions(neurons: pd.DataFrame, coords_csv: Path | None = None, seed: int = 0) -> np.ndarray:
    """(n, 2) positions: real coordinates if given, else the schematic region layout."""
    if coords_csv is not None and coords_csv.exists():
        c = pd.read_csv(coords_csv).set_index("root_id")
        xyz = c.reindex(neurons["root_id"])[["x", "y", "z"]].to_numpy(float)
        return np.column_stack([xyz[:, 0], -xyz[:, 1]])  # front view, dorsal up
    return neuron_positions_3d(neurons, seed=seed)[:, [0, 2]]


def neuron_positions_3d(neurons: pd.DataFrame, coords_csv: Path | None = None, seed: int = 0) -> np.ndarray:
    """(n, 3) positions as (left-right, anterior-posterior, dorsal-ventral).

    The downloaded table has no measured coordinates. Each neuron is placed in a 3-D cloud around
    its neuropil, mirrored by soma side, with the brain dorsal and the nerve cord ventral. A
    coordinates CSV (``root_id,x,y,z``) replaces this schematic layout.
    """
    if coords_csv is not None and coords_csv.exists():
        c = pd.read_csv(coords_csv).set_index("root_id")
        xyz = c.reindex(neurons["root_id"])[["x", "y", "z"]].to_numpy(float)
        return np.column_stack([xyz[:, 0], xyz[:, 1], xyz[:, 2]])
    rng = np.random.default_rng(seed)
    region = neurons["Top in/out region"].fillna("NO_CONS").str.split(".").str[0].str.upper()
    side = neurons["Soma side"].fillna("").map({"left": -1.0, "right": 1.0})
    side = side.fillna(pd.Series(rng.choice([-1.0, 1.0], len(neurons)), index=neurons.index)).to_numpy()
    counts = region.value_counts()
    pos = np.zeros((len(neurons), 3))
    for name, idx in region.groupby(region).groups.items():
        ax, height = REGION_XY.get(name, (0.6, 0.0))
        k = len(idx)
        spread = 0.1 + 0.002 * np.sqrt(counts[name])
        sx = spread * (0.6 if name in OPTIC else 1.0)
        rows = neurons.index.get_indexer(idx)
        pos[rows, 0] = side[rows] * ax + rng.normal(0, sx, k)
        pos[rows, 1] = 0.35 * height + rng.normal(0, spread * 0.7, k)
        pos[rows, 2] = height + rng.normal(0, spread * 0.8, k)
    return pos


def strongest_synapses(weights, limit: int = 8000):
    """Return presynaptic index, postsynaptic index and sign for the strongest synapses.

    ``weights[i, j]`` is the signed synapse count from i to j. The sample is what the 3-D view
    draws; the full matrix is about 1.5 million edges and cannot be rotated interactively.
    """
    strength = np.abs(np.asarray(weights.data))
    if strength.size == 0:
        empty = np.zeros(0, np.int64)
        return empty, empty, empty.astype(np.float32)
    pick = np.arange(strength.size) if strength.size <= limit else np.argpartition(strength, -limit)[-limit:]
    rows = np.repeat(np.arange(weights.shape[0]), np.diff(weights.indptr))
    return rows[pick].astype(np.int64), weights.indices[pick].astype(np.int64), np.sign(weights.data[pick])


def write_activity(path: Path, rates_hz: np.ndarray, time_s: float, awake: bool = True,
                   triggers: dict | None = None) -> bool:
    """Atomically publish the latest per-neuron firing rates for a live viewer.

    Never raises: on Windows the viewer (or OneDrive) may briefly hold the file, and a skipped frame
    must not stop the simulation. Returns whether the frame was written.
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.stem + ".tmp.npz")
        np.savez(tmp, rates=rates_hz.astype(np.float16), time=np.float32(time_s), awake=np.bool_(awake),
                 triggers=np.array(json.dumps(triggers or {}, default=float)))
        for _ in range(5):
            try:
                os.replace(tmp, path)
                return True
            except PermissionError:
                time.sleep(0.02)
    except OSError:
        pass
    return False


def read_activity(path: Path) -> tuple[np.ndarray, float, bool, dict]:
    """Load one published frame and release the file immediately (Windows file locking)."""
    with np.load(path) as f:
        awake = bool(f["awake"]) if "awake" in f.files else True
        triggers = json.loads(str(f["triggers"])) if "triggers" in f.files else {}
        return f["rates"].astype(np.float32), float(f["time"]), awake, triggers


def describe_triggers(t: dict) -> str:
    """Two-line summary of what the habitat is doing to the senses (empty outside the habitat)."""
    if not t:
        return ""
    return (f"smell: food {t.get('odor_food', 0):.2f}  aversive {t.get('odor_aversive', 0):.2f}   "
            f"taste: sugar {t.get('taste_sugar', 0):.1f}  water {t.get('taste_water', 0):.1f}  "
            f"bitter {t.get('taste_bitter', 0):.1f}\n"
            f"light {t.get('vision_mean', 0):.2f}   temperature {t.get('temperature_c', 0):.1f} C   "
            f"humidity {t.get('humidity', 0):.2f}   sound {t.get('sound_envelope', 0):.2f}   "
            f"world-driven neurons {t.get('environment_driven_neurons', 0):,}")


class BrainMap:
    """Dark figure: every neuron as a faint dot, firing neurons glowing by rate."""

    def __init__(self, positions: np.ndarray, title: str = "Noesis brain map", ax=None,
                 labels: bool = True):
        import matplotlib.pyplot as plt

        self.pos = positions
        if ax is None:
            self.fig, self.ax = plt.subplots(figsize=(9, 9), facecolor="black")
        else:
            self.fig, self.ax = ax.figure, ax
        self.ax.set_facecolor("black")
        self.ax.scatter(positions[:, 0], positions[:, 1], s=0.15, c="#3a3f58", alpha=0.35, linewidths=0)
        self.lit = self.ax.scatter([], [], s=[], c=[], cmap="plasma", vmin=0.0, vmax=np.log10(200.0),
                                   linewidths=0)
        self.text = self.ax.text(0.01, 0.99, "", transform=self.ax.transAxes, color="white",
                                 va="top", fontsize=10, family="monospace")
        self.world = self.ax.text(0.01, 0.01, "", transform=self.ax.transAxes, color="#ffd27f",
                                  va="bottom", fontsize=9, family="monospace")
        self.ax.set_title(title, color="white")
        if labels:
            for region, text in LABELS.items():
                x, y = REGION_XY[region]
                self.ax.text(x + 0.9, y, text, color="#9aa4c8", fontsize=7, va="center", ha="left")
        self.ax.set_aspect("equal")
        self.ax.axis("off")

    def update(self, rates_hz: np.ndarray, time_s: float | None = None, min_hz: float = 1.0,
               awake: bool = True, triggers: dict | None = None) -> None:
        on = np.flatnonzero(rates_hz > min_hz)
        r = rates_hz[on].astype(float)
        self.lit.set_offsets(self.pos[on] if len(on) else np.zeros((0, 2)))
        self.lit.set_array(np.log10(r + 1.0))
        self.lit.set_sizes(1.0 + 4.0 * np.clip(np.log10(r + 1.0) / 2.3, 0, 1))
        label = f"t = {time_s:6.3f} s   " if time_s is not None else ""
        state = "" if awake else "   (brain still asleep: waiting for birth)"
        self.text.set_text(f"{label}firing: {len(on):,} / {len(rates_hz):,} neurons{state}")
        self.world.set_text(describe_triggers(triggers or {}))


class BrainVolume:
    """Rotatable 3-D brain: every neuron is a point, the strongest real synapses are wires.

    Drag the view with the mouse. Gold wires are excitatory synapses and blue wires are
    inhibitory. A neuron that is firing above the background grows and turns hot.
    """

    def __init__(self, positions: np.ndarray, pre: np.ndarray, post: np.ndarray, sign: np.ndarray,
                 title: str = "Noesis 3D brain"):
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d.art3d import Line3DCollection

        self.pos = np.asarray(positions, float)
        self.pre, self.post = np.asarray(pre, np.int64), np.asarray(post, np.int64)
        self.sign = np.asarray(sign, float)
        self.fig = plt.figure(figsize=(11, 9), facecolor="black")
        self.ax = self.fig.add_subplot(111, projection="3d", facecolor="black")
        for axis in (self.ax.xaxis, self.ax.yaxis, self.ax.zaxis):
            axis.set_pane_color((0, 0, 0, 1))
            axis.line.set_color((0, 0, 0, 0))
            axis.set_ticks([])
        silent = self.pos if len(self.pos) <= 40000 else self.pos[:: max(1, len(self.pos) // 40000)]
        self.ax.scatter(silent[:, 0], silent[:, 1], silent[:, 2], s=0.4, c="#2c3350",
                        alpha=0.35, linewidths=0, depthshade=False)
        wires = np.stack([self.pos[self.pre], self.pos[self.post]], axis=1) if len(self.pre) else np.zeros((0, 2, 3))
        self.wires = Line3DCollection(wires, colors=[(0.25, 0.25, 0.3, 0.15)] * len(wires), linewidths=0.4)
        self.ax.add_collection3d(self.wires)
        self.lit = self.ax.scatter([], [], [], s=[], c=[], cmap="plasma", vmin=0.0, vmax=np.log10(200.0),
                                   linewidths=0, depthshade=False)
        self.text = self.ax.text2D(0.01, 0.98, "", transform=self.ax.transAxes, color="white",
                                   va="top", fontsize=10, family="monospace")
        self.world = self.ax.text2D(0.01, 0.02, "", transform=self.ax.transAxes, color="#ffd27f",
                                    va="bottom", fontsize=9, family="monospace")
        self.ax.text2D(0.01, 0.90, "drag to rotate   gold = excitatory synapse   blue = inhibitory",
                       transform=self.ax.transAxes, color="#9aa4c8", va="top", fontsize=8)
        self.ax.set_title(title, color="white")
        self.ax.set_box_aspect((1.6, 1.0, 1.8))
        self.ax.view_init(elev=18, azim=-70)
        self.ax.set_axis_off()

    def update(self, rates_hz: np.ndarray, time_s: float | None = None, min_hz: float = 12.0,
               awake: bool = True, triggers: dict | None = None) -> None:
        on = np.flatnonzero(rates_hz > min_hz)
        if len(on) > 20000:
            on = on[np.argpartition(rates_hz[on], -20000)[-20000:]]
        r = rates_hz[on].astype(float)
        pts = self.pos[on] if len(on) else np.zeros((0, 3))
        self.lit._offsets3d = (pts[:, 0], pts[:, 1], pts[:, 2])
        self.lit.set_array(np.log10(r + 1.0))
        self.lit.set_sizes(6.0 + 18.0 * np.clip(np.log10(r + 1.0) / 2.3, 0, 1))
        if len(self.pre):
            edge = np.maximum(rates_hz[self.pre], rates_hz[self.post])
            hot = np.clip((edge - min_hz) / 40.0, 0, 1)
            excite = self.sign >= 0
            colors = np.zeros((len(edge), 4))
            colors[:, 0] = np.where(excite, 0.15 + 0.85 * hot, 0.05)
            colors[:, 1] = np.where(excite, 0.12 + 0.55 * hot, 0.15 + 0.35 * hot)
            colors[:, 2] = np.where(excite, 0.05, 0.25 + 0.75 * hot)
            colors[:, 3] = 0.08 + 0.85 * hot
            self.wires.set_colors(colors)
            self.wires.set_linewidths(0.3 + 1.4 * hot)
        label = f"t = {time_s:6.3f} s   " if time_s is not None else ""
        state = "" if awake else "   (brain still asleep)"
        self.text.set_text(
            f"{label}firing: {int(np.count_nonzero(rates_hz > min_hz)):,} / {len(rates_hz):,}"
            f"   wires: {len(self.pre):,} strongest synapses{state}")
        self.world.set_text(describe_triggers(triggers or {}))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--coords", type=Path, help="optional CSV with root_id,x,y,z")
    ap.add_argument("--watch", type=Path, help="activity file written by a running simulation")
    ap.add_argument("--snapshot", type=Path, help="render this activity file once")
    ap.add_argument("--flat", action="store_true", help="2-D schematic instead of the rotatable 3-D network")
    ap.add_argument("--out", type=Path, default=Path("outputs/brain_map.png"))
    a = ap.parse_args()

    from noesis.connectome import load_cached

    cn = load_cached(a.data / "raw", a.data / "processed")
    neurons = cn.neurons
    use_3d = not a.flat
    layout = "real coordinates" if a.coords and a.coords.exists() else "schematic 3-D anatomy, real synapses"
    if use_3d:
        pos3 = neuron_positions_3d(neurons, a.coords)
        print(f"sampling the strongest synapses out of {cn.W.nnz:,} ...", flush=True)
        pre, post, sign = strongest_synapses(cn.W, limit=8000)
        print(f"3-D network: {len(neurons):,} neurons, {len(pre):,} wires. Drag the window to rotate.",
              flush=True)
    else:
        pos3 = pre = post = sign = None
    pos = neuron_positions(neurons, a.coords) if not use_3d else pos3
    if a.snapshot:
        import matplotlib
        matplotlib.use("Agg")
        if use_3d:
            m = BrainVolume(pos3, pre, post, sign, f"Noesis brain - {layout}")
        else:
            m = BrainMap(pos, f"Noesis brain map - {layout}", labels=not a.coords)
        rates, t, awake, triggers = read_activity(a.snapshot)
        m.update(rates, t, awake=awake, triggers=triggers)
        a.out.parent.mkdir(parents=True, exist_ok=True)
        m.fig.savefig(a.out, dpi=150, facecolor="black")
        print("saved", a.out)
        return
    if not a.watch:
        ap.error("use --watch FILE or --snapshot FILE")
    import matplotlib.pyplot as plt

    if use_3d:
        m = BrainVolume(pos3, pre, post, sign, f"Noesis brain - {layout}")
    else:
        m = BrainMap(pos, f"Noesis brain map - {layout}", labels=not a.coords)
    plt.ion()
    plt.show()
    m.text.set_text(f"waiting for the simulation to write {a.watch} ...")
    last = 0.0
    while plt.fignum_exists(m.fig.number):
        try:
            stamp = a.watch.stat().st_mtime
            if stamp != last:
                rates, t, awake, triggers = read_activity(a.watch)
                m.update(rates, t, awake=awake, triggers=triggers)
                last = stamp
        except (OSError, ValueError, KeyError, EOFError):
            pass  # file not written yet, or caught mid-replace
        plt.pause(0.25)
        time.sleep(0.05)


if __name__ == "__main__":
    main()
