"""Map of every downloaded neuron that lights up when the neuron fires.

The BANC tables carry no 3-D neuron positions, so the default layout is *schematic*: each neuron is
placed around a hand-placed point for its main brain region (``Top in/out region``), on its own side
of the body, brain above and nerve cord below -- shaped like the BANC overview, not anatomically exact.
If a coordinates table is supplied (``--coords`` CSV with ``root_id,x,y,z``), real positions are used.

Live use: the simulation writes recent firing rates with ``write_activity``; in another terminal

    python -m noesis.viz.brain_map --watch outputs/live_activity.npz

opens a window that redraws whenever the file changes. ``--snapshot`` renders one frame to a PNG.
"""

from __future__ import annotations

import argparse
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
    rng = np.random.default_rng(seed)
    region = neurons["Top in/out region"].fillna("NO_CONS").str.split(".").str[0].str.upper()
    side = neurons["Soma side"].fillna("").map({"left": -1.0, "right": 1.0})
    side = side.fillna(pd.Series(rng.choice([-1.0, 1.0], len(neurons)), index=neurons.index)).to_numpy()
    counts = region.value_counts()
    pos = np.zeros((len(neurons), 2))
    for name, idx in region.groupby(region).groups.items():
        ax, ay = REGION_XY.get(name, (0.6, 0.0))
        k = len(idx)
        spread = 0.1 + 0.002 * np.sqrt(counts[name])
        sx, sy = (spread * 0.6, spread * 1.6) if name in OPTIC else (spread, spread * 0.8)
        rows = neurons.index.get_indexer(idx)
        pos[rows, 0] = side[rows] * ax + rng.normal(0, sx, k)
        pos[rows, 1] = ay + rng.normal(0, sy, k)
    return pos


def write_activity(path: Path, rates_hz: np.ndarray, time_s: float, awake: bool = True) -> bool:
    """Atomically publish the latest per-neuron firing rates for a live viewer.

    Never raises: on Windows the viewer (or OneDrive) may briefly hold the file, and a skipped frame
    must not stop the simulation. Returns whether the frame was written.
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.stem + ".tmp.npz")
        np.savez(tmp, rates=rates_hz.astype(np.float16), time=np.float32(time_s), awake=np.bool_(awake))
        for _ in range(5):
            try:
                os.replace(tmp, path)
                return True
            except PermissionError:
                time.sleep(0.02)
    except OSError:
        pass
    return False


def read_activity(path: Path) -> tuple[np.ndarray, float, bool]:
    """Load one published frame and release the file immediately (Windows file locking)."""
    with np.load(path) as f:
        awake = bool(f["awake"]) if "awake" in f.files else True
        return f["rates"].astype(np.float32), float(f["time"]), awake


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
        self.ax.set_title(title, color="white")
        if labels:
            for region, text in LABELS.items():
                x, y = REGION_XY[region]
                self.ax.text(x + 0.9, y, text, color="#9aa4c8", fontsize=7, va="center", ha="left")
        self.ax.set_aspect("equal")
        self.ax.axis("off")

    def update(self, rates_hz: np.ndarray, time_s: float | None = None, min_hz: float = 1.0,
               awake: bool = True) -> None:
        on = np.flatnonzero(rates_hz > min_hz)
        r = rates_hz[on].astype(float)
        self.lit.set_offsets(self.pos[on] if len(on) else np.zeros((0, 2)))
        self.lit.set_array(np.log10(r + 1.0))
        self.lit.set_sizes(1.0 + 4.0 * np.clip(np.log10(r + 1.0) / 2.3, 0, 1))
        label = f"t = {time_s:6.3f} s   " if time_s is not None else ""
        state = "" if awake else "   (brain still asleep: waiting for birth)"
        self.text.set_text(f"{label}firing: {len(on):,} / {len(rates_hz):,} neurons{state}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--coords", type=Path, help="optional CSV with root_id,x,y,z")
    ap.add_argument("--watch", type=Path, help="activity file written by a running simulation")
    ap.add_argument("--snapshot", type=Path, help="render this activity file once")
    ap.add_argument("--out", type=Path, default=Path("outputs/brain_map.png"))
    a = ap.parse_args()

    from noesis.connectome import load_cached

    neurons = load_cached(a.data / "raw", a.data / "processed").neurons
    pos = neuron_positions(neurons, a.coords)
    layout = "real coordinates" if a.coords and a.coords.exists() else "schematic layout (not anatomical)"
    if a.snapshot:
        import matplotlib
        matplotlib.use("Agg")
        m = BrainMap(pos, f"Noesis brain map - {layout}", labels=not a.coords)
        rates, t, awake = read_activity(a.snapshot)
        m.update(rates, t, awake=awake)
        a.out.parent.mkdir(parents=True, exist_ok=True)
        m.fig.savefig(a.out, dpi=150, facecolor="black")
        print("saved", a.out)
        return
    if not a.watch:
        ap.error("use --watch FILE or --snapshot FILE")
    import matplotlib.pyplot as plt

    m = BrainMap(pos, f"Noesis brain map - {layout}", labels=not a.coords)
    plt.ion()
    plt.show()
    m.text.set_text(f"waiting for the simulation to write {a.watch} ...")
    last = 0.0
    while plt.fignum_exists(m.fig.number):
        try:
            stamp = a.watch.stat().st_mtime
            if stamp != last:
                rates, t, awake = read_activity(a.watch)
                m.update(rates, t, awake=awake)
                last = stamp
        except (OSError, ValueError, KeyError, EOFError):
            pass  # file not written yet, or caught mid-replace
        plt.pause(0.25)
        time.sleep(0.05)


if __name__ == "__main__":
    main()
