"""Map of every downloaded neuron that lights up when the neuron fires.

``Run-Fly.ps1`` opens the 3D network (``noesis.viz.brain_view3d``) by default. This module is the
2D schematic fallback (``Run-Fly.ps1 -Flat2D``) and the activity file the simulation writes for
either viewer.

The BANC tables carry no 3-D neuron positions, so the 2D layout is *schematic*: each neuron is
placed around a hand-placed point for its main brain region (``Top in/out region``), on its own side
of the body, brain above and nerve cord below -- shaped like the BANC overview, not anatomically exact.
If a coordinates table is supplied (``--coords`` CSV with ``root_id,x,y,z``), real positions are used.

Live use: the simulation writes recent firing rates with ``write_activity``; in another terminal

    python -m noesis.viz.brain_map --watch outputs/live_activity.npz

opens a window that redraws whenever the file changes. ``--snapshot`` renders one frame to a PNG.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import time
from dataclasses import dataclass
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


def spike_histogram(chunks: list[np.ndarray], n: int) -> np.ndarray:
    """Spike counts over one publish window. ``chunks`` are arrays of neuron indices."""
    counts = np.zeros(n, np.uint32)
    for idx in chunks:
        ids = np.asarray(idx, np.int64).ravel()
        if ids.size == 0:
            continue
        if ids.min() < 0 or ids.max() >= n:
            ids = ids[(ids >= 0) & (ids < n)]
        if ids.size:
            counts += np.bincount(ids, minlength=n).astype(np.uint32)
    return np.minimum(counts, np.uint16(65535)).astype(np.uint16)


def write_activity(path: Path, rates_hz: np.ndarray, time_s: float, awake: bool = True,
                   note: str = "", triggers: dict | None = None,
                   spikes: np.ndarray | None = None) -> bool:
    """Atomically publish the latest per-neuron firing rates for a live viewer.

    ``spikes``, when given and the same length as ``rates_hz``, is the spike count in this
    publish window (so a single spike can light up before the smoothed rate does).

    Never raises: on Windows the viewer (or OneDrive) may briefly hold the file, and a skipped frame
    must not stop the simulation. Returns whether the frame was written.
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.stem + ".tmp.npz")
        payload = {
            "rates": rates_hz.astype(np.float16),
            "time": np.float32(time_s),
            "awake": np.bool_(awake),
            "note": np.str_(note),
            "triggers": np.array(json.dumps(triggers or {}, default=float)),
        }
        if spikes is not None and np.asarray(spikes).shape == (len(rates_hz),):
            clipped = np.minimum(np.asarray(spikes), np.iinfo(np.uint16).max)
            payload["spikes"] = clipped.astype(np.uint16)
        np.savez(tmp, **payload)
        for _ in range(5):
            try:
                os.replace(tmp, path)
                return True
            except PermissionError:
                time.sleep(0.02)
    except OSError:
        pass
    return False


def read_activity(path: Path) -> tuple[np.ndarray, float, bool, str]:
    """Load one published frame and release the file immediately (Windows file locking)."""
    with np.load(path) as f:
        awake = bool(f["awake"]) if "awake" in f.files else True
        note = str(f["note"]) if "note" in f.files else ""
        return f["rates"].astype(np.float32), float(f["time"]), awake, note


def read_activity_full(path: Path) -> tuple[np.ndarray, float, bool, str, dict]:
    """Load rates, time, awake flag, note, and structured triggers/telemetry dictionary."""
    with np.load(path) as f:
        awake = bool(f["awake"]) if "awake" in f.files else True
        note = str(f["note"]) if "note" in f.files else ""
        triggers = json.loads(str(f["triggers"])) if "triggers" in f.files else {}
        return f["rates"].astype(np.float32), float(f["time"]), awake, note, triggers


@dataclass(frozen=True)
class ActivityFrame:
    """One published frame, including spike counts when the writer sent them."""

    rates: np.ndarray
    time_s: float
    awake: bool
    note: str
    triggers: dict
    spikes: np.ndarray


def read_activity_frame(path: Path) -> ActivityFrame:
    """Load a frame for the 3D viewer. Missing spike counts are zeros (older writers)."""
    with np.load(path) as f:
        rates = f["rates"].astype(np.float32)
        awake = bool(f["awake"]) if "awake" in f.files else True
        note = str(f["note"]) if "note" in f.files else ""
        triggers = json.loads(str(f["triggers"])) if "triggers" in f.files else {}
        if "spikes" in f.files and np.asarray(f["spikes"]).shape == rates.shape:
            spikes = np.asarray(f["spikes"]).astype(np.int32)
        else:
            spikes = np.zeros(rates.shape, np.int32)
        return ActivityFrame(rates, float(f["time"]), awake, note, triggers, spikes)


def describe_triggers(t: dict) -> str:
    """Summary of world sensory inputs, internal hemolymph state, and embodied posture."""
    if not t:
        return ""
    lines = []
    if "odor_food" in t or "vision_mean" in t:
        lines.append(
            f"world  | smell: food {t.get('odor_food', 0):.2f} aversive {t.get('odor_aversive', 0):.2f}  "
            f"taste: sugar {t.get('taste_sugar', 0):.1f} water {t.get('taste_water', 0):.1f} "
            f"bitter {t.get('taste_bitter', 0):.1f}  light {t.get('vision_mean', 0):.2f}  "
            f"T {t.get('temperature_c', 24):.1f}C  hum {t.get('humidity', 0.45):.2f}"
        )
    if "hemolymph_sugar" in t:
        lines.append(
            f"inside | hemolymph sugar {t.get('hemolymph_sugar', 0):.2f}  "
            f"hydration {t.get('hemolymph_hydration', 0):.2f}  O2 {t.get('hemolymph_oxygen', 0):.2f}  "
            f"heart {t.get('heart_rate_hz', 0):.1f} Hz  "
            f"MN9 {t.get('mn9_hz', 0):.1f} Hz  DILP {t.get('dilp_hz', 0):.1f} Hz"
        )
    if "body_z_mm" in t or "muscle_pos_mean" in t:
        lines.append(
            f"body   | z {t.get('body_z_mm', 0):.2f} mm  up {t.get('uprightness', 1):+.2f}  "
            f"muscles +{t.get('muscle_pos_mean', 0):.2f}/-{t.get('muscle_neg_mean', 0):.2f}  "
            f"DA {t.get('da_hz', 0):.1f} Hz  OA {t.get('oa_hz', 0):.1f} Hz  5-HT {t.get('ser_hz', 0):.1f} Hz"
        )
    return "\n".join(lines)


def position_window(fig, geometry: str | None = None) -> None:
    """Position the matplotlib Tk/Qt window on screen (e.g. '780x960+960+20' for right half)."""
    if not geometry:
        return
    mgr = getattr(fig.canvas, "manager", None)
    win = getattr(mgr, "window", None)
    if win is not None and hasattr(win, "wm_geometry"):
        with contextlib.suppress(RuntimeError, AttributeError, ValueError):
            win.wm_geometry(geometry)


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
                                 va="top", fontsize=9.5, family="monospace")
        self.world = self.ax.text(0.01, 0.01, "", transform=self.ax.transAxes, color="#ffd27f",
                                  va="bottom", fontsize=8.5, family="monospace")
        self.ax.set_title(title, color="white")
        if labels:
            for region, text in LABELS.items():
                x, y = REGION_XY[region]
                self.ax.text(x + 0.9, y, text, color="#9aa4c8", fontsize=7, va="center", ha="left")
        self.ax.set_aspect("equal")
        self.ax.axis("off")

    def update(self, rates_hz: np.ndarray, time_s: float | None = None, min_hz: float = 1.0,
               awake: bool = True, note: str = "", triggers: dict | None = None) -> None:
        on = np.flatnonzero(rates_hz > min_hz)
        r = rates_hz[on].astype(float)
        self.lit.set_offsets(self.pos[on] if len(on) else np.zeros((0, 2)))
        self.lit.set_array(np.log10(r + 1.0))
        self.lit.set_sizes(1.0 + 4.0 * np.clip(np.log10(r + 1.0) / 2.3, 0, 1))
        label = f"t = {time_s:6.3f} s   " if time_s is not None else ""
        state = "" if awake else "   (brain still asleep: waiting for birth)"
        extra = "\n" + note if note else ""
        self.text.set_text(f"{label}firing: {len(on):,} / {len(rates_hz):,} neurons{state}{extra}")
        self.world.set_text(describe_triggers(triggers or {}))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--coords", type=Path, help="optional CSV with root_id,x,y,z")
    ap.add_argument("--watch", type=Path, help="activity file written by a running simulation")
    ap.add_argument("--snapshot", type=Path, help="render this activity file once")
    ap.add_argument("--out", type=Path, default=Path("outputs/brain_map.png"))
    ap.add_argument("--geometry", type=str, default=None,
                    help="window geometry WxH+X+Y for side-by-side placement (e.g. 760x920+940+30)")
    a = ap.parse_args()

    from noesis.connectome import load_cached

    neurons = load_cached(a.data / "raw", a.data / "processed").neurons
    pos = neuron_positions(neurons, a.coords)
    layout = "real coordinates" if a.coords and a.coords.exists() else "schematic layout (not anatomical)"
    if a.snapshot:
        import matplotlib
        matplotlib.use("Agg")
        m = BrainMap(pos, f"Noesis brain map - {layout}", labels=not a.coords)
        rates, t, awake, note, triggers = read_activity_full(a.snapshot)
        m.update(rates, t, awake=awake, note=note, triggers=triggers)
        a.out.parent.mkdir(parents=True, exist_ok=True)
        m.fig.savefig(a.out, dpi=150, facecolor="black")
        print("saved", a.out)
        return
    if not a.watch:
        ap.error("use --watch FILE or --snapshot FILE")
    import matplotlib.pyplot as plt

    m = BrainMap(pos, f"Noesis brain map - {layout}", labels=not a.coords)
    position_window(m.fig, a.geometry)
    plt.ion()
    plt.show()
    m.text.set_text(f"waiting for the simulation to write {a.watch} ...")
    last = 0.0
    while plt.fignum_exists(m.fig.number):
        try:
            stamp = a.watch.stat().st_mtime
            if stamp != last:
                rates, t, awake, note, triggers = read_activity_full(a.watch)
                m.update(rates, t, awake=awake, note=note, triggers=triggers)
                last = stamp
        except (OSError, ValueError, KeyError, EOFError):
            pass  # file not written yet, or caught mid-replace
        plt.pause(0.25)
        time.sleep(0.05)


if __name__ == "__main__":
    main()
