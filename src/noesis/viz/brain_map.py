"""Rotatable 3-D view of the strongest downloaded connections and their neurons.

The default layout follows schematic neuropil anatomy; ``--network-layout`` instead spreads
neurons by their real connections. The tables carry no measured 3-D positions. Faint blue edges are
solid for excitatory and dotted for inhibitory. Click a node to inspect its incoming/outgoing edges and direction
arrows; Escape restores the overview. ``--edges`` controls how many connections are shown.
Drag to rotate, scroll to zoom. ``--flat`` keeps the older 2-D schematic. A coordinates CSV
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
    """Summarize measured motor/training status and habitat sensory input."""
    if not t:
        return ""
    status = ""
    if "motor_hz" in t:
        status = (f"training {'ON' if t.get('learning') else 'off'} | {t.get('brain', '')} | "
                  f"motor {t['motor_hz']:.1f} Hz | leg motion {t.get('joint_speed_rad_s', 0):.2f} rad/s | "
                  f"PAM {t.get('pam_hz', 0):.1f} / PPL1 {t.get('ppl1_hz', 0):.1f} Hz\n")
        if "environment_driven_neurons" not in t:
            return status.strip()
    return status + (f"smell: food {t.get('odor_food', 0):.2f}  aversive {t.get('odor_aversive', 0):.2f}   "
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


def prepare_matplotlib_toolkits() -> None:
    """Select matching toolkits before pyplot imports and registers 3-D axes."""
    import matplotlib
    import mpl_toolkits

    toolkit_dir = Path(matplotlib.__file__).resolve().parent.parent / "mpl_toolkits"
    if (toolkit_dir / "mplot3d").is_dir():
        mpl_toolkits.__path__ = [str(toolkit_dir)] + [
            p for p in mpl_toolkits.__path__ if Path(p).resolve() != toolkit_dir
        ]


def network_positions(positions, pre, post):
    """Spread the displayed graph by its real edges, without inventing connections."""
    import networkx as nx

    graph = nx.Graph()
    graph.add_edges_from(zip(pre.tolist(), post.tolist()))
    result = np.array(positions, dtype=float, copy=True)
    if len(graph):
        layout = nx.spring_layout(graph, dim=3, seed=7, iterations=20, scale=5)
        for node, point in layout.items():
            result[node] = point
    return result


def compact_anatomy_positions(neurons: pd.DataFrame, seed: int = 0) -> np.ndarray:
    """Bounded schematic neuropil volumes; display geometry, not measured morphology."""
    rng = np.random.default_rng(seed)
    regions = neurons["Top in/out region"].fillna("NO_CONS").str.split(".").str[0].str.upper()
    sides = neurons["Soma side"].map({"left": -1., "right": 1.}).fillna(0).to_numpy()
    sides = np.where(sides == 0, rng.choice([-1., 1.], len(neurons)), sides)
    result = np.zeros((len(neurons), 3))
    for region, indices in regions.groupby(regions).groups.items():
        rows = neurons.index.get_indexer(indices)
        x, z = REGION_XY.get(region, (.7, .5))
        unit = rng.normal(size=(len(rows), 3))
        unit /= np.linalg.norm(unit, axis=1, keepdims=True)
        unit *= rng.random((len(rows), 1)) ** (1 / 3)
        if region in OPTIC:
            center, radius = (2.65, 0., .9), (1.2, .65, .58)
        elif z < -2.:
            center, radius = (.28 * x, 0., -1.8 + .85 * (z + 2.8)), (.48, .5, .55)
        else:
            center, radius = (.48 * x, 0., 1.05 + .25 * z), (.65, .65, .53)
        result[rows] = unit * radius + center
        result[rows, 0] = sides[rows] * result[rows, 0]
    return result


class BrainVolume:
    """Rotatable 3-D brain: every neuron is a point, the strongest real synapses are wires.

    The reference-inspired portrait view uses small muted anatomical points and warm
    white activity. Drag to rotate; clicking reveals a neuron's real displayed edges.
    """

    def __init__(self, positions: np.ndarray, pre: np.ndarray, post: np.ndarray, sign: np.ndarray,
                 title: str = "Noesis 3D brain", root_ids=None, anatomical_context=False,
                 context_nodes=None):
        prepare_matplotlib_toolkits()
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d.art3d import Line3DCollection

        self.pos = np.asarray(positions, float)
        self.pre, self.post = np.asarray(pre, np.int64), np.asarray(post, np.int64)
        self.sign = np.asarray(sign, float)
        self.root_ids = root_ids
        self.selected = None
        self.focus_arrows = None
        self.rates = np.zeros(len(self.pos))
        self.nodes = np.unique(np.concatenate([self.pre, self.post]))
        self.degree = np.bincount(np.concatenate([self.pre, self.post]), minlength=len(self.pos))
        self.fig = plt.figure(figsize=(7, 11), facecolor="#0b1312")
        self.ax = self.fig.add_axes([0.0, 0.10, 1., 0.76], projection="3d", facecolor="#0b1312")
        self.ax.computed_zorder = False
        self.fig.text(0.06, 0.955, "N E U R A L   A C T I V I T Y", color="#7f9692", size=8)
        self.fig.text(0.06, 0.927, f"{len(self.pos):,} neurons", color="#e1e9e5", size=19, weight="bold")
        self.fig.text(0.06, 0.908, f"{len(self.pre):,} displayed connections · schematic anatomy",
                      color="#7f918e", size=8)
        self.info = self.fig.text(0.55, 0.895, "", color="#a8b8b3", size=8, va="top", linespacing=1.4)
        self.info.set_visible(False)
        for axis in (self.ax.xaxis, self.ax.yaxis, self.ax.zaxis):
            axis.set_pane_color((0, 0, 0, 1))
            axis.line.set_color((0, 0, 0, 0))
            axis.set_ticks([])
        if anatomical_context:
            # Dim context establishes the whole CNS shape without hiding real edges.
            indices = np.arange(len(self.pos)) if context_nodes is None else np.asarray(context_nodes)
            context = self.pos[indices[::max(1, len(indices) // 40000)]]
            self.ax.scatter(*context.T, s=.35, c="#7e9798", alpha=.32,
                            linewidths=0, depthshade=False, zorder=0)
        pts = self.pos[self.nodes]
        self.node_dots = self.ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                        s=.6, c="#779194",
                        linewidths=0,
                        depthshade=False, picker=5, zorder=3)
        wires = np.stack([self.pos[self.pre], self.pos[self.post]], axis=1) if len(self.pre) else np.zeros((0, 2, 3))
        self.wires = Line3DCollection(wires, linewidths=0.35, zorder=2)
        if len(self.sign):
            self.wires.set_linestyle(["solid" if s >= 0 else "dotted" for s in self.sign])
        self.ax.add_collection3d(self.wires)
        self.glow_outer = self.ax.scatter([], [], [], s=[], c="#d4ded1", alpha=.025,
                                          linewidths=0, depthshade=False, zorder=4)
        self.glow_inner = self.ax.scatter([], [], [], s=[], c="#e7e5cd", alpha=.065,
                                          linewidths=0, depthshade=False, zorder=5)
        self.lit = self.ax.scatter([], [], [], s=[], c="#f2f0dc",
                                   linewidths=0, depthshade=False, zorder=6)
        self.text = self.fig.text(0.06, 0.065, "", color="#9baba6", size=8)
        self.world = self.fig.text(0.06, 0.035, "", color="#6f827e", size=7)
        if anatomical_context:
            self.ax.set_proj_type("ortho")
            self.ax.set_xlim(-4.6, 4.6)
            self.ax.set_ylim(-1.4, 1.4)
            self.ax.set_zlim(-5.4, 2.5)
        self.ax.set_box_aspect((1.15, .35, 1.) if anatomical_context else (1.2, 1., 1.1), zoom=1.3)
        self.ax.view_init(elev=0 if anatomical_context else 18, azim=-90 if anatomical_context else -70)
        self.ax.set_axis_off()
        self.fig.canvas.manager.set_window_title(title)
        self.fig.canvas.mpl_connect("pick_event", self._pick)
        self.fig.canvas.mpl_connect("key_press_event", self._key)
        self.fig.canvas.mpl_connect("scroll_event", self._scroll)
        self.update(self.rates)

    def _pick(self, event):
        if event.artist is self.node_dots and len(event.ind):
            self.selected = int(self.nodes[event.ind[0]])
            self._style_connections()
            self.fig.canvas.draw_idle()

    def _key(self, event):
        if event.key == "escape":
            self.selected = None
            self._style_connections()
            self.fig.canvas.draw_idle()

    def _scroll(self, event):
        if event.inaxes is self.ax:
            factor = .85 if event.button == "up" else 1 / .85
            for get_limits, set_limits in ((self.ax.get_xlim3d, self.ax.set_xlim3d),
                                          (self.ax.get_ylim3d, self.ax.set_ylim3d),
                                          (self.ax.get_zlim3d, self.ax.set_zlim3d)):
                lo, hi = get_limits()
                center, radius = (lo + hi) / 2, (hi - lo) * factor / 2
                set_limits(center - radius, center + radius)
            self.fig.canvas.draw_idle()

    def _style_connections(self):
        if self.focus_arrows is not None:
            self.focus_arrows.remove()
            self.focus_arrows = None
        colors = np.tile([.43, .57, .56, .035], (len(self.sign), 1))
        widths = np.full(len(self.sign), .22)
        node_colors = np.tile([.46, .57, .58, .32], (len(self.nodes), 1))
        if self.selected is not None:
            self.info.set_visible(True)
            incident = (self.pre == self.selected) | (self.post == self.selected)
            colors[~incident, 3] = .006
            colors[incident, 3] = .60
            widths[incident] = .9
            neighbors = np.unique(np.concatenate([self.pre[incident], self.post[incident]]))
            starts = self.pos[self.pre[incident]]
            delta = self.pos[self.post[incident]] - starts
            arrow_starts = starts + .55 * delta
            if len(starts):
                self.focus_arrows = self.ax.quiver(*arrow_starts.T, *(.18 * delta).T,
                    color=colors[incident], arrow_length_ratio=.65, linewidth=.7)
            node_colors[~np.isin(self.nodes, neighbors), 3] = .025
            node_colors[self.nodes == self.selected] = [.45, .62, .85, .6]
            identifier = self.root_ids[self.selected] if self.root_ids is not None else self.selected
            self.info.set_text(f"NEURON {identifier}\nDisplayed incoming: {np.count_nonzero(self.post == self.selected)}\nDisplayed outgoing: {np.count_nonzero(self.pre == self.selected)}\nFiring: {self.rates[self.selected]:.1f} Hz")
        else:
            self.info.set_visible(False)
        self.wires.set_colors(colors)
        self.wires.set_linewidths(widths)
        self.node_dots.set_facecolors(node_colors)

    def update(self, rates_hz: np.ndarray, time_s: float | None = None, min_hz: float = 12.0,
               awake: bool = True, triggers: dict | None = None) -> None:
        self.rates = np.asarray(rates_hz)
        on = np.flatnonzero(rates_hz > min_hz) if awake else np.zeros(0, dtype=int)
        if len(on) > 2000:
            on = on[np.argpartition(rates_hz[on], -2000)[-2000:]]
        r = rates_hz[on].astype(float)
        pts = self.pos[on] if len(on) else np.zeros((0, 3))
        intensity = np.clip((r - min_hz) / 100., 0, 1)
        for artist, sizes in ((self.glow_outer, 3 + 9 * intensity),
                              (self.glow_inner, 1 + 3 * intensity),
                              (self.lit, .4 + 1.7 * intensity)):
            artist._offsets3d = (pts[:, 0], pts[:, 1], pts[:, 2])
            artist.set_sizes(sizes)
        self._style_connections()
        label = f"t = {time_s:6.3f} s   " if time_s is not None else ""
        state = "" if awake else "   (brain still asleep)"
        self.text.set_text(
            f"{label}Active: {int(np.count_nonzero(rates_hz > min_hz)):,} · 0—200+ Hz{state}")
        self.world.set_text(describe_triggers(triggers or {}))


def main() -> None:
    prepare_matplotlib_toolkits()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--coords", type=Path, help="optional CSV with root_id,x,y,z")
    ap.add_argument("--watch", type=Path, help="activity file written by a running simulation")
    ap.add_argument("--ready-file", type=Path, help="signal that the live window has rendered")
    ap.add_argument("--snapshot", type=Path, help="render this activity file once")
    ap.add_argument("--flat", action="store_true", help="2-D schematic instead of the rotatable 3-D network")
    ap.add_argument("--anatomy", action="store_true", help="use schematic anatomical positions (the default)")
    ap.add_argument("--network-layout", action="store_true", help="spread nodes by connectivity instead of the default anatomy")
    ap.add_argument("--edges", type=int, default=1800, help="number of strongest real connections to display")
    ap.add_argument("--out", type=Path, default=Path("outputs/brain_map.png"))
    a = ap.parse_args()
    if a.edges < 1:
        ap.error("--edges must be positive")

    from noesis.connectome import load_cached

    cn = load_cached(a.data / "raw", a.data / "processed")
    neurons = cn.neurons
    context_nodes = np.flatnonzero(~neurons["Top in/out region"].fillna("NO_CONS").str.upper().str.startswith("NO_CONS"))
    use_3d = not a.flat
    layout = "real coordinates" if a.coords and a.coords.exists() else "schematic 3-D anatomy, real synapses"
    if use_3d:
        pos3 = neuron_positions_3d(neurons, a.coords)
        if not a.coords and (not a.network_layout or a.anatomy):
            pos3 = compact_anatomy_positions(neurons)
        print(f"sampling the strongest synapses out of {cn.W.nnz:,} ...", flush=True)
        pre, post, sign = strongest_synapses(cn.W, limit=a.edges)
        if a.network_layout and not a.anatomy and not a.coords:
            print("arranging neurons by their connections ...", flush=True)
            pos3 = network_positions(pos3, pre, post)
            layout = "connection layout, real synapses"
        print(f"3-D network: {len(np.unique(np.concatenate([pre, post]))):,} visible neurons, "
              f"{len(pre):,} connections; {len(neurons):,} total. Click a node to inspect.",
              flush=True)
    else:
        pos3 = pre = post = sign = None
    pos = neuron_positions(neurons, a.coords) if not use_3d else pos3
    if a.snapshot:
        import matplotlib
        matplotlib.use("Agg")
        if use_3d:
            m = BrainVolume(pos3, pre, post, sign, f"Noesis brain - {layout}", neurons["root_id"].to_numpy(),
                            anatomical_context=not a.network_layout or a.anatomy or bool(a.coords),
                            context_nodes=context_nodes)
        else:
            m = BrainMap(pos, f"Noesis brain map - {layout}", labels=not a.coords)
        rates, t, awake, triggers = read_activity(a.snapshot)
        m.update(rates, t, awake=awake, triggers=triggers)
        a.out.parent.mkdir(parents=True, exist_ok=True)
        m.fig.savefig(a.out, dpi=150, facecolor=m.fig.get_facecolor())
        print("saved", a.out)
        return
    if not a.watch:
        ap.error("use --watch FILE or --snapshot FILE")
    import matplotlib.pyplot as plt

    if use_3d:
        m = BrainVolume(pos3, pre, post, sign, f"Noesis brain - {layout}", neurons["root_id"].to_numpy(),
                        anatomical_context=not a.network_layout or a.anatomy or bool(a.coords),
                        context_nodes=context_nodes)
    else:
        m = BrainMap(pos, f"Noesis brain map - {layout}", labels=not a.coords)
    plt.ion()
    plt.show()
    m.text.set_text("Network ready · waiting for live simulation activity")
    m.fig.canvas.draw()
    plt.pause(0.1)
    if a.ready_file:
        a.ready_file.write_text("ready", encoding="utf-8")
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
