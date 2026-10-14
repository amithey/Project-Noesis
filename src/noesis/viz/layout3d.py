"""3D positions for the BANC neurons that drive the simulation.

BANC v888 tables do not carry soma coordinates. FAFB ``coordinates.csv.gz`` uses
v783 root ids and matches none of the BANC neurons, so those numbers are not
used. Each neuron is a point inside an ellipsoid for its primary neuropil
(``Top in/out region``), on its own side of the body.

Axes of this layout, in relative micrometres (not a voxel space):

* ``x`` lateral, positive = the animal's right
* ``y`` dorsal, positive up — brain above the neck, nerve cord below
* ``z`` anterior, positive forward

A compound label such as ``ME.LO`` sits in the primary volume, pulled part of
the way toward the second neuropil. Neurons with no synapses (``NO_CONS``)
are parked beside the optic lobe, off the connectome.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import pandas as pd

from noesis.viz.brain_map import describe_triggers

# Transmitter colors. Missing predictions fall back to the neuropil family.
NT_RGB: dict[str, tuple[float, float, float]] = {
    "ACH": (0.25, 0.86, 1.00),
    "GABA": (0.78, 0.38, 1.00),
    "GLUT": (1.00, 0.50, 0.16),
    "HIST": (0.96, 0.90, 0.32),
    "DA": (0.28, 0.95, 0.42),
    "SER": (1.00, 0.30, 0.40),
    "OCT": (1.00, 0.74, 0.22),
    "TYR": (1.00, 0.48, 0.74),
}

# How long a spike stays visually hot, in seconds of viewer time.
FLASH_TAU_S = 0.35
# Share of the smoothed firing rate mixed under the spike flash.
RATE_MIX = 0.65
QUIET_FLOOR = 0.14

# The resting graph is a hint of connectivity, not the connectome. Long chords and
# a full edge upload are what make the live window stall and wash out to white.
BACKBONE_EDGES = 280
LIVE_EDGES = 90
LIVE_SOURCES = 48
QUIET_DRAWN = 6000
LOCAL_EDGE_LENGTH = 52.0
BRIDGE_EDGE_LENGTH = 115.0


class Neuropil(NamedTuple):
    """Right-side center (or the midline center when ``paired`` is false) and radii."""

    center: tuple[float, float, float]
    radii: tuple[float, float, float]
    paired: bool = True


# Hand-placed volumes in the axis convention above. Paired centers are the right side;
# the left side is the mirror image. Midline neuropils (central complex, GNG) are not mirrored.
NEUROPILS: dict[str, Neuropil] = {
    # optic lobes, lateral
    "ME": Neuropil((250.0, 48.0, -8.0), (62.0, 78.0, 42.0)),
    "LO": Neuropil((190.0, 36.0, 2.0), (28.0, 40.0, 26.0)),
    "LOP": Neuropil((162.0, 62.0, -26.0), (24.0, 28.0, 20.0)),
    "AME": Neuropil((178.0, 14.0, 30.0), (14.0, 14.0, 12.0)),
    # lateral and superior central brain
    "AVLP": Neuropil((130.0, 22.0, 24.0), (22.0, 18.0, 18.0)),
    "PVLP": Neuropil((140.0, 52.0, -6.0), (20.0, 16.0, 16.0)),
    "PLP": Neuropil((124.0, 82.0, -34.0), (18.0, 16.0, 16.0)),
    "LH": Neuropil((114.0, 112.0, -16.0), (16.0, 14.0, 14.0)),
    "SLP": Neuropil((84.0, 126.0, 2.0), (18.0, 14.0, 16.0)),
    "SMP": Neuropil((40.0, 132.0, 14.0), (22.0, 14.0, 18.0)),
    "SIP": Neuropil((52.0, 142.0, 38.0), (16.0, 12.0, 14.0)),
    "SCL": Neuropil((80.0, 92.0, 8.0), (14.0, 12.0, 12.0)),
    "ICL": Neuropil((66.0, 66.0, 16.0), (14.0, 12.0, 12.0)),
    "IB": Neuropil((50.0, 46.0, 24.0), (12.0, 12.0, 10.0)),
    "ATL": Neuropil((30.0, 116.0, 50.0), (10.0, 10.0, 10.0)),
    "CRE": Neuropil((48.0, 110.0, 24.0), (14.0, 12.0, 12.0)),
    "LAL": Neuropil((60.0, 26.0, 40.0), (16.0, 12.0, 14.0)),
    "VES": Neuropil((90.0, 4.0, 8.0), (14.0, 14.0, 12.0)),
    "SPS": Neuropil((98.0, 46.0, 10.0), (14.0, 12.0, 12.0)),
    "IPS": Neuropil((110.0, 26.0, 0.0), (14.0, 12.0, 12.0)),
    "WED": Neuropil((120.0, 6.0, -6.0), (14.0, 14.0, 12.0)),
    "AMMC": Neuropil((74.0, -8.0, 52.0), (16.0, 12.0, 14.0)),
    "SAD": Neuropil((48.0, -16.0, 34.0), (12.0, 10.0, 10.0)),
    "GOR": Neuropil((74.0, 56.0, 20.0), (10.0, 10.0, 8.0)),
    "EPA": Neuropil((88.0, 20.0, 18.0), (10.0, 10.0, 8.0)),
    "AOTU": Neuropil((110.0, 96.0, 30.0), (14.0, 12.0, 12.0)),
    "BU": Neuropil((54.0, 86.0, 32.0), (8.0, 8.0, 8.0)),
    "GA": Neuropil((42.0, 94.0, 26.0), (8.0, 8.0, 8.0)),
    # olfactory and learning centres
    "AL": Neuropil((40.0, 12.0, 86.0), (22.0, 20.0, 18.0)),
    "MB_CA": Neuropil((72.0, 138.0, -52.0), (16.0, 14.0, 14.0)),
    "MB_PED": Neuropil((56.0, 100.0, -6.0), (10.0, 22.0, 12.0)),
    "MB_VL": Neuropil((44.0, 122.0, 40.0), (10.0, 26.0, 10.0)),
    "MB_ML": Neuropil((24.0, 90.0, 46.0), (16.0, 8.0, 10.0)),
    # central complex and gnathal ganglion, on the midline
    "FB": Neuropil((0.0, 100.0, 10.0), (28.0, 16.0, 20.0), paired=False),
    "EB": Neuropil((0.0, 74.0, 48.0), (20.0, 10.0, 12.0), paired=False),
    "PB": Neuropil((0.0, 150.0, -14.0), (40.0, 8.0, 10.0), paired=False),
    "NO": Neuropil((0.0, 82.0, 4.0), (12.0, 12.0, 10.0), paired=False),
    "GNG": Neuropil((0.0, -40.0, 20.0), (36.0, 16.0, 22.0), paired=False),
    "PRW": Neuropil((16.0, -24.0, 10.0), (10.0, 8.0, 8.0)),
    "FLA": Neuropil((24.0, -6.0, 44.0), (10.0, 8.0, 8.0)),
    "CAN": Neuropil((32.0, -2.0, 20.0), (8.0, 8.0, 8.0)),
    # ventral nerve cord, neck then T1, T2, T3, abdomen
    "NTCT": Neuropil((18.0, -78.0, -6.0), (12.0, 14.0, 12.0)),
    "NTCT_UTCT_T1": Neuropil((20.0, -96.0, -14.0), (10.0, 10.0, 10.0)),
    "T1_PRONM": Neuropil((38.0, -120.0, -24.0), (18.0, 16.0, 14.0)),
    "LEGNP_T1": Neuropil((62.0, -128.0, -8.0), (14.0, 14.0, 12.0)),
    "AMNP": Neuropil((18.0, -150.0, -30.0), (12.0, 10.0, 10.0)),
    "WTCT": Neuropil((24.0, -168.0, -42.0), (14.0, 10.0, 12.0)),
    "WTCT_UTCT_T2": Neuropil((26.0, -180.0, -48.0), (10.0, 8.0, 8.0)),
    "T2_MVAC": Neuropil((30.0, -188.0, -32.0), (8.0, 8.0, 8.0)),
    "T2_MESONM": Neuropil((40.0, -210.0, -56.0), (18.0, 16.0, 14.0)),
    "INTTCT": Neuropil((16.0, -232.0, -62.0), (10.0, 8.0, 10.0)),
    "LTCT": Neuropil((18.0, -250.0, -68.0), (10.0, 8.0, 10.0)),
    "HTCT": Neuropil((24.0, -272.0, -78.0), (12.0, 10.0, 10.0)),
    "HTCT_UTCT_T3": Neuropil((26.0, -286.0, -84.0), (8.0, 8.0, 8.0)),
    "T3_METANM": Neuropil((42.0, -310.0, -92.0), (18.0, 16.0, 14.0)),
    "ABDNM": Neuropil((24.0, -370.0, -120.0), (16.0, 24.0, 14.0)),
    # not a neuropil: kept off to the side so background-only fragments stay visible
    "NO_CONS": Neuropil((360.0, -50.0, 30.0), (30.0, 40.0, 26.0)),
}

LABELS_3D: tuple[tuple[str, str], ...] = (
    ("ME", "optic lobe"),
    ("AL", "antennal lobe"),
    ("MB_CA", "mushroom body"),
    ("LH", "lateral horn"),
    ("FB", "central complex"),
    ("GNG", "GNG / SEZ"),
    ("T1_PRONM", "T1"),
    ("T2_MESONM", "T2"),
    ("T3_METANM", "T3"),
    ("ABDNM", "abdomen"),
)

_OPTIC = {"ME", "LO", "LOP", "AME"}
_MB = {"MB_CA", "MB_PED", "MB_VL", "MB_ML"}
_CX = {"FB", "EB", "PB", "NO"}
_SEZ = {"GNG", "PRW", "FLA", "CAN", "SAD", "AMMC"}
_T1 = {"T1_PRONM", "LEGNP_T1", "NTCT", "NTCT_UTCT_T1"}
_T2 = {"T2_MESONM", "T2_MVAC", "WTCT", "WTCT_UTCT_T2", "AMNP", "INTTCT"}
_T3 = {"T3_METANM", "HTCT", "HTCT_UTCT_T3", "LTCT"}
_FAMILY_RGB: dict[str, tuple[float, float, float]] = {
    "optic": (0.45, 0.62, 0.98),
    "al": (0.98, 0.58, 0.22),
    "mb": (0.30, 0.82, 0.48),
    "cx": (0.95, 0.82, 0.28),
    "lh": (0.98, 0.45, 0.62),
    "sez": (0.92, 0.38, 0.32),
    "t1": (0.25, 0.78, 0.72),
    "t2": (0.30, 0.62, 0.85),
    "t3": (0.62, 0.48, 0.90),
    "abd": (0.72, 0.52, 0.32),
    "none": (0.42, 0.44, 0.50),
    "brain": (0.58, 0.64, 0.78),
}


def primary_region(neurons: pd.DataFrame) -> np.ndarray:
    """First neuropil token of ``Top in/out region`` (``ME.LO`` -> ``ME``)."""
    if "Top in/out region" not in neurons.columns:
        return np.full(len(neurons), "NO_CONS", dtype=object)
    raw = neurons["Top in/out region"].fillna("NO_CONS").astype(str).str.upper()
    return raw.str.split(".").str[0].to_numpy()


def neuropil_center(name: str, side: str) -> np.ndarray:
    """Center of ``name`` on ``side`` (``left`` or ``right``)."""
    spec = NEUROPILS[name]
    sign = -1.0 if side == "left" else 1.0
    center = np.array(spec.center, np.float64)
    if spec.paired:
        center[0] *= sign
    else:
        center[0] += sign * spec.radii[0] * 0.15
    return center


def label_anchors() -> list[tuple[str, np.ndarray]]:
    """Short labels just lateral of the right-hand volume."""
    anchors = []
    for name, text in LABELS_3D:
        anchor = neuropil_center(name, "right")
        anchor = anchor.copy()
        anchor[0] += NEUROPILS[name].radii[0] + 16.0
        anchor[1] += NEUROPILS[name].radii[1] + 10.0
        anchors.append((text, anchor))
    return anchors


def _region_rgb(name: str) -> tuple[float, float, float]:
    if name in _OPTIC:
        return _FAMILY_RGB["optic"]
    if name == "AL":
        return _FAMILY_RGB["al"]
    if name in _MB:
        return _FAMILY_RGB["mb"]
    if name in _CX:
        return _FAMILY_RGB["cx"]
    if name == "LH":
        return _FAMILY_RGB["lh"]
    if name in _SEZ:
        return _FAMILY_RGB["sez"]
    if name in _T1:
        return _FAMILY_RGB["t1"]
    if name in _T2:
        return _FAMILY_RGB["t2"]
    if name in _T3:
        return _FAMILY_RGB["t3"]
    if name == "ABDNM":
        return _FAMILY_RGB["abd"]
    if name == "NO_CONS":
        return _FAMILY_RGB["none"]
    return _FAMILY_RGB["brain"]


def _sides(neurons: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
    n = len(neurons)
    side = np.zeros(n, np.float64)
    if "Soma side" in neurons.columns:
        raw = neurons["Soma side"].fillna("").astype(str).str.lower().to_numpy()
        side[raw == "right"] = 1.0
        side[raw == "left"] = -1.0
    missing = side == 0.0
    if missing.any():
        side[missing] = rng.choice(np.array([-1.0, 1.0]), int(missing.sum()))
    return side


def _in_ellipsoid(rng: np.random.Generator, n: int, radii: tuple[float, float, float]) -> np.ndarray:
    """Uniform points inside the ellipsoid, centered on the origin."""
    if n == 0:
        return np.zeros((0, 3), np.float32)
    direction = rng.normal(size=(n, 3))
    norm = np.linalg.norm(direction, axis=1, keepdims=True)
    direction /= np.maximum(norm, 1e-8)
    radius = rng.random(n) ** (1.0 / 3.0)
    scale = np.asarray(radii, np.float64)
    return (direction * radius[:, None] * scale).astype(np.float32)


def _place(spec: Neuropil, local: np.ndarray, signs: np.ndarray) -> np.ndarray:
    out = np.empty_like(local, dtype=np.float32)
    center = np.asarray(spec.center, np.float32)
    if spec.paired:
        out[:, 0] = signs * center[0] + local[:, 0]
    else:
        out[:, 0] = center[0] + local[:, 0] + signs * np.float32(spec.radii[0] * 0.15)
    out[:, 1] = center[1] + local[:, 1]
    out[:, 2] = center[2] + local[:, 2]
    return out


def neuron_positions_3d(neurons: pd.DataFrame, seed: int = 0) -> np.ndarray:
    """``(n, 3)`` layout positions, one per neuron row. See the module docstring."""
    rng = np.random.default_rng(seed)
    n = len(neurons)
    if "Top in/out region" in neurons.columns:
        raw = neurons["Top in/out region"].fillna("NO_CONS").astype(str).str.upper()
    else:
        raw = pd.Series(["NO_CONS"] * n)
    parts = raw.str.split(".", n=1, expand=True)
    primary = parts[0].to_numpy()
    if parts.shape[1] > 1:
        secondary = parts[1].fillna("").to_numpy()
    else:
        secondary = np.full(n, "", dtype=object)
    signs = _sides(neurons, rng)
    pos = np.zeros((n, 3), np.float32)
    groups = pd.DataFrame({"p": primary, "s": secondary, "i": np.arange(n)})
    for (pri, sec), sub in groups.groupby(["p", "s"], sort=False):
        pri, sec = str(pri), str(sec)
        rows = sub["i"].to_numpy()
        spec = NEUROPILS.get(pri, NEUROPILS["NO_CONS"])
        local = _in_ellipsoid(rng, len(rows), spec.radii)
        placed = _place(spec, local, signs[rows])
        if sec in NEUROPILS and sec != pri:
            anchor = _place(NEUROPILS[sec], np.zeros((len(rows), 3), np.float32), signs[rows])
            placed = placed * np.float32(0.65) + anchor * np.float32(0.35)
        pos[rows] = placed
    return pos


def base_colors(neurons: pd.DataFrame) -> np.ndarray:
    """``(n, 3)`` RGB in 0..1. Predicted transmitter wins; otherwise the neuropil family."""
    n = len(neurons)
    region = primary_region(neurons)
    if "Predicted NT type" in neurons.columns:
        nt = neurons["Predicted NT type"].fillna("").astype(str).str.upper().to_numpy()
    else:
        nt = np.full(n, "", dtype=object)
    codes, uniq = pd.factorize(pd.Series(region), sort=False)
    lut = np.array([_region_rgb(str(name)) for name in uniq], np.float32)
    rgb = lut[codes]
    nt_codes, nt_uniq = pd.factorize(pd.Series(nt), sort=False)
    nt_lut = np.array(
        [NT_RGB.get(str(name), (np.nan, np.nan, np.nan)) for name in nt_uniq], np.float32
    )
    known = np.isfinite(nt_lut).all(axis=1)
    use = known[nt_codes]
    rgb[use] = nt_lut[nt_codes[use]]
    return np.ascontiguousarray(rgb, dtype=np.float32)


def rate_glow(rates: np.ndarray) -> np.ndarray:
    """0..1 brightness from a smoothed firing rate. About 0.9 at 50 Hz (one fresh spike)."""
    level = np.maximum(np.asarray(rates, np.float32), 0.0)
    return (1.0 - np.exp(-level / np.float32(22.0))).astype(np.float32)


def decay_flash(flash: np.ndarray, dt_s: float, tau_s: float = FLASH_TAU_S) -> np.ndarray:
    """Exponential afterglow. ``dt_s`` is viewer time since the previous step."""
    factor = np.float32(np.exp(-max(float(dt_s), 0.0) / tau_s))
    return np.asarray(flash, np.float32) * factor


def absorb_spikes(
    flash: np.ndarray, spikes: np.ndarray | None, dt_s: float, tau_s: float = FLASH_TAU_S
) -> np.ndarray:
    """Decay the previous flash, then light every neuron that spiked in this window."""
    faded = decay_flash(flash, dt_s, tau_s)
    if spikes is None:
        return faded
    spiked = np.asarray(spikes) > 0
    if spiked.shape != faded.shape:
        return faded
    faded = faded.copy()
    faded[spiked] = 1.0
    return faded


def shown_glow(rates: np.ndarray, flash: np.ndarray) -> np.ndarray:
    """Brightness drawn for this frame: the spike flash sits above a dimmer rate glow."""
    return np.maximum(rate_glow(rates) * np.float32(RATE_MIX), np.asarray(flash, np.float32))


class Glow:
    """Per-neuron flash that jumps on a spike and fades between activity frames."""

    def __init__(self, n: int):
        self.flash = np.zeros(n, np.float32)
        self.rates = np.zeros(n, np.float32)

    def on_frame(self, rates: np.ndarray, spikes: np.ndarray | None, dt_s: float) -> np.ndarray:
        self.rates = np.asarray(rates, np.float32)
        self.flash = absorb_spikes(self.flash, spikes, dt_s)
        return self.shown()

    def on_tick(self, dt_s: float) -> np.ndarray:
        self.flash = decay_flash(self.flash, dt_s)
        return self.shown()

    def shown(self) -> np.ndarray:
        return shown_glow(self.rates, self.flash)


def rgba_from_glow(base_rgb: np.ndarray, glow: np.ndarray) -> np.ndarray:
    """Dim floor for quiet neurons, full transmitter color at glow 1."""
    level = np.clip(np.asarray(glow, np.float32), 0.0, 1.0)
    amp = (np.float32(QUIET_FLOOR) + np.float32(1.0 - QUIET_FLOOR) * level)[:, None]
    rgb = np.clip(np.asarray(base_rgb, np.float32) * amp, 0.0, 1.0)
    out = np.empty((len(level), 4), np.float32)
    out[:, :3] = rgb
    # Translucent body so a busy neuropil stays a cluster, not one white disk.
    out[:, 3] = np.float32(0.22) + np.float32(0.50) * level
    return out


def point_sizes(glow: np.ndarray) -> np.ndarray:
    """Pixel size. A spike is a few times a quiet dot, small enough that thousands stay distinct."""
    level = np.clip(np.asarray(glow, np.float32), 0.0, 1.0)
    return (np.float32(1.35) + np.float32(4.2) * level).astype(np.float32)


def quiet_display_index(names: np.ndarray, cap: int = QUIET_DRAWN) -> np.ndarray:
    """Row indices for the dim background cloud.

    Every neuropil keeps a few points, and larger ones keep more, but the total
    stays at ``cap``. Spiking neurons are drawn separately and are not dropped here.
    """
    names = np.asarray(names)
    n = len(names)
    if n == 0 or cap <= 0:
        return np.zeros(0, np.int32)
    codes, _ = pd.factorize(pd.Series(names), sort=False)
    counts = np.bincount(codes.astype(np.int32, copy=False))
    floor = np.minimum(counts, 12)
    if int(floor.sum()) >= cap or cap >= n:
        step = max(n // max(cap, 1), 1)
        return np.arange(0, n, step, dtype=np.int32)[:cap]
    remain = cap - int(floor.sum())
    extra_room = counts - floor
    weight = extra_room.astype(np.float64)
    extra = np.zeros(len(counts), np.int32)
    total = float(weight.sum())
    if total > 0 and remain > 0:
        raw = weight / total * remain
        extra = np.minimum(extra_room, np.floor(raw).astype(np.int32))
        short = remain - int(extra.sum())
        if short > 0:
            order = np.argsort(raw - np.floor(raw))[::-1]
            for code in order:
                if short <= 0:
                    break
                if extra[code] < extra_room[code]:
                    extra[code] += 1
                    short -= 1
    take = floor + extra
    chosen: list[np.ndarray] = []
    for code, k in enumerate(take):
        idx = np.flatnonzero(codes == code)
        k = int(min(int(k), idx.size))
        if k <= 0:
            continue
        if idx.size > k:
            idx = idx[np.linspace(0, idx.size - 1, k).astype(np.int32)]
        chosen.append(idx.astype(np.int32, copy=False))
    if not chosen:
        return np.zeros(0, np.int32)
    return np.concatenate(chosen)


def glowing_index(glow: np.ndarray, floor: float = 0.08) -> np.ndarray:
    """Neurons bright enough to redraw. Quiet ones stay on the static cloud."""
    level = np.asarray(glow, np.float32)
    return np.flatnonzero(level >= np.float32(floor)).astype(np.int32, copy=False)


def family_colors(names: np.ndarray) -> np.ndarray:
    """``(n, 3)`` neuropil-family color, used for the resting network rather than transmitter."""
    codes, uniq = pd.factorize(pd.Series(np.asarray(names)), sort=False)
    lut = np.array([_region_rgb(str(name)) for name in uniq], np.float32)
    return np.ascontiguousarray(lut[codes], dtype=np.float32)


# Wireframe volumes. Only the regions a watcher uses to orient; every neuron is still a point.
SHELL_REGIONS: tuple[str, ...] = (
    "ME", "LO", "AL", "MB_CA", "LH", "FB", "EB", "PB", "GNG",
    "T1_PRONM", "T2_MESONM", "T3_METANM", "ABDNM",
)


def _loop_segments(pts: np.ndarray, closed: bool) -> np.ndarray:
    if closed:
        nxt = np.roll(pts, -1, axis=0)
        src = pts
    else:
        src, nxt = pts[:-1], pts[1:]
    out = np.empty((len(src) * 2, 3), np.float32)
    out[0::2] = src
    out[1::2] = nxt
    return out


def _ellipsoid_loops(center: np.ndarray, radii: np.ndarray, n_arc: int) -> np.ndarray:
    """Latitude and longitude loops on an axis-aligned ellipsoid."""
    parts = []
    us = np.linspace(0.0, 2.0 * np.pi, n_arc, endpoint=False)
    for v in (-0.65, 0.0, 0.65):
        cv, sv = np.cos(v), np.sin(v)
        pts = np.stack(
            [radii[0] * np.cos(us) * cv, np.full(n_arc, radii[1] * sv), radii[2] * np.sin(us) * cv],
            axis=1,
        )
        parts.append(_loop_segments(pts + center, closed=True))
    vs = np.linspace(-np.pi / 2.0, np.pi / 2.0, max(n_arc // 2, 8))
    for u in (0.0, np.pi / 2.0, np.pi, 3.0 * np.pi / 2.0):
        cu, su = np.cos(u), np.sin(u)
        pts = np.stack(
            [radii[0] * cu * np.cos(vs), radii[1] * np.sin(vs), radii[2] * su * np.cos(vs)],
            axis=1,
        )
        parts.append(_loop_segments(pts + center, closed=False))
    return np.concatenate(parts)


def shell_wireframe(scale: float = 1.15, n_arc: int = 22) -> tuple[np.ndarray, np.ndarray]:
    """Faint ellipsoid shells around the orienting neuropils, left and right.

    Returns segment vertices ``(N, 3)`` and per-vertex RGBA. Paired neuropils are
    mirrored; midline ones (central complex, GNG) are drawn once.
    """
    segments: list[np.ndarray] = []
    colors: list[np.ndarray] = []
    for name in SHELL_REGIONS:
        spec = NEUROPILS[name]
        rgb = np.asarray(_region_rgb(name), np.float32)
        signs = (1.0,) if not spec.paired else (-1.0, 1.0)
        radii = np.asarray(spec.radii, np.float32) * np.float32(scale)
        for sign in signs:
            center = np.array(spec.center, np.float32)
            if spec.paired:
                center[0] *= np.float32(sign)
            loops = _ellipsoid_loops(center, radii, n_arc)
            segments.append(loops)
            col = np.empty((len(loops), 4), np.float32)
            col[:, :3] = rgb
            col[:, 3] = np.float32(0.42)
            colors.append(col)
    return np.concatenate(segments), np.concatenate(colors)


def format_hud(
    time_s: float,
    n: int,
    n_spiked: int,
    n_bright: int,
    awake: bool,
    note: str,
    triggers: dict | None,
) -> str:
    """Status block: clock, spikes, and the world / inside / body telemetry."""
    state = "" if awake else "   (brain still asleep: waiting for birth)"
    lines = [
        f"t = {time_s:6.3f} s   spiked {n_spiked:,}   bright {n_bright:,} / {n:,}{state}",
        "3D neuropil layout - BANC has no measured soma coordinates",
    ]
    if note:
        lines.append(note)
    body = describe_triggers(triggers or {})
    if body:
        lines.append(body)
    lines.append("ACh cyan   GABA violet   Glu orange   His yellow   DA green   5-HT red   OA amber")
    lines.append("drag orbit    right-drag pan    wheel zoom    R reset")
    return "\n".join(lines)


def _edge_lengths(positions: np.ndarray, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    delta = np.asarray(positions, np.float32)[rows] - np.asarray(positions, np.float32)[cols]
    return np.linalg.norm(delta, axis=1)


def backbone_from_csr(
    indptr: np.ndarray,
    indices: np.ndarray,
    data: np.ndarray,
    region_ids: np.ndarray,
    max_edges: int = BACKBONE_EDGES,
    seed: int = 0,
    positions: np.ndarray | None = None,
    max_length: float | None = None,
    bridge_length: float | None = None,
    within_fraction: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Capped strong edges. Most stay inside a neuropil; a few bridges connect regions.

    ``region_ids[i]`` is an integer neuropil code. Self-loops are dropped. When
    ``positions`` and a length limit are given, chords longer than that limit are
    left out so the drawing does not become a web across the whole nervous system.
    """
    indptr = np.asarray(indptr)
    indices = np.asarray(indices)
    weight = np.abs(np.asarray(data, np.float64))
    n = len(indptr) - 1
    if max_edges <= 0 or n == 0 or len(indices) == 0:
        return np.zeros(0, np.int32), np.zeros(0, np.int32)
    rows = np.repeat(np.arange(n, dtype=np.int32), np.diff(indptr))
    cols = indices.astype(np.int32, copy=False)
    valid = rows != cols
    rows, cols, weight = rows[valid], cols[valid], weight[valid]
    if rows.size == 0:
        return np.zeros(0, np.int32), np.zeros(0, np.int32)
    same = np.asarray(region_ids)[rows] == np.asarray(region_ids)[cols]
    rng = np.random.default_rng(seed)
    within_budget = int(max_edges * within_fraction)
    budgets = (
        (same, within_budget, max_length),
        (~same, max_edges - within_budget, bridge_length if bridge_length is not None else max_length),
    )
    chosen: list[np.ndarray] = []
    for mask, budget, limit in budgets:
        idx = np.flatnonzero(mask)
        if idx.size == 0 or budget <= 0:
            continue
        # Keep a strong pool first. Measuring every synapse in a 3-million-edge
        # cache on the UI thread is what makes the window stall before it draws.
        pool = max(budget * 24, budget)
        if idx.size > pool:
            w = weight[idx]
            cut = np.partition(w, -pool)[-pool]
            idx = idx[w >= cut]
            if idx.size > pool:
                idx = idx[np.argpartition(weight[idx], -pool)[-pool:]]
        if limit is not None and positions is not None and idx.size:
            length = _edge_lengths(positions, rows[idx], cols[idx])
            idx = idx[length <= limit]
        if idx.size == 0:
            continue
        if idx.size > budget:
            w = weight[idx]
            total = float(w.sum())
            prob = None if total <= 0.0 else w / total
            pick = rng.choice(idx.size, size=budget, replace=False, p=prob)
            idx = idx[pick]
        chosen.append(np.asarray(idx))
    if not chosen:
        return np.zeros(0, np.int32), np.zeros(0, np.int32)
    keep = np.concatenate(chosen)
    return rows[keep], cols[keep]


def live_synapse_edges(
    indptr: np.ndarray,
    indices: np.ndarray,
    data: np.ndarray,
    active: np.ndarray,
    max_edges: int = LIVE_EDGES,
    max_sources: int = LIVE_SOURCES,
    positions: np.ndarray | None = None,
    max_length: float | None = None,
    max_long: int = 40,
    long_length: float = BRIDGE_EDGE_LENGTH,
) -> tuple[np.ndarray, np.ndarray]:
    """Synapses among ``active`` neurons, strongest first, never more than ``max_edges``.

    With ``positions``, short edges fill the budget first and only ``max_long`` longer
    ones (still under ``long_length``) are kept, so a firing region shows its own web.
    """
    indptr = np.asarray(indptr)
    indices = np.asarray(indices)
    weight = np.abs(np.asarray(data, np.float64))
    n = len(indptr) - 1
    sources = np.unique(np.asarray(active, np.int64))
    sources = sources[(sources >= 0) & (sources < n)]
    if sources.size == 0 or max_edges <= 0:
        return np.zeros(0, np.int32), np.zeros(0, np.int32)
    if sources.size > max_sources:
        sources = sources[:max_sources]
    lit = np.zeros(n, np.bool_)
    lit[sources] = True
    pres: list[np.ndarray] = []
    posts: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    for i in sources:
        start, stop = int(indptr[i]), int(indptr[i + 1])
        if start == stop:
            continue
        cols = indices[start:stop]
        hit = lit[cols]
        if not np.any(hit):
            continue
        cols = cols[hit]
        w = weight[start:stop][hit]
        keep = cols != i
        cols, w = cols[keep], w[keep]
        if cols.size == 0:
            continue
        pres.append(np.full(cols.size, i, np.int32))
        posts.append(cols.astype(np.int32, copy=False))
        weights.append(w)
    if not pres:
        return np.zeros(0, np.int32), np.zeros(0, np.int32)
    pre = np.concatenate(pres)
    post = np.concatenate(posts)
    w = np.concatenate(weights)
    if positions is not None and max_length is not None and pre.size:
        length = _edge_lengths(positions, pre, post)
        short = length <= max_length
        bridge = (length > max_length) & (length <= long_length)
        keep_short = np.flatnonzero(short)
        keep_long = np.flatnonzero(bridge)
        short_budget = max(max_edges - max_long, 0)
        if short_budget == 0:
            keep_short = keep_short[:0]
        elif keep_short.size > short_budget:
            pick = np.argpartition(w[keep_short], -short_budget)[-short_budget:]
            keep_short = keep_short[pick]
        if max_long > 0 and keep_long.size > max_long:
            pick = np.argpartition(w[keep_long], -max_long)[-max_long:]
            keep_long = keep_long[pick]
        room = max(max_edges - int(keep_short.size), 0)
        if keep_long.size > room:
            if room == 0:
                keep_long = keep_long[:0]
            else:
                pick = np.argpartition(w[keep_long], -room)[-room:]
                keep_long = keep_long[pick]
        if keep_short.size == 0 and keep_long.size == 0:
            return np.zeros(0, np.int32), np.zeros(0, np.int32)
        chosen = np.concatenate([keep_short, keep_long])
        return pre[chosen], post[chosen]
    if pre.size > max_edges:
        pick = np.argpartition(w, -max_edges)[-max_edges:]
        pre, post = pre[pick], post[pick]
    return pre, post


def hottest_neurons(
    rates: np.ndarray, spikes: np.ndarray | None, limit: int = LIVE_SOURCES, min_score: float = 1.0
) -> np.ndarray:
    """Indices to consider as live-edge sources. A spike outranks a merely elevated rate."""
    score = np.asarray(rates, np.float32).astype(np.float32, copy=True)
    if spikes is not None:
        spiked = np.asarray(spikes) > 0
        if spiked.shape == score.shape:
            score[spiked] = np.maximum(score[spiked], np.float32(1000.0))
    idx = np.flatnonzero(score >= min_score)
    if idx.size > limit:
        pick = np.argpartition(score[idx], -limit)[-limit:]
        idx = idx[pick]
    return idx.astype(np.int64)


def segment_positions(positions: np.ndarray, pre: np.ndarray, post: np.ndarray) -> np.ndarray:
    """``(2E, 3)`` vertex buffer, pairs of endpoints, for ``connect='segments'``."""
    pre = np.asarray(pre, np.int64)
    post = np.asarray(post, np.int64)
    if pre.size == 0:
        return np.zeros((0, 3), np.float32)
    out = np.empty((pre.size * 2, 3), np.float32)
    out[0::2] = positions[pre]
    out[1::2] = positions[post]
    return out


def segment_colors(
    base_rgb: np.ndarray,
    pre: np.ndarray,
    alpha: float,
    gain: float,
    tint: tuple[float, float, float] = (0.0, 0.0, 0.0),
    tint_weight: float = 0.0,
) -> np.ndarray:
    """RGBA per vertex, taken from the presynaptic neuron and optionally pulled toward ``tint``."""
    if len(pre) == 0:
        return np.zeros((0, 4), np.float32)
    src = np.asarray(base_rgb, np.float32)[pre] * np.float32(gain)
    if tint_weight:
        src = src * np.float32(1.0 - tint_weight) + np.asarray(tint, np.float32) * np.float32(tint_weight)
    src = np.clip(src, 0.0, 1.0)
    col = np.empty((len(pre) * 2, 4), np.float32)
    col[0::2, :3] = src
    col[1::2, :3] = src
    col[:, 3] = np.float32(alpha)
    return col
