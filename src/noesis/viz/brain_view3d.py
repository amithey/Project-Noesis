"""Live 3D view of every neuron that drives the simulation.

``Run-Fly.ps1`` opens this beside the MuJoCo body. Drag to orbit, right-drag to
pan, the wheel zooms. A spike flashes in the neuron's transmitter color and fades;
quiet neurons stay dim. A spike is a bright core with a short colored afterglow.
Edges are a short capped backbone inside each neuropil, plus the synapses among
neurons that just fired — never the full connectome.

    python -m noesis.viz.brain_view3d --data DATA --watch ACTIVITY.npz
    python -m noesis.viz.brain_view3d --data DATA --snapshot demo --out outputs/brain_view3d.png
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from noesis.viz.brain_map import read_activity_frame
from noesis.viz.layout3d import (
    BACKBONE_EDGES,
    BRIDGE_EDGE_LENGTH,
    LIVE_EDGES,
    LOCAL_EDGE_LENGTH,
    Glow,
    backbone_from_csr,
    base_colors,
    family_colors,
    format_hud,
    hottest_neurons,
    label_anchors,
    live_synapse_edges,
    neuron_positions_3d,
    point_sizes,
    primary_region,
    rgba_from_glow,
    segment_colors,
    segment_positions,
    shell_wireframe,
)

LAYOUT_NOTE = "3D neuropil layout (not measured BANC coordinates)"


def parse_window_geometry(spec: str | None) -> tuple[tuple[int, int], tuple[int, int] | None]:
    """``760x920+940+30`` -> size ``(760, 920)`` and origin ``(940, 30)``."""
    if not spec:
        return (980, 1040), None
    import re

    match = re.fullmatch(r"(\d+)x(\d+)(?:\+(-?\d+)\+(-?\d+))?", spec.strip())
    if match is None:
        raise ValueError(f"geometry must look like 760x920+940+30, not {spec!r}")
    size = (int(match.group(1)), int(match.group(2)))
    if match.group(3) is None:
        return size, None
    return size, (int(match.group(3)), int(match.group(4)))


def load_neurons(data: Path) -> pd.DataFrame:
    """Neuron table in connectome-matrix order."""
    parquet = data / "processed" / "neurons.parquet"
    if parquet.exists():
        return pd.read_parquet(parquet)
    from noesis.connectome import load_cached

    return load_cached(data / "raw", data / "processed").neurons


def load_edge_matrix(data: Path, scaled: bool):
    """Cached sparse weights, or None when the cache is missing."""
    folder = data / "processed"
    if scaled:
        found = sorted(folder.glob("W_min5_scaled*.npz"), key=lambda path: path.stat().st_mtime)
        path = found[-1] if found else None
    else:
        path = folder / "W_min5.npz"
        if path is not None and not path.exists():
            path = None
    if path is None or not path.exists():
        return None
    from scipy import sparse

    return sparse.load_npz(path).tocsr()


def demo_activity(neurons: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, dict, str]:
    """One synthetic frame: antennal lobe and front-leg neuropil lit, plus a scatter."""
    n = len(neurons)
    spikes = np.zeros(n, np.uint16)
    spikes[::350] = 1
    region = primary_region(neurons)
    spikes[region == "AL"] = 1
    spikes[region == "T1_PRONM"] = 1
    rates = spikes.astype(np.float32) * np.float32(55.0)
    triggers = {
        "odor_food": 0.62, "odor_aversive": 0.05, "taste_sugar": 0.4, "taste_water": 0.0,
        "taste_bitter": 0.0, "vision_mean": 0.35, "temperature_c": 24.5, "humidity": 0.48,
        "hemolymph_sugar": 0.55, "hemolymph_hydration": 0.70, "hemolymph_oxygen": 0.93,
        "heart_rate_hz": 4.2, "mn9_hz": 6.0, "dilp_hz": 3.0,
        "body_z_mm": 1.10, "uprightness": 0.97, "muscle_pos_mean": 0.18, "muscle_neg_mean": 0.16,
        "da_hz": 2.0, "oa_hz": 3.5, "ser_hz": 1.4,
    }
    return rates, spikes, triggers, "demo frame - antennal lobe and T1 lit"


def fit_center(positions: np.ndarray, names: np.ndarray | None = None) -> tuple[np.ndarray, float]:
    """Camera center and span of the nervous system, brain and nerve cord together.

    Unconnected fragments (``NO_CONS``) sit far to the side and are left out of the
    frame so the neuropils fill the window.
    """
    use = positions
    if names is not None:
        keep = np.asarray(names) != "NO_CONS"
        if bool(keep.any()):
            use = positions[keep]
    lo, hi = use.min(axis=0), use.max(axis=0)
    return (lo + hi) / 2.0, float(np.max(hi - lo))


def _bind():
    import vispy

    vispy.use(app="glfw")
    from vispy import app, scene
    from vispy.util import keys

    return scene, app, keys


def _orbit_camera(scene, keys):
    """Left-drag orbits, right-drag pans, wheel zooms. Shift-left also pans."""

    class OrbitPanCamera(scene.TurntableCamera):
        def viewbox_mouse_event(self, event):
            if event.handled or not self.interactive:
                return
            scene.cameras.PerspectiveCamera.viewbox_mouse_event(self, event)
            if event.type == "mouse_release":
                self._event_value = None
            elif event.type == "mouse_press":
                event.handled = True
            elif event.type == "mouse_move" and event.press_event is not None:
                if 1 in event.buttons and 2 in event.buttons:
                    return
                modifiers = event.mouse_event.modifiers
                p1 = event.mouse_event.press_event.pos
                p2 = event.mouse_event.pos
                if 1 in event.buttons and not modifiers:
                    self._update_rotation(event)
                elif self._pan_gesture(event.buttons, modifiers, keys):
                    self._pan(p1, p2)

        @staticmethod
        def _pan_gesture(buttons, modifiers, keys) -> bool:
            right_or_middle = (2 in buttons or 3 in buttons) and not modifiers
            shift_left = 1 in buttons and keys.SHIFT in modifiers
            return right_or_middle or shift_left

        def _pan(self, p1, p2):
            norm = np.mean(self._viewbox.size)
            if norm <= 0:
                return
            if self._event_value is None or len(self._event_value) == 2:
                self._event_value = self.center
            dist = (p1 - p2) / norm * self._scale_factor
            dist[1] *= -1
            dx, dy, dz = self._dist_to_trans(dist)
            ff = self._flip_factors
            up, forward, right = self._get_dim_vectors()
            dx, dy, dz = right * dx + forward * dy + up * dz
            dx, dy, dz = ff[0] * dx, ff[1] * dy, dz * ff[2]
            c = self._event_value
            self.center = c[0] + dx, c[1] + dy, c[2] + dz

    return OrbitPanCamera


def _hidden_segment(center: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    seg = np.vstack([center, center]).astype(np.float32)
    col = np.zeros((2, 4), np.float32)
    return seg, col


def _edge_buffers(positions, base_rgb, pre, post, alpha, gain, center, tint=(0.0, 0.0, 0.0), tint_weight=0.0):
    if len(pre) == 0:
        return _hidden_segment(center)
    return (
        segment_positions(positions, pre, post),
        segment_colors(base_rgb, pre, alpha, gain, tint=tint, tint_weight=tint_weight),
    )


class NetworkView:
    """Point-sprite neurons, capped edges, and a telemetry strip."""

    def __init__(self, positions, base_rgb, matrix, region_ids, names, size, position, title):
        scene, self.app, keys = _bind()
        self.scene = scene
        self.positions = np.ascontiguousarray(positions, np.float32)
        self.base_rgb = np.ascontiguousarray(base_rgb, np.float32)
        self.names = np.asarray(names)
        self.family = family_colors(self.names)
        self.n = len(self.positions)
        self.glow = Glow(self.n)
        self.matrix = matrix
        self.region = region_ids
        self.backbone = self._backbone()
        self._fit_center, self._fit_span = fit_center(self.positions, self.names)
        kwargs = {"size": size, "bgcolor": "black", "keys": "interactive", "show": False, "title": title}
        if position is not None:
            kwargs["position"] = position
        self.canvas = scene.SceneCanvas(**kwargs)
        grid = self.canvas.central_widget.add_grid()
        self.view = grid.add_view(row=0, col=0)
        self.hud_view = grid.add_view(row=1, col=0)
        self.view.stretch = (1, 5)
        self.hud_view.stretch = (1, 1)
        self.view.bgcolor = "black"
        self.hud_view.bgcolor = "#10141c"
        self.camera = _orbit_camera(scene, keys)(
            fov=45, elevation=24, azimuth=-68, up="+y"
        )
        self.view.camera = self.camera
        self.home = None
        mean = self.positions.mean(axis=0)
        shell_pos, shell_col = shell_wireframe()
        self.shell_line = scene.visuals.Line(
            pos=shell_pos, color=shell_col, connect="segments", width=1.15, method="gl",
            parent=self.view.scene, antialias=True,
        )
        self.shell_line.set_gl_state(depth_test=False, blend=True)
        back_seg, back_col = _edge_buffers(
            self.positions, self.family, self.backbone[0], self.backbone[1], 0.30, 0.70, mean,
        )
        self.backbone_line = scene.visuals.Line(
            pos=back_seg, color=back_col, connect="segments", width=1.15, method="gl",
            parent=self.view.scene, antialias=False,
        )
        self.backbone_line.set_gl_state(depth_test=False, blend=True)
        seg, col = _hidden_segment(mean)
        self.live_line = scene.visuals.Line(
            pos=seg, color=col, connect="segments", width=2.4, method="gl",
            parent=self.view.scene, antialias=False,
        )
        self.live_line.set_gl_state(depth_test=False, blend=True)
        shown = np.zeros(self.n, np.float32)
        self.markers = scene.visuals.Markers(parent=self.view.scene)
        self.markers.set_gl_state(
            depth_test=True, blend=True, blend_func=("src_alpha", "one_minus_src_alpha")
        )
        self.markers.set_data(
            self.positions, face_color=rgba_from_glow(self.base_rgb, shown),
            size=point_sizes(shown), edge_width=0.0, symbol="o",
        )
        self.cores = scene.visuals.Markers(parent=self.view.scene)
        self.cores.set_gl_state(depth_test=False, blend=True)
        self.cores.set_data(
            self.positions[:1], face_color=np.zeros((1, 4), np.float32),
            size=np.array([0.1], np.float32), edge_width=0.0, symbol="o",
        )
        for text, anchor in label_anchors():
            scene.visuals.Text(
                text, pos=anchor, color=(0.90, 0.94, 1.0, 1.0), font_size=15,
                anchor_x="left", anchor_y="center", parent=self.view.scene,
            )
        self.hud_view.camera = scene.PanZoomCamera(rect=(0, 0, 100, 100), aspect=None)
        self.hud_view.camera.interactive = False
        self.hud_lines = []
        for i in range(8):
            self.hud_lines.append(scene.visuals.Text(
                "", color="#f2f6ff", font_size=11, anchor_x="left", anchor_y="top",
                parent=self.hud_view.scene, pos=(2.0, 84 - i * 9),
            ))
        self._lock_hud()
        self.canvas.events.key_press.connect(self._on_key)
        self._frame_home()

    def _backbone(self):
        if self.matrix is None or self.matrix.shape[0] != self.n:
            return np.zeros(0, np.int32), np.zeros(0, np.int32)
        return backbone_from_csr(
            self.matrix.indptr, self.matrix.indices, self.matrix.data, self.region,
            max_edges=BACKBONE_EDGES, positions=self.positions,
            max_length=LOCAL_EDGE_LENGTH, bridge_length=BRIDGE_EDGE_LENGTH,
            within_fraction=0.94,
        )

    def _frame_home(self) -> None:
        center, span = self._fit_center, self._fit_span
        self.camera.center = (float(center[0]), float(center[1]), float(center[2]))
        self.camera.fov = 45
        self.camera.elevation = 24
        self.camera.azimuth = -68
        self.camera.scale_factor = max(span, 1.0) * 0.72
        self.home = (
            self.camera.center, self.camera.scale_factor, self.camera.elevation, self.camera.azimuth
        )

    def _on_key(self, event) -> None:
        if getattr(event, "key", None) in ("R", "r") and self.home is not None:
            center, scale, elevation, azimuth = self.home
            self.camera.center = center
            self.camera.scale_factor = scale
            self.camera.elevation = elevation
            self.camera.azimuth = azimuth

    def paint(self, shown: np.ndarray) -> None:
        level = np.clip(np.asarray(shown, np.float32), 0.0, 1.0)
        self.markers.set_data(
            self.positions, face_color=rgba_from_glow(self.base_rgb, level),
            size=point_sizes(level), edge_width=0.0, symbol="o",
        )
        # Every spike in this window gets a sharp core, drawn in front of the cloud.
        hot = np.flatnonzero(level >= np.float32(0.72))
        if hot.size == 0:
            self.cores.set_data(
                self.positions[:1], face_color=np.zeros((1, 4), np.float32),
                size=np.array([0.1], np.float32), edge_width=0.0, symbol="o",
            )
            return
        color = np.empty((hot.size, 4), np.float32)
        color[:, :3] = np.clip(self.base_rgb[hot] * np.float32(0.28) + np.float32(0.78), 0.0, 1.0)
        color[:, 3] = 1.0
        self.cores.set_data(
            self.positions[hot], face_color=color,
            size=np.full(hot.size, 4.4, np.float32), edge_width=0.0, symbol="o",
        )

    def set_live_edges(self, pre: np.ndarray, post: np.ndarray) -> None:
        seg, col = _edge_buffers(
            self.positions, self.base_rgb, pre, post, 0.96, 1.05, self.positions.mean(axis=0),
            tint=(1.0, 0.96, 0.82), tint_weight=0.62,
        )
        self.live_line.set_data(pos=seg, color=col, connect="segments")

    def _lock_hud(self) -> None:
        """Keep the telemetry strip in a fixed 0..100 box so text stays left-aligned."""
        self.hud_view.camera.rect = (0, 0, 100, 100)

    def set_hud(self, text: str) -> None:
        lines = text.split("\n")
        for i, visual in enumerate(self.hud_lines):
            visual.text = lines[i] if i < len(lines) else ""
        self._lock_hud()

    def render_image(self) -> np.ndarray:
        self._lock_hud()
        self.app.process_events()
        self._lock_hud()
        return self.canvas.render(alpha=False)


def region_codes(names: np.ndarray) -> np.ndarray:
    codes, _ = pd.factorize(pd.Series(names), sort=False)
    return codes.astype(np.int32)


def build_view(neurons, positions, matrix, size, position, title) -> NetworkView:
    names = primary_region(neurons)
    return NetworkView(
        positions, base_colors(neurons), matrix, region_codes(names), names, size, position, title
    )


def apply_activity(view: NetworkView, rates, spikes, time_s, awake, note, triggers, dt_s) -> np.ndarray:
    """Push one activity frame into the view. Returns the brightness just drawn."""
    if len(rates) != view.n:
        raise ValueError(f"activity length {len(rates)} does not match {view.n} neurons")
    shown = view.glow.on_frame(rates, spikes, dt_s)
    view.paint(shown)
    if view.matrix is not None and view.matrix.shape[0] == view.n:
        sources = hottest_neurons(rates, spikes)
        pre, post = live_synapse_edges(
            view.matrix.indptr, view.matrix.indices, view.matrix.data, sources,
            max_edges=LIVE_EDGES, positions=view.positions, max_length=LOCAL_EDGE_LENGTH,
            max_long=16, long_length=BRIDGE_EDGE_LENGTH,
        )
        view.set_live_edges(pre, post)
    n_spiked = int(np.count_nonzero(spikes)) if spikes is not None else 0
    n_bright = int(np.count_nonzero(shown > 0.45))
    view.set_hud(format_hud(time_s, view.n, n_spiked, n_bright, awake, note, triggers))
    view.canvas.title = f"Noesis 3D network | t={time_s:.2f}s | spiked {n_spiked:,}"
    return shown


def render_still(neurons, positions, matrix, rates, spikes, triggers, note, out: Path,
                 size=(960, 1000)) -> Path:
    """One offscreen frame. Proves the points accept a spike update without a live sim."""
    view = build_view(neurons, positions, matrix, size, None, "Noesis 3D network")
    apply_activity(view, rates, spikes, 1.0, True, note, triggers, 0.0)
    image = view.render_image()
    out.parent.mkdir(parents=True, exist_ok=True)
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    plt.imsave(out, np.asarray(image))
    view.canvas.close()
    return out


def watch(neurons, positions, matrix, activity: Path, geometry: str | None) -> None:
    """Open the interactive window and follow ``activity`` until it closes."""
    size, position = parse_window_geometry(geometry)
    view = build_view(
        neurons, positions, matrix, size, position, f"Noesis 3D network — {LAYOUT_NOTE}"
    )
    view.set_hud(format_hud(0.0, view.n, 0, 0, False, f"waiting for {activity.name}", {}))
    view.canvas.show()
    state = {"stamp": 0, "wall": time.perf_counter(), "dirty": True}

    def on_timer(_event):
        now = time.perf_counter()
        dt = min(now - state["wall"], 0.25)
        state["wall"] = now
        fresh = None
        try:
            stamp = activity.stat().st_mtime_ns
        except OSError:
            stamp = state["stamp"]
        if stamp != state["stamp"]:
            try:
                fresh = read_activity_frame(activity)
            except (OSError, ValueError, KeyError, EOFError, json_error()):
                fresh = None
        if fresh is not None and len(fresh.rates) == view.n:
            try:
                apply_activity(
                    view, fresh.rates, fresh.spikes, fresh.time_s, fresh.awake, fresh.note,
                    fresh.triggers, dt,
                )
                state["stamp"] = stamp
                state["dirty"] = True
            except ValueError:
                view.glow.on_tick(dt)
        else:
            shown = view.glow.on_tick(dt)
            if state["dirty"] or float(shown.max()) > 0.02:
                view.paint(shown)
                state["dirty"] = float(shown.max()) > 0.02
        view._lock_hud()
        view.canvas.update()

    view._timer = view.app.Timer(interval=1 / 30, connect=on_timer, start=True)
    view.app.run()


def json_error():
    import json
    return json.JSONDecodeError


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--watch", type=Path, help="activity file written by the simulation")
    ap.add_argument("--snapshot", type=Path, help="render this activity file once, or 'demo'")
    ap.add_argument("--out", type=Path, default=Path("outputs/brain_view3d.png"))
    ap.add_argument("--geometry", type=str, default=None, help="WxH+X+Y, e.g. 760x920+940+30")
    ap.add_argument("--scaled", action="store_true",
                    help="backbone edges from the FAFB/MANC-scaled weight cache")
    ap.add_argument("--no-edges", action="store_true", help="points only")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.snapshot is None and args.watch is None:
        ap.error("use --watch FILE or --snapshot FILE")

    neurons = load_neurons(args.data)
    positions = neuron_positions_3d(neurons, seed=args.seed)
    matrix = None if args.no_edges else load_edge_matrix(args.data, args.scaled)
    n_edges = 0 if matrix is None else int(matrix.nnz)
    print(
        f"{LAYOUT_NOTE}: {len(neurons):,} neurons"
        + (f", weight cache {n_edges:,} synapses (edges are capped)" if matrix is not None else ""),
        flush=True,
    )
    if args.snapshot is not None and str(args.snapshot) == "demo":
        rates, spikes, triggers, note = demo_activity(neurons)
        path = render_still(neurons, positions, matrix, rates, spikes, triggers, note, args.out)
        print("saved", path, flush=True)
        return
    if args.snapshot is not None:
        frame = read_activity_frame(args.snapshot)
        path = render_still(
            neurons, positions, matrix, frame.rates, frame.spikes, frame.triggers, frame.note, args.out
        )
        print("saved", path, flush=True)
        return
    watch(neurons, positions, matrix, args.watch, args.geometry)


if __name__ == "__main__":
    main()
