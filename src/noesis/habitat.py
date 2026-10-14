"""A small sensory habitat, in mm and seconds, with deterministic spatial fields.

Odors, heat, humidity and sound envelopes are phenomenological fields, not CFD,
thermodynamics or acoustic wave solvers. Their sources also appear in the scene.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

ODORS = ("fruity_volatile", "yeasty_volatile", "alcoholic_fermentation_volatile",
         "decaying_fruit_volatile", "aversive_volatile", "plant_matter_volatile")
TASTES = ("sugar", "water", "bitter")


@dataclass(frozen=True)
class Patch:
    name: str
    xy: tuple[float, float]
    radius: float
    rgba: tuple[float, float, float, float]
    taste: str | None = None


class Habitat:
    """Scene specification and position-dependent receptor stimuli."""

    half_size = 9.0
    patches = (
        Patch("fruit", (3.2, 1.8), 1.1, (1.0, 0.65, 0.08, 1), "sugar"),
        Patch("water", (-3.2, 2.5), 1.0, (0.08, 0.5, 1.0, 1), "water"),
        Patch("bitter", (2.0, -4.0), 0.9, (0.65, 0.15, 0.85, 1), "bitter"),
        Patch("warm", (-3.5, -3.0), 1.2, (1.0, 0.25, 0.08, 1)),
        Patch("plant", (5.5, 4.7), 0.9, (0.15, 0.8, 0.25, 1)),
        Patch("cool", (-6.0, -1.0), 1.0, (0.5, 0.9, 1.0, 1)),
        Patch("dry", (6.0, -5.0), 1.0, (0.75, 0.55, 0.3, 1)),
    )
    odor_positions = np.array([[3.2, 1.8, 0.25], [2.0, -4.0, 0.25], [5.5, 4.7, 0.5]])
    odor_strengths = np.array([[1.0, 0.85, 0.45, 0.35, 0, 0],
                               [0, 0, 0, 0, 0.9, 0], [0, 0, 0, 0, 0, 0.9]])

    @staticmethod
    def _points(points):
        points = np.asarray(points, float)
        if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
            raise ValueError("sensor positions must be finite (n, 3) coordinates")
        return points

    def odors_at(self, points, time_s=0.0):
        """Concentration (odor channels, sensors); bounded even at the source."""
        points = self._points(points)
        if not np.isfinite(time_s):
            raise ValueError("time must be finite")
        delta = points[None, :, :] - self.odor_positions[:, None, :]
        # A gently shifting plume; source locations remain fixed.
        delta[:, :, 0] -= 0.7 * np.sin(2 * np.pi * time_s / 3)
        field = np.exp(-np.sum((delta / np.array([4.0, 2.5, 2.0])) ** 2, axis=2) / 2)
        pulse = 0.85 + 0.15 * np.sin(2 * np.pi * time_s / 4 + np.arange(3))
        return np.clip(self.odor_strengths.T @ (field * pulse[:, None]), 0, 1)

    def tastes_at(self, points):
        """Ground patches activate taste only when a sensor reaches the surface."""
        points = self._points(points)
        taste = np.zeros((len(points), len(TASTES)))
        for patch in self.patches:
            if patch.taste:
                inside = np.linalg.norm(points[:, :2] - patch.xy, axis=1) <= patch.radius
                touching = (points[:, 2] >= -0.002) & (points[:, 2] <= 0.18)
                taste[:, TASTES.index(patch.taste)] = (inside & touching).astype(float)
        return taste

    def local_conditions(self, position, time_s):
        position = self._points(np.asarray(position)[None, :])[0]
        if not np.isfinite(time_s):
            raise ValueError("time must be finite")
        warm = np.exp(-np.sum((position - [-3.5, -3.0, 0]) ** 2) / (2 * 1.5**2))
        wet = np.exp(-np.sum((position - [-3.2, 2.5, 0]) ** 2) / (2 * 1.8**2))
        cool = np.exp(-np.sum((position - [-6.0, -1.0, 0]) ** 2) / (2 * 1.2**2))
        dry = np.exp(-np.sum((position - [6.0, -5.0, 0]) ** 2) / (2 * 1.5**2))
        sound_distance = np.sum((position - [4.5, -1.8, 0.5]) ** 2)
        sound = np.exp(-sound_distance / 40) * (0.5 + 0.5 * np.cos(2 * np.pi * time_s / 1.5))
        return {"temperature_c": float(24 + 10 * warm - 8 * cool),
                "humidity": float(np.clip(0.45 + 0.4 * wet - 0.25 * dry, 0, 1)),
                "sound_envelope": float(sound), "illumination": self.illumination(time_s)}

    @staticmethod
    def illumination(time_s):
        return float(0.7 + 0.3 * np.cos(2 * np.pi * time_s / 8))

    @staticmethod
    def shadow(time_s):
        """An autonomous moving, approaching dark visual target; no direct neural trigger."""
        phase = (time_s % 3.0) / 3.0
        return (5.0 * np.cos(2 * np.pi * phase),
                3.0 * np.sin(2 * np.pi * phase), 2.3 + 1.5 * (1 + np.cos(2 * np.pi * phase)))

    def manifest(self):
        return {"half_size_mm": self.half_size, "patches": [asdict(p) for p in self.patches],
                "odor_channels": ODORS, "odor_source_positions_mm": self.odor_positions.tolist(),
                "odor_strengths": self.odor_strengths.tolist(),
                "model": "analytic spatial fields; rendered binocular vision; contact taste",
                "dynamic_sources": ["odor plume", "light cycle", "moving dark target", "sound envelope"]}


def habitat_arena():
    """Build the physical/visible arena lazily, so field tests need no renderer."""
    from flygym.arena import FlatTerrain

    class SensoryHabitatArena(FlatTerrain):
        def __init__(self):
            super().__init__(size=(12, 12))
            self.habitat = Habitat()
            self.time_s = 0.0
            self.root_element.find("material", "grid").reflectance = 0
            world = self.root_element.worldbody
            for patch in self.habitat.patches:
                world.add("geom", name=f"patch_{patch.name}", type="cylinder",
                          pos=(*patch.xy, 0.015), size=(patch.radius, 0.015), rgba=patch.rgba)
            # Physical fruit, vegetation, obstacle and enclosing walls.
            world.add("geom", name="fruit_piece", type="ellipsoid", pos=(3.2, 1.8, 0.45),
                      size=(0.55, 0.45, 0.4), rgba=(1, 0.6, 0.05, 1))
            world.add("geom", name="green_stem", type="cylinder", pos=(5.5, 4.7, 1.0),
                      size=(0.25, 1.0), rgba=(0.1, 0.7, 0.2, 1))
            world.add("geom", name="obstacle", type="box", pos=(4.5, -1.8, 0.6),
                      size=(0.65, 0.65, 0.6), rgba=(0.35, 0.25, 0.12, 1))
            for side in (-1, 1):
                world.add("geom", name=f"wall_x_{side}", type="box", pos=(side * 9, 0, 1.75),
                          size=(0.15, 9, 1.75), rgba=(0.15, 0.65, 0.8, 1))
                world.add("geom", name=f"wall_y_{side}", type="box", pos=(0, side * 9, 1.75),
                          size=(9, 0.15, 1.75), rgba=(0.85, 0.8, 0.25, 1))
            for i, (x, color) in enumerate([(-5, (0.1, 0.3, 1, 1)), (0, (1, 0.9, 0.1, 1)),
                                             (5, (0.1, 0.9, 0.3, 1))]):
                world.add("geom", name=f"landmark_{i}", type="box", pos=(x, 8.7, 1.75),
                          size=(0.8, 0.08, 1.2), rgba=color)
            moving = world.add("body", name="visual_target", mocap=True, pos=self.habitat.shadow(0))
            # FlyGym excludes names containing 'sensor' from explicit ground contact pairs.
            moving.add("geom", name="sensor_visual_target", type="sphere", size=(0.6,),
                       rgba=(0.025, 0.025, 0.035, 1), contype=0, conaffinity=0)

        @property
        def odor_dimensions(self):
            return len(ODORS)

        def get_olfaction(self, sensor_pos):
            return self.habitat.odors_at(sensor_pos, self.time_s)

        def step(self, dt, physics, **kwargs):
            self.time_s = float(physics.data.time + dt)
            physics.named.data.mocap_pos["visual_target"] = self.habitat.shadow(self.time_s)
            strength = self.habitat.illumination(self.time_s)
            for name in ("light_top", "light_bottom"):
                physics.named.model.light_diffuse[name] = (0.7 * strength,) * 3

    return SensoryHabitatArena()
