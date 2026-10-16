"""A leg-free 2D point agent steered by a brain-derived left/right signal.

Brain-agnostic: ``SteeringAgent`` only needs, each control step, the left/right firing rates of two
neuron groups. No physics, no legs, no odour recognition: the odour field is a single Gaussian plume and
the agent walks at a constant speed; only its heading changes. The sign convention (more activity on the
left turns the agent left) is an ASSUMPTION taken from the literature on DNa steering cells, not derived
from the connectome here; experiments therefore include a reversed-sign control.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import atan2, cos, exp, hypot, isfinite, pi, sin

import numpy as np


@dataclass(frozen=True)
class AgentParams:
    speed_mm_s: float = 8.0
    antenna_mm: float = 1.0          # lateral offset of each antenna from the body axis
    gain_rad_s: float = 40.0         # turn rate per unit steering signal
    max_turn_rad_s: float = 2.0
    smooth_s: float = 1.0            # low-pass time constant of the steering signal
    rot_diffusion_rad_sqrt_s: float = 0.3
    food_radius_mm: float = 3.0
    arena_radius_mm: float = 60.0

    def __post_init__(self):
        for name in self.__dataclass_fields__:
            v = getattr(self, name)
            if not isfinite(v) or v < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.smooth_s <= 0:
            raise ValueError("smooth_s must be positive")


def plume(x: float, y: float, food_xy=(0.0, 0.0), sigma_mm: float = 10.0) -> float:
    """Odour concentration in [0, 1]: Gaussian around the food source."""
    d2 = (x - food_xy[0]) ** 2 + (y - food_xy[1]) ** 2
    return exp(-d2 / (2 * sigma_mm ** 2))


def steering_signal(left_hz: float, right_hz: float) -> float:
    """(L - R) / (L + R) of two group rates; 0 when both are silent. Positive = more left activity."""
    if left_hz < 0 or right_hz < 0:
        raise ValueError("rates must be non-negative")
    total = left_hz + right_hz
    return 0.0 if total < 1e-9 else (left_hz - right_hz) / total


class SteeringAgent:
    def __init__(self, xy, heading_rad: float, params: AgentParams | None = None, seed: int = 0,
                 bias: float = 0.0, sign: float = 1.0):
        self.p = params or AgentParams()
        self.x, self.y = float(xy[0]), float(xy[1])
        self.theta = float(heading_rad)
        self.rng = np.random.default_rng(seed)
        self.bias, self.sign = bias, sign
        self.signal = 0.0          # smoothed steering signal
        self.path = [(self.x, self.y)]

    def antenna_positions(self):
        a = self.p.antenna_mm
        left = (self.x - a * sin(self.theta), self.y + a * cos(self.theta))
        right = (self.x + a * sin(self.theta), self.y - a * cos(self.theta))
        return left, right

    def step(self, dt_s: float, left_hz: float | None, right_hz: float | None) -> None:
        """Advance dt_s. ``None`` rates mean no steering input (diffusion-only control)."""
        if not isfinite(dt_s) or dt_s <= 0:
            raise ValueError("dt_s must be finite and positive")
        if left_hz is not None and right_hz is not None:
            raw = self.sign * (steering_signal(left_hz, right_hz) - self.bias)
            self.signal += (raw - self.signal) * (1 - exp(-dt_s / self.p.smooth_s))
        turn = float(np.clip(self.p.gain_rad_s * self.signal, -self.p.max_turn_rad_s, self.p.max_turn_rad_s))
        self.theta += turn * dt_s + self.rng.normal(0.0, self.p.rot_diffusion_rad_sqrt_s * dt_s ** 0.5)
        self.theta = (self.theta + pi) % (2 * pi) - pi
        self.x += self.p.speed_mm_s * dt_s * cos(self.theta)
        self.y += self.p.speed_mm_s * dt_s * sin(self.theta)
        self.path.append((self.x, self.y))

    def distance_to(self, xy) -> float:
        return hypot(self.x - xy[0], self.y - xy[1])

    def bearing_error(self, xy) -> float:
        """Signed angle (rad) from heading to the direction of ``xy``, in (-pi, pi]."""
        err = atan2(xy[1] - self.y, xy[0] - self.x) - self.theta
        return (err + pi) % (2 * pi) - pi


def path_efficiency(path) -> float:
    """Straight-line displacement divided by path length (1 = straight, 0 = no net displacement)."""
    pts = np.asarray(path, float)
    length = float(np.sum(np.hypot(*np.diff(pts, axis=0).T))) if len(pts) > 1 else 0.0
    return 0.0 if length == 0 else float(np.hypot(*(pts[-1] - pts[0])) / length)
