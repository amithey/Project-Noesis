"""Homeostatic internal state: energy and water fall with time and rise with ingestion.

Hunger and thirst are measured deficits, not feelings. They (a) scale sensory gain and (b) yield
reward/punishment signals for dopamine-gated plasticity. No neuron is addressed here; callers map
the numbers onto sensory drive weights and dopaminergic input.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

FOOD_CHANNELS = ("fruity_volatile", "yeasty_volatile", "alcoholic_fermentation_volatile",
                 "decaying_fruit_volatile")


def _check(name: str, value: float, low: float = 0.0) -> float:
    if not isfinite(value) or value < low:
        raise ValueError(f"{name} must be finite and >= {low}")
    return float(value)


@dataclass(frozen=True)
class DriveParams:
    energy_decay_per_s: float = 0.01
    water_decay_per_s: float = 0.015
    sugar_gain_per_s: float = 0.2      # per unit sugar contact intensity
    water_gain_per_s: float = 0.3
    bitter_cost_per_s: float = 0.0     # optional energy loss on bitter contact
    hunger_gain: float = 1.0           # food-channel gain = 1 + hunger_gain * hunger
    bitter_tolerance: float = 0.5      # bitter gain = max(0, 1 - bitter_tolerance * hunger)

    def __post_init__(self):
        for name in self.__dataclass_fields__:
            _check(name, getattr(self, name))


@dataclass
class StepResult:
    reward: float       # increase in total (energy + water) this step, >= 0
    punishment: float   # bitter contact intensity * dt, >= 0
    alive: bool


class HomeostaticState:
    def __init__(self, params: DriveParams | None = None, energy: float = 1.0, water: float = 1.0):
        self.p = params or DriveParams()
        for name, value in (("energy", energy), ("water", water)):
            if not isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be within [0, 1]")
        self.energy, self.water = float(energy), float(water)

    @property
    def hunger(self) -> float:
        return 1.0 - self.energy

    @property
    def thirst(self) -> float:
        return 1.0 - self.water

    @property
    def alive(self) -> bool:
        return self.energy > 0.0 and self.water > 0.0

    def step(self, dt_s: float, sugar: float = 0.0, water: float = 0.0, bitter: float = 0.0) -> StepResult:
        """Advance ``dt_s`` with contact intensities in [0, inf) for sugar, water and bitter."""
        dt_s = _check("dt_s", dt_s)
        sugar, water, bitter = _check("sugar", sugar), _check("water", water), _check("bitter", bitter)
        p = self.p
        before = self.energy + self.water
        energy = self.energy - (p.energy_decay_per_s + p.bitter_cost_per_s * bitter) * dt_s
        water_level = self.water - p.water_decay_per_s * dt_s
        # Ingestion cannot create reward beyond satiation: it saturates at 1.
        intake_e, intake_w = p.sugar_gain_per_s * sugar * dt_s, p.water_gain_per_s * water * dt_s
        self.energy = min(1.0, max(0.0, energy + intake_e))
        self.water = min(1.0, max(0.0, water_level + intake_w))
        reward = max(0.0, (self.energy + self.water) - before + (self.p.energy_decay_per_s
                     + self.p.water_decay_per_s) * dt_s)  # net intake beyond the passive decay
        return StepResult(reward=reward, punishment=bitter * dt_s, alive=self.alive)

    def sensory_gain(self, channel: str) -> float:
        """Multiplier for a sensory channel's drive weight given the current state."""
        if channel in FOOD_CHANNELS or channel == "sugar":
            return 1.0 + self.p.hunger_gain * self.hunger
        if channel == "water":
            return 1.0 + self.p.hunger_gain * self.thirst
        if channel in ("bitter", "aversive_volatile"):
            return max(0.0, 1.0 - self.p.bitter_tolerance * self.hunger)
        return 1.0
