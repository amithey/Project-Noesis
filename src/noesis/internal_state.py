"""Hemolymph and internal-organ physiological state coupled to BANC interoceptive/visceral circuits.

The BANC connectome includes annotated internal sensory and neurosecretory neurons:
- ``nutrient_sensory_neuron`` (5 cells: 4 ISN interoceptive, 1 SLP444 Gr43a hemolymph sugar sensor)
- ``oxygenation_neuron`` (20 abdominal oxygenation afferents)
- ``aorta_neuron`` (1 contractile aorta afferent)
- ``internal_taste_sensillum_gustatory_neuron`` (87 pharyngeal/digestive ingested-material sensors)
- ``internal_sensillum_tactile_neuron`` (51 pharyngeal tactile sensors)
- Efferent visceral/feeding motor & neurosecretory populations:
  ``proboscis_motor_neuron`` (including MN9), ``pharynx_motor_neuron`` (pharyngeal pumping),
  ``crop_motor_neuron``, ``salivary_motor_neuron``, ``pars_intercerebralis_neurosecretory_cell``
  (m_NSC_DILP insulin-like, m_NSC_DH44), ``pars_lateralis_neurosecretory_cell`` (l_NSC_CRZ, l_NSC_ITP),
  ``subesophageal_zone_neurosecretory_cell`` (SEZ_NSC_Hugin, SEZ_NSC_CAPA), and
  ``ventral_nerve_cord_neurosecretory_cell``.

All differential equations, time constants, and firing-rate transfer functions here are explicit
modelling assumptions (labelled as such); the connectome provides only the cell identities and wiring.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class InternalParams:
    """Assumed physiological time constants and gains for the internal hemolymph/gut loop."""

    initial_sugar: float = 0.55       # 0..1 normalised hemolymph trehalose/glucose reserve
    initial_hydration: float = 0.65   # 0..1 normalised hemolymph hydration / osmolality balance
    initial_oxygen: float = 0.90      # 0..1 normalised tracheal/hemolymph oxygenation
    basal_metabolic_s: float = 120.0  # s to deplete sugar at rest (assumption)
    muscle_metabolic_gain: float = 0.025  # additional sugar/oxygen demand per unit muscle activation
    evaporation_s: float = 180.0      # s to deplete hydration at dry conditions (assumption)
    intake_gain: float = 0.15         # sugar/water replenishment rate when proboscis/pharynx pump on patch
    base_hz: float = 2.0
    max_hz: float = 120.0


class InternalState:
    """Couples hemolymph sugar, hydration, oxygen, and gut intake to BANC internal neurons."""

    def __init__(self, neurons: pd.DataFrame, params: InternalParams | None = None):
        self.p = params or InternalParams()
        self.sugar = float(self.p.initial_sugar)
        self.hydration = float(self.p.initial_hydration)
        self.oxygen = float(self.p.initial_oxygen)
        self.crop_fill = 0.0
        self.heart_rate_hz = 4.0

        cls = neurons["Class"].fillna("")
        func = neurons["Function"].fillna("")
        ctype = neurons["Primary Cell Type"].fillna("")
        aff = neurons["Flow"].eq("afferent")
        eff = neurons["Flow"].eq("efferent")

        # Afferent internal sensors (mutually exclusive with leg, balance, and external environment senses)
        nut_mask = aff & cls.eq("nutrient_sensory_neuron")
        self.sugar_sensor_idx = np.flatnonzero((nut_mask & func.str.contains("hemolymph_sugar")).to_numpy())
        self.isn_idx = np.flatnonzero((nut_mask & ~func.str.contains("hemolymph_sugar")).to_numpy())
        self.oxygen_idx = np.flatnonzero((aff & cls.eq("oxygenation_neuron")).to_numpy())
        self.aorta_idx = np.flatnonzero((aff & cls.eq("aorta_neuron")).to_numpy())

        int_gust = aff & cls.eq("internal_taste_sensillum_gustatory_neuron")
        self.pharynx_water_idx = np.flatnonzero((int_gust & func.str.contains("water")).to_numpy())
        self.pharynx_nutrient_idx = np.flatnonzero((int_gust & ~func.str.contains("water")).to_numpy())
        self.pharynx_tactile_idx = np.flatnonzero(
            (aff & cls.eq("internal_sensillum_tactile_neuron")).to_numpy()
        )

        self.idx = np.concatenate([
            self.sugar_sensor_idx,
            self.isn_idx,
            self.oxygen_idx,
            self.aorta_idx,
            self.pharynx_water_idx,
            self.pharynx_nutrient_idx,
            self.pharynx_tactile_idx,
        ]).astype(np.int64)

        # Efferent visceral / neurosecretory readouts (driven by the brain)
        self.mn9_idx = np.flatnonzero(ctype.eq("MN9").to_numpy())
        self.proboscis_mn_idx = np.flatnonzero((eff & cls.eq("proboscis_motor_neuron")).to_numpy())
        self.pharynx_mn_idx = np.flatnonzero((eff & cls.eq("pharynx_motor_neuron")).to_numpy())
        self.crop_mn_idx = np.flatnonzero((eff & cls.eq("crop_motor_neuron")).to_numpy())
        self.salivary_mn_idx = np.flatnonzero((eff & cls.eq("salivary_motor_neuron")).to_numpy())
        self.dilp_idx = np.flatnonzero(ctype.str.contains("DILP").to_numpy())
        self.dh44_idx = np.flatnonzero(ctype.str.contains("DH44").to_numpy())
        self.hugin_idx = np.flatnonzero(ctype.str.contains("Hugin").to_numpy())
        self.crz_idx = np.flatnonzero(ctype.str.contains("CRZ").to_numpy())
        self.neurosecretory_idx = np.flatnonzero(
            (eff & cls.str.contains("neurosecretory_cell")).to_numpy()
        )

        # Modulatory monoamine populations for live telemetry
        nt = neurons["Predicted NT type"].fillna("")
        self.da_idx = np.flatnonzero(nt.eq("DA").to_numpy())
        self.oa_idx = np.flatnonzero(nt.eq("OCT").to_numpy())
        self.ser_idx = np.flatnonzero(nt.eq("SER").to_numpy())

        self.last_rates = np.full(len(self.idx), self.p.base_hz)

    @staticmethod
    def _mean_rate(rates_hz: np.ndarray, idx: np.ndarray) -> float:
        return float(rates_hz[idx].mean()) if len(idx) else 0.0

    def step(
        self,
        rates_hz: np.ndarray,
        dt_s: float,
        muscle_effort: float = 0.0,
        taste_sugar: float = 0.0,
        taste_water: float = 0.0,
        humidity: float = 0.5,
    ) -> tuple[np.ndarray, dict[str, float]]:
        """Advance hemolymph state by ``dt_s`` and return ``(drive_rates_hz, telemetry)``."""
        p = self.p
        effort = float(np.clip(muscle_effort, 0.0, 1.0))
        pharynx_hz = self._mean_rate(rates_hz, self.pharynx_mn_idx)
        prob_hz = self._mean_rate(rates_hz, self.proboscis_mn_idx)
        pump_gate = float(np.clip((pharynx_hz + 0.5 * prob_hz) / 25.0, 0.0, 1.0))

        # Swallowing requires both surface contact (sugar/water) and proboscis/pharyngeal motor activity
        swallowed_sugar = float(np.clip(taste_sugar, 0.0, 1.0)) * (0.2 + 0.8 * pump_gate)
        swallowed_water = float(np.clip(taste_water, 0.0, 1.0)) * (0.2 + 0.8 * pump_gate)

        sugar_use = dt_s / p.basal_metabolic_s + dt_s * p.muscle_metabolic_gain * effort
        water_loss = dt_s / p.evaporation_s * (1.2 - 0.8 * float(np.clip(humidity, 0.0, 1.0)))

        self.sugar = float(np.clip(self.sugar - sugar_use + dt_s * p.intake_gain * swallowed_sugar, 0.0, 1.0))
        self.hydration = float(
            np.clip(self.hydration - water_loss + dt_s * p.intake_gain * swallowed_water, 0.0, 1.0)
        )
        # Oxygen dips under high muscle exertion and recovers passively via tracheal diffusion
        o2_target = np.clip(0.95 - 0.35 * effort, 0.2, 1.0)
        self.oxygen = float(np.clip(self.oxygen + (o2_target - self.oxygen) * min(1.0, dt_s / 2.0), 0.0, 1.0))
        self.crop_fill = float(
            np.clip(self.crop_fill + dt_s * 0.5 * (swallowed_sugar + swallowed_water) - dt_s * 0.02, 0.0, 1.0)
        )

        crz_hz = self._mean_rate(rates_hz, self.crz_idx)
        self.heart_rate_hz = float(3.5 + 2.5 * effort + 0.05 * min(crz_hz, 60.0))

        span = p.max_hz - p.base_hz
        # Gr43a hemolymph sensor responds to circulating sugar; ISNs respond to energy/hydration signals
        r_sugar = np.full(len(self.sugar_sensor_idx), p.base_hz + span * self.sugar)
        hunger_thirst_drive = np.clip(0.5 * (1.0 - self.sugar) + 0.5 * (1.0 - self.hydration), 0.0, 1.0)
        r_isn = np.full(len(self.isn_idx), p.base_hz + span * hunger_thirst_drive)
        hypoxia = np.clip(1.0 - self.oxygen, 0.0, 1.0)
        r_oxy = np.full(len(self.oxygen_idx), p.base_hz + span * hypoxia)
        r_aorta = np.full(len(self.aorta_idx), p.base_hz + span * np.clip((self.heart_rate_hz - 3.0) / 5.0, 0.0, 1.0))
        r_pw = np.full(len(self.pharynx_water_idx), p.base_hz + span * swallowed_water)
        r_pn = np.full(len(self.pharynx_nutrient_idx), p.base_hz + span * swallowed_sugar)
        r_pt = np.full(
            len(self.pharynx_tactile_idx),
            p.base_hz + span * np.clip(0.5 * (swallowed_sugar + swallowed_water) + 0.3 * self.crop_fill, 0.0, 1.0),
        )

        self.last_rates = np.concatenate([r_sugar, r_isn, r_oxy, r_aorta, r_pw, r_pn, r_pt])
        telemetry = {
            "hemolymph_sugar": self.sugar,
            "hemolymph_hydration": self.hydration,
            "hemolymph_oxygen": self.oxygen,
            "crop_fill": self.crop_fill,
            "heart_rate_hz": self.heart_rate_hz,
            "mn9_hz": self._mean_rate(rates_hz, self.mn9_idx),
            "pharynx_motor_hz": pharynx_hz,
            "dilp_hz": self._mean_rate(rates_hz, self.dilp_idx),
            "dh44_hz": self._mean_rate(rates_hz, self.dh44_idx),
            "hugin_hz": self._mean_rate(rates_hz, self.hugin_idx),
            "neurosecretory_hz": self._mean_rate(rates_hz, self.neurosecretory_idx),
            "da_hz": self._mean_rate(rates_hz, self.da_idx),
            "oa_hz": self._mean_rate(rates_hz, self.oa_idx),
            "ser_hz": self._mean_rate(rates_hz, self.ser_idx),
        }
        return self.last_rates, telemetry
