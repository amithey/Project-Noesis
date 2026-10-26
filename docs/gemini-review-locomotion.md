# Independent Locomotion Review: Root Causes of Falls, Decisive Tests, Method & Learning Critique (2026-10-26)

**Author:** Gemini 4 Argon  
**Branch:** `gemini/locomotion-review`  
**Scope:** Independent root-cause analysis of why the embodied fly (`noesis.experiments.learning_lab`) falls across seeds and why training/evolution has not generalised, based on inspection of the body/control/plasticity code and new high-resolution (`5 ms`) analyses of the 30 recorded frozen episodes (`dev-eval-orig`/`orig-rerun`, `dev-eval-v26a`, `dev-eval-evo-final`) and MuJoCo forward kinematics.

---

## 0. Verification & Compliance Statement

- **What was measured and verified here (no new heavy simulations run):**
  1. Re-analysis of the **30 existing frozen 5 s episodes** (`1,000` steps at `5 ms` per episode; seeds `10–19` for `original`, `trained` [v26a 12-birth], and `evolved` [84-pool-gain CMA-ES]), plus their `.gait.json` motor-pool rate records.
  2. Direct inspection and forward-kinematics / contact queries of the MuJoCo model (`noesis.body.make_fly`, `MuscleModel`, `FootAdhesion`, `CollisionGuard`, `enable_wing_and_head_joints`, `configure_floor_contacts`).
- **What was NOT run or touched:**
  - No heavy full-connectome simulation episodes were launched (per the resource rule: announcement required before heavy runs).
  - Held-out seeds (`200, 201, 202`) and raw connectome tables (`data/`, `FABA/`, `MANC/`) were never read, copied, or touched.
  - No changes were made to defaults or to any other agent's worktree.

---

## 1. Crucial Correction to `docs/early-fall-analysis.md`: Why Falls Looked "Precursor-Free"

`docs/early-fall-analysis.md` reported two observations: (a) 9 episodes with `standing_fraction >= 0.5` also dipped below `uprightness < 0.3`, interpreted as *"brief losses of balance with recovery are common"*, and (b) comparing `0.30–0.05 s` (`300 ms` to `50 ms`) before `uprightness < 0.3` against control windows showed no precursor (`< 0.5 SD` difference).

Both conclusions change when inspected at the physical `5 ms` step scale:

### 1.1 `uprightness < 0` is a 100% absorbing state; 6 of the 9 "recovering" episodes actually flipped permanently
In the 30 frozen episodes, **15 episodes crossed `uprightness < 0` (`original`: 6/10, `trained`: 6/10, `evolved`: 3/10), and `0 of 15` ever recovered (`post_flip_standing = 0.000` in all 15; `min uprightness = -0.917` to `-1.000`)**.
- Six episodes that ended with `standing_fraction >= 0.5` (`orig s12` at `3.91 s`, `orig s13` at `4.74 s`, `orig s15` at `4.57 s`, `trained s16` at `4.70 s`, `trained s17` at `3.33 s`, `evolved s12` at `2.78 s`) **never recovered**: they flipped permanently onto their back in the second half of the episode, so their 5 s time-average remained `>= 0.5` (`0.53` to `0.91`).
- Only **3 of the 18** `< 0.3` dips (`orig s16` min `+0.240`, `trained s18` min `+0.063`, `evolved s13` min `+0.283`) stayed at `uprightness > 0` and recovered.
- Across all 30 episodes, **before** the first `uprightness < 0` crossing, the fly stands **90.9% of the time** (`pre_flip_stand`); **after** crossing `0`, it stands **0.0% of the time**. Consequently, 5-second `standing_fraction` has a **Pearson correlation of `r = 0.9953` ($R^2 = 99.1\%$)** with the right-censored survival time $T_{\text{flip}} / 4.95\text{ s}$.

### 1.2 The entire tip-over happens inside the `0.05–0.00 s` window that `early-fall-analysis.md` excluded
At fly scale ($h \approx 1.15\text{ mm}$, $g = 9810\text{ mm/s}^2$), the inverted-pendulum gravitational toppling time constant is:
$$\tau_{\text{topple}} = \sqrt{\frac{h}{g}} \approx \mathbf{10.8\text{ ms}} \quad (\approx 2\text{ control steps of } 5\text{ ms}).$$
Measuring the exact transition duration $\Delta t = t(\text{uprightness} < 0.3) - t_{\text{onset}}(\text{last step with uprightness} \ge 0.7)$ across all 18 falls in the 30 episodes:
- **Median tip-over duration is `30 ms`** (12 of 18 falls take **`15 to 45 ms`**; full range `15–85 ms`).
- Because `early-fall-analysis.md` stopped its lead-in window at `t_fall - 50 ms` (`0.30–0.05 s`), **it stopped before the tip-over even started** (`uprightness` was still `>= 0.7` at `t_fall - 35 ms`).

### 1.3 When measured in the `50 ms` before tip onset (`uprightness` still `>= 0.7`), a strong geometric precursor appears
Comparing the `50 ms` (`10` steps) **before `t_onset`** (while `uprightness >= 0.7` throughout) and the `50 ms` gap `[t_fall - 0.05 s, t_fall]` against matched time windows in the 12 episodes that never dipped below `0.3` (`10,000`-permutation two-sided test):

| Measure | `50 ms` before tip onset (`upright >= 0.7`, $n=15$) | Matched controls ($n=180$) | Perm. $p$ (pre-onset) | `50 ms` before `< 0.3` (`0.05–0.00 s` gap, $n=15$) | Perm. $p$ (`0.05–0.00 s`) |
|---|---|---|---|---|---|
| `supporting_feet` | `1.836` (sd `0.434`) | `2.310` (sd `0.537`) | **`0.0013`** | `1.833` (sd `0.541`) | **`0.0025`** |
| `unilat_zero` (all 3 `L` or all 3 `R` legs off ground) | `0.497` (sd `0.377`) | `0.282` (sd `0.266`) | **`0.0051`** | `0.667` (sd `0.352`) | **`0.0002`** |
| `fm_zero` (`(LF=0 & LM=0) \| (RF=0 & RM=0)`) | **`0.758`** (sd `0.214`) | **`0.398`** (sd `0.277`) | **`0.0001`** | **`0.980`** (sd `0.075`) | **`0.0001`** |
| `uprightness` | `0.813` (sd `0.060`) | `0.875` (sd `0.056`) | **`0.0002`** | `0.592` (sd `0.114`) | **`0.0001`** |

*(Even restricted to the 8 failed episodes with `standing < 0.5` used in `early-fall-analysis.md`, `fm_zero` in the `0.05–0.00 s` window is `0.963` vs `0.402`, $p = 0.0001$, and `0.727` vs `0.395`, $p = 0.0055$ before `t_onset`.)*

---

## 2. Ranked Root-Cause Hypotheses for the Falls and Yaw Spin

### Hypothesis 1 (Rank 1 — Primary Root Cause): Severe Bilateral & Antagonist Tonic Asymmetry in the Connectome-to-Muscle Mapping Leaves `RH` and `RF` Chronically Airborne and `RM` as a Single Point of Failure

#### Evidence from code and recorded data
1. **Chronic tonic firing asymmetry in `episode_000.gait.json` (10 original-weight episodes):**
   - **`RH` (Right Hind) is tonically flexed above the thorax:** On `RH_Femur`, the 8 extensor/depressor neurons (`+1` in `MUSCLE_JOINT`) fire at **`1.7 Hz`** (`a_pos = 0.065`) while the 10 flexor/levator neurons (`-1`) fire at **`42.7 Hz`** (`a_neg = 0.631`, `net_act = -0.57`). By contrast, `LH_Femur` is nearly balanced (`27.7 Hz` vs `23.6 Hz`, `net_act = +0.04`), and `LF/LM/RM/RF_Femur` are strongly extensor-dominated (`56–83 Hz` vs `0.5–5.9 Hz`, `net_act = +0.50` to `+0.75`).
   - **`1/N` pool-size normalization in `build_muscle_matrices` (`src/noesis/motor.py:238`) amplifies a 2-neuron `Tibia` extensor pool against a 16–18-neuron flexor pool:** Every leg has `16–18` `Tibia` flexor neurons firing at `0.2–10.0 Hz`, but **only `2` `Tibia` extensor neurons**, which fire tonically at **`88 to 268 Hz`** in 5 of 6 legs (`RF: 88.2 Hz`, `RH: 111.6 Hz`, `LH: 126.5 Hz`, `LM: 139.4 Hz`, `RM: 267.7 Hz`), driving `Tibia` to `63–78%` full extension (`net_act = -0.63` to `-0.78`), whereas `LF_Tibia` extensor fires at only `16.8 Hz` (`net_act = -0.38`).
   - **Anteroposterior leg crossing (`LF` vs `LM`):** `LF_Coxa` (`+1` retractor `61.2 Hz` vs `-1` protractor `1.5 Hz`) pulls `LF` backward by `-1.005 mm` (`x = +0.261 mm`), while `LM_Coxa` (`-1` protractor `131.3 Hz` vs `+1` retractor `31.9 Hz`) pulls `LM` forward by `+0.775 mm` (`x = +0.373 mm`), so **`LM`'s tonic equilibrium X-position crosses in front of `LF`**! Meanwhile `LH_Coxa_roll` adductor (`94.2 Hz` on 1 neuron) pulls `LHFemur` across the midline into collision with `RHFemur` (`dist = -0.042 mm` in MuJoCo FK).
2. **MuJoCo Forward Kinematics of the Tonic Mean Equilibrium Pose (thorax fixed at `z = 1.5 mm`):**

| Leg | `standing_pose()` foot $(x, y, z)$ mm | Tonic mean equilibrium foot $(x, y, z)$ mm | Tonic shift $(\Delta x, \Delta y, \Delta z)$ mm | Measured stance fraction while upright (`orig` / `trained` / `evolved`) |
|---|---|---|---|---|
| **LF** | `(+1.267, +0.750, +0.214)` | `(+0.261, +0.965, -0.372)` | `(-1.005, +0.215, -0.587)` | **`64.3%` / `62.7%` / `63.4%`** |
| **RM** | `(-0.402, -1.457, +0.195)` | `(+0.315, -1.545, -0.077)` | `(+0.717, -0.088, -0.272)` | **`67.3%` / `65.7%` / `65.7%`** |
| **LH** | `(-1.706, +0.514, +0.106)` | `(-1.903, +1.344, +0.759)` | `(-0.197, +0.830, +0.653)` | **`49.1%` / `51.4%` / `47.0%`** |
| **LM** | `(-0.402, +1.457, +0.195)` | `(+0.373, +1.598, -0.030)` | `(+0.775, +0.141, -0.225)` | **`32.9%` / `36.9%` / `35.8%`** |
| **RF** | `(+1.267, -0.750, +0.214)` | `(+0.680, -1.572, +0.224)` | `(-0.587, -0.822, +0.009)` | **`17.6%` / `25.9%` / `22.2%`** |
| **RH** | `(-1.706, -0.514, +0.106)` | `(-1.581, -1.165, +1.889)` | `(+0.125, -0.650, +1.783)` | **`8.5%` / `13.5%` / `7.2%`** |

3. **How this causes both the falls and the yaw spin:**
   - Because `RH` (`8.5%` stance) and `RF` (`17.6%` stance) are chronically held in the air, **55.8% of all upright steps across the 30 episodes have `<= 2` supporting feet** (`0 feet: 1.1%`, `1 foot: 13.2%`, `2 feet: 41.5%`). A 3D body on $\le 2$ contacts has zero static roll/pitch polygon area.
   - On the entire right side (`RF, RM, RH`), **`RM` is a single pillar of support**. At the exact onset `t_onset` of the 15 irreversible flips:
     - `RH` is off the ground in **15 / 15 flips** (`onset_RH = 0`).
     - `RF` is off the ground in **13 / 15 flips** (`onset_RF = 0`).
     - In **8 / 15 flips**, `RM` also lifts (`onset_RM = 0`), leaving **0 right-side legs on the ground** while `1–3` left legs push, rolling the fly onto its back in `15–55 ms`.
     - In **7 / 15 flips**, `LF` and `LM` both lose contact (`onset_LF = 0, onset_LM = 0`), leaving the fly balanced on the 2-point diagonal `[LH, RM]` or `[RF, RM]`.
   - Furthermore, asymmetric stance loading (`LF = 26.2, LH = 19.3, LM = 11.4` on the left vs `RM = 25.5, RF = 6.3, RH = 3.1` on the right) and opposing anteroposterior drives (`LF` retracting `-1.0 mm` while `RM` protracts `+0.72 mm`) generate a **systematic counter-clockwise (`+yaw`) torque**: all 30 episodes have positive net yaw rotation over 5 s (`+2.75` to `+20.60 rad`, mean **`+11.7 rad`**, or ~2 full CCW revolutions).

#### Falsifiable Prediction
If `MuscleModel` centers each pool around its resting tonic rate (or sets per-pool `r_half` = resting tonic rate of that pool so tonic background activity maps to `a_pos = a_neg` = `standing_pose()`), then in 10 frozen dev-seed episodes with original synaptic weights:
1. `RH` and `RF` upright stance fractions will increase from `8.5%` and `17.6%` to **`>= 35%` each**.
2. Mean `supporting_feet` while upright will rise from `2.40` to **`>= 3.20`**, and `fm_zero` (`0.398` control / `0.758` pre-fall) will drop below **`0.20`**.
3. The irreversible flip count on original weights will drop from **`6/10` seeds to `<= 2/10` seeds**, and mean standing fraction will exceed **`0.85`**.

---

### Hypothesis 2 (Rank 2 — Control & Adhesion Amplifier): `FootAdhesion` Overwrites the 2-Foot Rule (Dead Code) and Releases Up to 3 Stance Feet Based on World-Frame Servo Tracking Error (`20 µm`)

#### Evidence from code and recorded data
1. **Lines 360–367 of `src/noesis/experiments/learning_lab.py` are dead code under default settings:**
   ```python
   # learning_lab.py:360-369
   lift = np.array([desired[joints[leg, 'Femur']] - rest[joints[leg, 'Femur']] for leg in LEGS])
   swinging = np.flatnonzero(lift > 0.08)
   if len(swinging) > 2:
       swinging = swinging[np.argpartition(lift[swinging], -2)[-2:]]
   adhesion[swinging] = 0
   if foot_adhesion is not None:
       adhesion = foot_adhesion.command(target)  # <-- unconditionally overwrites adhesion!
   ```
   Because `--adhesion-mode` defaults to `'geometry'`, `foot_adhesion` is always active and overwrites `adhesion` using `max_swing = 3` and `threshold_mm = 0.02` (`20 µm`). Across all upright steps in the 30 episodes, `released_feet` is `2` (`33.4%`) or `3` (`33.3%`) — **66.7% of the time**. When `RH` and `RF` are already in the air (`adhesion` irrelevant because they do not touch the floor), releasing 2–3 additional feet removes adhesion from the remaining weight-bearing legs!
2. **`FootAdhesion.command(target)` (`src/noesis/adhesion.py:50-56`) measures servo tracking error (`z_world(target) - z_world(live.qpos)`), not commanded lift velocity:**
   ```python
   self.scratch.qpos[:] = self.live.qpos
   self.mj.mj_kinematics(self.model, self.scratch)
   current_z = self.scratch.xpos[self.feet, 2].copy()
   self.scratch.qpos[self.qadr] = target
   self.mj.mj_kinematics(self.model, self.scratch)
   lift = self.scratch.xpos[self.feet, 2] - current_z
   ```
   - `target` is an open-loop slewed setpoint (`target += clip(desired - target, -0.06, 0.06)`), whereas `live.qpos` reflects physical joint angles under ground reaction forces and body roll/pitch.
   - If a foot is in the air (`z = 0.5 mm`) and `target` is now moving **downward** toward the floor, as long as `target` hasn't yet caught up to `live.qpos` (`z(target) - z(live.qpos) > 0.02 mm`), `FootAdhesion` classifies it as "lifting", consuming one of the `max_swing = 3` slots.
   - Conversely, if the thorax tilts/rolls by a few degrees while a foot is on the floor, or if a stance joint lags `target` by `> 20 µm` upward in world Z, `FootAdhesion` cuts that foot's `40.0`-unit adhesion force to `0`, triggering an instant unilateral collapse.

#### Falsifiable Prediction
Running `dev_eval.py` (original weights, seeds 10–19) with `--adhesion-mode legacy` (which activates the dead-code 2-foot femur-lift rule on lines 360–367) or modifying `FootAdhesion` so that (a) lift is measured from the body-frame target step `z_body(target_t) - z_body(target_{t-1}) > 0.02 mm`, and (b) `max_swing = 2` (never releasing all 3 legs on the same lateral side) will reduce `unilat_zero` by **`>= 30%`** and increase `supporting_feet` by **`>= 0.3` feet**.

---

### Hypothesis 3 (Rank 3 — Sensory Blindness): `SensoryMap` Load Afferents (`campaniform`, `bristle`) Are 100% Saturated (`tanh(42 / 2.0) = 1.0`) by FlyGym's Adhesion Force, Depriving the VNC of Weight-Bearing Feedback

#### Evidence from code and MuJoCo inspection
1. **Fly weight vs adhesion force (`check_mujoco_params.py`):**
   - Total fly mass in MuJoCo is `1.025e-3 g` with `GRAVITY = -9810 mm/s^2`, so **total gravitational weight of the entire fly is `10.05` force units**.
   - Each `Tarsus5_adhesion` actuator has `gainprm = 40.0` — pulling each adhered foot into the floor with **`40.0` normal force units (`4x` the entire fly's body weight)**.
   - Consequently, in the 30 recorded CSVs, every stance foot has `{leg}_ground_load` of **`39 to 43` units** (`median = 41.3–42.5`).
2. **`load0 = 2.0` in `src/noesis/sensory.py:58,65`:**
   - Campaniform and bristle rates are `2.0 + 148.0 * tanh(load / 2.0)`.
   - At `load = 42.0`, $\tanh(42.0 / 2.0) = \tanh(21.0) = \mathbf{1.0000000000}$ (`150.0 Hz`). Even a lightly touching foot (`load = 6.0`) gives $\tanh(3.0) = 0.995$.
   - Thus **every campaniform sensillum and bristle neuron acts as a binary contact switch (`2 Hz` in air, `150 Hz` on contact) with zero graded sensitivity to actual body-weight transfer across legs**, while injecting a `150 Hz * 400 = 60,000` synapse-equivalent/s shock whenever a foot chatters on the stiff floor (`CONTACT_SOLREF = (2e-4, 1.0)`).

#### Falsifiable Prediction
Setting `load0 = 45.0` (or subtracting the `40.0` adhesion self-load before `tanh(max(0, load - 40.0) / 5.0)`) in `SensoryMap` will unsaturate campaniform/bristle firing rates from `150 Hz` (`tanh = 1.0`) to a graded `30–110 Hz` range that tracks true weight-bearing load on each leg.

---

### Hypothesis 4 (Rank 4 — Stiff Servo Scrubbing & Dorsal Floor Trapping): Stiff Position Servos (`kp = 45, forcerange = 65`) Moving at `2.5 rad/s` During Stance Overpower Friction and Vault the Thorax

#### Evidence from code and recorded data
1. **Stance legs move just as fast as swing legs:** Across all 30 episodes while upright (`>= 0.7`), mean joint speed `{leg}_speed_rad_s` when a leg is **in stance (`load > 1e-5`)** is `2.22–2.96 rad/s` (`LF: 2.48`, `LH: 2.96`, `RM: 2.22`), indistinguishable from when the leg is **in swing (`load == 0`)** (`1.95–2.58 rad/s`).
2. **Why `stance_slip_mm_s` is `15–17 mm/s` in every standing episode:** With `servo_kp = 45.0` and `forcerange = 65.0` over a `~1.5 mm` kinematic chain, opposing stance legs (`LF` pulling backward `-1.0 mm`, `RM`/`LM` pulling forward `+0.75 mm`) generate horizontal shear forces exceeding the friction cone ($\mu N \approx 1.0 \times 42 = 42$), causing continuous foot skidding at $v_{\text{slip}} \approx \dot{q} \cdot L_{\text{leg}} \approx 2.5\text{ rad/s} \times 6\text{ mm}_{\text{eff}} \approx 15\text{ mm/s}$.
3. **Why `CollisionGuard` and spawn height are NOT the causes of the falls:**
   - `CollisionGuard` (`src/noesis/body.py:137`) is **never instantiated or called in `learning_lab.py`** (`make_fly(..., body_proxies=False)` on line 182).
   - Spawn drop (`spawn_pos = (0, 0, 1.2)` -> thorax `z = 2.48 mm` at `t = 0`, landing at `t = 20 ms`, settling to `z = 1.098 mm, uprightness = 0.9998` with all 6 feet on the floor by `t = 50 ms`) is completely over before `birth = 0.05 s`, which is why `startup-study.md` (`birth = 0.5 s`) showed no improvement.
   - However, once a fall tips past `uprightness < 0`, `FLOOR_COLLISIONS = "all"` (`world.py:4`) and `enable_wing_and_head_joints` (`world.py:63`, actuating `0/LWing` and `0/RWing` with `kp = 15, forcerange = 20`) trap the fly on its dorsal thorax/wings with no righting reflex (`0/15` recoveries).

---

## 3. Smallest Cheap Decisive Tests (Original Weights Only, 10 Dev Seeds `10–19`)

Each test requires **no training**, uses **original weights only**, and can be evaluated with `scripts/dev_eval.py` against cached `outputs/dev-eval-orig` (or a 2-second / 10-seed pilot first to save compute):

| Test ID | Hypothesis Tested | Minimal Code / Flag Change | Confirming Outcome (on seeds `10–19`, paired with `original`) | Killing Outcome |
|---|---|---|---|---|
| **T1 (Tonic Baseline Centering)** | **H1** (Tonic motor asymmetry & 2-neuron `Tibia` extensor saturation cause `RH`/`RF` lift, yaw spin, and falls) | Opt-in flag `--muscle-baseline-center` in `MuscleModel`: record/use the per-pool mean tonic rate $r_{0,j}^{\pm}$ (or calibrate `r_half_pos[j]`, `r_half_neg[j]` to each pool's resting rate so resting tonic drive gives $a_{\text{pos}} = a_{\text{neg}} \Rightarrow \theta_{\text{eq}} = \theta_{\text{stand}}$) | `RH` & `RF` upright stance fraction rises from `8.5% / 17.6%` to **`>= 35%`**, mean `supporting_feet` rises from `2.4` to **`>= 3.2`**, irreversible flips drop from `6/10` to **`<= 2/10`**, paired standing CI lower bound **`> 0`** | `RH`/`RF` stance remains `< 25%` or paired standing difference CI includes `<= 0` |
| **T2 (Adhesion Mode & Body-Frame Delta-Target Lift)** | **H2** (`FootAdhesion` world-frame tracking-error rule & `max_swing=3` overwrite the 2-foot rule and drop stance legs) | **(T2a)** Run existing flag `--adhesion-mode legacy` (zero code changes; enables lines 360–367). **(T2b)** Opt-in `FootAdhesion` mode computing lift from body-frame target step $\Delta z_{\text{body}}(\text{target}_t - \text{target}_{t-1}) > 0.02\text{ mm}$ with `max_swing = 2` and unilateral guard (never release the last loaded foot on `L` or `R`) | `unilat_zero` drops from `0.28` to **`< 0.18`**, `supporting_feet` rises by **`>= +0.3`**, and paired standing fraction improves over default `geometry` adhesion | Standing fraction and `unilat_zero` unchanged within `±0.05` |
| **T3 (Unsaturate Campaniform Load Scale `load0`)** | **H3** (`load0 = 2.0` is 100% saturated by `40.0` adhesion force) | Set `load0 = 45.0` (or subtract `40.0` adhesion baseline) via an opt-in `--load0` flag in `SensoryMap` | Graded afferent rates (`30–110 Hz` instead of constant `150 Hz` in stance) and reduced step-to-step motor rate variance / improved standing | Motor pool rates and standing fraction unchanged (`|diff| < 0.03`) |
| **T4 (Stance Compliance / Damping)** | **H4** (`servo_kp = 45` in stance causes `16 mm/s` slip and vaulting) | Lower `servo_kp` from `45.0` to `20.0` (or `muscle_reach` from `0.9` to `0.5`) via opt-in `BodyParams` flag | Upright `stance_slip_mm_s` drops by **`>= 25%`** (`< 12 mm/s` on standing seeds) without reducing `standing_fraction` | Slip drops `< 10%` or body collapses (`height_mm < 0.5`) |

*(Before running a full 10-seed 5 s `dev_eval.py` [~50–70 min wall time], **T1** can be verified in **10 seconds of wall time** with a brain-free MuJoCo kinematic/physics test, and a 2-seed [`s10, s14`] frozen probe takes ~10 min.)*

---

## 4. Critique of the Evaluation Method (`dev_eval.py` & `train_validate.py`)

### 4.1 The protocol's stance-slip gate (`<= 0.8 * original_slip`) is mathematically inverted when `original` falls
In `train_validate.py:41`:
```python
slip = finite and trained['mean_stance_slip_mm_s'] <= .8 * baseline['mean_stance_slip_mm_s']
```
And in `embodied_metrics.py:103`:
```python
slip = float(np.mean(np.linalg.norm(feet[support, :2], axis=1))) if support.any() else 0.
```
When the fly flips onto its back (`uprightness < 0`), its feet are in the air (`support.any() == False`), so `stance_slip_mm_s` is **`0.0 mm/s`**, dragging the episode mean down!
- Across the 30 episodes, episode-mean `stance_slip_mm_s` correlates **positively with `standing_fraction` (`r = +0.76`)**:
  - On seed 10, `original` falls at `2.37 s` (`standing = 0.442`), so its episode-mean slip is **`9.37 mm/s`**, setting the gate at **`<= 7.49 mm/s`**.
  - On seed 11, `original` stands (`standing = 0.929`), and its episode-mean slip is **`16.7 mm/s`**.
  - Every candidate that successfully stands on seed 10 (`V23: 0.920`, `V25: 0.934`, `v26a: 0.975`, `evolved: 0.993`) has upright slip of `14.0–16.8 mm/s` and **automatically fails the seed-10 slip gate because `original` lay on its back for half of seed 10**!
- **Fix:** Measure stance slip **conditioned on upright standing steps (`uprightness >= 0.7` and `supporting_feet >= 1`)**, and compare against the original weights' **upright stance slip across the 10-seed bank (`~16.2 mm/s`)**, not against a fallen single-seed episode average.

### 4.2 `standing_fraction` over 5 s is a right-censored absorption time ($R^2 = 99.1\%$ with $T_{\text{flip}}$), and "paired seeds" are not Common Random Numbers (CRN)
1. **Why paired differences have huge variance (`sd(diffs) = 0.49–0.51`, wider than unpaired `sd = 0.24`):**
   - In `LIFNetwork._run` (`src/noesis/lif.py:102-106`), `np.random.random()` is drawn sequentially inside Numba for each sensory afferent `stim_idx` and background Poisson event.
   - The instant a candidate's weight or gain alters a single spike at `t = 55 ms`, the body posture shifts by $10^{-5}$, changing `stim_p` and desynchronizing the chaotic network trajectory in the saturated `bg_weight = 40` regime.
   - Consequently, across seeds `10–19`, candidate and original `standing_fraction` are **uncorrelated or negatively correlated (`r = -0.51` in `v26a`, `r = -0.22` in `evolved`)**! Pairing by seed provides **zero variance reduction**; in fact $\text{Var}(C - O) = \text{Var}(C) + \text{Var}(O) - 2\text{Cov}(C, O)$ is *larger* than unpaired variance when $\text{Cov}(C, O) < 0$.
   - With $\text{sd}(\text{diffs}) \approx 0.50$ and $N = 10$ seeds, the 95% bootstrap CI half-width is $\pm 1.96 \times 0.50 / \sqrt{10} \approx \mathbf{\pm 0.31}$. Thus **even a true $+0.20$ improvement in standing fraction cannot achieve a 95% CI lower bound $> 0$ at $N = 10$ unless episode-to-episode flip variance is reduced first**.
2. **What to report alongside `standing_fraction`:**
   - **Flip count / survival rate** (number of seeds that never cross `uprightness < 0`, or Kaplan-Meier / log-rank survival time $T_{\text{flip}}$). Note that `evolved` reduced irreversible flips from `6/10` (`original`) to `3/10` (`evolved`), which `standing_fraction` (`0.750` vs `0.738`) obscured because 3 of `original`'s flips happened late (`3.91, 4.57, 4.74 s`).
   - **Pre-flip support stability** on upright frames: mean `supporting_feet`, bilateral support fraction `1 - unilat_zero` (both `L` and `R` having $\ge 1$ foot on the ground), and `1 - fm_zero`.

---

## 5. Learning-Rule Review (`MotorPlasticity`, `MotorExploration`, `SepCMA`)

### 5.1 Why V24 depended on exploration noise (`0.93` with `1.5 mV/ms` vs `0.08–0.19` frozen)
In `MotorExploration.step` (`src/noesis/embodiment.py:82`), an Ornstein-Uhlenbeck current of amplitude `1.5 mV/ms` (`±0.75 mV` per `0.5 ms` LIF step) is injected identically into all neurons of each `(leg, muscle)` pool.
- Because the LIF spike threshold is a **rectifying nonlinearity** (`v >= -45 mV`, reset `-52 mV`, `t_ref = 2.2 ms`), zero-mean current fluctuations have a **strong positive rectified mean effect on subthreshold/silent pools** (`LF/RF_Tibia` flexors firing at `0.2–0.7 Hz`, `RH_Femur` extensors firing at `1.7 Hz`), waking them up to `10–25 Hz`, while **saturated pools** (`RM/LM/LH/RH_Tibia` 2-neuron extensors firing at `110–268 Hz`, near the refractory ceiling) are compressed or inhibited during negative swings!
- Thus `1.5 mV/ms` exploration acted as a **prosthetic tonic equalizer** that partially corrected **Hypothesis 1** during training; turning it off in frozen evaluation collapsed the subthreshold pools back to silence.

### 5.2 Is a single scalar reward per 5 s episode enough for perturbation learning across 73,594 motor synapses?
**No — for four quantitative reasons:**
1. **Signal-to-noise ratio across $M = 55$ perturbed muscle pools:** `MotorExploration` perturbs $M \approx 55$ independent `(leg, muscle)` pools simultaneously with $\tau_{\text{OU}} = 100\text{ ms}$, while `MotorPlasticity` broadcasts a **single global scalar** `error(t)` (`src/noesis/plasticity.py:153`). In simultaneous perturbation stochastic approximation (REINFORCE / node perturbation), attributing a scalar reward to $M$ independent noise channels injects cross-talk variance proportional to $M - 1 = 54$. Over a `5 s` episode (`50` independent `100 ms` correlation times), the SNR per pool in a single birth is $\sim \sqrt{50 / 55} \approx \mathbf{0.95 < 1}$ (pure noise!).
2. **Unmeasured chaotic intrinsic noise dominates `exploration.credit()`:** At low exploration (`0.35 -> 0.1 mV/ms` in V25/v26a), the motor neurons are driven primarily by the connectome's own chaotic background/recurrent spikes (`bg_hz = 3 Hz, bg_weight = 40`). Those intrinsic rate fluctuations cause the body to fall or stand, **but `MotorPlasticity` correlates reward error only with `exploration.credit()`**, blaming whatever random OU innovation happened to be active when the chaotic recurrent network triggered a fall!
3. **Time-scale mismatch between eligibility (`100 ms`), reward baseline (`2000 ms`), and fall dynamics (`30 ms`):** When a fall occurs, `standing` drops from `1` to `0`, plunging `reward` from `+0.2` to `-0.8`. Because `reward_baseline` has $\tau = 2000\text{ ms}$, `error` stays pinned at `-1.0` for the entire remaining fallen duration (or `100 ms` cutoff), punishing whichever motor actions happen *after* the fly is already tipping or on its back.
4. **Weight bounds (`±50%`, sign-preserving) cannot fix a 100:1 tonic pool-rate imbalance:** Even if `MotorPlasticity` pushes every excitatory input to `RH_Femur` extensor to `+50%` and every input to `RM_Tibia` extensor to `-50%`, a `1.5x` synaptic scaling in a bistable/saturated network cannot turn a `1.7 Hz` silent pool into `40 Hz` or a `268 Hz` 2-neuron pool into `15 Hz`.

### 5.3 Why the 84-parameter `SepCMA` evolution (`docs/evolution-pool-gains.md`) did not converge
- **Underdetermined sample budget ($64 < 84$):** `SepCMA` was run with dimension $n = 84$, population $\lambda = 8$, for $G = 8$ generations = **$64$ total candidate evaluations** (`sigma` moved only from `0.250` to `0.214`). In $n = 84$ dimensions, separable CMA-ES requires at least $15n$ to $30n$ ($\approx 1,200–2,500$) evaluations even on noise-free quadratics, and more under high seed noise ($\text{sd} \approx 0.30$ across 3-seed 2 s rollouts).
- **Yet notice a telling signal:** Even with 64 noisy evaluations, adjusting only the 84 pool gains (`noesis.pool_genome`) raised the count of dev seeds with `standing >= 0.9` from `4/10` to `7/10` and cut irreversible flips from `6/10` to `3/10`, because pool gains directly act on the **Hypothesis 1** bottleneck (the actuator interface).

### 5.4 Better-Conditioned Parameterisations & Learning Rules to Try Next
1. **Stage 0 — Deterministic Homeostatic / Tonic-Centering Calibration (0 free search parameters):**
   - Measure each agonist/antagonist pool's resting tonic firing rate $\bar{r}_{j,\text{pos}}, \bar{r}_{j,\text{neg}}$ over a short (`1–2 s`) uncoupled brain run (or use the recorded mean rates), and set per-pool half-activation rates $r_{\text{half},j}^{\pm} = \max(\bar{r}_{j}^{\pm}, r_{\min})$ (or subtract $\bar{r}_j^{\pm}$ before computing $\Delta a$) so that **at the connectome's natural tonic operating point, $a_{\text{pos},j} = a_{\text{neg},j}$ and $\theta_{\text{eq}} = \theta_{\text{stand}}$ for all 42 leg joints**.
   - This eliminates the 84-parameter blind search needed just to find the standing equilibrium pose.
2. **Stage 1 — Low-Dimensional Bilateral / Per-DOF Parameterisation ($7–14$ parameters instead of $84$ or $73,594$):**
   - Enforce left-right symmetry on residual gains or learn 14 parameters (7 DOFs $\times$ [common gain, L/R balance offset]) scored on continuous **support polygon quality** ($\text{mean}(\min(L_{\text{legs}}, R_{\text{legs}} \ge 1) \cdot \text{uprightness})$) rather than binary right-censored flip time.
3. **Stage 2 — Per-Leg Local Credit Assignment for Synaptic Plasticity ($M = 9$ per leg instead of $M = 55$ global):**
   - Replace the single global scalar `reward` in `MotorPlasticity` with **6 leg-specific reward signals** $r_{\text{leg}}(t)$ (crediting leg $L$ for maintaining stance load without horizontal foot slip when `adhesion[L] == 1`, and for clean vertical clearance when `adhesion[L] == 0`), so each leg's motor synapses only correlate against that leg's own 9 muscle perturbations ($6\times$ variance reduction).
4. **Decouple PRNG streams (`LIFNetwork`) for true Common Random Numbers:**
   - Give background Poisson noise its own `np.random.Generator` (fixed per step) separate from state-dependent sensory afferent `stim_p` draws, so that small postural differences do not scramble the entire brain's background spike train after `50 ms`.

---

## 6. Proposed Next Step (Awaiting Agreement)

Before touching plasticity or running multi-hour evolutions, **Test T1 (Tonic Baseline Centering in `MuscleModel`) + Test T2 (Body-Frame Delta-Target `FootAdhesion` with `max_swing=2` and unilateral support guard)** directly address the measured physical root causes (**H1 + H2**) with `< 30` lines of opt-in code.

If agreed, I will:
1. Implement opt-in flags for **T1** and **T2** on `gemini/locomotion-review` with unit tests (`pytest` + `ruff`), keeping all default behaviour 100% unchanged.
2. Run a fast kinematic/short-episode verification first, and—with your approval for the compute budget—run the paired 10-seed `scripts/dev_eval.py` comparison against `outputs/dev-eval-orig` and report the exact bootstrap CI, sign test, flip count, and upright slip in `docs/`.
