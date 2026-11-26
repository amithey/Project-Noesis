"""Fixed EC1 episode lineage and training-panel criteria."""
import numpy as np

TRAIN = (1010, 1011, 1012, 1013, 1014, 1015)
BANKS = ("stage1", "carry", "reset")
PHYSICAL = ("mean_PS1_quality", "mean_COM_speed_mm_s",
            "mean_angular_speed_rad_s", "mean_non_tarsal_fraction")


def lineage(bank):
    if bank not in BANKS:
        raise ValueError("Unknown EC1 training stage")
    return ("parent", 1010) if bank == "stage1" else ("stage1" if bank == "carry" else "parent", 1011)


def jobs():
    return [dict(name="baseline-1010", phase="frozen", bank="parent", seed=1010)] + [
        dict(name=f"train-{bank}", phase="train", bank=bank, seed=lineage(bank)[1]) for bank in BANKS
    ] + [dict(name=f"frozen-{bank}-{seed}", phase="frozen", bank=bank, seed=seed)
         for bank in BANKS for seed in TRAIN]


def decision(banks):
    if set(banks) != {"parent", *BANKS} or any(set(v) != set(TRAIN) for v in banks.values()):
        raise ValueError("Incomplete EC1 panel")
    for bank in banks.values():
        for cell in bank.values():
            values = [cell["return"], *[cell["physical"][key] for key in PHYSICAL],
                      cell["summary"]["standing_fraction"], cell["summary"]["mean_stance_slip_mm_s"],
                      cell["anatomical"]["mode_fractions"]["tarsal_only"]]
            if not np.isfinite(values).all():
                raise ValueError("Nonfinite EC1 evidence")
    from s2_contract import standing_retention
    endpoint = banks["carry"]
    comparisons = {}
    for control in ("parent", "reset", "stage1"):
        strict = control != "stage1"
        passed, anatomy_pass = [], []
        gains = []
        for seed in TRAIN:
            candidate, reference = endpoint[seed], banks[control][seed]
            gains.append(candidate["return"] - reference["return"])
            changes = [candidate["physical"][key] - reference["physical"][key] for key in PHYSICAL]
            physical_ok = changes[0] > 0 and all(value < 0 for value in changes[1:]) if strict else (
                changes[0] >= 0 and all(value <= 0 for value in changes[1:]))
            passed.append(physical_ok)
            ca, ra = candidate["anatomical"], reference["anatomical"]
            tarsal = ca["mode_fractions"]["tarsal_only"] - ra["mode_fractions"]["tarsal_only"]
            non_tarsal = ca["mean_non_tarsal_fraction"] - ra["mean_non_tarsal_fraction"]
            anatomy_pass.append(tarsal > 0 and non_tarsal < 0 if strict else tarsal >= 0 and non_tarsal <= 0)
        retention = standing_retention({s: banks[control][s]["summary"] for s in TRAIN},
                                       {s: endpoint[s]["summary"] for s in TRAIN}, TRAIN)
        comparisons[control] = dict(
            contexts_physical_pass=int(sum(passed)), contexts_anatomical_pass=int(sum(anatomy_pass)),
            contexts_reward_improved=int(np.count_nonzero(np.asarray(gains) > 1e-6)),
            worst_reward_gate=bool(min(gains) > 1e-6 if strict else min(gains) >= 0),
            retention=retention,
            passes=bool(all(passed) and all(anatomy_pass) and retention["passed"] and
                        (min(gains) > 1e-6 if strict else min(gains) >= 0)))
    return dict(comparisons=comparisons, endpoint_meets_training_criteria=all(c["passes"] for c in comparisons.values()),
                fixed_endpoint="carry", model_promoted=False, walking_proven=False)
