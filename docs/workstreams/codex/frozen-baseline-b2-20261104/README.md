# B2 — second-host baseline reproduced exactly (4 November 2026)

The USB transfer removed the exact-data blocker. Four BANC/FAFB/MANC files matched the source SHA256 values in guide0e560229. The short frozen smoke passed, then two full5s parent runs completed.

## Registered conditions and outcomes

Registration: PR24comment6635203833. Pilot75cdc34, core42c2076, completed parent9fd7d411. CPU1world,5ms control, bg3Hz x40, no learning/exploration. Development seeds10006/10007 only.600s process watchdog per run,1200s aggregate, canonical exclusive lease, no concurrent Python workers at admission and at least2GiB available RAM before each job.

| Seed | Standing | Slip mm/s | Heading mm | Net forward mm | Regime | Every raw array identical to historical S1 parent |
| --- | ---: | ---: | ---: | ---: | --- | --- |
|10006|96.2626%|3.05817|0.298359|0.022346|STEPPING_IN_PLACE|yes|
|10007|94.4444%|2.49016|0.331793|0.383999|STEPPING_IN_PLACE|yes|

Both processes exit0, full1000control chunks/5s, unchanged weights, finite numerical observation arrays. Aggregate120.719s. Parent and both source trees remained unchanged; the owned canonical lease was released. Raw arrays were compared field-by-field, not only by summary.

The current environment uses Numba0.68.1 rather than historical0.68.0. FlyGym2.1.0, MuJoCo3.9.0, NumPy2.5.4, core source digest and upstream body commit match. Despite the Numba patch difference, every recorded numerical observation array was exactly equal in both probes. This establishes these two replays, not universal cross-version reproducibility.

## Scientific consequence

The stronger host can reproduce the existing frozen baseline with the exact data/checkpoint. It does not establish training improvement, stable walking, retention or generalization. Both regimes remain stepping in place. Historical T1B/L2/S1 negative retention results still apply; no model/stage or validated-best promotion.

The next intervention must be a separately registered hypothesis with matched parent/control runs. Do not repeat the failed duration/LR/profile sweep as an assumed fix or use final seeds10/11/200/201/202 for development. Claude diagnostic reviewPR27 and sensory-hook proposal remain separate from the unmodified runtime.

## Evidence and publication

This folder contains the exact bounded runner, protocol and complete compact result with per-field comparisons and native summaries.
Local archive: frozen-baseline-b2-20261104-evidence.zip;10evidence files plus internal SHA256 manifest;1,870,230bytes; SHA2568419516829a5b630075ab46b9920cf92886a4f531d36dca45352932d6ffd94ae. ZIP CRC and every archived evidence file SHA were verified. Raw licensed connectome files, graph caches, environments and credentials are excluded.

**GitHub Release upload is pending.** The connector has no release-upload tool; installed GitHub CLI reports its account token invalid. No reauthentication, token extraction, public release or upload was performed. Code and compact results are committed; the verified full archive remains in this chat's outputs for transfer/publication. Do not claim a remotely verified artifact receipt.

Goal remains unmet: robust standing/balance, fly-like stable walking, environmental response and retained improvements.
