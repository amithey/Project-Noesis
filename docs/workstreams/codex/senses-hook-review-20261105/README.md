# Review of the thermosensory hook

Scope: offline review of Claude's proposal at d2160a9 and Gemini's manifest at 2c3dca9. Run `audit.py` from the repository with both commits available. Budget: 60 seconds, no simulation.

The controller patch passes `git apply --check` on the Codex pilot branch at 0e56022. Keep the interface proposal under review until the control isolation defect in the caller is repaired. The default hook returns an empty list, and the patch routes the additional parts through the existing overlap-checking merge.

The committed `_extra_drive_parts` eagerly evaluates every value in the mode dictionary. The exact method body calls the field in all four modes and raises on a missing position even in off, zero and constant. The probe uses a spy field and a stub adapter to isolate control flow. This is a software defect, not evidence of divergence in an ordinary physical rollout with valid position.

Use lazy branch selection and add a regression with a field that raises when called in each control mode. Compare a patched default controller to a pristine controller, then compare off and zero to the patched default. The supplied tests compare the latter paths but the plain brain already uses the patched controller. Their spike observable is a total over all neurons. Record the injected afferent indices, rates and target-cell spikes separately before concluding that the selected afferents respond.

Before physical allocation, pin the geometry units, starting field values and dose controls. A fixed constant is an informative matched control only for the registered contrast and documented exposure. Report exposure and observed response, including cases where the active field barely changes. Keep this sensory intervention separate from load calibration and learning, with the shared completed parent and development seeds.

All six Gemini manifest entries match exact Git bytes and the declared CRLF transforms. Their working-copy lengths match the corresponding declared variant. This verifies file identity. The S1 report's causal wording and the distinction between aggregate case summaries and raw replay remain open.
