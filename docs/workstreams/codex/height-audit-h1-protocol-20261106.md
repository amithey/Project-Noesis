# H1: offline height/contact/coordinate audit

Prospective2026-11-06, issue18 claim6689926413. Codex new height_audit_h1.py,
test_height_audit_h1.py and H1docs, branch codex/height-audit-h1-20261106,
baseea6d06b55b8552a92aa871a816e172364bf0afc2. No shared metrics, core, learner,
reward, body/controller/default or peer worktree edited. No model construction,
neural/physics step, new training, threshold relaxation or success rescoring.

SR1found supported upright high-height failures in new births alongside an older
sustained upright/support collapse. H1 distinguishes what the recorded signals
can establish before changing training. It does not assume the height gate is
wrong or that supported samples establish balance/absence of flight between steps.

Inputs: ALL22SR1reported parent/child/shared-development/own-seed frozen traces,
plus both new25s training-observation banks for exact reward replay. Verify SR1
signed archive SHA256e23d61e932062a5172391232b2d6e2513b77c6cd011c494c6a50168614352941
and every selected file against its manifest, native20/core40 against TP2source
manifest, and hash actual upstream/analysis source. Upstream pinned
ca65a510c2afe6ac61c51df4f274c8d190c2f95f; the three consulted upstream files have
no tracked diff at that HEAD. Input/source identities rechecked after analysis.

Source interpretation to audit/document:
- Native position is selected thorax body `xpos`, global millimeters, not whole-
  body COM. FlatGroundWorld plane is at z=0. Absolute thorax z is therefore height
  above the floor in this experiment; no coordinate/scale substitution observed.
- Foot support is tarsus5XBODY-to-floor contact-found plus norm(force)>1e-5.
  The norm is not a signed vertical force. Feet positions are body-frame origins,
  not exact contact points or geometry clearance.
- Standing mask and native v7reward both use uprightness>=.7 and height.3..1.5
  with any loaded claw. Maintain that definition and all existing verdicts.

For each trace retain50ms warmup/5s horizon and exactly recompute original failure
mask. Analyze high-height-only frames(all other standing gates true): height
excess quantiles, supporting-claw counts, uprightness, backward5ms finite-
difference vertical thorax speed, complete consecutive-bout durations. No fitted
stationarity threshold, support-polygon/COM certification or inferred penetration
from foot origins. Report raw quantiles and all cases, not a selected snapshot.

Evaluate existing v7score on each real transition from measured rows10..999.
For every high-only sample evaluate the SAME scorer at thorax z=1.5+/-1e-6mm,
holding all other inputs fixed; report bounded reward discontinuity. This is an
artificial scorer-input boundary probe, NOT a simulated physical transition or
changed reward. Recompute4999recorded reward rows in EACH new training bank
(first row lacks its initial before-observation) and require exact equality.

Limits:5ms contact snapshots cannot rule out intervening flight; recorded42active
joint angles omit passive tarsus states; no full solver state, actual contact
points/normals or whole-body COM stored. No exact pose replay or final conclusion
about hopping/overextension/biological standing from these data alone.

Budget180s contained offline worker, canonical atomic host lease, fresh>=2GiB RAM
and no competing Python. No heavy simulation allocated. Stop input/source/hash/
nonfinite/shape/original-mask/reward-consistency/timeout failure; no automatic retry
or retune. Preserve receipt, empty job tree and owned-lease cleanup. Three focused
tests pass0.08s including real native scorer boundary behavior and no input mutation;
source compiles. Publish before data analysis. Prepared command:
work/noesis-env/Scripts/python.exe work/height_audit_h1.py --root <workspace>.

Code/protocol/compact results in Git, selected signed local evidence; Release
upload pending authenticated asset upload. No promotion or walking claim. Full
retained-standing/balance/walking/environment-response goal remains active.
